"""
Phase 2 Physics Engine - Differentiable Rolling Constraint
D-MDA: Differentiable Mechanical Differential Analyzer

Key addition: Proper rolling-without-slipping constraint implemented as
differentiable Taichi kernels for gradient-based optimization.

Rolling constraint physics:
- Wheel must stay on disk surface (position constraint)
- Relative velocity at contact point must be zero (velocity constraint)
  v_contact_disk + v_contact_wheel = 0
"""

import taichi as ti
import numpy as np
import config

# Initialize Taichi with gradient support
ti.init(
    arch=ti.cpu,
    default_fp=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    debug=False
)

# =============================================================================
# STATE FIELDS (Differentiable)
# =============================================================================
N = config.NUM_BODIES

pos = ti.Vector.field(n=2, dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                      shape=N, needs_grad=True)
angle = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                 shape=N, needs_grad=True)
vel = ti.Vector.field(n=2, dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                      shape=N, needs_grad=True)
ang_vel = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                   shape=N, needs_grad=True)
mass = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                shape=N, needs_grad=False)
moment_of_inertia = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                             shape=N, needs_grad=False)
radius = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                  shape=N, needs_grad=False)

# Force accumulators
force = ti.Vector.field(n=2, dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                        shape=N, needs_grad=True)
torque = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                  shape=N, needs_grad=True)

# =============================================================================
# CONSTRAINT PARAMETERS (NEW: Optimizable)
# =============================================================================
# Wheel offset from disk center - this is what we'll optimize
# The wheel can be placed at different radii on the disk
wheel_offset_distance = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                                 shape=(), needs_grad=True)

# Constraint violation tracking (for debugging/analysis)
constraint_violation_pos = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                                    shape=())
constraint_violation_vel = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                                    shape=())

# =============================================================================
# ODE TARGET SYSTEM (NEW)
# =============================================================================
# Target ODE: dy/dt = f(t, y)
# We'll try to make the differential analyzer solve this

# Time parameter for ODE
current_time = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                        shape=(), needs_grad=True)

# Target function evaluation
target_integral = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                           shape=(), needs_grad=True)
actual_integral = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                           shape=(), needs_grad=True)

# Loss accumulator
loss = ti.field(dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32, 
                shape=(), needs_grad=True)

# =============================================================================
# INITIALIZATION
# =============================================================================

@ti.kernel
def initialize_bodies():
    """Initialize all rigid bodies."""
    # Disk at center
    pos[0] = ti.Vector(config.INITIAL_POSITIONS[0])
    angle[0] = 0.0
    vel[0] = ti.Vector([0.0, 0.0])
    ang_vel[0] = config.INITIAL_ANGULAR_VELOCITIES[0]
    mass[0] = config.DISK_MASS
    moment_of_inertia[0] = config.DISK_MOMENT_OF_INERTIA
    radius[0] = config.DISK_RADIUS
    force[0] = ti.Vector([0.0, 0.0])
    torque[0] = 0.0
    
    # Wheel at configurable offset
    wheel_offset_distance[None] = config.DISK_RADIUS * 0.5  # Start at half radius
    pos[1] = ti.Vector([
        config.INITIAL_POSITIONS[0][0] + wheel_offset_distance[None],
        config.INITIAL_POSITIONS[0][1]
    ])
    angle[1] = 0.0
    vel[1] = ti.Vector([0.0, 0.0])
    ang_vel[1] = config.INITIAL_ANGULAR_VELOCITIES[1]
    mass[1] = config.WHEEL_MASS
    moment_of_inertia[1] = config.WHEEL_MOMENT_OF_INERTIA
    radius[1] = config.WHEEL_RADIUS
    force[1] = ti.Vector([0.0, 0.0])
    torque[1] = 0.0
    
    # Shaft
    pos[2] = ti.Vector(config.INITIAL_POSITIONS[2])
    angle[2] = 0.0
    vel[2] = ti.Vector([0.0, 0.0])
    ang_vel[2] = config.INITIAL_ANGULAR_VELOCITIES[2]
    mass[2] = config.SHAFT_MASS
    moment_of_inertia[2] = config.SHAFT_MOMENT_OF_INERTIA
    radius[2] = config.SHAFT_RADIUS
    force[2] = ti.Vector([0.0, 0.0])
    torque[2] = 0.0
    
    # Initialize time and tracking
    current_time[None] = 0.0
    target_integral[None] = 0.0
    actual_integral[None] = 0.0
    loss[None] = 0.0


def reset_simulation():
    """Reset all state to initial conditions."""
    # Clear gradients
    pos.grad.fill(0)
    angle.grad.fill(0)
    vel.grad.fill(0)
    ang_vel.grad.fill(0)
    force.grad.fill(0)
    torque.grad.fill(0)
    wheel_offset_distance.grad.fill(0)
    current_time.grad.fill(0)
    target_integral.grad.fill(0)
    actual_integral.grad.fill(0)
    loss.grad.fill(0)
    
    initialize_bodies()


# =============================================================================
# ODE TARGET FUNCTIONS (NEW)
# =============================================================================

@ti.func
def ode_rhs(t: ti.f64) -> ti.f64:
    """
    Right-hand side of target ODE: dy/dt = f(t)
    
    For Phase 2, we use a simple test function:
    dy/dt = sin(t)  -> solution should be y = -cos(t) + C
    
    This is easy to verify analytically.
    """
    return ti.sin(t)


@ti.func
def target_solution(t: ti.f64) -> ti.f64:
    """
    Analytical solution to the ODE for comparison.
    For dy/dt = sin(t), with y(0) = 0:
    y(t) = 1 - cos(t)
    """
    return 1.0 - ti.cos(t)


# =============================================================================
# ROLLING CONSTRAINT (NEW - PHASE 2 CORE)
# =============================================================================

@ti.func
def get_contact_point_disk(disk_pos: ti.template(), disk_angle: ti.f64, 
                           wheel_pos: ti.template()) -> ti.Vector:
    """
    Get the contact point on the disk surface.
    This is where the wheel touches the disk.
    """
    # Vector from disk center to wheel center
    to_wheel = wheel_pos - disk_pos
    dist = to_wheel.norm()
    
    # Contact point is on the disk surface in direction of wheel
    # Use ti.select to avoid conditional return
    direction = ti.Vector([1.0, 0.0])  # Default fallback
    valid_dist = dist > 1e-10
    
    # Compute direction only if valid distance
    inv_dist = 1.0 / (dist + 1e-10)  # Avoid division by zero
    dir_x = to_wheel[0] * inv_dist
    dir_y = to_wheel[1] * inv_dist
    
    direction = ti.Vector([dir_x, dir_y])
    
    # Contact point
    # The wheel sits on the disk surface at the current offset distance
    # NOT at the disk edge (config.DISK_RADIUS)
    contact = disk_pos + direction * dist  # dist is the actual offset distance
    
    # Fallback if at center (shouldn't normally happen)
    fallback = disk_pos + ti.Vector([config.DISK_RADIUS, 0.0])
    
    return ti.select(valid_dist, contact, fallback)


@ti.func
def get_contact_point_wheel(disk_pos: ti.template(), disk_angle: ti.f64,
                            wheel_pos: ti.template(), wheel_angle: ti.f64) -> ti.Vector:
    """
    Get the contact point on the wheel surface.
    This is where the wheel touches the disk (opposite side from disk center).
    """
    to_wheel = wheel_pos - disk_pos
    dist = to_wheel.norm()
    
    # Compute direction
    inv_dist = 1.0 / (dist + 1e-10)
    dir_x = to_wheel[0] * inv_dist
    dir_y = to_wheel[1] * inv_dist
    
    direction = ti.Vector([dir_x, dir_y])
    valid_dist = dist > 1e-10
    
    # Contact point on wheel surface, facing toward disk
    contact = wheel_pos - direction * config.WHEEL_RADIUS
    
    # Fallback
    fallback = wheel_pos - ti.Vector([config.WHEEL_RADIUS, 0.0])
    
    return ti.select(valid_dist, contact, fallback)


@ti.func
def velocity_at_point(body_idx: int, point: ti.template()) -> ti.Vector:
    """
    Calculate velocity of a specific point on a rigid body.
    
    v_point = v_center + omega x r
    where r is vector from center to point
    """
    r = point - pos[body_idx]
    # 2D cross product: omega x r = [-omega * r_y, omega * r_x]
    rotational_vel = ti.Vector([-ang_vel[body_idx] * r[1], 
                                 ang_vel[body_idx] * r[0]])
    return vel[body_idx] + rotational_vel


@ti.kernel
def compute_rolling_constraint_violation():
    """
    Compute how much the rolling constraint is violated.
    For debugging and loss calculation.
    """
    disk_idx = 0
    wheel_idx = 1
    
    # Get contact points
    contact_disk = get_contact_point_disk(pos[disk_idx], angle[disk_idx], pos[wheel_idx])
    contact_wheel = get_contact_point_wheel(pos[disk_idx], angle[disk_idx], 
                                            pos[wheel_idx], angle[wheel_idx])
    
    # Velocities at contact points
    v_disk = velocity_at_point(disk_idx, contact_disk)
    v_wheel = velocity_at_point(wheel_idx, contact_wheel)
    
    # For rolling without slipping: v_disk should equal v_wheel at contact
    # The relative velocity should be zero
    relative_vel = v_disk - v_wheel
    
    # Store violation magnitude
    constraint_violation_vel[None] = relative_vel.norm()


@ti.kernel
def apply_position_constraint():
    """
    Enforce position constraint: wheel stays on disk surface.
    Uses direct position projection with relaxation.
    """
    disk_idx = 0
    wheel_idx = 1
    
    # Vector from disk to wheel
    delta = pos[wheel_idx] - pos[disk_idx]
    dist = delta.norm()
    
    if dist > 1e-10:
        # Target: wheel center is at wheel_offset_distance from disk center
        # This is the optimizable parameter!
        target_dist = wheel_offset_distance[None]
        
        # Position error
        error = dist - target_dist
        
        # Direction to wheel
        direction = delta / dist
        
        # Relaxation factor (prevents overshooting)
        relaxation = 0.5  # Only correct 50% of error per iteration
        
        # Position correction (direct projection with relaxation)
        correction = direction * error * relaxation
        pos[wheel_idx] -= correction
        
        # Update velocity to match position change (soft constraint)
        # This prevents jitter but doesn't fight the position correction
        vel_correction = correction / config.DT * 0.1
        vel[wheel_idx] -= vel_correction
        
        constraint_violation_pos[None] = ti.abs(error)


@ti.kernel
def apply_velocity_constraint():
    """
    Simplified rolling without slipping constraint.
    
    Uses direct velocity projection (damping) instead of impulses.
    Much more stable for differentiable programming.
    """
    disk_idx = 0
    wheel_idx = 1
    
    # Contact point
    contact_point = get_contact_point_disk(pos[disk_idx], angle[disk_idx], pos[wheel_idx])
    
    # Current velocities at contact
    v_disk_contact = velocity_at_point(disk_idx, contact_point)
    v_wheel_contact = velocity_at_point(wheel_idx, contact_point)
    
    # Relative velocity (what causes slip)
    relative_vel = v_disk_contact - v_wheel_contact
    
    # Simplified approach: directly damp relative velocity toward zero
    # This is a soft constraint that gradually eliminates slip
    damping = 0.1  # Damping factor (0 to 1)
    
    # Compute correction
    correction = relative_vel * damping
    
    # Apply correction to both bodies (split evenly)
    # Disk loses some velocity, wheel gains some
    vel[disk_idx] -= correction * 0.5
    vel[wheel_idx] += correction * 0.5
    
    # Angular velocity correction using actual offset distance
    # NOT config.DISK_RADIUS - that was the bug!
    delta = pos[wheel_idx] - pos[disk_idx]
    r_offset = delta.norm()  # Actual distance from disk center to wheel center
    r_wheel = config.WHEEL_RADIUS
    
    # Surface velocities at contact point
    # Disk surface velocity at offset distance = ω_disk * r_offset
    # Wheel surface velocity = ω_wheel * r_wheel
    v_surface_disk = ang_vel[disk_idx] * r_offset
    v_surface_wheel = ang_vel[wheel_idx] * r_wheel
    
    # Target: surface velocities should match for rolling without slipping
    v_diff = v_surface_disk - v_surface_wheel
    
    # Apply damping to angular velocities
    # The correction distributes the velocity difference
    if r_offset > 1e-10:
        ang_vel[disk_idx] -= v_diff * damping * 0.5 / r_offset
    ang_vel[wheel_idx] += v_diff * damping * 0.5 / r_wheel
    
    # Track violation for debugging
    constraint_violation_vel[None] = relative_vel.norm()


# =============================================================================
# TIME INTEGRATION (REFACTORED FOR PHASE 2)
# =============================================================================

@ti.kernel
def clear_forces():
    """Reset force accumulators."""
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])
        torque[i] = 0.0


@ti.kernel
def integrate_step():
    """
    Semi-implicit Euler integration.
    Updates velocities first, then positions.
    """
    dt = config.DT
    
    for i in range(N):
        # Linear motion
        acceleration = force[i] / mass[i]
        damping_force = -config.FRICTION_DAMPING * vel[i]
        acceleration += damping_force / mass[i]
        
        vel[i] += acceleration * dt
        pos[i] += vel[i] * dt
        
        # Angular motion
        angular_acceleration = torque[i] / moment_of_inertia[i]
        angular_damping = -config.FRICTION_DAMPING * ang_vel[i]
        angular_acceleration += angular_damping
        
        ang_vel[i] += angular_acceleration * dt
        angle[i] += ang_vel[i] * dt
        
        # Normalize angle to [0, 2*pi] using modulo (avoids while loops)
        two_pi = 2.0 * ti.math.pi
        angle[i] = ti.math.mod(angle[i], two_pi)


@ti.kernel
def update_time_kernel():
    """Update simulation time (separate kernel for autodiff compatibility)."""
    current_time[None] += config.DT


@ti.kernel
def update_ode_kernel():
    """Update ODE tracking (separate kernel for autodiff compatibility)."""
    t = current_time[None]
    target_integral[None] = target_solution(t)
    actual_integral[None] = angle[1]  # Wheel rotation tracks integral


@ti.kernel
def compute_loss_kernel():
    """Compute L2 loss between target and actual integral."""
    diff = actual_integral[None] - target_integral[None]
    loss[None] = diff * diff


# =============================================================================
# MAIN SIMULATION STEP (KERNEL-BASED FOR GRADIENT TRACKING)
# =============================================================================

def step():
    """
    Advance simulation by one time step.
    All operations are kernel-based for full gradient tracking.
    """
    # Clear forces
    clear_forces()
    
    # Integrate physics first (move bodies)
    integrate_step()
    
    # Apply position constraint multiple times to enforce geometric constraints
    # This must happen AFTER integration to correct drift
    for _ in range(10):  # Multiple solver iterations
        apply_position_constraint()
    
    # Apply velocity constraint after position is fixed
    apply_velocity_constraint()
    
    # Update time and ODE tracking
    update_time_kernel()
    update_ode_kernel()
    
    # Compute loss
    compute_loss_kernel()
    
    # Track constraint violation
    compute_rolling_constraint_violation()


def step_with_gradients():
    """
    Advance simulation with gradient computation enabled.
    This is used during optimization.
    """
    with ti.ad.Tape(loss):
        step()


# =============================================================================
# STATE ACCESS
# =============================================================================

def get_state():
    """Extract current state as NumPy arrays."""
    return {
        'pos': pos.to_numpy(),
        'angle': angle.to_numpy(),
        'vel': vel.to_numpy(),
        'ang_vel': ang_vel.to_numpy(),
        'mass': mass.to_numpy(),
        'radius': radius.to_numpy(),
        'moment_of_inertia': moment_of_inertia.to_numpy(),
        'time': current_time.to_numpy(),
        'target_integral': target_integral.to_numpy(),
        'actual_integral': actual_integral.to_numpy(),
        'loss': loss.to_numpy(),
        'wheel_offset': wheel_offset_distance.to_numpy(),
        'constraint_violation_pos': constraint_violation_pos.to_numpy(),
        'constraint_violation_vel': constraint_violation_vel.to_numpy(),
    }


def set_angular_velocity(body_idx: int, omega: float):
    """Manually set angular velocity."""
    ang_vel[body_idx] = omega


def apply_torque(body_idx: int, tau: float):
    """Apply external torque."""
    torque[body_idx] += tau


def get_wheel_offset() -> float:
    """Get current wheel offset distance."""
    return wheel_offset_distance.to_numpy()


def set_wheel_offset(offset: float):
    """Set wheel offset distance (for optimization)."""
    wheel_offset_distance[None] = offset
    # Re-position wheel based on new offset
    pos[1] = [config.INITIAL_POSITIONS[0][0] + offset, config.INITIAL_POSITIONS[0][1]]


def reset_gradients():
    """Clear all gradients."""
    pos.grad.fill(0)
    angle.grad.fill(0)
    vel.grad.fill(0)
    ang_vel.grad.fill(0)
    force.grad.fill(0)
    torque.grad.fill(0)
    wheel_offset_distance.grad.fill(0)
    current_time.grad.fill(0)
    target_integral.grad.fill(0)
    actual_integral.grad.fill(0)
    loss.grad.fill(0)
