"""
Physics Engine for Differentiable Mechanical Differential Analyzer (D-MDA)
Phase 1: Physics Environment Setup

This module implements:
- Rigid body state management with gradient tracking
- Semi-implicit Euler time integration
- Constraint solver for mechanical connections
- Differentiable physics (backpropagation through time)
"""

import taichi as ti
import numpy as np
import config

# Initialize Taichi with gradient support
ti.init(
    arch=ti.cpu,  # Use ti.cuda if GPU available
    default_fp=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    debug=False
)

# =============================================================================
# STATE FIELDS (Differentiable Tensors)
# =============================================================================
# These fields store the physical state of all rigid bodies
# needs_grad=True enables automatic differentiation through these values

# Number of bodies from configuration
N = config.NUM_BODIES

# Position: 2D vector [x, y] for each body
# Shape: (N,) where each element is a 2D vector
pos = ti.Vector.field(
    n=2,                    # 2D: x and y coordinates
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True         # CRITICAL: Enables backpropagation
)

# Rotation angle (radians) for each body
# Positive = counterclockwise rotation
angle = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True
)

# Linear velocity: 2D vector [vx, vy]
vel = ti.Vector.field(
    n=2,
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True
)

# Angular velocity (radians/second)
# Positive = counterclockwise rotation
ang_vel = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True
)

# Mass for each body (scalar)
mass = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=False  # Mass typically constant
)

# Moment of inertia for each body (scalar)
moment_of_inertia = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=False
)

# Radius for visualization and collision (scalar)
radius = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=False
)

# =============================================================================
# FORCE ACCUMULATORS
# =============================================================================
# These store forces/torques applied during each time step

# External forces (reset each step)
force = ti.Vector.field(
    n=2,
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True
)

# External torques (reset each step)
torque = ti.field(
    dtype=ti.f64 if config.FLOAT_TYPE == "f64" else ti.f32,
    shape=N,
    needs_grad=True
)

# =============================================================================
# INITIALIZATION
# =============================================================================

@ti.kernel
def initialize_bodies():
    """
    Initialize all rigid bodies to their starting configuration.
    Uses values from config.py.
    """
    # Body 0: Main disk
    pos[0] = ti.Vector(config.INITIAL_POSITIONS[0])
    angle[0] = 0.0
    vel[0] = ti.Vector([0.0, 0.0])
    ang_vel[0] = config.INITIAL_ANGULAR_VELOCITIES[0]
    mass[0] = config.DISK_MASS
    moment_of_inertia[0] = config.DISK_MOMENT_OF_INERTIA
    radius[0] = config.DISK_RADIUS
    force[0] = ti.Vector([0.0, 0.0])
    torque[0] = 0.0
    
    # Body 1: Wheel
    pos[1] = ti.Vector(config.INITIAL_POSITIONS[1])
    angle[1] = 0.0
    vel[1] = ti.Vector([0.0, 0.0])
    ang_vel[1] = config.INITIAL_ANGULAR_VELOCITIES[1]
    mass[1] = config.WHEEL_MASS
    moment_of_inertia[1] = config.WHEEL_MOMENT_OF_INERTIA
    radius[1] = config.WHEEL_RADIUS
    force[1] = ti.Vector([0.0, 0.0])
    torque[1] = 0.0
    
    # Body 2: Output shaft
    pos[2] = ti.Vector(config.INITIAL_POSITIONS[2])
    angle[2] = 0.0
    vel[2] = ti.Vector([0.0, 0.0])
    ang_vel[2] = config.INITIAL_ANGULAR_VELOCITIES[2]
    mass[2] = config.SHAFT_MASS
    moment_of_inertia[2] = config.SHAFT_MOMENT_OF_INERTIA
    radius[2] = config.SHAFT_RADIUS
    force[2] = ti.Vector([0.0, 0.0])
    torque[2] = 0.0


def reset_simulation():
    """
    Reset all state to initial conditions.
    Call this to restart the simulation.
    """
    # Clear all gradients first
    pos.grad.fill(0)
    angle.grad.fill(0)
    vel.grad.fill(0)
    ang_vel.grad.fill(0)
    force.grad.fill(0)
    torque.grad.fill(0)
    
    # Re-initialize bodies
    initialize_bodies()


# =============================================================================
# TIME INTEGRATION (Semi-Implicit Euler)
# =============================================================================
# This is the core physics loop. Semi-implicit Euler:
# 1. Update velocity using forces (v = v + a*dt)
# 2. Update position using new velocity (x = x + v*dt)
# More stable than explicit Euler for oscillatory systems

@ti.kernel
def integrate_step():
    """
    Single step of semi-implicit Euler integration.
    Updates velocities first, then positions.
    """
    dt = config.DT
    
    for i in range(N):
        # --- UPDATE LINEAR MOTION ---
        # F = ma, so a = F/m
        acceleration = force[i] / mass[i]
        
        # Apply damping (viscous friction)
        # F_damping = -c * v
        damping_force = -config.FRICTION_DAMPING * vel[i]
        acceleration += damping_force / mass[i]
        
        # Update velocity: v_new = v_old + a*dt
        vel[i] += acceleration * dt
        
        # Update position: x_new = x_old + v_new*dt
        # Using NEW velocity (semi-implicit)
        pos[i] += vel[i] * dt
        
        # --- UPDATE ANGULAR MOTION ---
        # tau = I*alpha, so alpha = tau/I
        angular_acceleration = torque[i] / moment_of_inertia[i]
        
        # Apply angular damping
        angular_damping = -config.FRICTION_DAMPING * ang_vel[i]
        angular_acceleration += angular_damping
        
        # Update angular velocity
        ang_vel[i] += angular_acceleration * dt
        
        # Update angle
        angle[i] += ang_vel[i] * dt
        
        # Keep angle in [0, 2*pi] for numerical stability
        # Using modulo for gradients (Taichi handles this)
        two_pi = 2.0 * ti.math.pi
        while angle[i] >= two_pi:
            angle[i] -= two_pi
        while angle[i] < 0.0:
            angle[i] += two_pi


@ti.kernel
def clear_forces():
    """Reset force accumulators to zero."""
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])
        torque[i] = 0.0


# =============================================================================
# CONSTRAINTS (Simplified for Phase 1)
# =============================================================================
# In full implementation, constraints enforce rolling without slipping
# For Phase 1, we provide the structure

@ti.kernel
def apply_wheel_disk_constraint():
    """
    Enforce rolling without slipping between wheel and disk.
    
    This is the key constraint for the differential analyzer:
    The wheel rolls on the disk surface without sliding.
    """
    # Simplified for Phase 1: Just keep wheel on disk surface
    # Full implementation would compute contact velocity and enforce zero slip
    
    disk_idx = 0
    wheel_idx = 1
    
    # Vector from disk center to wheel
    delta_pos = pos[wheel_idx] - pos[disk_idx]
    distance = delta_pos.norm()
    
    # Target distance (disk radius)
    target_dist = radius[disk_idx]
    
    # If wheel is too far or too close, project it to surface
    if distance > 0.001:  # Avoid division by zero
        direction = delta_pos / distance
        # Position correction (Baumgarte stabilization)
        error = distance - target_dist
        pos[wheel_idx] = pos[disk_idx] + direction * target_dist


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def step():
    """
    Advance simulation by one time step.
    This is the function called in the main loop.
    """
    # Clear forces from previous step
    clear_forces()
    
    # Apply any external forces/torques here
    # (Phase 1: minimal external forces, just initial spin)
    
    # Apply constraints
    apply_wheel_disk_constraint()
    
    # Integrate physics
    integrate_step()


def get_state():
    """
    Extract current state as NumPy arrays for analysis/visualization.
    Returns dictionary with all state variables.
    """
    return {
        'pos': pos.to_numpy(),
        'angle': angle.to_numpy(),
        'vel': vel.to_numpy(),
        'ang_vel': ang_vel.to_numpy(),
        'mass': mass.to_numpy(),
        'radius': radius.to_numpy(),
        'moment_of_inertia': moment_of_inertia.to_numpy(),
    }


def set_angular_velocity(body_idx: int, omega: float):
    """
    Manually set angular velocity of a specific body.
    Useful for applying control inputs.
    """
    ang_vel[body_idx] = omega


def apply_torque(body_idx: int, tau: float):
    """
    Apply external torque to a body.
    """
    torque[body_idx] += tau
