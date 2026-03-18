"""
Configuration file for Differentiable Mechanical Differential Analyzer (D-MDA)
Phase 1: Physics Environment Setup

This file contains all hyperparameters and constants for the simulation.
Modify values here to adjust simulation behavior without changing core code.
"""

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Time step for integration (seconds)
# Smaller = more accurate but slower simulation
# Larger = faster but may become unstable
DT = 0.001  # 1 millisecond steps (1000 steps per simulated second)

# Total simulation time (seconds)
SIMULATION_TIME = 10.0  # Run for 10 seconds by default

# Number of rigid bodies in the simulation
# Phase 1: 3 bodies (disk, wheel, output shaft placeholders)
NUM_BODIES = 3

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

# Gravity (m/s^2)
# Set to 0 for 2D top-down view where gravity doesn't act
GRAVITY = 0.0

# Friction coefficients
# Coulomb friction: static threshold before motion begins
FRICTION_STATIC = 0.1

# Viscous friction: proportional to velocity (damping)
FRICTION_DAMPING = 0.01

# Rolling friction: resistance to rolling motion
FRICTION_ROLLING = 0.001

# =============================================================================
# RIGID BODY GEOMETRY
# =============================================================================

# Disk parameters (the main integrator disk)
DISK_RADIUS = 0.5  # meters
DISK_MASS = 1.0    # kg
DISK_MOMENT_OF_INERTIA = 0.5 * DISK_MASS * DISK_RADIUS ** 2  # I = 0.5 * m * r^2

# Wheel parameters (the integrating wheel)
WHEEL_RADIUS = 0.05  # meters (10x smaller than disk)
WHEEL_MASS = 0.1     # kg
WHEEL_MOMENT_OF_INERTIA = 0.5 * WHEEL_MASS * WHEEL_RADIUS ** 2

# Output shaft parameters
SHAFT_RADIUS = 0.02  # meters
SHAFT_MASS = 0.05    # kg
SHAFT_MOMENT_OF_INERTIA = 0.5 * SHAFT_MASS * SHAFT_RADIUS ** 2

# =============================================================================
# CONSTRAINT PARAMETERS
# =============================================================================

# Stiffness of spring constraints (for constraint solver)
# Higher = more rigid constraints, but may cause numerical instability
CONSTRAINT_STIFFNESS = 1e4  # Reduced from 1e6

# Damping for constraint stabilization
CONSTRAINT_DAMPING = 0.1

# Baumgarte stabilization constant (prevents constraint drift)
# Typical range: 0.1 to 0.3
BAUMGARTE_ALPHA = 0.1  # Reduced from 0.2

# Position constraint solver iterations
POSITION_CONSTRAINT_ITERATIONS = 5

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Window size in pixels
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Virtual simulation space size (meters)
# The window shows this much physical space
SIMULATION_BOUNDS = 2.0  # 2 meters x 2 meters

# Color scheme
COLOR_BACKGROUND = 0x1a1a2e  # Dark blue
COLOR_DISK = 0xe94560       # Red-pink
COLOR_WHEEL = 0x0f3460      # Deep blue
COLOR_SHAFT = 0x533483      # Purple
COLOR_VELOCITY_LOW = 0x00ffff   # Cyan (slow)
COLOR_VELOCITY_HIGH = 0xff0000  # Red (fast)

# Circle rendering detail (number of segments)
CIRCLE_RESOLUTION = 50

# =============================================================================
# GRADIENT/LEARNING PARAMETERS
# =============================================================================

# Precision for gradient computation
# Use ti.f64 for higher precision in backpropagation
FLOAT_TYPE = "f64"  # "f32" or "f64"

# Maximum time steps for gradient checkpointing
# Longer sequences need checkpointing to manage memory
CHECKPOINT_INTERVAL = 100

# Gradient clipping (prevents exploding gradients)
GRADIENT_CLIP_VALUE = 1e6

# =============================================================================
# BODY TYPE INDICES
# =============================================================================

# Index mapping for the three bodies
DISK_INDEX = 0
WHEEL_INDEX = 1
SHAFT_INDEX = 2

# Initial positions (center of simulation space)
INITIAL_POSITIONS = [
    [1.0, 1.0],    # Disk at center
    [1.0, 1.5],    # Wheel at top of disk
    [1.5, 1.0],    # Shaft to the right
]

# Initial angular velocities (radians/second)
INITIAL_ANGULAR_VELOCITIES = [
    1.0,   # Disk spinning at 1 rad/s
    0.0,   # Wheel starts stationary
    0.0,   # Shaft starts stationary
]