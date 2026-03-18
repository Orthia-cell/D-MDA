"""
Visualization Module for Differentiable Mechanical Differential Analyzer (D-MDA)
Phase 1: Physics Environment Setup

This module provides real-time visualization of the physics simulation:
- Top-down view of rotating disks
- Color-coded angular velocity
- Real-time GUI with pause functionality
"""

import taichi as ti
import numpy as np
import config
import physics_engine

# =============================================================================
# COLOR INTERPOLATION
# =============================================================================

def velocity_to_color(ang_vel):
    """
    Map angular velocity to a color between blue (slow) and red (fast).
    
    Args:
        ang_vel: Angular velocity in radians/second
    
    Returns:
        Hex color code
    """
    # Normalize velocity to [0, 1] range
    # Assume max velocity of 10 rad/s for full red
    max_vel = 10.0
    t = min(abs(ang_vel) / max_vel, 1.0)
    
    # Interpolate between cyan (slow) and red (fast)
    # Cyan: 0x00ffff (R=0, G=255, B=255)
    # Red: 0xff0000 (R=255, G=0, B=0)
    
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    b = int(255 * (1.0 - t))
    
    return (r << 16) | (g << 8) | b


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class Visualizer:
    """
    Real-time visualization using Taichi's GUI system.
    """
    
    def __init__(self):
        """Initialize visualization window."""
        # Create GUI window
        self.gui = ti.GUI(
            title="Differential Analyzer - Phase 1",
            res=(config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            background_color=config.COLOR_BACKGROUND
        )
        
        # Simulation bounds
        self.bounds = config.SIMULATION_BOUNDS
        
        # Control flags
        self.paused = False
        self.show_trails = True
        self.trail_history = [[] for _ in range(config.NUM_BODIES)]
        self.max_trail_length = 100
        
    def world_to_screen(self, pos):
        """
        Convert world coordinates to screen coordinates.
        
        World: (0,0) to (bounds, bounds)
        Screen: (0,0) to (WINDOW_WIDTH, WINDOW_HEIGHT)
        """
        x = pos[0] / self.bounds * config.WINDOW_WIDTH
        y = pos[1] / self.bounds * config.WINDOW_HEIGHT
        return np.array([x, y])
    
    def draw_body(self, idx, pos, angle, radius, ang_vel, color):
        """
        Draw a single rigid body (disk/wheel/shaft).
        
        Args:
            idx: Body index
            pos: 2D position [x, y]
            angle: Rotation angle (radians)
            radius: Body radius
            ang_vel: Angular velocity (for color)
            color: Base color
        """
        # Convert to screen coordinates
        screen_pos = self.world_to_screen(pos)
        screen_radius = radius / self.bounds * config.WINDOW_WIDTH
        
        # Determine color based on angular velocity
        if abs(ang_vel) > 0.1:  # Only color if spinning
            display_color = velocity_to_color(ang_vel)
        else:
            display_color = color
        
        # Draw main circle
        self.gui.circle(
            pos=screen_pos,
            radius=screen_radius,
            color=display_color
        )
        
        # Draw rotation indicator (line showing orientation)
        # Line extends from center to edge at current angle
        end_x = screen_pos[0] + screen_radius * np.cos(angle)
        end_y = screen_pos[1] + screen_radius * np.sin(angle)
        
        self.gui.line(
            begin=screen_pos,
            end=np.array([end_x, end_y]),
            radius=2,
            color=0xffffff  # White line
        )
        
        # Draw center dot
        self.gui.circle(
            pos=screen_pos,
            radius=3,
            color=0xffffff
        )
    
    def draw_labels(self, state):
        """Draw text labels showing current state."""
        # Get angular velocities
        ang_vels = state['ang_vel']
        angles = state['angle']
        
        # Label for disk
        disk_text = f"Disk: ω={ang_vels[0]:.2f} rad/s, θ={angles[0]:.2f} rad"
        self.gui.text(
            content=disk_text,
            pos=[0.02, 0.95],
            font_size=20,
            color=0xffffff
        )
        
        # Label for wheel
        wheel_text = f"Wheel: ω={ang_vels[1]:.2f} rad/s, θ={angles[1]:.2f} rad"
        self.gui.text(
            content=wheel_text,
            pos=[0.02, 0.90],
            font_size=20,
            color=0xffffff
        )
        
        # Label for shaft
        shaft_text = f"Shaft: ω={ang_vels[2]:.2f} rad/s, θ={angles[2]:.2f} rad"
        self.gui.text(
            content=shaft_text,
            pos=[0.02, 0.85],
            font_size=20,
            color=0xffffff
        )
        
        # Instructions
        self.gui.text(
            content="Controls: SPACE=Pause, R=Reset, Q=Quit",
            pos=[0.02, 0.05],
            font_size=16,
            color=0xaaaaaa
        )
        
        if self.paused:
            self.gui.text(
                content="PAUSED",
                pos=[0.45, 0.5],
                font_size=30,
                color=0xff0000
            )
    
    def update_trails(self, state):
        """Update position trails for visualization."""
        if not self.show_trails:
            return
        
        positions = state['pos']
        for i in range(config.NUM_BODIES):
            self.trail_history[i].append(positions[i].copy())
            if len(self.trail_history[i]) > self.max_trail_length:
                self.trail_history[i].pop(0)
    
    def draw_trails(self):
        """Draw motion trails."""
        if not self.show_trails:
            return
        
        colors = [config.COLOR_DISK, config.COLOR_WHEEL, config.COLOR_SHAFT]
        
        for i in range(config.NUM_BODIES):
            if len(self.trail_history[i]) < 2:
                continue
            
            # Draw trail as fading line
            for j in range(len(self.trail_history[i]) - 1):
                alpha = j / len(self.trail_history[i])  # Fade older points
                p1 = self.world_to_screen(self.trail_history[i][j])
                p2 = self.world_to_screen(self.trail_history[i][j + 1])
                
                # Draw faint trail
                self.gui.line(
                    begin=p1,
                    end=p2,
                    radius=1,
                    color=colors[i]
                )
    
    def render(self, state):
        """
        Render one frame of the simulation.
        
        Args:
            state: Dictionary from physics_engine.get_state()
        """
        # Clear and set background
        self.gui.clear(config.COLOR_BACKGROUND)
        
        # Update trails
        if not self.paused:
            self.update_trails(state)
        
        # Draw trails
        self.draw_trails()
        
        # Draw connections between bodies (mechanical linkages)
        positions = state['pos']
        self.gui.line(
            begin=self.world_to_screen(positions[0]),
            end=self.world_to_screen(positions[1]),
            radius=1,
            color=0x555555  # Gray line showing wheel on disk
        )
        
        # Draw each body
        colors = [config.COLOR_DISK, config.COLOR_WHEEL, config.COLOR_SHAFT]
        for i in range(config.NUM_BODIES):
            self.draw_body(
                idx=i,
                pos=state['pos'][i],
                angle=state['angle'][i],
                radius=state['radius'][i],
                ang_vel=state['ang_vel'][i],
                color=colors[i]
            )
        
        # Draw labels
        self.draw_labels(state)
        
        # Show frame
        self.gui.show()
    
    def handle_events(self):
        """
        Handle user input events.
        
        Returns:
            bool: False if user wants to quit, True otherwise
        """
        for e in self.gui.get_events():
            if e.key == ti.GUI.SPACE:
                self.paused = not self.paused
                print(f"Simulation {'paused' if self.paused else 'resumed'}")
            
            elif e.key == ti.GUI.ESCAPE or e.key == 'q':
                print("Quit requested")
                return False
            
            elif e.key == 'r':
                print("Reset requested")
                physics_engine.reset_simulation()
                self.trail_history = [[] for _ in range(config.NUM_BODIES)]
            
            elif e.key == 't':
                self.show_trails = not self.show_trails
                print(f"Trails: {'on' if self.show_trails else 'off'}")
        
        return True
    
    def close(self):
        """Close visualization window."""
        self.gui.close()


# =============================================================================
# SIMPLE VISUALIZATION (No GUI, for headless testing)
# =============================================================================

def visualize_matplotlib(state_history, save_path=None):
    """
    Create static plots using matplotlib.
    Useful for analysis and saving results.
    
    Args:
        state_history: List of state dictionaries over time
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Extract time series
    num_steps = len(state_history)
    time = np.arange(num_steps) * config.DT
    
    angles = np.array([s['angle'] for s in state_history])
    ang_vels = np.array([s['ang_vel'] for s in state_history])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Angular positions over time
    ax = axes[0, 0]
    ax.plot(time, angles[:, 0], label='Disk', color='#e94560')
    ax.plot(time, angles[:, 1], label='Wheel', color='#0f3460')
    ax.plot(time, angles[:, 2], label='Shaft', color='#533483')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Angular Position')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Angular velocities
    ax = axes[0, 1]
    ax.plot(time, ang_vels[:, 0], label='Disk', color='#e94560')
    ax.plot(time, ang_vels[:, 1], label='Wheel', color='#0f3460')
    ax.plot(time, ang_vels[:, 2], label='Shaft', color='#533483')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Phase space (angle vs velocity) for disk
    ax = axes[1, 0]
    ax.plot(angles[:, 0], ang_vels[:, 0], color='#e94560')
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Disk Phase Space')
    ax.grid(True)
    
    # Plot 4: Trajectories in 2D space
    ax = axes[1, 1]
    positions = np.array([s['pos'] for s in state_history])
    ax.plot(positions[:, 0, 0], positions[:, 0, 1], label='Disk', color='#e94560')
    ax.plot(positions[:, 1, 0], positions[:, 1, 1], label='Wheel', color='#0f3460')
    ax.plot(positions[:, 2, 0], positions[:, 2, 1], label='Shaft', color='#533483')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Body Trajectories')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    return fig
