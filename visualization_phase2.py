"""
Phase 2 Visualization - ODE Integration Visualization
D-MDA: Differentiable Mechanical Differential Analyzer

Visualizes the differential analyzer's integration performance
compared to analytical ODE solution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import config


def visualize_ode_integration(state_history, save_path='phase2_ode_comparison.png'):
    """
    Create comprehensive visualization comparing differential analyzer
    output to analytical ODE solution.
    
    Args:
        state_history: List of state dictionaries from simulation
        save_path: Where to save the plot
        
    Returns:
        matplotlib figure
    """
    # Extract data
    times = [s['time'] for s in state_history]
    target_integrals = [s['target_integral'] for s in state_history]
    actual_integrals = [s['actual_integral'] for s in state_history]
    losses = [s['loss'] for s in state_history]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: ODE Solution Comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, target_integrals, 'b-', linewidth=2, label='Target: 1 - cos(t)')
    ax1.plot(times, actual_integrals, 'r--', linewidth=2, label='Differential Analyzer')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Integral Value', fontsize=12)
    ax1.set_title('ODE Integration: Target vs Actual', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Over Time
    ax2 = fig.add_subplot(gs[1, 0])
    errors = np.array(actual_integrals) - np.array(target_integrals)
    ax2.plot(times, errors, 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Integration Error', fontsize=12)
    ax2.set_title('Integration Error Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Over Time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(times, losses, 'm-', linewidth=1.5)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Loss (log scale)', fontsize=12)
    ax3.set_title('Loss Function Over Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mechanism Trajectory
    ax4 = fig.add_subplot(gs[2, 0])
    if len(state_history) > 0:
        # Get final state positions
        disk_pos = state_history[-1]['pos'][0]
        wheel_pos = state_history[-1]['pos'][1]
        
        # Plot disk
        disk = Circle(disk_pos, config.DISK_RADIUS, fill=False, 
                     color='blue', linewidth=2, label='Disk')
        ax4.add_patch(disk)
        
        # Plot wheel
        wheel = Circle(wheel_pos, config.WHEEL_RADIUS, fill=False,
                      color='red', linewidth=2, label='Wheel')
        ax4.add_patch(wheel)
        
        # Plot trajectory history
        disk_x = [s['pos'][0][0] for s in state_history[::10]]
        disk_y = [s['pos'][0][1] for s in state_history[::10]]
        wheel_x = [s['pos'][1][0] for s in state_history[::10]]
        wheel_y = [s['pos'][1][1] for s in state_history[::10]]
        
        ax4.plot(disk_x, disk_y, 'b-', alpha=0.3, label='Disk path')
        ax4.plot(wheel_x, wheel_y, 'r-', alpha=0.3, label='Wheel path')
        
        ax4.set_xlim(0, 2)
        ax4.set_ylim(0, 2)
        ax4.set_aspect('equal')
        ax4.set_xlabel('X Position (m)', fontsize=12)
        ax4.set_ylabel('Y Position (m)', fontsize=12)
        ax4.set_title('Mechanism Configuration', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Phase Space (Error vs Error Rate)
    ax5 = fig.add_subplot(gs[2, 1])
    if len(errors) > 1:
        error_rate = np.gradient(errors, times)
        ax5.plot(errors, error_rate, 'purple', linewidth=1, alpha=0.7)
        ax5.scatter(errors[::50], error_rate[::50], c=times[::50], 
                   cmap='viridis', s=20, alpha=0.6)
        ax5.set_xlabel('Error', fontsize=12)
        ax5.set_ylabel('Error Rate', fontsize=12)
        ax5.set_title('Error Phase Space', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.suptitle('D-MDA Phase 2: Differentiable ODE Integration', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def visualize_optimization_history(history, save_path='phase2_optimization.png'):
    """
    Visualize the optimization process.
    
    Args:
        history: Dictionary with 'loss', 'wheel_offset' lists
        save_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = range(len(history['loss']))
    
    # Plot 1: Loss over iterations
    ax = axes[0, 0]
    ax.semilogy(iterations, history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss During Optimization', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Wheel offset over iterations
    ax = axes[0, 1]
    ax.plot(iterations, history['wheel_offset'], 'g-', linewidth=2)
    ax.axhline(y=config.DISK_RADIUS * 0.5, color='r', linestyle='--', 
              alpha=0.5, label='Initial guess')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Wheel Offset (m)', fontsize=12)
    ax.set_title('Wheel Offset Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Loss vs Offset (trajectory)
    ax = axes[1, 0]
    scatter = ax.scatter(history['wheel_offset'], history['loss'], 
                        c=iterations, cmap='plasma', s=30, alpha=0.6)
    ax.plot(history['wheel_offset'], history['loss'], 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Wheel Offset (m)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Optimization Trajectory', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Iteration')
    
    # Plot 4: Gradient magnitude (approximate from loss change)
    ax = axes[1, 1]
    if len(history['loss']) > 1:
        # Approximate gradient magnitude from loss changes
        loss_changes = np.abs(np.diff(history['loss']))
        ax.semilogy(iterations[1:], loss_changes, 'm-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('|ΔLoss| (log scale)', fontsize=12)
        ax.set_title('Loss Change Rate', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('D-MDA Phase 2: Optimization Convergence', 
                fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def visualize_comparison(grid_results, grad_history, save_path='phase2_comparison.png'):
    """
    Compare grid search vs gradient descent.
    
    Args:
        grid_results: List of dicts from grid search
        grad_history: Dict from gradient descent
        save_path: Where to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid search results
    ax = axes[0]
    offsets = [r['offset'] for r in grid_results]
    losses = [r['loss'] for r in grid_results]
    ax.plot(offsets, losses, 'bo-', linewidth=2, markersize=8, label='Grid search')
    ax.set_xlabel('Wheel Offset (m)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Grid Search: Loss vs Offset', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gradient descent trajectory
    ax = axes[1]
    ax.plot(grad_history['wheel_offset'], grad_history['loss'], 
           'ro-', linewidth=2, markersize=6, label='Gradient descent')
    ax.set_xlabel('Wheel Offset (m)', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Gradient Descent: Loss vs Offset', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('D-MDA Phase 2: Optimization Methods Comparison', 
                fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig
