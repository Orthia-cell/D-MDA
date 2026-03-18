"""
Phase 4 Visualization - Architecture Search Results
D-MDA: Differentiable Mechanical Differential Analyzer

Visualizes the evolutionary search process and discovered architectures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict
import json


def visualize_evolution(history: List[Dict], save_path: str = 'phase4_evolution.png'):
    """
    Visualize the evolutionary search progress.
    
    Args:
        history: List of generation statistics
        save_path: Where to save the plot
    """
    generations = [h['generation'] for h in history]
    best_losses = [h['best_loss'] for h in history]
    mean_losses = [-h['mean_fitness'] for h in history]
    worst_losses = [-h['worst_fitness'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss over generations
    ax = axes[0, 0]
    ax.semilogy(generations, best_losses, 'b-', linewidth=2, label='Best')
    ax.semilogy(generations, mean_losses, 'g--', linewidth=1.5, label='Mean')
    ax.semilogy(generations, worst_losses, 'r:', linewidth=1.5, label='Worst')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Loss Evolution Over Generations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Fitness distribution per generation
    ax = axes[0, 1]
    # Show best, mean, and worst as filled area
    ax.fill_between(generations, worst_losses, best_losses, alpha=0.3, color='blue')
    ax.plot(generations, best_losses, 'b-', linewidth=2, label='Best')
    ax.plot(generations, mean_losses, 'g-', linewidth=1.5, label='Mean')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Fitness Distribution (Best to Worst)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Convergence rate
    ax = axes[1, 0]
    if len(best_losses) > 1:
        improvements = np.diff(best_losses)
        ax.bar(generations[1:], improvements, color='purple', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Loss Improvement', fontsize=12)
        ax.set_title('Improvement per Generation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    if history:
        final_best = history[-1]['best_loss']
        initial_best = history[0]['best_loss']
        improvement = (initial_best - final_best) / initial_best * 100
        
        best_gene = history[-1]['best_gene']
        
        summary_text = f"""
ARCHITECTURE SEARCH RESULTS

Generations: {len(history)}
Population size: {len(history)}

Initial best loss: {initial_best:.6f}
Final best loss: {final_best:.6f}
Improvement: {improvement:.1f}%

BEST MECHANISM FOUND:
Number of wheels: {best_gene['num_wheels']}
Wheel offsets: {[f"{o:.3f}" for o in best_gene['wheel_offsets']]} m
Gear ratios: {[f"{r:.2f}" for r in best_gene['gear_ratios']]}
Connection type: {best_gene['connection_type']}

The evolutionary algorithm successfully
discovered a mechanism configuration
that minimizes integration error!
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Phase 4: Evolutionary Architecture Search', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def visualize_mechanism_comparison(
    mechanisms: List[Dict],
    save_path: str = 'phase4_mechanism_comparison.png'
):
    """
    Compare different discovered mechanisms.
    
    Args:
        mechanisms: List of mechanism data dicts
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    num_wheels = [m['gene']['num_wheels'] for m in mechanisms]
    losses = [m['loss'] for m in mechanisms]
    offsets = [m['gene']['wheel_offsets'][0] if m['gene']['wheel_offsets'] else 0 
               for m in mechanisms]
    
    # Plot 1: Loss vs Number of Wheels
    ax = axes[0, 0]
    wheel_counts = sorted(set(num_wheels))
    wheel_losses = {nw: [] for nw in wheel_counts}
    for nw, loss in zip(num_wheels, losses):
        wheel_losses[nw].append(loss)
    
    positions = []
    labels = []
    data = []
    for nw in wheel_counts:
        positions.append(nw)
        labels.append(f"{nw} wheel{'s' if nw > 1 else ''}")
        data.append(wheel_losses[nw])
    
    bp = ax.boxplot(data, positions=positions, widths=0.5)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Performance by Number of Wheels', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Loss vs Wheel Offset
    ax = axes[0, 1]
    scatter = ax.scatter(offsets, losses, c=num_wheels, cmap='viridis', 
                        s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Wheel Offset (m)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Performance vs Wheel Offset', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Wheels')
    
    # Plot 3: Top 5 mechanisms
    ax = axes[1, 0]
    sorted_mechs = sorted(mechanisms, key=lambda x: x['loss'])[:5]
    
    y_pos = np.arange(len(sorted_mechs))
    bars = ax.barh(y_pos, [m['loss'] for m in sorted_mechs], color='green', alpha=0.7)
    
    # Color best one differently
    bars[0].set_color('gold')
    bars[0].set_edgecolor('orange')
    bars[0].set_linewidth(2)
    
    labels = [f"{m['gene']['num_wheels']}w, "
              f"o={[f'{o:.2f}' for o in m['gene']['wheel_offsets']]}, "
              f"r={[f'{r:.1f}' for r in m['gene']['gear_ratios']]}"
              for m in sorted_mechs]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Loss', fontsize=12)
    ax.set_title('Top 5 Discovered Mechanisms', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Connection type distribution
    ax = axes[1, 1]
    conn_types = {}
    for m in mechanisms:
        ct = m['gene']['connection_type']
        conn_types[ct] = conn_types.get(ct, 0) + 1
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax.pie(conn_types.values(), labels=conn_types.keys(), autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Connection Type Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('Phase 4: Discovered Mechanism Analysis', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def visualize_best_mechanism_simulation(
    state_history: List[Dict],
    gene: Dict,
    save_path: str = 'phase4_best_mechanism.png'
):
    """
    Visualize the simulation of the best discovered mechanism.
    
    Args:
        state_history: Simulation state history
        gene: Gene of best mechanism
        save_path: Where to save the plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    times = [s['time'] for s in state_history]
    targets = [s['target_integral'] for s in state_history]
    actuals = [s['actual_integral'] for s in state_history]
    losses = [s['loss'] for s in state_history]
    
    # Plot 1: ODE Solution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, targets, 'b-', linewidth=2, label='Target: 1 - cos(t)')
    ax1.plot(times, actuals, 'r--', linewidth=2, label='Discovered Mechanism')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Integral Value', fontsize=12)
    ax1.set_title('ODE Integration: Target vs Discovered Mechanism', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2 = fig.add_subplot(gs[0, 2])
    errors = np.array(actuals) - np.array(targets)
    ax2.plot(times, errors, 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Integration Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mechanism schematic
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    
    # Draw disk
    disk = plt.Circle((0, 0), 0.5, fill=False, color='blue', linewidth=3, label='Disk')
    ax3.add_patch(disk)
    
    # Draw wheels
    for i, (offset, ratio) in enumerate(zip(gene['wheel_offsets'], gene['gear_ratios'])):
        # Position wheels around disk
        angle = 2 * np.pi * i / gene['num_wheels']
        wx = offset * np.cos(angle)
        wy = offset * np.sin(angle)
        
        wheel = plt.Circle((wx, wy), 0.05, fill=False, color='red', linewidth=2)
        ax3.add_patch(wheel)
        
        # Draw connection line
        ax3.plot([0, wx], [0, wy], 'k--', alpha=0.5, linewidth=1)
        
        # Label
        ax3.text(wx, wy + 0.15, f'r={ratio:.1f}', ha='center', fontsize=8)
    
    ax3.set_title('Discovered Mechanism Layout', fontsize=13, fontweight='bold')
    ax3.set_xlabel('X Position (m)', fontsize=11)
    ax3.set_ylabel('Y Position (m)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Configuration details
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    
    final_error = abs(actuals[-1] - targets[-1]) if actuals and targets else 0
    
    config_text = f"""
DISCOVERED MECHANISM CONFIGURATION

Architecture Parameters:
  Number of wheels: {gene['num_wheels']}
  Connection type: {gene['connection_type']}
  
Wheel Configurations:
"""
    for i, (offset, ratio) in enumerate(zip(gene['wheel_offsets'], gene['gear_ratios'])):
        config_text += f"  Wheel {i+1}: offset={offset:.3f}m, gear_ratio={ratio:.2f}\n"
    
    config_text += f"""
Performance:
  Final loss: {losses[-1]:.6f}
  Final error: {final_error:.6f}
  
This configuration was discovered through
evolutionary architecture search, not designed
by hand. The algorithm "learned" which
topology best solves the ODE!
    """
    
    ax4.text(0.1, 0.5, config_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Phase 4: Best Discovered Mechanism', 
                fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def load_and_visualize_results(json_path: str = 'phase4_architecture_search.json'):
    """
    Load results from JSON and create all visualizations.
    
    Args:
        json_path: Path to results JSON file
    """
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {json_path}")
    print(f"Generations: {len(results['generation_history'])}")
    print(f"Best loss: {results['best_loss']:.6f}")
    
    # Create visualizations
    visualize_evolution(results['generation_history'], 'phase4_evolution.png')
    
    if results['best_gene']:
        print(f"\nBest mechanism: {results['best_gene']}")
    
    print("\nVisualizations complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        load_and_visualize_results(sys.argv[1])
    else:
        print("Usage: python phase4_visualization.py <results_json>")
        print("  or run after architecture_search.py to visualize results")
