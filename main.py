"""
Main Entry Point for Differentiable Mechanical Differential Analyzer (D-MDA)
Phase 1: Physics Environment Setup

This is the main script to run the simulation.
Usage:
    python main.py              # Run with visualization
    python main.py --test       # Run validation tests
    python main.py --headless   # Run without GUI, save plots
"""

import sys
import argparse
import numpy as np

import config
import physics_engine
import visualization
import test_gradients


def run_simulation_with_visualization(duration_seconds=10.0):
    """
    Run the physics simulation with real-time visualization.
    
    Args:
        duration_seconds: How long to run the simulation
    """
    print("="*60)
    print("DIFFERENTIAL ANALYZER - Phase 1 Demo")
    print("="*60)
    print(f"Simulation time: {duration_seconds}s")
    print(f"Time step: {config.DT}s")
    print(f"Total steps: {int(duration_seconds / config.DT)}")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset simulation")
    print("  T - Toggle trails")
    print("  Q/ESC - Quit")
    print("="*60)
    
    # Initialize physics
    physics_engine.initialize_bodies()
    
    # Set initial spin on disk
    physics_engine.ang_vel[0] = 1.0  # 1 rad/s
    
    # Create visualizer
    vis = visualization.Visualizer()
    
    # Main simulation loop
    step = 0
    max_steps = int(duration_seconds / config.DT)
    state_history = []
    
    running = True
    while running and step < max_steps:
        # Handle user input
        running = vis.handle_events()
        
        # Step physics (if not paused)
        if not vis.paused:
            physics_engine.step()
            step += 1
            
            # Record state every 10 steps
            if step % 10 == 0:
                state = physics_engine.get_state()
                state_history.append(state)
        
        # Render
        state = physics_engine.get_state()
        vis.render(state)
    
    # Cleanup
    vis.close()
    
    print(f"\nSimulation completed: {step} steps")
    print(f"Final disk angle: {state['angle'][0]:.4f} rad")
    print(f"Final disk velocity: {state['ang_vel'][0]:.4f} rad/s")
    
    # Save analysis plot
    if len(state_history) > 0:
        print("\nGenerating analysis plots...")
        fig = visualization.visualize_matplotlib(
            state_history,
            save_path="/root/.openclaw/workspace/differential_analyzer_env/phase1_results.png"
        )
        print("Saved: phase1_results.png")
    
    return state_history


def run_headless_simulation(duration_seconds=10.0):
    """
    Run simulation without GUI (for servers/background execution).
    
    Args:
        duration_seconds: How long to run
    """
    print("Running headless simulation...")
    
    # Initialize
    physics_engine.initialize_bodies()
    physics_engine.ang_vel[0] = 1.0
    
    # Run
    max_steps = int(duration_seconds / config.DT)
    state_history = []
    
    for step in range(max_steps):
        physics_engine.step()
        
        # Record every 100 steps
        if step % 100 == 0:
            state_history.append(physics_engine.get_state())
        
        # Progress report
        if step % 1000 == 0:
            print(f"Step {step}/{max_steps}")
    
    # Save plot
    print("Generating plots...")
    visualization.visualize_matplotlib(
        state_history,
        save_path="/root/.openclaw/workspace/differential_analyzer_env/phase1_headless.png"
    )
    print("Saved: phase1_headless.png")
    
    return state_history


def main():
    """Main entry point with command-line argument handling."""
    parser = argparse.ArgumentParser(
        description="Differentiable Mechanical Differential Analyzer - Phase 1"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run validation tests instead of simulation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for servers)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run validation suite
        success = test_gradients.run_all_tests()
        sys.exit(0 if success else 1)
    
    elif args.headless:
        # Run without GUI
        run_headless_simulation(args.duration)
    
    else:
        # Run with visualization
        try:
            run_simulation_with_visualization(args.duration)
        except Exception as e:
            print(f"\nError: {e}")
            print("\nIf visualization failed, try running with --headless")
            sys.exit(1)


if __name__ == "__main__":
    main()
