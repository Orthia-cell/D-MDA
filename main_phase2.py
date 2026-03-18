"""
Main Entry Point for D-MDA (Phase 1 + Phase 2)
Differentiable Mechanical Differential Analyzer

Usage:
    python main.py --phase 1              # Run Phase 1 demo
    python main.py --phase 2              # Run Phase 2 ODE integration
    python main.py --phase 2 --test       # Run Phase 2 tests
    python main.py --phase 2 --optimize   # Run optimization
    python main.py --phase 2 --compare    # Compare optimization methods
"""

import sys
import argparse
import numpy as np

import config


def run_phase1_demo(duration=10.0):
    """Run Phase 1 physics demo with original modules."""
    print("\n" + "="*60)
    print("D-MDA PHASE 1: Physics Environment Demo")
    print("="*60)
    
    import physics_engine as pe
    import visualization as vis
    
    pe.initialize_bodies()
    pe.ang_vel[0] = 1.0
    
    print(f"Running {duration}s simulation...")
    print("Controls: SPACE=pause, R=reset, T=trails, Q=quit")
    
    # Create visualizer
    visualizer = vis.Visualizer()
    
    step = 0
    max_steps = int(duration / config.DT)
    state_history = []
    
    running = True
    while running and step < max_steps:
        running = visualizer.handle_events()
        
        if not visualizer.paused:
            pe.step()
            step += 1
            
            if step % 10 == 0:
                state_history.append(pe.get_state())
        
        state = pe.get_state()
        visualizer.render(state)
    
    visualizer.close()
    
    print(f"\nCompleted {step} steps")
    print(f"Final disk angle: {state['angle'][0]:.4f} rad")
    
    if state_history:
        vis.visualize_matplotlib(
            state_history,
            save_path="phase1_results.png"
        )
    
    return state_history


def run_phase2_simulation(duration=10.0):
    """Run Phase 2 ODE integration simulation."""
    print("\n" + "="*60)
    print("D-MDA PHASE 2: ODE Integration Demo")
    print("="*60)
    print(f"Target ODE: dy/dt = sin(t)")
    print(f"Solution: y(t) = 1 - cos(t)")
    print("="*60)
    
    import physics_engine_phase2 as pe
    import visualization_phase2 as vis2
    
    # Initialize
    pe.reset_simulation()
    pe.ang_vel[0] = 1.0  # Spin disk at 1 rad/s
    
    print(f"\nRunning {duration}s simulation...")
    
    num_steps = int(duration / config.DT)
    state_history = []
    
    for step in range(num_steps):
        pe.step()
        
        # Record every 100 steps
        if step % 100 == 0:
            state_history.append(pe.get_state())
        
        # Progress
        if step % 1000 == 0:
            t = step * config.DT
            loss = pe.loss.to_numpy()
            print(f"  t={t:.2f}s, loss={loss:.6e}")
    
    # Final state
    final_state = pe.get_state()
    print(f"\nFinal Results:")
    print(f"  Time: {final_state['time']:.4f} s")
    print(f"  Target integral: {final_state['target_integral']:.6f}")
    print(f"  Actual integral: {final_state['actual_integral']:.6f}")
    print(f"  Error: {abs(final_state['actual_integral'] - final_state['target_integral']):.6f}")
    print(f"  Constraint violation: {final_state['constraint_violation_pos']:.6e}")
    
    # Visualize
    print("\nGenerating visualizations...")
    vis2.visualize_ode_integration(state_history, 'phase2_ode_integration.png')
    
    return state_history


def run_phase2_tests():
    """Run Phase 2 validation tests."""
    import test_phase2
    success = test_phase2.run_all_tests()
    return success


def run_phase2_optimization():
    """Run gradient-based optimization."""
    import optimization as opt
    
    optimizer = opt.MechanismOptimizer(
        learning_rate=0.05,
        max_iterations=50
    )
    
    best_offset, best_loss, history = optimizer.optimize(
        duration=5.0,
        verbose=True
    )
    
    # Visualize optimization
    import visualization_phase2 as vis2
    vis2.visualize_optimization_history(history, 'phase2_optimization.png')
    
    return best_offset, best_loss, history


def run_phase2_comparison():
    """Compare grid search vs gradient descent."""
    import optimization as opt
    results = opt.compare_optimization_methods(duration=3.0)
    
    # Visualize comparison
    import visualization_phase2 as vis2
    grid_offset, grid_loss, grid_results = results['grid']
    grad_offset, grad_loss, grad_history = results['gradient']
    
    vis2.visualize_comparison(
        grid_results, 
        grad_history,
        'phase2_method_comparison.png'
    )
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Differentiable Mechanical Differential Analyzer"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Which phase to run (1 or 2, default: 2)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run validation tests"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization (Phase 2 only)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare optimization methods (Phase 2 only)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.phase == 1:
        # Phase 1 operations
        if args.test:
            import test_gradients
            success = test_gradients.run_all_tests()
            sys.exit(0 if success else 1)
        else:
            run_phase1_demo(args.duration)
    
    else:
        # Phase 2 operations
        if args.test:
            success = run_phase2_tests()
            sys.exit(0 if success else 1)
        
        elif args.optimize:
            run_phase2_optimization()
        
        elif args.compare:
            run_phase2_comparison()
        
        else:
            # Default: run simulation
            run_phase2_simulation(args.duration)


if __name__ == "__main__":
    main()
