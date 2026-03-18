"""
Design of Experiments (DoE) for Rolling Constraint Parameter Tuning
D-MDA Phase 2 - Numerical Stability Investigation

Systematically tests combinations of constraint parameters to find
stable configurations that prevent NaN/Inf explosions.
"""

import numpy as np
import itertools
import json
from datetime import datetime
import physics_engine_phase2 as pe
import config


class ConstraintParameterDoE:
    """
    Design of Experiments for constraint solver tuning.
    
    Tests combinations of:
    - Constraint stiffness (how strongly to enforce constraints)
    - Constraint damping (velocity correction factor)
    - Baumgarte stabilization (position drift correction)
    - Impulse scaling (if needed)
    """
    
    def __init__(self):
        # Parameter ranges to test
        # Using logarithmic spacing for stiffness (varies over orders of magnitude)
        self.param_ranges = {
            'CONSTRAINT_STIFFNESS': np.logspace(3, 8, 6),  # 1e3 to 1e8
            'CONSTRAINT_DAMPING': [0.01, 0.05, 0.1, 0.2, 0.5],  # 0.01 to 0.5
            'BAUMGARTE_ALPHA': [0.05, 0.1, 0.2, 0.3, 0.5],  # 0.05 to 0.5
        }
        
        self.results = []
        
    def run_single_test(self, stiffness, damping, baumgarte, duration=0.5):
        """
        Run a single simulation with given parameters.
        
        Returns:
            dict with success flag and metrics
        """
        # Temporarily override config values
        original_stiffness = config.CONSTRAINT_STIFFNESS
        original_damping = config.CONSTRAINT_DAMPING
        original_baumgarte = config.BAUMGARTE_ALPHA
        
        config.CONSTRAINT_STIFFNESS = stiffness
        config.CONSTRAINT_DAMPING = damping
        config.BAUMGARTE_ALPHA = baumgarte
        
        try:
            # Reset and initialize
            pe.reset_simulation()
            pe.ang_vel[0] = 1.0
            
            num_steps = int(duration / config.DT)
            max_loss = 0.0
            nan_detected = False
            constraint_violations = []
            
            for step in range(num_steps):
                pe.step()
                
                state = pe.get_state()
                loss = state['loss']
                
                # Check for NaN/Inf
                if np.isnan(loss) or np.isinf(loss):
                    nan_detected = True
                    break
                
                max_loss = max(max_loss, loss)
                constraint_violations.append(state['constraint_violation_pos'])
                
                # Early termination if explosion detected
                if loss > 1e6:
                    nan_detected = True
                    break
            
            # Restore original config
            config.CONSTRAINT_STIFFNESS = original_stiffness
            config.CONSTRAINT_DAMPING = original_damping
            config.BAUMGARTE_ALPHA = original_baumgarte
            
            return {
                'success': not nan_detected,
                'completed_steps': step + 1,
                'max_loss': max_loss,
                'final_constraint_violation': constraint_violations[-1] if constraint_violations else float('inf'),
                'mean_constraint_violation': np.mean(constraint_violations) if constraint_violations else float('inf'),
            }
            
        except Exception as e:
            # Restore config on error
            config.CONSTRAINT_STIFFNESS = original_stiffness
            config.CONSTRAINT_DAMPING = original_damping
            config.BAUMGARTE_ALPHA = original_baumgarte
            
            return {
                'success': False,
                'error': str(e),
                'completed_steps': 0,
                'max_loss': float('inf'),
            }
    
    def run_full_factorial(self, duration=0.5):
        """
        Run full factorial design (all combinations).
        
        For 6 x 5 x 5 = 150 combinations
        """
        print("="*70)
        print("DESIGN OF EXPERIMENTS: Constraint Parameter Tuning")
        print("="*70)
        print(f"Parameter ranges:")
        for param, values in self.param_ranges.items():
            print(f"  {param}: {len(values)} values from {min(values):.2e} to {max(values):.2e}")
        print(f"Total combinations: {np.prod([len(v) for v in self.param_ranges.values()])}")
        print(f"Test duration: {duration}s per run")
        print("="*70)
        
        # Generate all combinations
        param_names = list(self.param_ranges.keys())
        param_values = [self.param_ranges[p] for p in param_names]
        
        total = np.prod([len(v) for v in param_values])
        completed = 0
        successful = 0
        
        for combo in itertools.product(*param_values):
            stiffness, damping, baumgarte = combo
            
            result = self.run_single_test(stiffness, damping, baumgarte, duration)
            
            self.results.append({
                'CONSTRAINT_STIFFNESS': stiffness,
                'CONSTRAINT_DAMPING': damping,
                'BAUMGARTE_ALPHA': baumgarte,
                **result
            })
            
            completed += 1
            if result['success']:
                successful += 1
            
            if completed % 10 == 0:
                print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) - Success rate: {100*successful/completed:.1f}%")
        
        print("="*70)
        print(f"DoE Complete: {successful}/{total} configurations stable ({100*successful/total:.1f}%)")
        print("="*70)
        
        return self.results
    
    def find_best_configuration(self):
        """
        Find the best parameter configuration from successful runs.
        
        Criteria:
        1. Must complete without NaN/Inf
        2. Lowest final constraint violation
        3. Lowest max loss
        """
        successful = [r for r in self.results if r.get('success', False)]
        
        if not successful:
            print("No successful configurations found!")
            return None
        
        # Sort by constraint violation (primary) and loss (secondary)
        successful.sort(key=lambda x: (x['final_constraint_violation'], x['max_loss']))
        
        return successful[0]
    
    def save_results(self, filename='doe_results.json'):
        """Save all results to JSON."""
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'param_ranges': {k: [float(v) for v in vals] for k, vals in self.param_ranges.items()},
                'results': self.results,
            }, f, indent=2)
        print(f"Results saved to: {filename}")
    
    def print_summary(self):
        """Print summary of findings."""
        print("\n" + "="*70)
        print("DOE SUMMARY")
        print("="*70)
        
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Successful (no NaN/Inf): {len(successful)} ({100*len(successful)/len(self.results):.1f}%)")
        print(f"Failed (NaN/Inf): {len(failed)} ({100*len(failed)/len(self.results):.1f}%)")
        
        if successful:
            best = self.find_best_configuration()
            print("\nBest Configuration:")
            print(f"  CONSTRAINT_STIFFNESS: {best['CONSTRAINT_STIFFNESS']:.2e}")
            print(f"  CONSTRAINT_DAMPING: {best['CONSTRAINT_DAMPING']:.2f}")
            print(f"  BAUMGARTE_ALPHA: {best['BAUMGARTE_ALPHA']:.2f}")
            print(f"  Max Loss: {best['max_loss']:.6e}")
            print(f"  Final Constraint Violation: {best['final_constraint_violation']:.6e}")
        
        # Analyze failure patterns
        if failed:
            print("\nFailure Pattern Analysis:")
            
            # Group by stiffness
            stiffness_failures = {}
            for r in failed:
                s = r['CONSTRAINT_STIFFNESS']
                stiffness_failures[s] = stiffness_failures.get(s, 0) + 1
            
            print("  Failures by stiffness:")
            for s, count in sorted(stiffness_failures.items()):
                total_at_stiffness = len([r for r in self.results if r['CONSTRAINT_STIFFNESS'] == s])
                print(f"    {s:.2e}: {count}/{total_at_stiffness} failed ({100*count/total_at_stiffness:.0f}%)")


def quick_tune():
    """
    Quick tuning with fewer combinations for faster results.
    """
    print("Running quick tuning (subset of parameter space)...")
    
    doe = ConstraintParameterDoE()
    
    # Reduce ranges for quick test
    doe.param_ranges = {
        'CONSTRAINT_STIFFNESS': [1e4, 1e5, 1e6, 1e7],
        'CONSTRAINT_DAMPING': [0.05, 0.1, 0.2, 0.5],
        'BAUMGARTE_ALPHA': [0.1, 0.2, 0.3],
    }
    
    doe.run_full_factorial(duration=0.3)  # Shorter duration
    doe.print_summary()
    doe.save_results('doe_quick_results.json')
    
    return doe.find_best_configuration()


def full_tune():
    """
    Full tuning with complete parameter space.
    """
    print("Running full parameter space exploration...")
    
    doe = ConstraintParameterDoE()
    doe.run_full_factorial(duration=0.5)
    doe.print_summary()
    doe.save_results('doe_full_results.json')
    
    return doe.find_best_configuration()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        best = full_tune()
    else:
        best = quick_tune()
    
    if best:
        print("\n" + "="*70)
        print("RECOMMENDED CONFIGURATION:")
        print("="*70)
        print(f"config.CONSTRAINT_STIFFNESS = {best['CONSTRAINT_STIFFNESS']:.2e}")
        print(f"config.CONSTRAINT_DAMPING = {best['CONSTRAINT_DAMPING']:.2f}")
        print(f"config.BAUMGARTE_ALPHA = {best['BAUMGARTE_ALPHA']:.2f}")
        print("="*70)
