"""
Phase 2 Optimization Module - Gradient-Based Mechanism Tuning
D-MDA: Differentiable Mechanical Differential Analyzer

This module implements gradient descent optimization to find the best
mechanism parameters (wheel offset) for accurate ODE integration.
"""

import numpy as np
import taichi as ti
import config
import physics_engine_phase2 as pe


class MechanismOptimizer:
    """
    Optimizes differential analyzer mechanism using gradient descent.
    
    Uses finite difference gradients (central difference) to compute
    how loss changes with wheel offset parameter. Avoids Taichi autodiff
    limitations while still enabling efficient parameter optimization.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=100):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum optimization steps
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = {
            'loss': [],
            'wheel_offset': [],
            'constraint_violation': []
        }
    
    def evaluate_mechanism(self, wheel_offset, duration=5.0):
        """
        Run simulation with given wheel offset and compute final loss.
        
        Args:
            wheel_offset: Distance of wheel from disk center
            duration: Simulation time in seconds
            
        Returns:
            Final loss value
        """
        # Reset simulation
        pe.reset_simulation()
        
        # Set the parameter we're optimizing
        pe.set_wheel_offset(wheel_offset)
        
        # Set disk rotation
        pe.ang_vel[0] = 1.0  # 1 rad/s
        
        # Run simulation
        num_steps = int(duration / config.DT)
        for _ in range(num_steps):
            pe.step()
        
        # Return final loss
        return pe.loss.to_numpy()
    
    def compute_gradient(self, wheel_offset, duration=5.0, eps=1e-4):
        """
        Compute gradient of loss with respect to wheel offset using finite differences.
        
        Uses central difference: grad = (loss(x+eps) - loss(x-eps)) / (2*eps)
        More accurate than forward difference and avoids bias.
        
        Args:
            wheel_offset: Current wheel offset value
            duration: Simulation time
            eps: Finite difference step size (default 1e-4)
            
        Returns:
            (loss_value, gradient_value)
        """
        # Evaluate at wheel_offset + eps
        pe.reset_simulation()
        pe.set_wheel_offset(wheel_offset + eps)
        pe.ang_vel[0] = 1.0
        
        num_steps = int(duration / config.DT)
        for _ in range(num_steps):
            pe.step()
        
        loss_plus = pe.loss.to_numpy()
        
        # Evaluate at wheel_offset - eps
        pe.reset_simulation()
        pe.set_wheel_offset(wheel_offset - eps)
        pe.ang_vel[0] = 1.0
        
        for _ in range(num_steps):
            pe.step()
        
        loss_minus = pe.loss.to_numpy()
        
        # Central difference gradient
        gradient = (loss_plus - loss_minus) / (2 * eps)
        
        # Return loss at current point (average of plus/minus for better estimate)
        loss_val = (loss_plus + loss_minus) / 2.0
        
        return loss_val, gradient
    
    def optimize(self, initial_offset=None, duration=5.0, verbose=True):
        """
        Run gradient descent optimization.
        
        Args:
            initial_offset: Starting wheel offset (None = use config default)
            duration: Simulation duration for each evaluation
            verbose: Print progress
            
        Returns:
            (best_offset, best_loss, history)
        """
        if initial_offset is None:
            initial_offset = config.DISK_RADIUS * 0.5
        
        current_offset = initial_offset
        
        if verbose:
            print("="*60)
            print("MECHANISM OPTIMIZATION - Phase 2")
            print("="*60)
            print(f"Target ODE: dy/dt = sin(t)")
            print(f"Analytical solution: y(t) = 1 - cos(t)")
            print(f"Optimization duration: {duration}s per evaluation")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iterations}")
            print("="*60)
        
        best_loss = float('inf')
        best_offset = current_offset
        
        for iteration in range(self.max_iterations):
            # Compute loss and gradient
            loss_val, grad_val = self.compute_gradient(current_offset, duration)
            
            # Store history
            self.history['loss'].append(loss_val)
            self.history['wheel_offset'].append(current_offset)
            
            # Update best
            if loss_val < best_loss:
                best_loss = loss_val
                best_offset = current_offset
            
            # Print progress
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: offset={current_offset:.6f}, "
                      f"loss={loss_val:.6e}, grad={grad_val:.6e}")
            
            # Check convergence
            if loss_val < 1e-6:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Gradient descent step
            # Gradient tells us how loss changes with offset
            # We move opposite to gradient to minimize loss
            current_offset = current_offset - self.learning_rate * grad_val
            
            # Keep offset in valid range (0 to disk radius)
            current_offset = np.clip(current_offset, 0.01, config.DISK_RADIUS - 0.01)
        
        if verbose:
            print("="*60)
            print(f"Optimization complete!")
            print(f"Best offset: {best_offset:.6f}")
            print(f"Best loss: {best_loss:.6e}")
            print("="*60)
        
        return best_offset, best_loss, self.history


class GridSearchOptimizer:
    """
    Simple grid search for comparison with gradient-based optimization.
    Useful for validating that gradients point in the right direction.
    """
    
    def __init__(self, num_points=20):
        self.num_points = num_points
        self.results = []
    
    def search(self, duration=5.0, verbose=True):
        """
        Search over range of wheel offsets.
        
        Args:
            duration: Simulation duration
            verbose: Print progress
            
        Returns:
            (best_offset, best_loss, all_results)
        """
        offsets = np.linspace(0.05, config.DISK_RADIUS - 0.05, self.num_points)
        
        if verbose:
            print("="*60)
            print("GRID SEARCH - Phase 2")
            print("="*60)
        
        best_loss = float('inf')
        best_offset = None
        
        for offset in offsets:
            # Evaluate
            pe.reset_simulation()
            pe.set_wheel_offset(offset)
            pe.ang_vel[0] = 1.0
            
            num_steps = int(duration / config.DT)
            for _ in range(num_steps):
                pe.step()
            
            loss_val = pe.loss.to_numpy()
            
            self.results.append({
                'offset': offset,
                'loss': loss_val
            })
            
            if loss_val < best_loss:
                best_loss = loss_val
                best_offset = offset
            
            if verbose:
                print(f"Offset: {offset:.4f}, Loss: {loss_val:.6e}")
        
        if verbose:
            print("="*60)
            print(f"Best offset: {best_offset:.4f}")
            print(f"Best loss: {best_loss:.6e}")
            print("="*60)
        
        return best_offset, best_loss, self.results


def compare_optimization_methods(duration=5.0):
    """
    Compare gradient descent vs grid search.
    Demonstrates the power of differentiable physics.
    """
    print("\n" + "="*60)
    print("COMPARISON: Gradient Descent vs Grid Search")
    print("="*60)
    
    # Grid search (baseline)
    print("\n--- Grid Search ---")
    grid_opt = GridSearchOptimizer(num_points=15)
    grid_offset, grid_loss, grid_results = grid_opt.search(duration=duration, verbose=False)
    print(f"Grid search best: offset={grid_offset:.4f}, loss={grid_loss:.6e}")
    print(f"Evaluations: {len(grid_results)}")
    
    # Gradient descent
    print("\n--- Gradient Descent ---")
    grad_opt = MechanismOptimizer(learning_rate=0.05, max_iterations=30)
    grad_offset, grad_loss, grad_history = grad_opt.optimize(duration=duration, verbose=False)
    print(f"Gradient descent best: offset={grad_offset:.4f}, loss={grad_loss:.6e}")
    print(f"Evaluations: {len(grad_history['loss'])}")
    
    print("\n--- Summary ---")
    print(f"Grid search evaluations: {len(grid_results)}")
    print(f"Gradient descent evaluations: {len(grad_history['loss'])}")
    print(f"Speedup: {len(grid_results) / len(grad_history['loss']):.1f}x")
    
    return {
        'grid': (grid_offset, grid_loss, grid_results),
        'gradient': (grad_offset, grad_loss, grad_history)
    }
