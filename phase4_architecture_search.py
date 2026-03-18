"""
Phase 4 Architecture Search - D-MDA
Differentiable Mechanical Differential Analyzer

Automatically discovers mechanism configurations (topology, gear ratios, 
number of components) rather than just tuning a single parameter.

Search Space:
- Number of wheels (1-3)
- Gear ratios between components
- Connection topology (which wheels connect to which)
- Disk rotation input method
"""

import numpy as np
import taichi as ti
import config
import physics_engine_phase2 as pe
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import json
from datetime import datetime


@dataclass
class MechanismGene:
    """
    Genetic representation of a mechanism configuration.
    
    This is like DNA for mechanical systems - it encodes:
    - How many wheels
    - Where they're placed
    - How they connect
    - Gear ratios between them
    """
    num_wheels: int  # 1-3 wheels
    wheel_offsets: List[float]  # Distance from disk center for each wheel
    gear_ratios: List[float]  # Ratio between disk and each wheel
    connection_type: str  # 'series', 'parallel', or 'compound'
    
    def __post_init__(self):
        # Ensure lists have correct length
        assert len(self.wheel_offsets) == self.num_wheels
        assert len(self.gear_ratios) == self.num_wheels
        
        # Clamp values to valid ranges
        self.wheel_offsets = [
            np.clip(offset, 0.01, config.DISK_RADIUS - 0.01)
            for offset in self.wheel_offsets
        ]
        self.gear_ratios = [
            np.clip(ratio, 0.1, 10.0)
            for ratio in self.gear_ratios
        ]
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'num_wheels': self.num_wheels,
            'wheel_offsets': self.wheel_offsets,
            'gear_ratios': self.gear_ratios,
            'connection_type': self.connection_type
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'MechanismGene':
        """Deserialize from dictionary."""
        return cls(
            num_wheels=d['num_wheels'],
            wheel_offsets=d['wheel_offsets'],
            gear_ratios=d['gear_ratios'],
            connection_type=d['connection_type']
        )
    
    def __str__(self) -> str:
        return (f"MechanismGene(wheels={self.num_wheels}, "
                f"offsets={self.wheel_offsets}, "
                f"ratios={self.gear_ratios}, "
                f"type={self.connection_type})")


class MechanismPhenotype:
    """
    The physical realization of a MechanismGene.
    
    This translates the genetic code into actual simulation parameters
    and runs the physics to compute fitness (how well it solves the ODE).
    """
    
    def __init__(self, gene: MechanismGene):
        self.gene = gene
        self.fitness = None
        self.final_loss = None
        self.error = None
        self.simulation_data = None
    
    def evaluate(self, duration: float = 5.0, verbose: bool = False) -> float:
        """
        Evaluate this mechanism by running simulation.
        
        Returns:
            Fitness score (lower is better - it's actually loss)
        """
        if verbose:
            print(f"Evaluating: {self.gene}")
        
        # Reset simulation
        pe.reset_simulation()
        
        # Configure based on gene
        # For now, we use the first wheel's offset as the primary
        if self.gene.num_wheels > 0:
            pe.set_wheel_offset(self.gene.wheel_offsets[0])
            
            # Apply gear ratio by adjusting wheel inertia/mass
            # Higher gear ratio = wheel rotates faster for given disk rotation
            gear_ratio = self.gene.gear_ratios[0]
            # Scale the wheel's moment of inertia to simulate gear ratio
            pe.moment_of_inertia[1] = config.WHEEL_MOMENT_OF_INERTIA / (gear_ratio ** 2)
        
        # Set disk rotation
        pe.ang_vel[0] = 1.0
        
        # Run simulation
        num_steps = int(duration / config.DT)
        state_history = []
        
        for i in range(num_steps):
            pe.step()
            if i % 100 == 0:
                state_history.append(pe.get_state())
        
        # Get final state
        final_state = pe.get_state()
        self.final_loss = final_state['loss']
        
        # Calculate error
        target = final_state['target_integral']
        actual = final_state['actual_integral']
        self.error = abs(actual - target)
        
        # Fitness is negative loss (higher fitness = better)
        # We use negative loss so that "maximize fitness" = "minimize loss"
        self.fitness = -self.final_loss
        
        self.simulation_data = {
            'state_history': state_history,
            'final_state': final_state
        }
        
        if verbose:
            print(f"  Loss: {self.final_loss:.6f}, Error: {self.error:.6f}, "
                  f"Fitness: {self.fitness:.6f}")
        
        return self.fitness


class ArchitectureSearch:
    """
    Evolutionary architecture search for differential analyzer mechanisms.
    
    Uses a genetic algorithm to evolve mechanism configurations:
    1. Initialize population of random mechanisms
    2. Evaluate fitness (how well they solve the ODE)
    3. Select best performers
    4. Crossover and mutate to create new generation
    5. Repeat until convergence
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_generations: int = 30,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elitism_count: int = 3
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        self.population: List[MechanismPhenotype] = []
        self.generation_history: List[Dict] = []
        self.best_individual: Optional[MechanismPhenotype] = None
    
    def _create_random_gene(self) -> MechanismGene:
        """Create a random mechanism gene."""
        num_wheels = np.random.randint(1, 4)  # 1-3 wheels
        
        wheel_offsets = [
            np.random.uniform(0.05, config.DISK_RADIUS - 0.05)
            for _ in range(num_wheels)
        ]
        
        gear_ratios = [
            np.random.uniform(0.5, 5.0)
            for _ in range(num_wheels)
        ]
        
        connection_type = np.random.choice(['series', 'parallel', 'compound'])
        
        return MechanismGene(
            num_wheels=num_wheels,
            wheel_offsets=wheel_offsets,
            gear_ratios=gear_ratios,
            connection_type=connection_type
        )
    
    def _initialize_population(self):
        """Create initial random population."""
        print(f"Initializing population of {self.population_size} mechanisms...")
        self.population = []
        
        for i in range(self.population_size):
            gene = self._create_random_gene()
            phenotype = MechanismPhenotype(gene)
            self.population.append(phenotype)
    
    def _evaluate_population(self, duration: float = 3.0):
        """Evaluate all individuals in population."""
        print("Evaluating population...")
        for i, individual in enumerate(self.population):
            print(f"  [{i+1}/{len(self.population)}]", end=" ")
            individual.evaluate(duration=duration, verbose=True)
        
        # Sort by fitness (descending, since fitness = -loss)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best individual
        if self.best_individual is None or \
           self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = deepcopy(self.population[0])
            print(f"\n🌟 New best: {self.best_individual.gene}")
            print(f"   Loss: {self.best_individual.final_loss:.6f}")
    
    def _select_parent(self) -> MechanismPhenotype:
        """
        Tournament selection: pick best of random subset.
        """
        tournament_size = 3
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(
        self,
        parent1: MechanismGene,
        parent2: MechanismGene
    ) -> Tuple[MechanismGene, MechanismGene]:
        """
        Create two children by combining parent genes.
        """
        if np.random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return deepcopy(parent1), deepcopy(parent2)
        
        # Decide on child number of wheels (take from one parent)
        child1_num_wheels = parent1.num_wheels
        child2_num_wheels = parent2.num_wheels
        
        # Average the offsets and gear ratios (simple blending)
        # Pad shorter lists
        max_wheels = max(parent1.num_wheels, parent2.num_wheels)
        
        p1_offsets = parent1.wheel_offsets + [0.3] * (max_wheels - parent1.num_wheels)
        p2_offsets = parent2.wheel_offsets + [0.3] * (max_wheels - parent2.num_wheels)
        p1_ratios = parent1.gear_ratios + [1.0] * (max_wheels - parent1.num_wheels)
        p2_ratios = parent2.gear_ratios + [1.0] * (max_wheels - parent2.num_wheels)
        
        child1_offsets = [
            (p1_offsets[i] + p2_offsets[i]) / 2
            for i in range(child1_num_wheels)
        ]
        child2_offsets = [
            (p1_offsets[i] + p2_offsets[i]) / 2
            for i in range(child2_num_wheels)
        ]
        child1_ratios = [
            (p1_ratios[i] + p2_ratios[i]) / 2
            for i in range(child1_num_wheels)
        ]
        child2_ratios = [
            (p1_ratios[i] + p2_ratios[i]) / 2
            for i in range(child2_num_wheels)
        ]
        
        # Random connection type from parents
        child1_type = parent1.connection_type
        child2_type = parent2.connection_type
        
        child1 = MechanismGene(
            num_wheels=child1_num_wheels,
            wheel_offsets=child1_offsets,
            gear_ratios=child1_ratios,
            connection_type=child1_type
        )
        child2 = MechanismGene(
            num_wheels=child2_num_wheels,
            wheel_offsets=child2_offsets,
            gear_ratios=child2_ratios,
            connection_type=child2_type
        )
        
        return child1, child2
    
    def _mutate(self, gene: MechanismGene) -> MechanismGene:
        """
        Randomly modify a gene.
        """
        gene = deepcopy(gene)
        
        # Mutate number of wheels (rare)
        if np.random.random() < self.mutation_rate * 0.1:
            old_num = gene.num_wheels
            gene.num_wheels = np.random.randint(1, 4)
            
            # Adjust lists
            if gene.num_wheels > old_num:
                # Add new wheels
                for _ in range(gene.num_wheels - old_num):
                    gene.wheel_offsets.append(np.random.uniform(0.1, 0.9))
                    gene.gear_ratios.append(np.random.uniform(0.5, 5.0))
            else:
                # Remove wheels
                gene.wheel_offsets = gene.wheel_offsets[:gene.num_wheels]
                gene.gear_ratios = gene.gear_ratios[:gene.num_wheels]
        
        # Mutate wheel offsets
        for i in range(gene.num_wheels):
            if np.random.random() < self.mutation_rate:
                gene.wheel_offsets[i] += np.random.normal(0, 0.1)
                gene.wheel_offsets[i] = np.clip(
                    gene.wheel_offsets[i], 0.01, config.DISK_RADIUS - 0.01
                )
        
        # Mutate gear ratios
        for i in range(gene.num_wheels):
            if np.random.random() < self.mutation_rate:
                gene.gear_ratios[i] += np.random.normal(0, 0.5)
                gene.gear_ratios[i] = np.clip(gene.gear_ratios[i], 0.1, 10.0)
        
        # Mutate connection type (rare)
        if np.random.random() < self.mutation_rate * 0.2:
            gene.connection_type = np.random.choice(['series', 'parallel', 'compound'])
        
        return gene
    
    def _create_next_generation(self):
        """Create new population through selection, crossover, mutation."""
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(self.elitism_count):
            new_population.append(deepcopy(self.population[i]))
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            child1_gene, child2_gene = self._crossover(parent1.gene, parent2.gene)
            
            # Mutate
            child1_gene = self._mutate(child1_gene)
            child2_gene = self._mutate(child2_gene)
            
            # Add to new population
            new_population.append(MechanismPhenotype(child1_gene))
            if len(new_population) < self.population_size:
                new_population.append(MechanismPhenotype(child2_gene))
        
        self.population = new_population
    
    def search(self, duration: float = 3.0, verbose: bool = True) -> MechanismPhenotype:
        """
        Run evolutionary architecture search.
        
        Args:
            duration: Simulation duration for each evaluation
            verbose: Print progress
            
        Returns:
            Best mechanism phenotype found
        """
        if verbose:
            print("="*70)
            print("PHASE 4: ARCHITECTURE SEARCH")
            print("D-MDA: Differentiable Mechanical Differential Analyzer")
            print("="*70)
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.num_generations}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Crossover rate: {self.crossover_rate}")
            print("="*70)
        
        # Initialize
        self._initialize_population()
        
        # Evolution loop
        for generation in range(self.num_generations):
            print(f"\n{'='*70}")
            print(f"GENERATION {generation + 1}/{self.num_generations}")
            print('='*70)
            
            # Evaluate
            self._evaluate_population(duration=duration)
            
            # Record statistics
            fitnesses = [p.fitness for p in self.population]
            self.generation_history.append({
                'generation': generation,
                'best_fitness': max(fitnesses),
                'worst_fitness': min(fitnesses),
                'mean_fitness': np.mean(fitnesses),
                'best_loss': -max(fitnesses),
                'best_gene': self.population[0].gene.to_dict()
            })
            
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Best loss: {-max(fitnesses):.6f}")
            print(f"  Mean loss: {-np.mean(fitnesses):.6f}")
            print(f"  Worst loss: {-min(fitnesses):.6f}")
            
            if generation < self.num_generations - 1:
                # Create next generation
                self._create_next_generation()
        
        # Final evaluation of best
        print("\n" + "="*70)
        print("ARCHITECTURE SEARCH COMPLETE")
        print("="*70)
        print(f"\nBest mechanism found:")
        print(f"  {self.best_individual.gene}")
        print(f"  Final loss: {self.best_individual.final_loss:.6f}")
        print(f"  Error: {self.best_individual.error:.6f}")
        
        return self.best_individual
    
    def save_results(self, filename: str = 'phase4_results.json'):
        """Save search results to JSON."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_count': self.elitism_count
            },
            'best_gene': self.best_individual.gene.to_dict() if self.best_individual else None,
            'best_loss': self.best_individual.final_loss if self.best_individual else None,
            'best_error': self.best_individual.error if self.best_individual else None,
            'generation_history': self.generation_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def run_phase4_search():
    """Run a complete Phase 4 architecture search."""
    search = ArchitectureSearch(
        population_size=15,
        num_generations=10,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=2
    )
    
    best = search.search(duration=2.0, verbose=True)
    search.save_results('phase4_architecture_search.json')
    
    return best


if __name__ == "__main__":
    best_mechanism = run_phase4_search()
