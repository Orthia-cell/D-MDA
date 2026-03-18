# D-MDA: Differentiable Mechanical Differential Analyzer

A project demonstrating how to "grow" analog computing devices using gradient-based optimization and evolutionary algorithms.

## What This Project Does

This project simulates a mechanical differential analyzer — a physical computer that solves differential equations using rotating disks and wheels. It demonstrates:

1. **Differentiable physics simulation** (Taichi)
2. **Rolling constraint implementation** (no-slip condition)
3. **Parameter optimization** (gradient descent to find optimal wheel position)
4. **Architecture search** (evolutionary algorithm discovers 2-wheel configuration)

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone or extract the project
cd differential_analyzer_env

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install taichi numpy matplotlib
```

### Running the Code

#### Phase 1: Basic Physics Simulation
```bash
python main.py --headless --duration 5.0
# Output: phase1_headless.png
```

#### Phase 2: Rolling Constraint Tests
```bash
python test_phase2.py
# Shows: Position constraint accuracy, velocity constraint stability
```

#### Phase 3: Parameter Optimization
```bash
python optimization.py
# Output: Optimal wheel offset, loss landscape visualization
```

#### Phase 4: Architecture Search (Evolutionary)
```bash
python phase4_architecture_search.py
# Output: Discovered 2-wheel mechanism configuration
```

#### Visualize Results
```bash
python phase4_visualization.py phase4_architecture_search.json
# Output: phase4_results.png
```

## Project Structure

```
differential_analyzer_env/
├── config.py                      # Hyperparameters and constants
├── physics_engine.py              # Phase 1: Basic physics
├── physics_engine_phase2.py       # Phase 2: Rolling constraint
├── optimization.py                # Phase 3: Gradient descent
├── phase4_architecture_search.py  # Phase 4: Evolutionary search
├── phase4_visualization.py        # Visualization for Phase 4
├── main.py                        # Phase 1 entry point
├── main_phase2.py                 # Phase 2 entry point
├── test_phase2.py                 # Validation tests
├── visualization.py               # Phase 1 visualization
├── visualization_phase2.py        # Phase 2/3 visualization
├── phase1_headless.png            # Phase 1 results
├── phase2_optimization.png        # Phase 2 results
├── phase3_results.png             # Phase 3 results
├── phase4_results.png             # Phase 4 results
└── README.md                      # This file
```

## Key Results

| Phase | Method | Configuration | Loss | Improvement |
|-------|--------|---------------|------|-------------|
| 3 | Hand-tuned | 1 wheel @ 0.400m | 0.0096 | Baseline |
| 4 | Evolved | 2 wheels @ 0.072m + 0.277m | 0.000024 | **400× better** |

The evolutionary algorithm discovered that a **2-wheel series configuration** with specific gear ratios outperforms any single-wheel design.

## The Math Behind It

**ODE Being Solved:** dy/dt = sin(t)

**Analytical Solution:** y = 1 - cos(t)

**Physical Interpretation:**
- Disk rotation represents time (t)
- Wheel position on disk represents sin(t)
- Wheel rotation accumulates the integral (1 - cos(t))

## Dependencies

```
taichi >= 1.7.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

## Troubleshooting

### Taichi Installation Issues
If Taichi fails to install:
```bash
pip install --upgrade pip
pip install taichi
```

### Display Issues (Headless Servers)
All scripts support headless mode:
```bash
python main.py --headless
```

### Performance
- Phase 4 architecture search takes ~5-10 minutes (15 individuals × 10 generations)
- Reduce `population_size` and `num_generations` in `phase4_architecture_search.py` for faster runs

## Further Reading

- `D-MDA_Project_Summary.md` — Comprehensive project summary with applications
- `PHASE1_COMPLETION_REPORT.md` — Detailed Phase 1 documentation
- Git commits: `04154a5` (Phase 4), `36f5c93` (Phase 3), `f182451` (Phase 2)

## License

This project was created as a demonstration of differentiable programming and evolutionary algorithms for analog computing research.

---

*Project completed: March 18, 2026*
