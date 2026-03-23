# Wheel-Disk Integrator Visualization

An interactive 3D simulation of a mechanical wheel-disk integrator — the fundamental building block of differential analyzers.

## Overview

This visualization demonstrates how a simple mechanical system can perform **integration**, one of the core operations in calculus and differential equations. The wheel-disk integrator was a key component in early analog computers like the Differential Analyzer.

## The Math

The system computes:

```
dθ_wheel = (r/R) × dθ_disk
Output = ∫ r × ω_disk dt
```

Where:
- **r** = position of wheel from center (0 to R)
- **R** = disk radius
- **ω_disk** = angular velocity of disk (input function)
- **Output** = accumulated rotation of the wheel

## How It Works

1. **The Big Disk** (blue) spins at a rate representing your input function
2. **The Small Wheel** (orange) rolls on the disk without slipping
3. **Position matters**: Moving the wheel to different radii changes how much it rotates per disk revolution
4. **At radius 0** (center): wheel doesn't rotate at all
5. **At radius R** (edge): wheel rotates at maximum rate

## Usage

Open `integrator.html` in any modern web browser. No installation required — Three.js loads automatically from CDN.

### Controls

| Control | Description |
|---------|-------------|
| **Radius Slider** | Moves wheel from center (0) to edge (0.9) |
| **Speed Slider** | Sets disk rotation rate (input function) |
| **Presets** | Constant, Sine Wave, Ramp, or Stop |
| **Pause/Play** | Toggle animation |
| **Reset** | Clear all accumulated values |

### Visual Elements

- 🔵 **Blue Disk** — Input variable, your function to integrate
- 🟠 **Orange Wheel** — Output, accumulates rotation
- 🟢 **Green Dot** — Contact point (rolling without slipping)
- 🟡 **Yellow Pointer** — Shows wheel rotation
- 📈 **Graph** — Integration output over time

## Try This

1. Set **radius = 0.5** and let the disk spin one full revolution
2. The wheel will rotate exactly **half a turn** — demonstrating the radius ratio
3. Try **Sine Wave** preset to see the integral of sin(t) = -cos(t)

## Background

This visualization was created as part of the [D-MDA (Differentiable Mechanical Differential Analyzer)](../) project — exploring how mechanical analog computers work and how they might be "grown" using modern differentiable programming techniques.

## References

- [Differential Analyzer - Wikipedia](https://en.wikipedia.org/wiki/Differential_analyser)
- [Vannevar Bush's 1931 Paper](https://en.wikipedia.org/wiki/Differential_analyser#History)
- Three.js visualization library: https://threejs.org/
