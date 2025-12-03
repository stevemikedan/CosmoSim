CosmoSim
A Polymorphic, Differentiable, JAX-Accelerated Universe Simulation Engine

Sprints 0–6.6 Complete • Physics & Visualization Stable • Ready for Sprint 7

🌌 Overview

CosmoSim is an extensible cosmological simulation engine built using a JAX-powered ECS architecture.
It enables research, experimentation, and comparison of cosmological models across:

Continuous vector physics (N-body gravity)

Discrete lattice / voxel worlds (future Sprints)

Multiple topologies (Flat, Spherical, Toroidal, future Organic Manifolds)

Differentiable physics and metrics for AI-driven optimization

CosmoSim is designed for developers, researchers, and agentic AI workflows (e.g., Google Antigravity).

🧠 Core Scientific Ideas
1. Differentiable Universe State (PyTree ECS)

All state is contained within a JAX PyTree — enabling:

JIT-accelerated physics

Differentiable updates

Vectorized operations

Static memory layout (required by JAX)

2. Polymorphic Topologies

CosmoSim cleanly separates Metric Space from Physics Rules.

The engine never assumes Euclidean space.

Supported so far:

Flat (Euclidean)

Sphere (Riemannian)

Torus (toroidal wrap-around)

Planned:

Hyperbolic spaces

Organic tetrahedral manifold

Arbitrary user-defined geometries

3. Physics Router

A strategy layer dynamically dispatches physics kernels:

VECTOR mode

LATTICE mode

VOXEL / FIELD (reserved)

CUSTOM (future)

🏗️ High-Level Architecture
                      ┌─────────────────────────────────────┐
                      │             UniverseConfig           │
                      │ (topology, physics_mode, constants)  │
                      └─────────────────────────────────────┘
                                     │
                                     ▼
       ┌────────────────────────────────────────────────────────────┐
       │                      UniverseState (PyTree)                 │
       │-------------------------------------------------------------│
       │  time, radius, curvature_k, dt                               │
       │                                                             │
       │  entity_active[N]                                           │
       │  entity_pos[N,2]                                            │
       │  entity_vel[N,2]                                            │
       │  entity_mass[N]                                             │
       │                                                             │
       │  lattice buffers (preallocated, optional)                   │
       │                                                             │
       │  All fields JAX arrays → stable static shapes               │
       └────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
               ┌──────────────────────────────────────┐
               │            Physics Router             │
               │   dispatch_physics(state,cfg)         │
               └──────────────────────────────────────┘
                      │                │
     ┌────────────────┘                └────────────────┐
     ▼                                                 ▼
┌───────────────┐                            ┌────────────────┐
│ update_vector │                            │ update_lattice │
│  (N-body)     │                            │   (placeholder)│
└───────────────┘                            └────────────────┘
       │                                                 
       ▼                                                 
┌──────────────────────────────────────────────────────────┐
│         Metric Layer (compute_distance, boundaries)       │
└──────────────────────────────────────────────────────────┘


📂 Project Structure
CosmoSim/
│
├── state.py
├── kernel.py
├── topology.py
├── entities.py
│
├── run_sim.py
├── jit_run_sim.py
│
├── trajectory_plot.py
├── snapshot_plot.py
├── energy_plot.py
├── visualize.py
│
├── test_architecture.py
├── test_visualization.py
│
├── outputs/
│   ├── trajectories/
│   ├── snapshots/
│   ├── energy/
│   └── animations/
│
├── tools/
│   ├── clean_outputs.py
│   └── run_all_visualizations.py
│
├── task.md
└── README.md

🧪 Running Tests

Run entire suite:

pytest -q


Or run engine-only tests:

python test_architecture.py


Visualization tests:

python test_visualization.py

🎨 Visualization Tools
Trajectory Plot
python trajectory_plot.py


Outputs to: outputs/trajectories/

Snapshot Plot
python snapshot_plot.py


Outputs to: outputs/snapshots/

Energy Diagnostics
python energy_plot.py


Outputs to: outputs/energy/

Real-Time Animation (saved frame)
python visualize.py


Outputs to: outputs/animations/

Or run all:
python tools/run_all_visualizations.py

🚀 Running the app

The following steps show how to run CosmoSim from the terminal (PowerShell).

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Install required packages (no requirements.txt in this repo)
pip install jax matplotlib pytest

# Run a scenario (e.g., bulk_ring) with a debug view
python cosmosim.py --scenario bulk_ring --view debug
```

🔧 Installation

Create a virtual environment:

python -m venv .venv


Activate:

PowerShell:

.\.venv\Scripts\Activate.ps1


Install dependencies:

pip install jax matplotlib pytest


Or minimal required:

pip install jax matplotlib pytest

🚀 Completed Sprints (0–6.6)
Sprint	Description	Status
0	Project skeleton + directory structure	✅
1	UniverseState + ECS memory model	✅
2	Metric Layer (topology & distance)	✅
3	Physics Router (lax.cond)	✅
4	Boundaries integrated	✅
5	Real vector physics (N-body gravity)	✅
6	Snapshot + Energy + Animation visualizers	✅
6.6	Unified visualization architecture	✅
🧭 Upcoming Roadmap
🚧 Sprint 7 — Physics Stabilization

Add gravitational softening parameter (ε²)

Add energy drift tolerance tests

Improve numerical integrators:

Leapfrog

Verlet

RK2 or RK4

Long-run stability diagnostics

Regression trajectories

🚧 Sprint 8 — Topological Expansion

Organic tetrahedral manifold

Hyperbolic models

Advanced coordinate transforms

Non-Euclidean light cones

🚧 Sprint 9 — Configurable Simulation Loader

JSON/DSL-based universe definitions

User-defined physics modules

🚧 Sprint 10 — GUI / Web Viewer

Interactive 2D/3D visualization

Simulation controls and live tweaking

🤝 Contributing

Even if this is currently a single-developer repo, this section future-proofs the project:

All code must pass both test suites

Engine modules should not be modified outside their designated sprint

It’s designed to grow with your vision — including your long-term cosmological and philosophical frameworks.