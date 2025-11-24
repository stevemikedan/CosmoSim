# CosmoSim Development Task Log

This file tracks all Sprints, constraints, and implementation rules
for ongoing development of the CosmoSim project.

---

## Sprint 0 — Project Skeleton
**Status:** Complete  
- Created base modules: `state.py`, `kernel.py`, `entities.py`, `topology.py`
- Verified basic structure with initial tests.

---

## Sprint 1 — Cosmological ECS (Engine Foundations)
**Status:** Complete  
- Implemented UniverseConfig and UniverseState
- Preallocated buffers for vector + lattice models
- Added Physics Router and Null kernels

---

## Sprint 2 — Metric & Topology Layer
**Status:** Complete  
- Implemented compute_distance() and enforce_boundaries()
- Added support for FLAT, SPHERE, TORUS

---

## Sprint 3 — Physics Router
**Status:** Complete  
- Added dispatch system for vector/lattice physics
- Added JIT-safe routing logic
- Implemented basic update_entities()

---

## Sprint 4 — Boundary Integration
**Status:** Complete  
- enforce_boundaries now part of physics pipeline
- Added associated regression tests

---

## Sprint 5 — Real Vector Physics (N-body Gravity)
**Status:** Complete  
- Fully vectorized gravity
- Semi-implicit Euler integrator
- JIT-compatible and differentiable
- Regression test suite updated

---

## Sprint 6 — Visualization System
**Status:** Complete  
- Added:
  - `snapshot_plot.py`
  - `energy_plot.py`
  - `visualize.py`
  - `trajectory_plot.py` (replaces simple_plot.py)
- Standardized directory structure: outputs/{snapshots,energy,trajectories,animations}
- Enforced Agg backend + timestamped filenames

---

## Sprint 6.6 — Visualization Architecture Finalization
**Status:** Complete  
- Removed outdated visualization scripts
- Updated all visualizers to unified naming and folder structure
- Added Sprint 6 Visualization Test Suite

---

## Sprint 7 — Physics Stabilization (NEXT)
**Status:** Pending  
Goals:
- Add gravitational softening parameter
- Ensure total energy drift < threshold
- Improve integrators (Verlet or Runge-Kutta)
- Add stability tests + energy tolerance thresholds

Constraints:
- Do NOT modify visualization scripts during Sprint 7
- Do NOT modify topology until Sprint 8

---

## Repository Constraints
- Engine files (`state.py`, `kernel.py`, `entities.py`, `topology.py`) may only change if the Sprint explicitly allows it.
- Visualization scripts follow unified output structure and timestamp naming.
- Tests must be updated only when the Sprint introduces new behavior.

