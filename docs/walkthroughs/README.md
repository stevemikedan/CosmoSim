# CosmoSim Walkthroughs

This directory contains detailed walkthroughs of major features and implementations in CosmoSim.

## Available Walkthroughs

### [Physics Stabilization PS1.1 & PS1.2](physics_stabilization_ps1.1_ps1.2.md)
Implementation of softened gravity and integrator abstraction:
- **PS1.1**: Softened gravity to prevent numerical blowups at close range
- **PS1.2**: Integrator abstraction with Euler and Leapfrog support
- Includes test results, design decisions, and diff summaries

### [Topology Overlay Implementation](topology_overlay_implementation.md)
Complete implementation of the topology overlay system:
- Flat, Torus, Sphere, and Bubble topologies
- Visual overlays for the Three.js viewer
- Comprehensive test coverage

### [Test Suite Updates](test_suite_updates.md)
Updates to test suite for run_sim.py CLI transformation:
- Removed run_sim from scenario interface tests
- Fixed JAX string type errors in mocking
- Created new CLI-specific tests
- All 70 tests passing

## Purpose

These walkthroughs serve as:
- **Documentation** of major features and their implementation
- **Reference** for understanding design decisions
- **Proof of work** showing what was tested and validated
- **Onboarding material** for new contributors

## Format

Each walkthrough typically includes:
- Summary of changes
- Files modified/created
- Code examples and diffs
- Test results
- Design decisions and rationale
- Verification steps
