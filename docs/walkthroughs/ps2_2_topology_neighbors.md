# PS2.2: Topology-Aware Neighbor Query System

## Overview
Implemented a centralized neighbor query engine that unifies topology-aware offset calculations across all topologies (flat, torus, sphere, bubble) and prepares for PS2.3 spatial partitioning.

**Status**: ✅ Complete (with known limitation)

> **⚠️ IMPORTANT**: The neighbor engine is currently **disabled by default** (`enable_neighbor_engine = False`) due to JAX JIT incompatibility. It works correctly in non-JIT contexts and will be made JIT-compatible in a future update.

## Changes Made

### 1. New Module: `environment/topology_neighbors.py`
Created centralized neighbor engine with:
- `compute_topology_offset()`: Topology-aware offset calculations
- `generate_neighbor_pairs()`: Iterator for all neighbor pairs with offsets
- PS2.3 placeholder functions (empty, ready for future implementation)

#### Topology Offset Calculations
- **FLAT (0)**: Standard Euclidean `offset = pos_j - pos_i`
- **TORUS (1)**: Minimum image convention with wraparound `offset = dx - round(dx/L) * L`
- **SPHERE (2)**: Great-circle tangent displacement using cross products
- **BUBBLE (3)**: Euclidean offset (curvature handled in distance metric)

### 2. Modified: `state.py`
Added `enable_neighbor_engine: bool = False` to `UniverseConfig` for backward compatibility.

### 3. Modified: `physics_utils.py`  
Updated `compute_gravity_forces()` to use new neighbor engine when enabled.

### 4. Test Suite: `tests/test_topology_neighbors.py`
Created comprehensive tests (8 tests, all passing).

## Known Limitation: JAX JIT Incompatibility

The current implementation uses Python `for` loops with `if` statements which don't work with JAX JIT compilation.

### Workaround
The neighbor engine is disabled by default, so all existing JIT-compiled code paths use the proven vectorized implementation.

## Validation
✅ All 8 new tests pass  
✅ Force calculations identical to legacy implementation  
✅ No physics behavior changes  
✅ All existing tests pass (198/198)
