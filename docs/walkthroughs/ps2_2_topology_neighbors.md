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
- Default is `False` due to current JIT incompatibility
- Can be set to `True` for non-JIT usage

### 3. Modified: `physics_utils.py`  
Updated `compute_gravity_forces()` to:
- Use new neighbor engine when `enable_neighbor_engine=True`
- Fall back to vectorized implementation when `False` (default)
- Maintain identical physics behavior
- Preserve all softening, G constant, and force symmetry

### 4. Test Suite: `tests/test_topology_neighbors.py`
Created comprehensive tests (8 tests, all passing):
- Flat topology offsets
- Torus wraparound
- Sphere directionality and tangent correctness
- Bubble curvature behavior
- Symmetric pair generation
- No self-pairs
- Active mask respected
- Backward compatibility flag

## Known Limitation: JAX JIT Incompatibility

### Issue
The current implementation of `generate_neighbor_pairs()` uses Python `for` loops with `if` statements:
```python
for i in range(N):
    if not active_mask[i]:  # ❌ Causes TracerBoolConversionError during JIT
        continue
```

This doesn't work with JAX's JIT compilation because JAX can't trace Python control flow on array values.

### Impact
- **Non-JIT usage**: Works perfectly ✅
- **JIT usage**: Raises `TracerBoolConversionError` ❌

### Workaround
The neighbor engine is disabled by default (`enable_neighbor_engine = False`), so all existing JIT-compiled code paths use the proven vectorized implementation.

### Future Fix (PS2.3)
The neighbor engine will be rewritten using:
- JAX-compatible vectorized operations
- `jax.lax.scan` or `jax.lax.fori_loop` for iteration
- Spatial partitioning for performance

## Example: Torus Wraparound

**Setup:**
```python
config = UniverseConfig(..., topology_type=1, torus_size=10.0, enable_neighbor_engine=True)
pos_i = [1.0, 1.0]
pos_j = [9.0, 9.0]
```

**Behavior:**
- Raw offset: `[8.0, 8.0]`
- Minimum image wraps to: `[-2.0, -2.0]`
- Particles are 2.8 units apart (not 11.3 through the longer path)

## Example: Sphere Tangent Offset

**Setup:**
```python
config = UniverseConfig(..., topology_type=2, radius=10.0, enable_neighbor_engine=True)
pos_i = [10.0, 0.0, 0.0]
pos_j = [0.0, 10.0, 0.0]
```

**Behavior:**
- Offset is tangent to sphere at `pos_i`
- Perpendicular to radius: `dot(offset, pos_i) ≈ 0`
- Magnitude is arc length: `R * θ = 10 * (π/2) ≈ 15.7`

## Backward Compatibility
- Default (`enable_neighbor_engine = False`): Uses original vectorized implementation ✅
- Explicit enable (`enable_neighbor_engine = True`): Uses new neighbor engine (non-JIT only) ✅

## PS2.3 Preparation
Added empty placeholder functions:
- `start_spatial_partition()`
- `get_partition_neighbors()`
- `spatial_partition_debug_info()`

These will be implemented in PS2.3 with JAX-compatible spatial partitioning.

## Validation
✅ All 8 new tests pass  
✅ Force calculations identical to legacy implementation  
✅ No physics behavior changes  
✅ No breaking changes to existing tests  
✅ All existing tests pass (198/198)  
✅ Architecture properly separated for future PS2.3 work

## Next Steps
PS2.3 will:
1. Rewrite neighbor engine using JAX-compatible operations
2. Implement spatial partitioning (grid/octree)
3. Enable JIT compilation
4. Set `enable_neighbor_engine = True` by default
