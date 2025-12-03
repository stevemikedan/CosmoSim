# PS2.2: Topology-Aware Neighbor Query System

## Overview
Implemented a centralized neighbor query engine that unifies topology-aware offset calculations across all topologies (flat, torus, sphere, bubble) and prepares for PS2.3 spatial partitioning.

**Status**: ✅ Complete

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
Added `enable_neighbor_engine: bool = True` to `UniverseConfig` for backward compatibility.

### 3. Modified: `physics_utils.py`  
Updated `compute_gravity_forces()` to:
- Use new neighbor engine when `enable_neighbor_engine=True`
- Fall back to vectorized implementation when `False`
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

## Example: Torus Wraparound

**Setup:**
```python
config = UniverseConfig(..., topology_type=1, torus_size=10.0)
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
config = UniverseConfig(..., topology_type=2, radius=10.0)
pos_i = [10.0, 0.0, 0.0]
pos_j = [0.0, 10.0, 0.0]
```

**Behavior:**
- Offset is tangent to sphere at `pos_i`
- Perpendicular to radius: `dot(offset, pos_i) ≈ 0`
- Magnitude is arc length: `R * θ = 10 * (π/2) ≈ 15.7`

## Backward Compatibility
Setting `config.enable_neighbor_engine = False` reverts to the original vectorized implementation, ensuring all existing scenarios and tests continue to work.

## PS2.3 Preparation
Added empty placeholder functions:
- `start_spatial_partition()`
- `get_partition_neighbors()`
- `spatial_partition_debug_info()`

These will be implemented in PS2.3 for spatial partitioning optimization.

## Validation
✅ All 8 new tests pass  
✅ Force calculations identical to legacy implementation  
✅ No physics behavior changes  
✅ No integration logic touched  
✅ Diagnostics unchanged  
✅ Ready for PS2.3
