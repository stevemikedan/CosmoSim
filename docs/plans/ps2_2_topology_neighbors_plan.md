# PS2.2 Implementation Plan: Topology-Aware Neighbor Query System

## Overview
Unify and stabilize neighbor/offset queries across all topologies (flat, torus, sphere, bubble) by creating a centralized neighbor engine. This prepares for PS2.3 spatial partitioning without changing physics behavior.

## Architecture

### 1. Neighbor Engine (`environment/topology_neighbors.py`)
Core functions:
- `compute_topology_offset(pos_i, pos_j, config)`: Returns topology-correct offset vector
- `generate_neighbor_pairs(positions, active_mask, config)`: Iterator of (i, j, offset_ij)

### 2. Topology Offset Calculations
- **Flat**: `offset = pos_j - pos_i`
- **Torus**: Apply wrap-around independently per axis using domain bounds
- **Sphere**: Compute great-circle tangent displacement on sphere surface
- **Bubble**: Compute displacement using bubble radius embedding in higher-dimensional space

### 3. Force Integration
Replace nested loops in `compute_gravity_forces()` with:
```python
for i, j, offset in generate_neighbor_pairs(pos, active, config):
    r = norm(offset)
    force_ij = G * m_i * m_j * offset / (r^3 + eps)
    # Accumulate forces
```

### 4. Backward Compatibility
Add `enable_neighbor_engine` flag to `UniverseConfig`:
- Default: `True`
- If `False`: Use original Euclidean pairwise computation

## Implementation Phases

### Phase 1: Neighbor Engine
Create `environment/topology_neighbors.py` with all topology calculations.

### Phase 2: Integration
Modify `physics_utils.py::compute_gravity_forces()` to use neighbor engine.

### Phase 3: Testing
Create comprehensive test suite covering all topologies and edge cases.

### Phase 4: Validation
Verify no physics changes except topology corrections, all tests pass.

## Constraints
- NO changes to: integration, energy formulas, JSON export, viewer, scenarios
- NO performance optimizations (PS2.3/PS2.4)
- NO breaking changes to test suite
- MAINTAIN exact physics behavior except topology offsets

## PS2.3 Preparation
Add empty placeholder functions (no implementation):
- `start_spatial_partition()`
- `get_partition_neighbors()`
- `spatial_partition_debug_info()`
