# PS2.2: Topology-Aware Neighbor Query System

## Phase 1: New Neighbor Engine Module
- [x] Create `environment/topology_neighbors.py`
- [x] Implement `compute_topology_offset(pos_i, pos_j, config)`
  - [x] Flat topology (simple subtraction)
  - [x] Torus topology (wrap-around on each axis)
  - [x] Sphere topology (great-circle tangent displacement)
  - [x] Bubble topology (displacement using bubble radius embedding)
- [x] Implement `generate_neighbor_pairs(positions, active_mask, config)`
  - [x] Skip i == j
  - [x] Return symmetric pairs (i,j) AND (j,i)
  - [x] Respect active_mask
  - [x] Use topology-aware offsets
- [x] Add PS2.3 placeholder functions
  - [x] `start_spatial_partition(config)`
  - [x] `get_partition_neighbors(partition, pos, active, config)`
  - [x] `spatial_partition_debug_info(partition)`

## Phase 2: Integrate into Force Calculation
- [x] Modify `compute_gravity_forces()` in `physics_utils.py`
- [x] Replace nested loop with `generate_neighbor_pairs()`
- [x] Ensure same G constant, epsilon, force symmetry
- [x] Verify same output shape and dtype

## Phase 3: Backward Compatibility
- [x] Add `enable_neighbor_engine: bool = True` to `UniverseConfig`
- [x] Add fallback logic in `compute_gravity_forces()`

## Phase 4: Test Suite
- [x] Create `tests/test_topology_neighbors.py`
- [x] test_flat_topology_offsets_basic()
- [x] test_torus_wraparound_offsets()
- [x] test_sphere_offset_directionality()
- [x] test_bubble_offset_curvature_behavior()
- [x] test_symmetric_pairs()
- [x] test_no_self_pairs()
- [x] test_active_mask_respected()
- [x] test_neighbor_engine_flag_off_restores_original_behavior()

## Phase 5: Validation
- [x] All new tests pass (8/8)
- [x] Flat topology produces identical results
- [x] Torus seam interactions are symmetric
- [x] Sphere & bubble offsets behave smoothly
- [x] No integration logic touched
- [x] Diagnostics remain unchanged
- [x] All placeholder PS2.3 hooks present but unused

## Status: ✅ COMPLETE
Topology-aware neighbor system implemented and verified.
