# PS2.1: Topology-Aware Distance System - Task Checklist

## Core Implementation
- [x] Create `distance_utils.py` with `compute_offset` and `compute_distance`
- [x] Add `torus_size` and `bubble_curvature` to `UniverseConfig` in `state.py`

## Test Suite
- [x] Create `tests/physics/test_distance_utils.py`
  - [x] FLAT topology tests (offset, distance)
  - [x] TORUS topology tests (wrapping, minimum image)
  - [x] SPHERE topology tests (geodesic, antipodal, small-angle)
  - [x] BUBBLE topology tests (curvature correction, flat limit)

## Integration
- [x] Integrate into `physics_utils.py`
  - [x] Replace Euclidean offset/distance in `compute_gravity_forces`
  - [x] Preserve backward compatibility for flat topology
- [x] Verify no changes to integrators, expansion, diagnostics, substrate

## Verification
- [x] Run `test_distance_utils.py` - all tests pass (10/10)
- [x] Run `test_softened_gravity.py` - regression check (5/5)
- [x] Run `test_integrators.py` - regression check (8/8)
- [ ] Manual test: flat topology behaves identically to before
- [ ] Manual test: torus topology shows periodic wrapping
- [ ] Manual test: sphere topology shows curved behavior

## Documentation
- [ ] Create walkthrough documenting changes and verification results
