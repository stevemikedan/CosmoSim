# Physics Enhancement Phase PS2.1: Topology-Aware Distance System

## Goal
Create a unified, topology-aware API for computing distances and offsets between entities. Replace all direct Euclidean calculations with functions that correctly handle flat, torus, sphere, and bubble topologies.

## User Review Required

> [!IMPORTANT]
> This change modifies the **core distance calculation** used in all force computations. While backward-compatible for flat topology, torus/sphere/bubble will produce different results.

> [!WARNING]
> Existing simulations using torus/sphere topologies may produce different physics behavior after this change due to corrected distance calculations.

## Proposed Changes

### [NEW] [distance_utils.py](file:///c:/Users/steve/dev/CosmoSim/distance_utils.py)

Core module providing topology-aware distance calculations.

#### Functions

**`compute_offset(pos_i, pos_j, config) -> ndarray`**
- Returns separation vector Δx respecting topology
- **flat**: `pos_j - pos_i`
- **torus**: Minimum image convention with periodic wrapping
- **sphere**: Tangent-plane offset along great circle
- **bubble**: Euclidean offset (curvature handled in distance)
- Includes NaN guards and zero-division protection

**`compute_distance(pos_i, pos_j, config) -> float`**
- Returns scalar distance respecting topology
- **flat**: Euclidean norm of offset
- **torus**: Euclidean norm after minimum image
- **sphere**: Geodesic distance `R * arccos(dot(u,v))`
- **bubble**: First-order curvature correction `r * (1 + k*r²/6)`
- Falls back to Euclidean for unknown topologies

---

### [MODIFY] [state.py](file:///c:/Users/steve/dev/CosmoSim/state.py)

Add new config fields to `UniverseConfig`:

```python
# Topology distance parameters
torus_size: float | None = None        # Periodic box size (default: radius*2)
bubble_curvature: float = 0.0          # Radial curvature k (0 = flat)
```

**Backward compatibility:** Both fields default to flat behavior.

---

### [MODIFY] [physics_utils.py](file:///c:/Users/steve/dev/CosmoSim/physics_utils.py)

Update `compute_gravity_forces` to use topology-aware distances.

**Current code:**
```python
dx = pos_j - pos_i
r = np.sqrt(np.sum(dx**2))
```

**Updated code:**
```python
from distance_utils import compute_offset, compute_distance

offset = compute_offset(pos_i, pos_j, config)
r = compute_distance(pos_i, pos_j, config)
```

**Impact:** ~10 line changes in force calculation loop.

---

### [NEW] [tests/physics/test_distance_utils.py](file:///c:/Users/steve/dev/CosmoSim/tests/physics/test_distance_utils.py)

Comprehensive test suite validating all topology modes.

#### Test Coverage

**Flat Topology:**
- ✓ Offset equals direct subtraction
- ✓ Distance matches Euclidean norm
- ✓ Backward compatibility verified

**Torus Topology:**
- ✓ Points > L/2 apart wrap correctly
- ✓ Distance is minimal after wrapping
- ✓ No wrap when |dx| < L/2
- ✓ Works in all dimensions

**Sphere Topology:**
- ✓ Small-angle distance ≈ Euclidean locally
- ✓ Antipodal points give πR
- ✓ No NaN for normalized directions
- ✓ Great circle distance is symmetric

**Bubble Topology:**
- ✓ Curvature=0 matches flat Euclidean
- ✓ Curvature>0 gives distance > Euclidean
- ✓ No overflow or NaN for large r
- ✓ First-order approximation is stable

---

## Verification Plan

### Automated Tests
```bash
pytest tests/physics/test_distance_utils.py -v
```

Expected: All tests pass (~20 tests total).

### Integration Verification
```bash
# Run existing physics tests to ensure no regressions
pytest tests/physics/test_softened_gravity.py -v
pytest tests/physics/test_integrators.py -v
```

Expected: All existing tests still pass (backward compatibility).

### Manual Verification

1. **Flat topology (baseline):**
   ```bash
   python run_sim.py --steps 100 --topology flat
   ```
   Expected: Identical behavior to before (Euclidean distances).

2. **Torus topology (wrapping):**
   ```bash
   python run_sim.py --steps 100 --topology torus --radius 10
   ```
   Expected: Particles near boundaries interact across periodic wrap.

3. **Sphere topology (geodesic):**
   ```bash
   python run_sim.py --steps 100 --topology sphere --radius 10
   ```
   Expected: Particles follow curved paths on sphere surface.
