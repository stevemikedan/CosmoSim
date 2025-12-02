# Codebase Architecture Reorganization Plan (Revised)

## Current State Analysis

### ✅ **What's Good** (Keep Structure)
- Existing packages: `tests/`, `scenarios/`, `topologies/`, `substrate/`, `environment/`, `exporters/`

### ⚠️ **What Needs Organization**

**Root Level Clutter:**
- 2 utility modules: `physics_utils.py`, `distance_utils.py`
- 4 plotting scripts: `energy_plot.py`, `snapshot_plot.py`, `trajectory_plot.py`, `visualize.py`
- 4 documentation files: `ENVIRONMENT_ENGINE.md`, `education_module.md`, etc.
- 3 output directories: `frames/`, `sim_output/`, `outputs/`
- 2 temporary files: `test_output.txt`, `task.md`

**Misplaced Dev Artifacts:**
- `tasks/`, `implementation_plans/`, `walkthroughs/` - Should be in `docs/`

---

## Proposed Structure

```
CosmoSim/
├── cosmosim.py              # Main CLI entry point
├── run_sim.py               # Direct simulation runner
├── state.py                 # Core state management
├── entities.py              # Entity spawning/management
├── kernel.py                # Simulation kernel
├── topology.py              # Topology utilities
│
├── utils/                   # NEW - Utility modules package
│   ├── __init__.py
│   ├── physics_utils.py     # Physics calculations
│   └── distance_utils.py    # Topology-aware distances
│
├── plotting/                # NEW - Plotting scripts
│   ├── energy_plot.py
│   ├── snapshot_plot.py
│   ├── trajectory_plot.py
│   └── visualize.py
│
├── docs/                    # Consolidated documentation
│   ├── README.md
│   ├── architecture/
│   │   └── ENVIRONMENT_ENGINE.md
│   ├── education/
│   │   ├── education_module.md
│   │   └── education_module_roadmap.md
│   ├── implementation_plans/   # MOVED from root
│   │   ├── README.md
│   │   └── ps2.1_topology_aware_distance.md
│   ├── tasks/                  # MOVED from root
│   │   ├── README.md
│   │   └── ps2.1_topology_aware_distance.md
│   └── walkthroughs/           # MOVED from root
│       ├── README.md
│       └── (all walkthrough files)
│
├── outputs/                 # Consolidated output directory
│   ├── simulations/         # Simulation results
│   ├── frames/              # Frame data
│   └── plots/               # Generated plots
│
├── tests/                   # Test suite (existing)
├── scenarios/               # Scenario definitions (existing)
├── topologies/              # Topology handlers (existing)
├── substrate/               # Substrate system (existing)
├── environment/             # Environment handlers (existing)
├── exporters/               # Export utilities (existing)
├── tools/                   # CLI tools (existing)
└── viewer/                  # Visualization (existing)
```

---

## Import Impact Analysis

### Creating `utils/` Package

**Files that need import updates (8 total):**
1. `kernel.py`
2. `run_sim.py`
3. `physics_utils.py` (self-import for distance_utils)
4. `tests/physics/test_softened_gravity.py`
5. `tests/physics/test_integrators.py`
6. `tests/physics/test_distance_utils.py`
7. `tests/physics/test_energy_diagnostics.py`
8. `tests/physics/test_adaptive_dt.py`

**Change pattern:**
```python
# Before
from physics_utils import compute_gravity_forces
from distance_utils import compute_offset

# After
from utils.physics_utils import compute_gravity_forces
from utils.distance_utils import compute_offset
```

**Benefit:** Clean root, scalable for future utils (`collision_utils.py`, `field_utils.py`, etc.)

---

## Implementation Phases

### Phase 1: Documentation & Outputs (LOW RISK)
```bash
# Create docs structure
mkdir docs/architecture docs/education
mv ENVIRONMENT_ENGINE.md docs/architecture/
mv education_module*.md docs/education/
mv tasks docs/
mv implementation_plans docs/
mv walkthroughs docs/

# Consolidate outputs
mkdir -p outputs/simulations outputs/frames outputs/plots
mv frames/* outputs/frames/
mv sim_output/* outputs/simulations/
rmdir frames sim_output

# Clean temps
rm test_output.txt task.md
rm json_exporter.py  # Thin wrapper, use exporters.json_export directly
```

**Risk:** Near zero - no code changes

### Phase 2: Create `utils/` Package (MEDIUM RISK)
```bash
# Create package
mkdir utils
touch utils/__init__.py
mv physics_utils.py utils/
mv distance_utils.py utils/
```

**Then update imports in 8 files** (automated with sed/PowerShell or manual)

**Risk:** Moderate - breaks if imports not updated correctly  
**Mitigation:** Run `pytest` after each change to verify

### Phase 3: Create `plotting/` Directory (LOW RISK)
```bash
# Create directory
mkdir plotting

# Move scripts
mv energy_plot.py plotting/
mv snapshot_plot.py plotting/
mv trajectory_plot.py plotting/
mv visualize.py plotting/
mv jit_run_sim.py plotting/  # Experimental script
```

**Add to each script:**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

**Risk:** Low - these are scripts, not imported modules

---

## Validation Steps

After each phase:
```bash
# Run tests
pytest tests/physics/ -v

# Test CLI
python cosmosim.py --help

# Test runner
python run_sim.py --help

# Test plotting (Phase 3)
python plotting/energy_plot.py --help
```

---

## Alternative: Staged Approach

**Option A - Full Cleanup (Phases 1-3)**
- Cleanest result
- ~1 hour work
- Medium risk Phase 2

**Option B - Safe First (Phase 1 only)**
- Docs & outputs only
- ~15 min work
- Zero risk
- Do Phase 2-3 later

**Option C - Phased (1 → wait → 2 → wait → 3)**
- Phase 1 today
- Validate, then Phase 2 later
- Validate, then Phase 3 later

---

## Recommendation

**Start with Phase 1 + Phase 3** (skip utils for now):
- Move docs & consolidate outputs ✅
- Create `plotting/` directory ✅
- **Skip `utils/` package** - defer until we accumulate more utils

This gives immediate cleanup with minimal risk. Create `utils/` package later when:
- We have 3+ utility modules, OR
- Starting a major refactor anyway

**Your call:** Which approach feels right?

