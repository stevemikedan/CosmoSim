# PSS Implementation Walkthrough: Phases 0.x through 1.1

## Overview

Successfully implemented the foundational Parameterized Scenario System (PSS) enabling runtime configuration of scenarios via CLI parameters.

**Status**: ✅ Complete - All 182 tests passing (155 baseline + 27 PSS)

## What Was Implemented

### PSS0.x: CLI Parameter Support

**CLI Argument**:
```bash
--params "N=500,radius=20,spread=0.3"
```

**Parser Implementation**:
```python
def parse_param_string(param_str: str | None) -> dict:
    # Parses comma-separated key=value pairs
    # Returns dict with string values (type conversion in merge_params)
```

**Features**:
- Comma-separated parsing
- Whitespace trimming
- Graceful handling of malformed input
- Empty value support

### PSS1.0: Schema Format Definition

**Schema Structure**:
```python
SCENARIO_PARAMS = {
    "param_name": {
        "type": "int" | "float" | "bool" | "str",
        "default": <value>,
        "min": <optional>,
        "max": <optional>,
        "description": <optional>
    },
}
```

**Example** (in [`scenarios/bulk_ring.py`](file:///c:/Users/steve/dev/CosmoSim/scenarios/bulk_ring.py)):
```python
SCENARIO_PARAMS = {
    "N": {
        "type": "int",
        "default": 64,
        "min": 1,
        "max": 128,
        "description": "Number of entities in the ring"
    },
    "radius": {
        "type": "float",
        "default": 8.0,
        "min": 1.0,
        "max": 50.0,
        "description": "Orbital radius of the ring"
    },
    # ... more params
}
```

### PSS1.1: Schema Discovery & Merging

**Discovery**:
```python
def load_scenario_schema(module) -> dict | None:
    # Reads SCENARIO_PARAMS if present
    # Validates format
    # Returns None if not defined
```

**Merging & Type Conversion**:
```python
def merge_params(schema: dict | None, cli_params: dict) -> dict:
    # Start with schema defaults
    # Apply CLI overrides with type conversion
    # Enforce bounds (min/max clamping)
    # Warn for unknown/invalid params
```

**Type Conversion Rules**:
- `int`: `int(value)`
- `float`: `float(value)`
- `bool`: `value.lower() in ("true", "1", "yes", "y")`
- `str`: Keep as-is

**Bounds Enforcement**:
- If value < min → clamp to min (with warning)
- If value > max → clamp to max (with warning)

### Integration into cosmosim.py

**Workflow**:
```python
# In run_scenario()
schema = load_scenario_schema(module)
cli_params = parse_param_string(args.params)
merged_params = merge_params(schema, cli_params)

# PSS Logging
if schema:
    print(f"[PSS] Loaded schema with {len(schema)} parameters")
if cli_params:
    print(f"[PSS] Applied CLI overrides: {cli_params}")
if merged_params:
    print(f"[PSS] Final merged parameters: {merged_params}")

# Pass to scenario
sig = inspect.signature(module.build_initial_state)
if 'params' in sig.parameters:
    state = module.build_initial_state(cfg, merged_params)
else:
    state = module.build_initial_state(cfg)  # Backward compatible
```

## Usage Examples

### Basic Usage with Defaults
```bash
python cosmosim.py --scenario bulk_ring --steps 300 --view none
```
Output:
```
[PSS] Loaded schema with 4 parameters
[PSS] Final merged parameters: {'N': 64, 'radius': 8.0, 'speed': 0.8, 'mass': 1.0}
```

### Override Parameters
```bash
python cosmosim.py --scenario bulk_ring --params N=100,radius=15.0 --steps 300 --view none
```
Output:
```
[PSS] Loaded schema with 4 parameters
[PSS] Applied CLI overrides: {'N': '100', 'radius': '15.0'}
[PSS] Final merged parameters: {'N': 100, 'radius': 15.0, 'speed': 0.8, 'mass': 1.0}
```

### Bounds Clamping
```bash
python cosmosim.py --scenario bulk_ring --params N=500 --steps 10 --view none
```
Output:
```
[PSS] Warning: N=500 above max=128, clamping
[PSS] Final merged parameters: {'N': 128, ...}
```

### Unknown Parameter
```bash
python cosmosim.py --scenario bulk_ring --params unknown=value --steps 10 --view none
```
Output:
```
[PSS] Warning: Unknown parameter 'unknown' ignored
```

### Scenario Without Schema
```bash
python cosmosim.py --scenario manual_run --params N=100 --steps 10 --view none
```
Output:
```
[PSS] Warning: CLI params provided but scenario has no schema
```
Scenario runs normally with its defaults.

## Implementation Details

### Files Modified

**Core**:
- [`cosmosim.py`](file:///c:/Users/steve/dev/CosmoSim/cosmosim.py) (+120 lines)
  - `parse_param_string()`
  - `load_scenario_schema()`
  - `merge_params()`
  - Integration in `run_scenario()`
  - `--params` CLI argument

**Example Scenario**:
- [`scenarios/bulk_ring.py`](file:///c:/Users/steve/dev/CosmoSim/scenarios/bulk_ring.py) (+50 lines)
  - `SCENARIO_PARAMS` schema
  - Updated `build_initial_state(cfg, params=None)`
  - Params usage for N, radius, speed, mass

**Tests**:
- [`tests/test_scenario_params.py`](file:///c:/Users/steve/dev/CosmoSim/tests/test_scenario_params.py) (27 tests, ~350 lines)

### Test Coverage

**27 PSS Tests** (all passing):

| Category | Tests |
|----------|-------|
| Parsing | 6 tests (basic, empty, whitespace, malformed, single) |
| Schema Loading | 3 tests (valid, missing, no attribute) |
| Type Conversion | 5 tests (int, float, bool, str, invalid) |
| Bounds & Validation | 5 tests (min/max clamping, unknown params) |
| Merging Logic | 4 tests (basic, missing schema, multiple overrides) |
| Integration | 4 tests (without schema, with schema, with params, end-to-end) |

**Test Results**:
```bash
pytest tests/test_scenario_params.py -v
# 27/27 passing

pytest -q
# 182/182 passing (155 baseline + 27 PSS)
```

## Backward Compatibility

| Scenario Type | Behavior |
|---------------|----------|
| No SCENARIO_PARAMS | ✅ Runs exactly as before, ignores --params |
| Has SCENARIO_PARAMS, no --params | ✅ Uses all defaults from schema |
| Has SCENARIO_PARAMS, with --params | ✅ Applies overrides |
| `build_initial_state(config)` | ✅ Called without params (backward compatible) |
| `build_initial_state(config, params)` | ✅ Called with merged params |

**Zero Breaking Changes**: All 155 baseline tests pass unchanged.

## Constraints Preserved

✅ No physics modifications  
✅ No JSON output changes  
✅ Existing scenarios work unchanged  
✅ All baseline tests pass  
✅ Backward compatible  
✅ Optional system (scenarios can ignore PSS)  

## PSS Logging Examples

### Successful Parameter Override
```
Building configuration for 'bulk_ring'...
[PSS] Loaded schema with 4 parameters
[PSS] Applied CLI overrides: {'N': '32', 'radius': '12.0'}
[PSS] Final merged parameters: {'N': 32, 'radius': 12.0, 'speed': 0.8, 'mass': 1.0}
Initializing universe state...
```

### Type Conversion Warning
```
[PSS] Warning: Invalid type for parameter 'N'; using default
[PSS] Final merged parameters: {'N': 64, ...}
```

### Bounds Clamping Warning
```
[PSS] Warning: radius=100.0 above max=50.0, clamping
[PSS] Final merged parameters: {..., 'radius': 50.0, ...}
```

## Future Enhancements (Not in PSS0.x-PSS1.1)

Planned for later phases:
- **PSS1.2**: Advanced validation (custom validators)
- **PSS1.3**: Parameter presets
- **PSS1.4**: Parameter documentation generation
- **PSS2.0**: Dynamic parameter UI

## Files Changed Summary

**Added**:
- `tests/test_scenario_params.py` (350 lines)

**Modified**:
- `cosmosim.py` (+120 lines)
- `scenarios/bulk_ring.py` (+50 lines)

**Total**: ~520 lines of new code

## Validation

### Automated Tests
```bash
pytest tests/test_scenario_params.py -v
# 27/27 PSS tests passing

pytest -q
# 182/182 total tests passing
```

### Manual Verification
✅ Default params: `--scenario bulk_ring`  
✅ Override params: `--params N=20,radius=5.0`  
✅ Bounds clamping: `--params N=500` (clamps to 128)  
✅ Unknown param warning: `--params unknown=value`  
✅ Scenario without schema: `--scenario manual_run` (works normally)  
✅ Type conversion: bool, int, float, str all working  

## See Also

- [CLI Consolidation](file:///c:/Users/steve/dev/CosmoSim/docs/walkthroughs/cli_consolidation.md) - Entry point documentation
- [Scenario Interface](file:///c:/Users/steve/dev/CosmoSim/scenarios/README.md) - Scenario development guide
