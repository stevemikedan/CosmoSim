# PSS1.2: Parameter Validation Layer

## Overview
Implemented a robust parameter validation layer for the Parameterized Scenario System (PSS). This ensures that scenario parameters supplied via CLI or other interfaces are type-checked, bounds-checked, and validated against the schema before being passed to the simulation.

**Status**: ✅ Complete

## Features

### 1. Robust Validation Logic
The `merge_params` function now performs:
- **Type Conversion**: Safely converts strings to `int`, `float`, `bool` with fallback to default on failure.
- **Bounds Checking**: Clamps numeric values to `min`/`max` limits.
- **Allowed Values**: Enforces `allowed` list constraints.
- **Required Checks**: Warns if `required` parameters are missing.
- **Unknown Parameters**: Warns and ignores unknown keys.

### 2. Schema Validation
The `validate_schema` function checks the `SCENARIO_PARAMS` definition itself for errors:
- Invalid types
- Inconsistent defaults (e.g., default > max)
- Logical errors (min > max)

### 3. CLI Messaging
Improved CLI output with strict logging formats:
- `[PSS] ...` for informational messages
- `[PSS WARNING] ...` for validation warnings (non-fatal)

### 4. Safe Boolean Parsing
Accepts multiple boolean formats:
- True: `true`, `1`, `yes`, `y`
- False: `false`, `0`, `no`, `n`

## Usage Example

**Scenario Schema (`bulk_ring.py`):**
```python
SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 64, "min": 1, "max": 128},
    "radius": {"type": "float", "default": 12.0, "max": 50.0},
    "speed": {"type": "float", "default": 0.8}
}
```

**CLI Command:**
```bash
python cosmosim.py --scenario bulk_ring --params N=500,radius=100,speed=fast
```

**Output:**
```
[PSS WARNING] N=500 above max=128, clamping
[PSS WARNING] radius=100 above max=50.0, clamping
[PSS WARNING] Invalid type for parameter 'speed'; using default
[PSS] Applied CLI overrides: {'N': '500', 'radius': '100', 'speed': 'fast'}
[PSS] Final merged parameters: {'N': 128, 'radius': 50.0, 'speed': 0.8, ...}
```

## Verification
- **New Tests**: `tests/test_scenario_params_validation.py` (10 tests covering all edge cases)
- **Regression Tests**: `tests/test_scenario_params.py` (24 tests passed)
- **Manual Verification**: Confirmed warnings appear correctly in CLI.
