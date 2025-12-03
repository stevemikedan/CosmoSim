# PSS Implementation Plan: Phases 0.x through 1.1

## Overview
Implement foundational Parameterized Scenario System to allow runtime configuration of scenarios via CLI parameters.

## Architecture

### PSS0.x: CLI Interface
```python
# cosmosim.py
parser.add_argument('--params', help='Comma-separated key=value pairs (e.g., N=500,radius=20)')

def parse_param_string(param_str: str) -> dict:
    """Parse 'key=value,key2=value2' into dict."""
    # Returns: {"key": "value", "key2": "value2"} (strings for now)
```

### PSS1.0: Schema Format
```python
# scenarios/bulk_ring.py (optional)
SCENARIO_PARAMS = {
    "N": {
        "type": "int",
        "default": 200,
        "min": 1,
        "max": 2000,
        "description": "Number of entities"
    },
    "radius": {
        "type": "float",
        "default": 20.0,
        "min": 1.0,
        "description": "Ring radius"
    },
    # ... more params
}
```

### PSS1.1: Discovery & Merging
```python
def load_scenario_schema(module) -> dict | None:
    """Extract SCENARIO_PARAMS from module if present."""
    
def merge_params(schema: dict, cli_params: dict) -> dict:
    """
    Merge CLI overrides with schema defaults.
    - Type conversion based on schema
    - Bounds checking (min/max clamping)
    - Warnings for unknown/invalid params
    """
```

## Implementation Steps

### 1. Add CLI Support (PSS0.x)

**File**: `cosmosim.py`

Add argument:
```python
parser.add_argument(
    '--params',
    help='Scenario parameters as key=value pairs (comma-separated)'
)
```

Add parser function:
```python
def parse_param_string(param_str: str | None) -> dict:
    if not param_str:
        return {}
    
    params = {}
    for pair in param_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        params[key.strip()] = value.strip()
    
    return params
```

### 2. Define Schema Format (PSS1.0)

**File**: `scenarios/bulk_ring.py`

Add at module level:
```python
SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 200, "min": 1, "max": 2000},
    "radius": {"type": "float", "default": 20.0, "min": 1.0},
    "thickness": {"type": "float", "default": 3.0, "min": 0.1},
    "mass_min": {"type": "float", "default": 0.5},
    "mass_max": {"type": "float", "default": 2.0},
}
```

### 3. Schema Discovery & Merging (PSS1.1)

**File**: `cosmosim.py`

Add helper functions:
```python
def load_scenario_schema(module) -> dict | None:
    """Load SCENARIO_PARAMS from module."""
    if hasattr(module, 'SCENARIO_PARAMS'):
        schema = module.SCENARIO_PARAMS
        # Validate schema format
        for key, spec in schema.items():
            if 'type' not in spec or 'default' not in spec:
                print(f"Warning: Invalid schema for param '{key}'")
                continue
        return schema
    return None

def merge_params(schema: dict | None, cli_params: dict) -> dict:
    """Merge CLI params with schema defaults."""
    if not schema:
        if cli_params:
            print("[PSS] Warning: CLI params provided but scenario has no schema")
        return {}
    
    merged = {}
    
    # Start with defaults
    for key, spec in schema.items():
        merged[key] = spec['default']
    
    # Apply CLI overrides
    for key, value_str in cli_params.items():
        if key not in schema:
            print(f"[PSS] Warning: Unknown parameter '{key}' ignored")
            continue
        
        spec = schema[key]
        param_type = spec['type']
        
        # Type conversion
        try:
            if param_type == 'int':
                value = int(value_str)
            elif param_type == 'float':
                value = float(value_str)
            elif param_type == 'bool':
                value = value_str.lower() in ('true', '1', 'yes', 'y')
            else:  # str
                value = value_str
            
            # Bounds checking
            if param_type in ('int', 'float'):
                if 'min' in spec and value < spec['min']:
                    print(f"[PSS] Warning: {key}={value} below min={spec['min']}, clamping")
                    value = spec['min']
                if 'max' in spec and value > spec['max']:
                    print(f"[PSS] Warning: {key}={value} above max={spec['max']}, clamping")
                    value = spec['max']
            
            merged[key] = value
            
        except ValueError:
            print(f"[PSS] Warning: Invalid type for parameter '{key}'; using default")
            merged[key] = spec['default']
    
    return merged
```

### 4. Integration into run_scenario()

**File**: `cosmosim.py:run_scenario()`

```python
def run_scenario(module: Any, args: argparse.Namespace, scenario_name: str) -> None:
    # ... existing code ...
    
    # PSS Integration
    schema = load_scenario_schema(module)
    cli_params = parse_param_string(args.params)
    merged_params = merge_params(schema, cli_params)
    
    if schema:
        print(f"[PSS] Loaded schema with {len(schema)} parameters")
    if cli_params:
        print(f"[PSS] Applied CLI overrides: {cli_params}")
    if merged_params:
        print(f"[PSS] Final merged parameters: {merged_params}")
    
    # Build config (existing)
    cfg = module.build_config()
    
    # ... existing config overrides ...
    
    # Build initial state with params
    sig = inspect.signature(module.build_initial_state)
    if 'params' in sig.parameters:
        state = module.build_initial_state(cfg, merged_params)
    else:
        state = module.build_initial_state(cfg)
    
    # ... rest of function ...
```

### 5. Update bulk_ring.py to Use Params

**File**: `scenarios/bulk_ring.py`

Update signature:
```python
def build_initial_state(config: UniverseConfig, params: dict | None = None):
    if params is None:
        params = {k: v['default'] for k, v in SCENARIO_PARAMS.items()}
    
    N = params.get('N', 200)
    radius = params.get('radius', 20.0)
    # ... use params ...
```

## Testing Strategy

### Unit Tests (`tests/test_scenario_params.py`)

1. **parse_param_string tests**:
   - Basic parsing
   - Empty string
   - Malformed input
   - Multiple values

2. **Schema tests**:
   - Load valid schema
   - Handle missing schema
   - Validate schema format

3. **merge_params tests**:
   - Type conversion (int, float, bool, str)
   - Invalid type handling
   - Unknown param warning
   - Bounds checking
   - Missing schema

4. **Integration tests**:
   - Scenario without schema runs
   - Scenario with schema + params runs
   - Backward compatibility

## Backward Compatibility

| Scenario Type | Behavior |
|---------------|----------|
| No SCENARIO_PARAMS | Runs exactly as before, ignores --params |
| Has SCENARIO_PARAMS, no --params | Uses all defaults |
| Has SCENARIO_PARAMS, with --params | Applies overrides |
| build_initial_state(config) | Called without params |
| build_initial_state(config, params) | Called with merged params |

## Constraints Preserved

✅ No physics modifications  
✅ No JSON output changes  
✅ Existing scenarios work unchanged  
✅ All 155 baseline tests pass  
✅ Backward compatible  

## Timeline Estimate

1. **PSS0.x (CLI)**: 20 min
2. **PSS1.0 (Schema)**: 15 min  
3. **PSS1.1 (Discovery/Merge)**: 45 min
4. **Integration**: 30 min
5. **Testing**: 40 min
6. **Documentation**: 20 min

**Total**: ~2.5-3 hours
