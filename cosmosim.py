#!/usr/bin/env python3
"""
CosmoSim Unified CLI Entrypoint

This module provides a command-line interface for running all CosmoSim
simulations and scenarios through a unified interface.
"""

import argparse
import importlib
import inspect
import os
import sys
import pathlib
import datetime
from typing import Any

# JSON exporter import
from exporters.json_export import export_simulation
from topologies.mobius_topology import MobiusTopology

############################################################
# NEW: Single-file JSON exporter for the web viewer
############################################################
def export_simulation_single(cfg, state, outfile: str, steps: int = 300):
    """
    Run simulation and export all frames into a single JSON file.
    Viewer-friendly format:
        {
            "config": cfg.to_dict(),
            "frames": [ {pos, vel, mass, active}, ... ]
        }
    """
    from exporters.json_export import get_frame_dict
    import kernel

    import dataclasses
    
    # Serialize config (handle chex.dataclass/dataclass)
    if hasattr(cfg, "to_dict"):
        config_dict = cfg.to_dict()
    elif dataclasses.is_dataclass(cfg):
        config_dict = dataclasses.asdict(cfg)
    else:
        config_dict = {}

    data = {
        "config": config_dict,
        "frames": []
    }

    print(f"[EXPORT] Generating {steps} frames for single-file export...")
    for i in range(steps):
        # Get frame dict using the existing helper
        frame_dict = get_frame_dict(state)
        frame_dict["frame"] = i
        data["frames"].append(frame_dict)
        
        # Step simulation
        state = kernel.step_simulation(state, cfg)

    import json
    with open(outfile, "w") as f:
        json.dump(data, f)

    print(f"[EXPORT] Saved single JSON file: {outfile}")

__version__ = "0.2"

# =====================================================================
# CORE PHYSICS PARAMETERS
# =====================================================================
# These parameters are available to ALL scenarios and map directly to
# UniverseConfig fields. They can be overridden via CLI --params.

CORE_PHYSICS_PARAMS = {
    "dt": {
        "type": "float",
        "default": 0.2,
        "min": 0.001,
        "max": 10.0,
        "description": "Simulation timestep (smaller = more accurate but slower)"
    },
    "G": {
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 1000.0,
        "description": "Gravitational constant"
    },
    "c": {
        "type": "float",
        "default": 1.0,
        "min": 0.001,
        "max": 100.0,
        "description": "Speed of light (limits max velocity)"
    },
    "topology_type": {
        "type": "int",
        "default": 0,
        "allowed": [0, 1, MobiusTopology.MOBIUS_TOPOLOGY],
        "description": "Topology: 0=flat, 1=toroidal, 5=mobius"
    },
    "physics_mode": {
        "type": "int",
        "default": 0,
        "allowed": [0, 1],
        "description": "Physics mode: 0=newtonian, 1=relativistic"
    },
    "max_entities": {
        "type": "int",
        "default": 100,
        "min": 1,
        "max": 100000,
        "description": "Maximum number of entities in simulation"
    },
    "radius": {
        "type": "float",
        "default": 10.0,
        "min": 0.1,
        "max": 1000.0,
        "description": "Universe radius (for bounded topologies)"
    },
    "dim": {
        "type": "int",
        "default": 3,
        "allowed": [2, 3],
        "description": "Spatial dimensionality (2 or 3)"
    },
}



def load_scenarios() -> dict[str, str]:
    """
    Discover all available scenarios.

    Returns:
        Dictionary mapping short scenario names to full module paths.
    """
    # Manually seed short names for backward compatibility
    scenarios = {
        "visualize": "plotting.visualize",
        "snapshot_plot": "plotting.snapshot_plot",
        "trajectory_plot": "plotting.trajectory_plot",
        "energy_plot": "plotting.energy_plot",
    }

    # Add other root-level modules
    root_modules = [
        "run_sim",
        "jit_run_sim",
    ]

    for name in root_modules:
        scenarios[name] = name

    # Also add full plotting.* paths for discovery
    scenarios["plotting.visualize"] = "plotting.visualize"
    scenarios["plotting.snapshot_plot"] = "plotting.snapshot_plot"
    scenarios["plotting.trajectory_plot"] = "plotting.trajectory_plot"
    scenarios["plotting.energy_plot"] = "plotting.energy_plot"

    # Auto-discover scenarios in scenarios/ package
    base_dir = pathlib.Path(os.getcwd())
    scenarios_dir = base_dir / "scenarios"
    if scenarios_dir.exists() and scenarios_dir.is_dir():
        for py_file in scenarios_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            short_name = py_file.stem
            full_module = f"scenarios.{short_name}"

            # Don't override manually-specified names
            if short_name not in scenarios:
                # Check for DEVELOPER_SCENARIO flag
                try:
                    mod = importlib.import_module(full_module)
                    if getattr(mod, "DEVELOPER_SCENARIO", False):
                        continue
                except ImportError:
                    continue
                    
                scenarios[short_name] = full_module

    return scenarios


# =====================================================================
# PSS (Parameterized Scenario System) Support
# =====================================================================

def parse_param_string(param_str: str | None) -> dict:
    """
    Parse comma-separated key=value pairs into a dictionary.
    
    Example: "N=500,radius=20.5,active=true" -> {"N": "500", "radius": "20.5", "active": "true"}
    
    Returns:
        dict with string values (type conversion happens in merge_params)
    """
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


def validate_schema(schema: dict) -> list[str]:
    """
    Validate the schema definition itself.
    
    Checks:
    - Structural correctness (type, default)
    - Min/max applicability
    - Allowed values consistency
    - Default value validity
    
    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            warnings.append(f"Invalid schema for param '{key}' (not a dict)")
            continue
            
        # Check required fields
        if 'type' not in spec:
            warnings.append(f"Param '{key}' missing 'type'")
            continue
        if 'default' not in spec and not spec.get('required', False):
            warnings.append(f"Param '{key}' missing 'default' (and not required)")
            
        param_type = spec['type']
        
        # Check min/max applicability
        if param_type not in ('int', 'float'):
            if 'min' in spec or 'max' in spec:
                warnings.append(f"Param '{key}' has min/max but type is {param_type}")
                
        # Check min < max
        if 'min' in spec and 'max' in spec:
            if spec['min'] > spec['max']:
                warnings.append(f"Param '{key}' has min > max ({spec['min']} > {spec['max']})")
                
        # Check default value validity
        if 'default' in spec:
            default_val = spec['default']
            
            # Type check default
            try:
                if param_type == 'int' and not isinstance(default_val, int):
                    warnings.append(f"Param '{key}' default {default_val} is not int")
                elif param_type == 'float' and not isinstance(default_val, (float, int)):
                    warnings.append(f"Param '{key}' default {default_val} is not float")
                elif param_type == 'bool' and not isinstance(default_val, bool):
                    warnings.append(f"Param '{key}' default {default_val} is not bool")
            except Exception:
                pass
                
            # Bounds check default
            if param_type in ('int', 'float'):
                if 'min' in spec and default_val < spec['min']:
                    warnings.append(f"Param '{key}' default {default_val} < min {spec['min']}")
                if 'max' in spec and default_val > spec['max']:
                    warnings.append(f"Param '{key}' default {default_val} > max {spec['max']}")
                    
            # Allowed check default
            if 'allowed' in spec:
                if default_val not in spec['allowed']:
                    warnings.append(f"Param '{key}' default {default_val} not in allowed values")
                    
    return warnings


def safe_convert_type(value_str: str, target_type: str) -> Any:
    """
    Safely convert string value to target type.
    
    Args:
        value_str: String value from CLI
        target_type: 'int', 'float', 'bool', or 'str'
        
    Returns:
        Converted value
        
    Raises:
        ValueError: If conversion fails
    """
    if target_type == 'int':
        return int(value_str)
    elif target_type == 'float':
        return float(value_str)
    elif target_type == 'bool':
        lower_val = value_str.lower()
        if lower_val in ('true', 'yes', 'y', '1'):
            return True
        if lower_val in ('false', 'no', 'n', '0'):
            return False
        raise ValueError(f"Invalid boolean value: {value_str}")
    else:
        return value_str


def load_scenario_schema(module: Any) -> dict | None:
    """
    Extract SCENARIO_PARAMS schema from module if present.
    
    Returns:
        dict of parameter specs or None if not defined
    """
    if not hasattr(module, 'SCENARIO_PARAMS'):
        return None
    
    schema = module.SCENARIO_PARAMS
    if not isinstance(schema, dict):
        return None
    
    # Validate schema and print warnings
    warnings = validate_schema(schema)
    for warning in warnings:
        print(f"[PSS WARNING] {warning}")
    
    return schema


def load_scenario_presets(module: Any) -> dict | None:
    """
    Extract SCENARIO_PRESETS from module if present.
    
    Returns:
        dict of presets or None if not defined
    """
    if not hasattr(module, 'SCENARIO_PRESETS'):
        return None
    
    presets = module.SCENARIO_PRESETS
    if not isinstance(presets, dict):
        print(f"[PSS WARNING] Invalid presets format in module '{module.__name__}' (not a dict)")
        return None
        
    return presets


def merge_params(schema: dict | None, cli_params: dict) -> dict:
    """
    Merge CLI parameters with schema defaults using robust validation.
    
    Args:
        schema: Parameter schema from SCENARIO_PARAMS
        cli_params: Parsed CLI parameters (strings)
        
    Returns:
        dict with typed, validated, and bounded parameters
    """
    if not schema:
        # Warning handled in run_scenario for Case C
        return {}
    
    merged = {}
    
    # 1. Initialize with defaults
    for key, spec in schema.items():
        if 'default' in spec:
            merged[key] = spec['default']
            
    # 2. Apply CLI overrides
    for key, value_str in cli_params.items():
        if key not in schema:
            print(f"[PSS WARNING] Unknown parameter '{key}' ignored")
            continue
        
        spec = schema[key]
        param_type = spec.get('type', 'str')
        
        try:
            # Type Conversion
            value = safe_convert_type(value_str, param_type)
            
            # Allowed Values Check
            if 'allowed' in spec and value not in spec['allowed']:
                print(f"[PSS WARNING] Value '{value}' not allowed for '{key}'; using default")
                continue
            
            # Bounds Checking
            if param_type in ('int', 'float'):
                if 'min' in spec and value < spec['min']:
                    print(f"[PSS WARNING] {key}={value} below min={spec['min']}, clamping")
                    value = spec['min']
                if 'max' in spec and value > spec['max']:
                    print(f"[PSS WARNING] {key}={value} above max={spec['max']}, clamping")
                    value = spec['max']
            
            merged[key] = value
            
        except ValueError:
            print(f"[PSS WARNING] Invalid type for parameter '{key}'; using default")
            # Keep default value
            
    # 3. Check Required Parameters
    for key, spec in schema.items():
        if spec.get('required', False) and key not in cli_params:
            if 'default' in spec:
                print(f"[PSS WARNING] Required param '{key}' missing; using default")
                # Ensure default is set (it should be from step 1, but safe to ensure)
                if key not in merged:
                    merged[key] = spec['default']
            else:
                print(f"[PSS WARNING] Required param '{key}' missing and no default provided")
                
    return merged


def validate_interface(module: Any) -> bool:
    """
    Validate that a module conforms to the CosmoSim interface.
    """
    required_functions = ["build_config", "build_initial_state", "run"]
    missing = []

    for func_name in required_functions:
        if not hasattr(module, func_name):
            missing.append(func_name)
        elif not callable(getattr(module, func_name)):
            missing.append(func_name)

    if missing:
        print(f"Error: Module '{module.__name__}' is missing required functions:", file=sys.stderr)
        for func in missing:
            print(f"  - {func}()", file=sys.stderr)
        print("\nAll scenarios must implement:", file=sys.stderr)
        print("  - build_config() -> UniverseConfig", file=sys.stderr)
        print("  - build_initial_state(config) -> UniverseState", file=sys.stderr)
        print("  - run(config, state, ...) -> UniverseState", file=sys.stderr)
        return False

    return True


def print_banner(args, scenario_name, output_path=None):
    """Print runtime configuration banner."""
    print("=" * 60)
    print(f"COSMOSIM v{__version__}")
    print("=" * 60)
    print(f"Scenario:   {scenario_name}")
    print(f"View Mode:  {args.view}")
    print(f"Steps:      {args.steps if args.steps else 'Default'}")
    if args.topology: print(f"Topology:   {args.topology}")
    if args.substrate: print(f"Substrate:  {args.substrate}")
    if args.expansion: print(f"Expansion:  {args.expansion}")
    if output_path:     print(f"Output:     {output_path}")
    print("=" * 60)
    print()


def run_scenario(module: Any, args: argparse.Namespace, scenario_name: str) -> None:
    """
    Execute a scenario module with the given arguments.
    """
    # Step 1: PSS - Load and merge parameter schemas
    # Merge CORE_PHYSICS_PARAMS with scenario-specific params
    schema = load_scenario_schema(module)
    
    # Create unified schema: core params + scenario params
    # Scenario params can override core param definitions if needed
    full_schema = {**CORE_PHYSICS_PARAMS}
    if schema:
        full_schema.update(schema)
    
    presets = load_scenario_presets(module)
    cli_params = parse_param_string(getattr(args, 'params', None))
    preset_name = getattr(args, 'preset', None)
    
    merged_params = {}
    
    # 1. Start with Schema Defaults (includes both core and scenario defaults)
    for key, spec in full_schema.items():
        if 'default' in spec:
            merged_params[key] = spec['default']

    # 2. Apply Preset Overrides
    if preset_name:
        if not presets:
            print("[PSS WARNING] Scenario does not define SCENARIO_PRESETS; ignoring --preset")
        elif preset_name not in presets:
            print(f"[PSS WARNING] Unknown preset '{preset_name}' for scenario '{scenario_name}'")
        else:
            print(f"[PSS] Using preset '{preset_name}'")
            preset_values = presets[preset_name]
            
            for key, value in preset_values.items():
                merged_params[key] = value
                
            print(f"[PSS] Preset overrides: {preset_values}")

    # 3. Apply CLI Overrides
    if cli_params:
        if preset_name:
            print("[PSS] CLI overrides applied on top of preset")
        else:
            print(f"[PSS] Applied CLI overrides: {cli_params}")
            
        for key, value in cli_params.items():
            merged_params[key] = value

    # 4. Final Validation using unified schema
    final_params = {}
    for key, value in merged_params.items():
        if key not in full_schema:
            if cli_params and key in cli_params:
                 print(f"[PSS WARNING] Unknown parameter '{key}' ignored")
            continue
            
        spec = full_schema[key]
        param_type = spec.get('type', 'str')
        
        try:
            # Ensure value is correct type (convert strings from CLI)
            if isinstance(value, str):
                value = safe_convert_type(value, param_type)
            
            # Allowed Values Check
            if 'allowed' in spec and value not in spec['allowed']:
                print(f"[PSS WARNING] Value '{value}' not allowed for '{key}'; using default")
                if 'default' in spec:
                    value = spec['default']
                else:
                    continue # Skip if no default
            
            # Bounds Checking
            if param_type in ('int', 'float'):
                if 'min' in spec and value < spec['min']:
                    print(f"[PSS WARNING] {key}={value} below min={spec['min']}, clamping")
                    value = spec['min']
                if 'max' in spec and value > spec['max']:
                    print(f"[PSS WARNING] {key}={value} above max={spec['max']}, clamping")
                    value = spec['max']
            
            final_params[key] = value
            
        except ValueError:
            print(f"[PSS WARNING] Invalid type for parameter '{key}'; using default")
            if 'default' in spec:
                final_params[key] = spec['default']

    # Check Required Parameters (after validation loop)
    for key, spec in full_schema.items():
        if spec.get('required', False) and key not in final_params:
            if 'default' in spec:
                print(f"[PSS WARNING] Required param '{key}' missing; using default")
                final_params[key] = spec['default']
            else:
                print(f"[PSS WARNING] Required param '{key}' missing and no default provided")

    merged_params = final_params

    # PSS Logging Final
    if merged_params:
        print(f"[PSS] Final merged parameters: {merged_params}")
    elif full_schema and not cli_params and not preset_name:
         print(f"[PSS] Using schema defaults")

    # Step 2: Build configuration with params
    print(f"Building configuration for '{scenario_name}'...")
    
    # Check if build_config accepts params
    sig = inspect.signature(module.build_config)
    if 'params' in sig.parameters:
        cfg = module.build_config(merged_params if merged_params else None)
    else:
        cfg = module.build_config()

    # Config dump
    if args.config_dump:
        print("\nConfiguration:")
        print(cfg)
        print()


    # Step 3: Build initial state
    print("Initializing universe state...")
    
    # Check if scenario supports params parameter
    sig = inspect.signature(module.build_initial_state)
    if 'params' in sig.parameters:
        state = module.build_initial_state(cfg, merged_params if merged_params else None)
    else:
        state = module.build_initial_state(cfg)

    # Determine View Mode
    view_mode = args.view
    if view_mode == "auto":
        if args.interactive:
            view_mode = "debug"
        else:
            view_mode = "none"
    
    # Handle legacy flags mapping to view modes
    if args.interactive and view_mode == "auto": view_mode = "debug"

    # Print Banner
    print_banner(args, scenario_name)

    # -----------------------------------------------------------------
    # ROUTING: DEBUG VIEWER
    # -----------------------------------------------------------------
    if view_mode == "debug":
        from viewer.viewer import Viewer
        print(f"Launching interactive viewer...")
        viewer = Viewer(cfg, state, scenario_name=scenario_name, pss_params=merged_params)
        viewer.run()
        return

    # -----------------------------------------------------------------
    # ROUTING: WEB VIEWER (Export)
    # -----------------------------------------------------------------
    # ============================================================
    # NEW: Standalone JSON export (single consolidated file)
    # Triggered whenever --export-json is passed
    # ============================================================
    if args.export_json:
        # Add timestamp to filename to prevent overwriting
        steps_value = args.steps if args.steps is not None else getattr(cfg, "steps", 300)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{scenario_name}_{steps_value}_steps_{timestamp}.json"
        
        outfile = (
            pathlib.Path(args.output_dir) / filename
            if args.output_dir else pathlib.Path("outputs") / filename
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)

        export_simulation_single(cfg, state, str(outfile), steps=steps_value)

        print(f"[PSS] JSON export complete: {outfile}")
        return

    if view_mode == "web":
        print("[INFO] Web view enabled. JSON output prepared for external viewer.")

    # -----------------------------------------------------------------
    # ROUTING: HEADLESS / NONE
    # -----------------------------------------------------------------
    # Prepare run arguments
    run_kwargs = {}
    if args.steps is not None:
        sig = inspect.signature(module.run)
        params = sig.parameters
        if "steps" in params:
            run_kwargs["steps"] = args.steps
        elif "num_steps" in params:
            run_kwargs["num_steps"] = args.steps
        else:
            print(f"Warning: --steps specified but '{scenario_name}' doesn't accept 'steps' or 'num_steps' parameter")

    # Set headless env var
    os.environ["COSMOSIM_HEADLESS"] = "1"
    
    # Output dir override
    if args.output_dir:
        output_path = pathlib.Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        os.environ["COSMOSIM_OUTPUT_DIR"] = str(output_path.resolve())

    # Execute
    print(f"Running scenario headless...\n")
    try:
        if run_kwargs:
            module.run(cfg, state, **run_kwargs)
        else:
            run_sig = inspect.signature(module.run)
            if "steps" in run_sig.parameters:
                # Use args.steps if provided, else try to get from config, else default
                steps_val = args.steps if args.steps is not None else getattr(cfg, "steps", 300)
                module.run(cfg, state, steps=steps_val)
            else:
                module.run(cfg, state)
    except Exception as e:
        print("\nError during scenario execution:", file=sys.stderr)
        print(f"   {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nScenario '{scenario_name}' completed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cosmosim",
        description="CosmoSim - Unified CLI for N-body physics simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core Arguments
    parser.add_argument("--scenario", "-s", help="Scenario to run (e.g. 'bulk_ring')")
    parser.add_argument("--steps", "-n", type=int, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, help="Time step size")
    parser.add_argument("--entities", "-N", type=int, help="Number of entities")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Scenario Parameters (PSS)
    parser.add_argument(
        "--params",
        help="Scenario parameters as comma-separated key=value pairs (e.g., N=500,radius=20)"
    )
    parser.add_argument("--preset", help="Name of a preset configuration to use")
    
    # Physics Parameters
    parser.add_argument("--topology", help="Topology type (flat, torus, etc)")
    parser.add_argument("--substrate", help="Substrate type")
    parser.add_argument("--expansion", help="Expansion mode")

    # View / Output Control
    parser.add_argument(
        "--view", 
        choices=["auto", "debug", "web", "none"], 
        default="auto",
        help="Viewer mode: 'debug' (interactive), 'web' (export for browser), 'none' (headless)"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Alias for --view debug")
    parser.add_argument("--export-json", action="store_true", help="Export simulation as single JSON file")
    parser.add_argument("--export-frames", action="store_true", help="Export simulation as per-step frame files")
    
    parser.add_argument("--headless", action="store_true", help="Force headless mode (deprecated, use --view none)")
    parser.add_argument("--output-dir", "-o", help="Override output directory")
    parser.add_argument("--config-dump", action="store_true", help="Print config before running")
    parser.add_argument("--list", "-l", action="store_true", help="List scenarios")
    parser.add_argument("--version", action="version", version=f"CosmoSim CLI v{__version__}")

    args = parser.parse_args(argv)

    # Load scenarios
    scenarios = load_scenarios()

    if args.list:
        print("Available scenarios:\n")
        print(f"{'Short Name':<25} -> {'Full Module Path'}")
        print("=" * 60)
        for short_name in sorted(scenarios.keys()):
            print(f"{short_name:<25} -> {scenarios[short_name]}")
        return 0

    if not args.scenario:
        parser.print_help()
        print("\nError: --scenario is required.", file=sys.stderr)
        return 1

    # Resolve Scenario
    scenario_name = args.scenario
    module_path = scenarios.get(scenario_name)
    
    if not module_path:
        # Try prefixing with scenarios.
        if "." not in scenario_name:
            candidate = f"scenarios.{scenario_name}"
            try:
                importlib.import_module(candidate)
                module_path = candidate
            except ImportError:
                pass
        
        if not module_path:
            # Try as direct path
            module_path = scenario_name

    # Import
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        if module_path not in scenarios.values() and scenario_name not in scenarios:
            print(f"Error: Unknown scenario '{scenario_name}'", file=sys.stderr)
            print("\nUse --list to see available scenarios, or specify a full module path.", file=sys.stderr)
        else:
            print(f"Error: Failed to import scenario '{scenario_name}' ({module_path}):", file=sys.stderr)
            print(f"   {e}", file=sys.stderr)
        return 1

    # Validate
    import cosmosim # Self-import for validation hook if needed
    if not validate_interface(module):
        return 1

    # Run
    run_scenario(module, args, scenario_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
