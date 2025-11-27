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
# from pathlib import Path
from typing import Any

# JSON exporter import
from exporters.json_export import export_simulation, export_frame

__version__ = "0.1"


def load_scenarios() -> dict[str, str]:
    """
    Discover all available scenarios.

    Returns:
        Dictionary mapping short scenario names to full module paths.
    """
    scenarios = {}

    # Manually add root-level simulation modules
    root_modules = [
        "run_sim",
        "jit_run_sim",
        "visualize",
        "snapshot_plot",
        "trajectory_plot",
        "energy_plot",
    ]

    for name in root_modules:
        scenarios[name] = name

    # Auto-discover scenarios in scenarios/ package
    # Use os.getcwd() for test compatibility so tests can patch os.getcwd if needed
    # or rely on the fact that tests run from project root
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
                scenarios[short_name] = full_module

    return scenarios


def validate_interface(module: Any) -> bool:
    """
    Validate that a module conforms to the CosmoSim interface.

    Args:
        module: The module to validate

    Returns:
        True if valid, False otherwise (also prints errors)
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


def run_scenario(module: Any, args: argparse.Namespace, scenario_name: str) -> None:
    """
    Execute a scenario module with the given arguments.

    Args:
        module: The scenario module to run
        args: Parsed command-line arguments
        scenario_name: The short scenario name for display
    """
    # Step 1: Build configuration
    print(f"Building configuration for '{scenario_name}'...")
    cfg = module.build_config()

    # Step 2: Config dump if requested
    if args.config_dump:
        print("\nConfiguration:")
        print(cfg)
        print()

    # Step 3: Build initial state
    print("Initializing universe state...")
    state = module.build_initial_state(cfg)

    # -----------------------------------------------------------------
    # JSON Export Mode
    # -----------------------------------------------------------------
    if getattr(args, "export_json", False):
        import datetime

        # Determine number of frames to export
        steps_value = args.steps if args.steps is not None else 100

        # Generate timestamped export directory name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        export_dir_name = f"{scenario_name}_{steps_value}_steps_{timestamp}"
        
        # Always use frames as the root directory
        full_export_dir = pathlib.Path("frames") / export_dir_name

        # Ensure directory exists
        full_export_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable for other tools
        os.environ["COSMOSIM_EXPORT_JSON_DIR"] = str(full_export_dir.resolve())

        # Export simulation to the timestamped directory
        export_simulation(
            cfg,
            state,
            steps=steps_value,
            output_dir=str(full_export_dir),
        )

        print(f"\nExported {steps_value} JSON frames to: {full_export_dir}")
        return

    # Step 4: Prepare run arguments
    run_kwargs = {}

    # Check if run() accepts steps override
    if args.steps is not None:
        sig = inspect.signature(module.run)
        params = sig.parameters

        if "steps" in params:
            run_kwargs["steps"] = args.steps
        elif "num_steps" in params:
            run_kwargs["num_steps"] = args.steps
        else:
            print(f"Warning: --steps specified but '{scenario_name}' doesn't accept 'steps' or 'num_steps' parameter")

    # Step 5: Set headless mode
    if args.headless:
        os.environ["COSMOSIM_HEADLESS"] = "1"
        print("Running in headless mode...")

    # Step 6: Set output directory
    if args.output_dir:
        output_path = pathlib.Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        os.environ["COSMOSIM_OUTPUT_DIR"] = str(output_path.resolve())
        print(f"Output directory: {output_path.resolve()}")

    # Step 7: Execute scenario
    print(f"Running scenario '{scenario_name}'...\n")

    try:
        # Only pass run_kwargs if we have recognized parameters
        if run_kwargs:
            final_state = module.run(cfg, state, **run_kwargs)
        else:
            final_state = module.run(cfg, state)
    except Exception as e:
        print("\nError during scenario execution:", file=sys.stderr)
        print(f"   {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 8: Success summary
    print(f"\nScenario '{scenario_name}' completed successfully.")


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="cosmosim",
        description="CosmoSim - Unified CLI for N-body physics simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenario", "-s",
        help="Scenario to run (short name or full module path)"
    )

    parser.add_argument(
        "--steps", "-n",
        type=int,
        help="Override number of simulation steps"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (sets COSMOSIM_HEADLESS=1)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        help="Override output directory (sets COSMOSIM_OUTPUT_DIR)"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available scenarios and exit"
    )

    parser.add_argument(
        "--config-dump",
        action="store_true",
        help="Print UniverseConfig before running"
    )

    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export simulation frames to JSON instead of running the scenario normally."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"CosmoSim CLI v{__version__}"
    )

    args = parser.parse_args(argv)

    # Load available scenarios
    scenarios = load_scenarios()

    # Handle --list
    if args.list:
        print("Available scenarios:\n")
        print(f"{'Short Name':<25} -> {'Full Module Path'}")
        print("=" * 60)
        for short_name in sorted(scenarios.keys()):
            full_path = scenarios[short_name]
            print(f"{short_name:<25} -> {full_path}")
        return 0

    # Require --scenario unless --list
    if not args.scenario:
        parser.print_help()
        print("\nError: --scenario is required (use --list to see available scenarios)", file=sys.stderr)
        return 1

    # Resolve scenario name to module path
    scenario_display_name = args.scenario
    if args.scenario in scenarios:
        module_path = scenarios[args.scenario]
    elif "." in args.scenario:
        # Treat as full module path
        module_path = args.scenario
        scenario_display_name = module_path
    else:
        print(f"Error: Unknown scenario '{args.scenario}'", file=sys.stderr)
        print("\nUse --list to see available scenarios, or specify a full module path.", file=sys.stderr)
        return 1

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: Failed to import module '{module_path}':", file=sys.stderr)
        print(f"   {e}", file=sys.stderr)
        return 1

    # Validate interface
    # Import cosmosim to allow patching in tests
    import cosmosim

    if not cosmosim.validate_interface(module):
        return 1

    # Run the scenario
    run_scenario(module, args, scenario_display_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
