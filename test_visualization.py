"""
Sprint 6 Visualization Test Suite for CosmoSim.

These tests verify that all visualization scripts:
- Import correctly
- Create the correct output directories
- Produce PNG files with timestamp-based names
- Do not crash when run
- Use non-interactive matplotlib backend (Agg)
"""

import os
import re
import importlib
import subprocess
from pathlib import Path

OUTPUT_ROOT = Path("outputs")
SUBDIRS = ["snapshots", "energy", "trajectories", "animations"]

VIS_SCRIPTS = [
    "trajectory_plot",
    "snapshot_plot",
    "energy_plot",
    "visualize",
]


def test_imports():
    """All visualization modules must import successfully."""
    for module in VIS_SCRIPTS:
        try:
            importlib.import_module(module)
        except Exception as e:
            raise AssertionError(f"Failed to import {module}: {e}")


def test_output_directories_exist():
    """Visualization scripts should create output subdirectories."""
    for folder in SUBDIRS:
        path = OUTPUT_ROOT / folder
        assert path.exists(), f"Missing expected directory: {path}"


def run_script(script_name):
    """Helper: run a script via subprocess and ensure no crash."""
    import sys
    cmd = [sys.executable, f"{script_name}.py"] 
    try:
        subprocess.run(cmd, check=True, timeout=20)
    except Exception as e:
        raise AssertionError(f"{script_name}.py failed to execute: {e}")


def test_scripts_execute_without_crashing():
    """All visualization scripts should run without raising errors."""
    for script in VIS_SCRIPTS:
        run_script(script)


def test_png_output_created():
    """Each script should produce at least one PNG with correct naming."""
    png_pattern = re.compile(r"(trajectory|snapshot|energy|animation)_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_steps?\d+\.png")

    found_any = False

    for folder in SUBDIRS:
        for file in (OUTPUT_ROOT / folder).glob("*.png"):
            if png_pattern.match(file.name):
                found_any = True

    assert found_any, "No valid PNG outputs found after running scripts."


def test_matplotlib_agg_backend():
    """Verify that scripts use matplotlib Agg backend."""
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg", \
        f"Matplotlib backend must be Agg, not {matplotlib.get_backend()}"
