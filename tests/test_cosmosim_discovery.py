"""
Tests for cosmosim.py scenario discovery functionality.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for cosmosim import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cosmosim


def test_load_scenarios_includes_root_modules():
    """Verify that load_scenarios includes all manually-seeded root modules."""
    scenarios = cosmosim.load_scenarios()
    
    required_root_modules = [
        "run_sim",
        "jit_run_sim",
        "visualize",
        "snapshot_plot",
        "trajectory_plot",
        "energy_plot",
    ]
    
    for module_name in required_root_modules:
        assert module_name in scenarios, f"Missing root module: {module_name}"
        assert scenarios[module_name] == module_name


def test_load_scenarios_discovers_scenario_package_modules():
    """Verify that load_scenarios auto-discovers modules in scenarios/ package."""
    scenarios = cosmosim.load_scenarios()
    
    expected_scenarios = [
        "manual_run",
        "random_nbody",
        "scenario_runner",
    ]
    
    for scenario_name in expected_scenarios:
        assert scenario_name in scenarios, f"Missing scenario: {scenario_name}"
        assert scenarios[scenario_name] == f"scenarios.{scenario_name}"


def test_load_scenarios_maps_short_names_correctly():
    """Verify short names map to correct full module paths."""
    scenarios = cosmosim.load_scenarios()
    
    # Root modules map to themselves
    assert scenarios["run_sim"] == "run_sim"
    assert scenarios["visualize"] == "visualize"
    
    # Scenarios package modules have 'scenarios.' prefix
    assert scenarios["manual_run"] == "scenarios.manual_run"
    assert scenarios["random_nbody"] == "scenarios.random_nbody"


def test_load_scenarios_ignores_init_files(tmp_path):
    """Verify that __init__.py files are ignored during discovery."""
    # Create a fake scenarios directory
    fake_scenarios = tmp_path / "scenarios"
    fake_scenarios.mkdir()
    
    # Create __init__.py and a valid scenario file
    (fake_scenarios / "__init__.py").write_text("# init file")
    (fake_scenarios / "test_scenario.py").write_text("# scenario")
    
    # Mock the scenarios directory path in cosmosim
    # We need to patch pathlib.Path because cosmosim uses it directly
    with patch("cosmosim.pathlib.Path") as mock_path_class:
        # Configure the mock instance returned by Path(...)
        mock_path_instance = mock_path_class.return_value
        
        # Configure the glob method on the scenarios_dir (which is base_dir / "scenarios")
        mock_scenarios_dir = MagicMock()
        mock_scenarios_dir.exists.return_value = True
        mock_scenarios_dir.is_dir.return_value = True
        mock_scenarios_dir.glob.return_value = [
            tmp_path / "scenarios" / "__init__.py",
            tmp_path / "scenarios" / "test_scenario.py",
        ]
        
        # When base_dir / "scenarios" is called, return mock_scenarios_dir
        # base_dir is mock_path_instance
        # base_dir / "scenarios" calls __truediv__
        mock_path_instance.__truediv__.return_value = mock_scenarios_dir
        
        scenarios = cosmosim.load_scenarios()
        
        # __init__ should never appear as a scenario name
        assert "__init__" not in scenarios
        assert "test_scenario" in scenarios


def test_load_scenarios_does_not_override_manual_seeds():
    """Verify that auto-discovered scenarios don't override manually-seeded names."""
    # This test verifies the behavior is correct in the real implementation
    scenarios = cosmosim.load_scenarios()
    
    # If there was a 'run_sim.py' in scenarios/, it shouldn't override the root run_sim
    # The root module should map to itself, not to scenarios.run_sim
    assert scenarios["run_sim"] == "run_sim"
    assert scenarios["jit_run_sim"] == "jit_run_sim"
