import pytest
import importlib
import sys
import os
from unittest.mock import patch
import matplotlib.pyplot as plt

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import UniverseConfig, UniverseState

def test_printing_behavior_run_sim(capsys):
    """Confirm run_sim.py prints 'Step' lines."""
    import run_sim
    cfg = run_sim.build_config()
    state = run_sim.build_initial_state(cfg)
    
    # Patch step_simulation to be fast
    with patch("kernel.step_simulation", return_value=state):
        run_sim.run(cfg, state)
        
    captured = capsys.readouterr()
    assert "Step" in captured.out
    assert "Initializing Standard Simulation" in captured.out

def test_printing_behavior_manual_run(capsys):
    """Confirm scenarios.manual_run prints 'Step' lines."""
    from scenarios import manual_run
    cfg = manual_run.build_config()
    state = manual_run.build_initial_state(cfg)
    
    # Patch step_simulation to be fast
    with patch("kernel.step_simulation", return_value=state):
        with patch("jax.jit", side_effect=lambda f: f):
            manual_run.run(cfg, state)
            
    captured = capsys.readouterr()
    assert "Step" in captured.out
    assert "Running manual physics test" in captured.out

FILE_OUTPUT_MODULES = [
    "visualize",
    "snapshot_plot",
    "trajectory_plot",
    "energy_plot",
]

@pytest.mark.parametrize("module_name", FILE_OUTPUT_MODULES)
def test_file_output_behavior(module_name, tmp_path):
    """Confirm that file-generating modules create a file in the expected directory.
    We monkeypatch os.path.join to redirect 'outputs' to tmp_path.
    """
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)
    
    # Real os.path.join to use in our fake
    real_join = os.path.join
    
    def fake_join(*args):
        # If the path starts with "outputs", redirect to tmp_path
        if args and args[0] == "outputs":
            return real_join(str(tmp_path), *args[1:])
        return real_join(*args)
    
    # We also need to patch plt.savefig to actually save (or we can verify the call)
    # But the requirement is "Assert that a file was created".
    # So we must let plt.savefig run.
    # We assume plt.savefig works if the path is valid.
    
    with patch("os.path.join", side_effect=fake_join):
        with patch("jax.jit", side_effect=lambda f, *args, **kwargs: f):
            with patch("kernel.step_simulation", return_value=state):
                 # For energy_plot, we also need to patch compute_energy if it's jitted
                 # But we patched jax.jit to identity, so it runs Python code.
                 # Python code for compute_energy works fine.
                 
                 module.run(cfg, state)
                 
    # Check if any file was created in tmp_path (recursively or flat)
    # The modules create subdirs like tmp_path/animations, tmp_path/snapshots, etc.
    files_found = []
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            files_found.append(os.path.join(root, file))
            
    assert len(files_found) > 0, f"{module_name} did not create any output file in {tmp_path}"

def test_silent_execution_scenario_runner(capsys):
    """Confirm scenario_runner.py executes silently."""
    from scenarios import scenario_runner
    cfg = scenario_runner.build_config()
    state = scenario_runner.build_initial_state(cfg)
    
    scenario_runner.run(cfg, state)
    
    captured = capsys.readouterr()
    assert captured.out == "", "scenario_runner.py should be silent"
