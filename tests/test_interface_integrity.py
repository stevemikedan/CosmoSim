import pytest
import importlib
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state import UniverseConfig, UniverseState

TARGET_MODULES = [
    "run_sim",
    "jit_run_sim",
    "visualize",
    "snapshot_plot",
    "trajectory_plot",
    "energy_plot",
    "scenarios.manual_run",
    "scenarios.random_nbody",
    "scenarios.scenario_runner",
]

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_interface_existence(module_name):
    """Validate that the module exposes the required interface functions."""
    module = importlib.import_module(module_name)
    
    assert hasattr(module, "build_config"), f"{module_name} missing build_config"
    assert hasattr(module, "build_initial_state"), f"{module_name} missing build_initial_state"
    assert hasattr(module, "run"), f"{module_name} missing run"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_build_config_returns_config(module_name):
    """Validate that build_config returns a UniverseConfig."""
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    assert isinstance(cfg, UniverseConfig), f"{module_name}.build_config did not return UniverseConfig"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_build_initial_state_returns_state(module_name):
    """Validate that build_initial_state returns a UniverseState."""
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)
    assert isinstance(state, UniverseState), f"{module_name}.build_initial_state did not return UniverseState"

@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_run_executes_and_returns_state(module_name):
    """Validate that run executes without error and returns UniverseState.
    
    We patch jax.jit to avoid compilation overhead and patch step_simulation
    to be a no-op to avoid computational overhead.
    """
    module = importlib.import_module(module_name)
    cfg = module.build_config()
    state = module.build_initial_state(cfg)

    # Mocking to speed up tests and avoid long loops/files
    # 1. Patch jax.jit to return the function as-is (identity)
    # 2. Patch kernel.step_simulation to return state immediately (no-op)
    # 3. Patch matplotlib.pyplot.savefig to no-op (avoid file IO)
    # 4. Patch matplotlib.pyplot.show to no-op
    
    with patch("jax.jit", side_effect=lambda f, *args, **kwargs: f):
        # We need to patch the step_simulation imported in the module, or the one in kernel
        # Patching kernel.step_simulation is safer as it covers all imports
        with patch("kernel.step_simulation", return_value=state):
             with patch("matplotlib.pyplot.savefig"):
                 with patch("matplotlib.pyplot.show"):
                     # For visualize.py, we might want to reduce FRAMES if possible, 
                     # but since step_simulation is no-op, 300 iterations is fast.
                     
                     final_state = module.run(cfg, state)
                     
                     assert isinstance(final_state, UniverseState), f"{module_name}.run did not return UniverseState"
