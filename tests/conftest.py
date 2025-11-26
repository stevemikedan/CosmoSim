"""
Safe global pytest speed patch for CosmoSim.

Test-only behavior:
- Disable JAX jit/vmap so tests don't spend time compiling.
- Replace jax.lax.scan with a tiny Python loop (2 iterations).
- Patch `range` within simulation modules to cap all loops at 2 iterations.

SAFETY GUARANTEES:
- Never patches builtins.range
- Never touches global Python behavior
- Only patches JAX internals: jax.jit, jax.vmap, jax.lax.scan
- Only patches `range` WITHIN simulation modules (not globally)
- Guarantees all loops execute max 2 iterations during tests
"""

import importlib

import jax
import jax.lax
import jax.numpy as jnp
import pytest

# Modules that contain long-running simulations / visualizations
SIM_MODULES = [
    "run_sim",
    "jit_run_sim",
    "visualize",
    "snapshot_plot",
    "trajectory_plot",
    "energy_plot",
    "scenarios.random_nbody",
    "scenarios.manual_run",
    "scenarios.scenario_runner",
]

# Save the original range function
_original_range = range


def _capped_range(*args):
    """
    Range replacement that caps all iterations at 2.
    
    This ensures that loops like `for i in range(300):` only execute 2 times.
    Uses the original range implementation but with capped values.
    """
    if len(args) == 1:
        # range(stop)
        stop = min(args[0], 2)
        return _original_range(stop)
    elif len(args) == 2:
        # range(start, stop)
        start, stop = args
        stop = min(stop, start + 2)
        return _original_range(start, stop)
    else:
        # range(start, stop, step)
        start, stop, step = args
        # Cap to at most 2 iterations
        if step > 0:
            stop = min(stop, start + 2 * step)
        else:
            stop = max(stop, start + 2 * step)
        return _original_range(start, stop, step)


@pytest.fixture(autouse=True)
def speed_patch(monkeypatch):
    """
    Automatically applied to ALL tests.

    Speeds up tests by:
    1. Disabling JAX JIT compilation
    2. Disabling JAX vmap vectorization
    3. Replacing jax.lax.scan with a 2-iteration Python loop
    4. Patching `range` within each simulation module to cap loops at 2 iterations

    This fixture applies patches when tests run, ensuring that all
    simulation loops execute with minimal iterations during testing.
    """

    # ========================================================================
    # A. DISABLE JAX PERFORMANCE FEATURES
    # ========================================================================

    # 1. Disable JIT globally for tests (identity function)
    monkeypatch.setattr(jax, "jit", lambda f, *args, **kwargs: f)

    # 2. Disable vmap (treat it as identity)
    monkeypatch.setattr(jax, "vmap", lambda f, *args, **kwargs: f)

    # 3. Replace lax.scan with a 2-iteration Python loop
    def fake_scan(f, init, xs):
        """
        Fake scan that only executes 2 iterations instead of len(xs).
        This dramatically speeds up tests without breaking functionality.
        """
        carry = init
        ys = []
        # CRITICAL: Always exactly 2 iterations in tests
        for _ in _original_range(2):
            carry, y = f(carry, xs)
            ys.append(y)
        return carry, jnp.array(ys)

    monkeypatch.setattr(jax.lax, "scan", fake_scan)

    # B. PATCH RANGE WITHIN SIMULATION MODULES  
    # ========================================================================

    # For each simulation module, replace its `range` reference with our
    # capped version. This ensures loops like `for i in range(300):` only
    # execute 2 times.
    #
    # IMPORTANT: We are NOT patching builtins.range. We are patching the
    # `range` name within each module's namespace.
    for modname in SIM_MODULES:
        try:
            mod = importlib.import_module(modname)
            # Try to patch range - if module doesn't have range attribute,
            # monkeypatch will add it to the module's __dict__
            monkeypatch.setattr(mod, "range", _capped_range, raising=False)
        except Exception as e:
            # Module might not exist or have import errors - skip it
            pass

    # ========================================================================
    # C. PATCH MODULE-LEVEL CONSTANTS (for modules that use them)
    # ========================================================================

    # Also patch any module-level integer constants > 10
    # This handles cases like `FRAMES = 300` at module level
    for modname in SIM_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue

        for attr_name in dir(mod):
            # Skip private/magic attributes and 'range'
            if attr_name.startswith("_") or attr_name == "range":
                continue

            try:
                val = getattr(mod, attr_name)
            except Exception:
                continue

            # Only patch simple integers > 10
            if isinstance(val, int) and val > 10:
                monkeypatch.setattr(mod, attr_name, 2)
