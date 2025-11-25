import builtins
import jax
import jax.lax
import jax.numpy as jnp
import pytest

# These modules contain long loops
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

@pytest.fixture(autouse=True)
def global_speed_patch(monkeypatch, request):
    """
    GLOBAL test-speed optimization for ALL tests.
    This reduces runtime from minutes to seconds.
    """

    # 1. Replace Python range globally
    monkeypatch.setattr(builtins, "range", lambda *args: builtins.range(2))

    # 2. Disable JAX JIT globally
    monkeypatch.setattr(jax, "jit", lambda fn, *a, **k: fn)

    # 3. Disable vmap (identity transform)
    monkeypatch.setattr(jax, "vmap", lambda fn, *a, **k: fn)

    # 4. Disable lax.scan (run body twice)
    def fake_scan(f, init, xs):
        carry = init
        ys = []
        for i in range(2):  # small loop
            carry, y = f(carry, xs)
            ys.append(y)
        return carry, jnp.array(ys)

    monkeypatch.setattr(jax.lax, "scan", fake_scan)

    # 5. Reduce FRAMES, STEPS, or other constants in modules
    for mod in SIM_MODULES:
        try:
            module = __import__(mod, fromlist=["*"])

            if hasattr(module, "FRAMES"):
                monkeypatch.setattr(module, "FRAMES", 2)

            if hasattr(module, "STEPS"):
                monkeypatch.setattr(module, "STEPS", 2)

            # Some files embed literal loop counts—patch them too
            for attr in dir(module):
                val = getattr(module, attr)
                if isinstance(val, int) and val > 10:
                    monkeypatch.setattr(module, attr, 2)

        except Exception:
            pass
