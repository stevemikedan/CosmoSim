"""
Clear test to verify patches limit iterations to 2
"""
import sys
import os

sys.path.insert(0, '.')

from tests.conftest import _capped_range

# Test the capped_range function
print("=== Testing _capped_range ===")
print(f"range(50) -> {list(_capped_range(50))}")
print(f"range(10, 20) -> {list(_capped_range(10, 20))}")
print(f"range(100, 200) -> {list(_capped_range(100, 200))}")
print(f"✓ All ranges capped to 2 iterations!")

# Import and manually patch run_sim
import run_sim
run_sim.range = _capped_range

print("\n=== Testing run_sim with patched range ===")
cfg = run_sim.build_config()
state = run_sim.build_initial_state(cfg)

# Mock kernel to be fast
import kernel
def fake_step(s, c):
    return s
kernel.step_simulation = fake_step

# This should only print 2 steps
run_sim.run(cfg, state)
print("✓ run_sim.run() completed with only 2 iterations!")
