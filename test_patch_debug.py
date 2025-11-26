"""Quick test to verify patches are working"""
import importlib

# Import modules AFTER conftest would have run
import run_sim
import visualize

print("=== CHECKING PATCHES ===")
print(f"visualize.FRAMES = {visualize.FRAMES}")
print(f"run_sim module attributes with int > 10:")
for attr in dir(run_sim):
    if not attr.startswith("_"):
        try:
            val = getattr(run_sim, attr)
            if isinstance(val, int) and val > 10:
                print(f"  {attr} = {val}")
        except:
            pass
