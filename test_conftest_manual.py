"""Test if conftest patches actually work"""
import sys
sys.path.insert(0, "tests")

# Import conftest manually
from conftest import speed_patch, _capped_range

print("=== Testing _capped_range ===")
print(f"range(50): {list(_capped_range(50))}")
print(f"range(10, 20): {list(_capped_range(10, 20))}")  
print(f"range(0, 100, 10): {list(_capped_range(0, 100, 10))}")

# Test that modules can be imported
print("\n=== Testing module imports ===")
try:
    import run_sim
    print(f"✓ run_sim imported")
except Exception as e:
    print(f"✗ run_sim failed: {e}")

try:
    import visualize
    print(f"✓ visualize imported, FRAMES={visualize.FRAMES}")
except Exception as e:
    print(f"✗ visualize failed: {e}")
