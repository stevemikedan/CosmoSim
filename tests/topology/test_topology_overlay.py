"""Quick test of topology overlay system."""

from topology_overlay import get_topology
from state import UniverseConfig

# Create a test config
cfg = UniverseConfig(
    physics_mode=0,
    radius=10.0,
    max_entities=5,
    max_nodes=1,
    dt=0.1,
    c=1.0,
    G=1.0,
    dim=2,
    bounds=10.0
)

# Test grid overlay
print("Testing GridOverlay...")
grid = get_topology('grid', cfg)
grid_data = grid.generate()
print(f"  Type: {grid_data['type']}")
print(f"  Lines: {len(grid_data['lines'])}")
print(f"  Color: {grid_data['color']}")
print(f"  Opacity: {grid_data['opacity']}")

# Test bounds overlay
print("\nTesting BoundsOverlay...")
bounds = get_topology('bounds', cfg)
bounds_data = bounds.generate()
print(f"  Type: {bounds_data['type']}")
print(f"  Lines: {len(bounds_data['lines'])}")
print(f"  Color: {bounds_data['color']}")
print(f"  Opacity: {bounds_data['opacity']}")

# Test none
print("\nTesting None...")
none_overlay = get_topology(None, cfg)
print(f"  Result: {none_overlay}")

print("\n✅ All topology overlay tests passed!")
