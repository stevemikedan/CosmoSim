"""Quick test of topology system."""

import jax.numpy as jnp
from topologies import get_topology_handler
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
    dim=3,
    bounds=10.0
)

# Test positions
p1 = jnp.array([5.0, 0.0, 0.0])
p2 = jnp.array([-5.0, 0.0, 0.0])
out_of_bounds = jnp.array([15.0, 0.0, 0.0])

print("=" * 60)
print("TOPOLOGY SYSTEM TESTS")
print("=" * 60)

# Test Flat Topology
print("\n1. FlatTopology:")
flat = get_topology_handler("flat", cfg)
wrapped = flat.wrap_position(out_of_bounds)
dist = flat.distance(p1, p2)
print(f"   Wrapped [15,0,0]: {wrapped}")
print(f"   Distance [5,0,0] to [-5,0,0]: {dist:.2f}")

# Test Torus Topology
print("\n2. TorusTopology:")
torus = get_topology_handler("torus", cfg)
wrapped = torus.wrap_position(out_of_bounds)
dist = torus.distance(p1, p2)
print(f"   Wrapped [15,0,0]: {wrapped}")
print(f"   Minimum image distance: {dist:.2f}")

# Test Sphere Topology
print("\n3. SphereTopology:")
sphere = get_topology_handler("sphere", cfg)
wrapped = sphere.wrap_position(out_of_bounds)
r = jnp.sqrt(jnp.sum(wrapped ** 2))
dist = sphere.distance(p1, p2)
print(f"   Projected [15,0,0]: {wrapped}")
print(f"   Radius after projection: {r:.2f}")
print(f"   Geodesic distance: {dist:.2f}")

# Test Bubble Topology
print("\n4. BubbleTopology:")
bubble = get_topology_handler("bubble", cfg, curvature_k=0.1)
wrapped = bubble.wrap_position(out_of_bounds)
dist = bubble.distance(p1, p2)
print(f"   Constrained [15,0,0]: {wrapped}")
print(f"   Metric distance: {dist:.2f}")

# Test Hyperbolic (placeholder)
print("\n5. HyperbolicTopology (placeholder):")
hyp = get_topology_handler("hyperbolic", cfg)
wrapped = hyp.wrap_position(p1)
dist = hyp.distance(p1, p2)
print(f"   Position (unchanged): {wrapped}")
print(f"   Distance (Euclidean for now): {dist:.2f}")

print("\n" + "=" * 60)
print("✅ All topology tests completed!")
print("=" * 60)
