"""
Test script for EnvironmentEngine.

Verifies basic functionality of topology coordination, expansion,
and environment pipeline.
"""

import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_environment_basics():
    """Test basic EnvironmentEngine functionality."""
    print("=" * 60)
    print("ENVIRONMENT ENGINE TESTS")
    print("=" * 60)
    
    # Create config
    cfg = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,  # flat
        bounds=10.0
    )
    
    # Create environment
    env = EnvironmentEngine(cfg)
    
    # Test positions, velocities, forces
    pos = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vel = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    force = jnp.zeros_like(pos)
    
    # Create dummy state
    state = initialize_state(cfg)
    
    print("\n1. Test without expansion:")
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    print(f"   Original pos: {pos[0]}")
    print(f"   After env:    {p2[0]}")
    print(f"   Unchanged: {jnp.allclose(pos, p2)}")
    
    print("\n2. Test with expansion:")
    env.set_expansion_rate(0.01)
    p3, v3, f3 = env.apply_environment(pos, vel, force, state)
    print(f"   Original pos: {pos[0]}")
    print(f"   After expansion: {p3[0]}")
    
    # Check that distance increased
    r_before = jnp.linalg.norm(pos[0])
    r_after = jnp.linalg.norm(p3[0])
    print(f"   Distance before: {r_before:.4f}")
    print(f"   Distance after:  {r_after:.4f}")
    assert r_after > r_before, "Expansion should increase distance!"
    
    print("\n3. Test topology wrapping:")
    env.set_expansion_rate(0.0)  # Disable expansion
    out_of_bounds = jnp.array([[15.0, 0.0, 0.0]])  # Outside bounds
    vel_zero = jnp.zeros_like(out_of_bounds)
    force_zero = jnp.zeros_like(out_of_bounds)
    
    p4, v4, f4 = env.apply_environment(out_of_bounds, vel_zero, force_zero, state)
    print(f"   Out of bounds: {out_of_bounds[0]}")
    print(f"   After topology: {p4[0]}")
    print(f"   Clipped to bounds: {jnp.all(jnp.abs(p4) <= cfg.bounds)}")
    
    print("\n4. Test distance calculation:")
    p_a = jnp.array([5.0, 0.0, 0.0])
    p_b = jnp.array([-5.0, 0.0, 0.0])
    dist = env.compute_distance(p_a, p_b)
    print(f"   Distance [5,0,0] to [-5,0,0]: {dist:.2f}")
    expected = 10.0
    assert jnp.allclose(dist, expected), f"Expected {expected}, got {dist}"
    
    print("\n" + "=" * 60)
    print("✅ All EnvironmentEngine tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_environment_basics()
