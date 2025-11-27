"""
Test expansion system - Phase 2.

Tests anisotropic (direction-dependent) expansion including:
- X-only expansion
- Y-only expansion
- Mixed-axis expansion
- Anisotropic stretching effects
"""

import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_anisotropic_expansion_x_only():
    """
    Test expansion only along X axis.
    
    Points on +X should move outward, while points on Y-axis
    with x=0 should not move in X direction.
    """
    print("\n1. Test Anisotropic Expansion (X-only):")
    
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
        bounds=10.0,
        expansion_type="anisotropic",
        expansion_axes=(0.1, 0.0, 0.0),  # Expand only along X
        expansion_center=(0.0, 0.0, 0.0),
    )
    
    env = EnvironmentEngine(cfg)
    
    # One point on +X, one point on +Y
    pos = jnp.array([
        [1.0, 0.0, 0.0],  # Should move outward along X
        [0.0, 1.0, 0.0],  # Should not move in X (x=0)
    ])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # First particle: x should increase
    assert p2[0, 0] > pos[0, 0], "X-axis particle should move outward"
    print(f"   Particle on X-axis:")
    print(f"     Original: {pos[0]}")
    print(f"     Expanded: {p2[0]}")
    
    # Second particle: x should remain ~0 (no X component if x=0)
    assert jnp.abs(p2[1, 0]) < 1e-6, "Y-axis particle should not gain X component"
    print(f"   Particle on Y-axis:")
    print(f"     Original: {pos[1]}")
    print(f"     Expanded: {p2[1]}")
    
    print("   ✓ X-only anisotropic expansion verified")


def test_anisotropic_expansion_mixed():
    """Test expansion with different rates on different axes."""
    print("\n2. Test Anisotropic Expansion (Mixed):")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        bounds=10.0,
        expansion_type="anisotropic",
        expansion_axes=(0.05, 0.10, 0.02),  # Different rates per axis
        expansion_center=(0.0, 0.0, 0.0),
    )
    
    env = EnvironmentEngine(cfg)
    
    # Point at (1, 1, 1)
    pos = jnp.array([[1.0, 1.0, 1.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Calculate displacement in each direction
    dx = p2[0, 0] - pos[0, 0]
    dy = p2[0, 1] - pos[0, 1]
    dz = p2[0, 2] - pos[0, 2]
    
    # Y should expand most (rate=0.10), then X (rate=0.05), then Z (rate=0.02)
    assert dy > dx > dz, "Expansion should be largest along Y, then X, then Z"
    
    print(f"   Original pos: {pos[0]}")
    print(f"   Expanded pos: {p2[0]}")
    print(f"   Displacement: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
    print(f"   Rates: Hx=0.05, Hy=0.10, Hz=0.02")
    print("   ✓ Mixed anisotropic expansion verified")


def test_anisotropic_stretching():
    """Test that anisotropic expansion stretches space correctly."""
    print("\n3. Test Anisotropic Stretching:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        bounds=10.0,
        expansion_type="anisotropic",
        expansion_axes=(0.2, 0.0, 0.0),  # Strong X expansion, no Y/Z
        expansion_center=(0.0, 0.0, 0.0),
    )
    
    env = EnvironmentEngine(cfg)
    
    # Two points symmetrically displaced from origin
    pos = jnp.array([
        [-2.0, 0.0, 0.0],  # Left of origin
        [ 2.0, 0.0, 0.0],  # Right of origin
    ])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Left point should move further left (negative x increases in magnitude)
    assert p2[0, 0] < pos[0, 0], "Left point should move further left"
    
    # Right point should move further right
    assert p2[1, 0] > pos[1, 0], "Right point should move further right"
    
    # Distance between points should increase
    dist_before = pos[1, 0] - pos[0, 0]
    dist_after = p2[1, 0] - p2[0, 0]
    assert dist_after > dist_before, "Distance should increase under expansion"
    
    print(f"   Distance before: {dist_before:.4f}")
    print(f"   Distance after:  {dist_after:.4f}")
    print(f"   Stretching factor: {dist_after/dist_before:.4f}")
    print("   ✓ Anisotropic stretching verified")


def test_compare_linear_vs_anisotropic():
    """Compare linear and anisotropic expansion with same rate."""
    print("\n4. Test Linear vs Anisotropic Equivalence:")
    
    # Linear expansion
    cfg_linear = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        bounds=10.0,
        expansion_type="linear",
        expansion_rate=0.05,
    )
    
    # Anisotropic expansion with same rate on all axes
    cfg_aniso = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        bounds=10.0,
        expansion_type="anisotropic",
        expansion_axes=(0.05, 0.05, 0.05),  # Same on all axes
    )
    
    env_linear = EnvironmentEngine(cfg_linear)
    env_aniso = EnvironmentEngine(cfg_aniso)
    
    pos = jnp.array([[1.0, 2.0, 3.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state_linear = initialize_state(cfg_linear)
    state_aniso = initialize_state(cfg_aniso)
    
    p_linear, v_linear, _ = env_linear.apply_environment(pos, vel, force, state_linear)
    p_aniso, v_aniso, _ = env_aniso.apply_environment(pos, vel, force, state_aniso)
    
    # Should produce identical results
    assert jnp.allclose(p_linear, p_aniso), "Linear and isotropic anisotropic should match"
    assert jnp.allclose(v_linear, v_aniso), "Velocities should match"
    
    print(f"   Linear result:     {p_linear[0]}")
    print(f"   Anisotropic result: {p_aniso[0]}")
    print("   ✓ Linear and isotropic anisotropic are equivalent")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPANSION SYSTEM - PHASE 2 TESTS")
    print("=" * 60)
    
    test_anisotropic_expansion_x_only()
    test_anisotropic_expansion_mixed()
    test_anisotropic_stretching()
    test_compare_linear_vs_anisotropic()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 2 expansion tests passed!")
    print("=" * 60)
