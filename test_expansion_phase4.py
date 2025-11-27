"""
Test expansion system - Phase 4.

Tests bubble expansion with topology-aware dynamics:
- Radial expansion from center
- Bubble radius growth (inflation)
- Curvature-aware expansion velocity
"""

import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_bubble_expansion_basic():
    """Test basic radial expansion in a bubble."""
    print("\n1. Test Bubble Expansion - Basic Radial:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=3,  # bubble
        bounds=10.0,
        expansion_type="bubble",
        expansion_rate=0.05,
        bubble_radius=10.0,
        bubble_expand=False,
        curvature_k=0.0,  # Flat interior
        expansion_center=(0.0, 0.0, 0.0),
    )
    
    env = EnvironmentEngine(cfg)
    
    # Start a particle near the bubble edge
    pos = jnp.array([[9.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    # Apply expansion for several steps
    for i in range(5):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
    
    # Should move outward
    assert pos[0, 0] > 9.0, "Bubble expansion should push particles outward"
    print(f"   Original pos: 9.0000")
    print(f"   Final pos:    {pos[0, 0]:.4f}")
    
    print("   ✓ Basic bubble expansion verified")


def test_bubble_inflation():
    """Test that the bubble itself expands."""
    print("\n2. Test Bubble Inflation:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=3,
        bounds=10.0,
        expansion_type="bubble",
        expansion_rate=0.10,
        bubble_radius=10.0,
        bubble_expand=True,  # Enable bubble inflation
        curvature_k=0.0,
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[5.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    state = initialize_state(cfg)
    
    print(f"   Initial radius: {env.expansion.bubble_radius:.4f}")
    
    # Run expansion
    for i in range(5):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
    
    final_radius = env.expansion.bubble_radius
    print(f"   Final radius:   {final_radius:.4f}")
    
    # Bubble radius should grow: R_new = R_old * (1 + rate*dt)
    # After 5 steps: R ≈ 10 * (1 + 0.1*0.1)^5 ≈ 10 * 1.01^5 ≈ 10.51
    assert final_radius > 10.0, "Bubble radius should expand"
    
    print("   ✓ Bubble inflation verified")


def test_curvature_effect():
    """Test that curvature affects expansion velocity."""
    print("\n3. Test Curvature Effect:")
    
    # Flat bubble (k=0)
    cfg_flat = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=3,
        bounds=10.0,
        expansion_type="bubble",
        expansion_rate=0.10,
        bubble_radius=10.0,
        curvature_k=0.0,
    )
    
    # Curved bubble (k=0.005)
    cfg_curved = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=3,
        bounds=10.0,
        expansion_type="bubble",
        expansion_rate=0.10,
        bubble_radius=10.0,
        curvature_k=0.005,  # Positive curvature slows expansion near edge
    )
    
    env_flat = EnvironmentEngine(cfg_flat)
    env_curved = EnvironmentEngine(cfg_curved)
    
    # Particle near edge where curvature effect is strongest
    pos = jnp.array([[9.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state_flat = initialize_state(cfg_flat)
    state_curved = initialize_state(cfg_curved)
    
    # Single step
    p_flat, _, _ = env_flat.apply_environment(pos, vel, force, state_flat)
    p_curved, _, _ = env_curved.apply_environment(pos, vel, force, state_curved)
    
    # Calculate displacement
    disp_flat = p_flat[0, 0] - pos[0, 0]
    disp_curved = p_curved[0, 0] - pos[0, 0]
    
    print(f"   Flat displacement:   {disp_flat:.6f}")
    print(f"   Curved displacement: {disp_curved:.6f}")
    
    # Curvature factor sqrt(1 - k*r^2) < 1, so expansion should be slower
    assert disp_curved < disp_flat, "Positive curvature should slow expansion near edge"
    
    print("   ✓ Curvature effect verified")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPANSION SYSTEM - PHASE 4 TESTS")
    print("=" * 60)
    
    test_bubble_expansion_basic()
    test_bubble_inflation()
    test_curvature_effect()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 4 expansion tests passed!")
    print("=" * 60)
