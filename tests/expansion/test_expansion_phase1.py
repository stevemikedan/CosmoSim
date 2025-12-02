"""
Test expansion system - Phase 1.

Tests the basic expansion functionality including:
- LinearExpansion model
- EnvironmentEngine integration
- No expansion (baseline)
"""

import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_no_expansion():
    """Test that no expansion leaves everything unchanged."""
    print("\n1. Test No Expansion:")
    
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
        expansion_type="none",  # No expansion
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    assert jnp.allclose(pos, p2), "Position should not change without expansion"
    assert jnp.allclose(vel, v2), "Velocity should not change without expansion"
    
    print("   ✓ No expansion verified")


def test_linear_expansion():
    """Test linear (Hubble-like) expansion."""
    print("\n2. Test Linear Expansion:")
    
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
        expansion_type="linear",
        expansion_rate=0.05,  # H = 0.05
        expansion_center=(0.0, 0.0, 0.0),
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Position should increase (moving away from center)
    assert p2[0, 0] > pos[0, 0], "Position should increase under expansion"
    print(f"   Original pos: {pos[0, 0]:.4f}")
    print(f"   Expanded pos: {p2[0, 0]:.4f}")
    
    # Velocity should increase (Hubble flow)
    assert v2[0, 0] > vel[0, 0], "Velocity should increase under expansion"
    print(f"   Original vel: {vel[0, 0]:.4f}")
    print(f"   Expanded vel: {v2[0, 0]:.4f}")
    
    print("   ✓ Linear expansion verified")


def test_expansion_from_different_center():
    """Test expansion from non-origin center."""
    print("\n3. Test Expansion from Custom Center:")
    
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
        expansion_type="linear",
        expansion_rate=0.05,
        expansion_center=(2.0, 0.0, 0.0),  # Center at x=2
    )
    
    env = EnvironmentEngine(cfg)
    
    # Position at x=1 (to the left of center)
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Position should move AWAY from center (x=2)
    # So x should decrease (move left)
    assert p2[0, 0] < pos[0, 0], "Position should move away from custom center"
    
    print(f"   Center: x={cfg.expansion_center[0]}")
    print(f"   Original pos: x={pos[0, 0]:.4f}")
    print(f"   Expanded pos: x={p2[0, 0]:.4f}")
    print("   ✓ Custom center expansion verified")


def test_dynamic_expansion_rate():
    """Test changing expansion rate dynamically."""
    print("\n4. Test Dynamic Expansion Rate:")
    
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
        expansion_type="linear",
        expansion_rate=0.01,  # Start with slow expansion
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    state = initialize_state(cfg)
    
    # Apply with slow expansion
    p1, v1, f1 = env.apply_environment(pos, vel, force, state)
    delta1 = p1[0, 0] - pos[0, 0]
    
    # Increase expansion rate
    env.set_expansion_rate(0.10)
    
    # Apply with fast expansion
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    delta2 = p2[0, 0] - pos[0, 0]
    
    # Fast expansion should move position more
    assert delta2 > delta1, "Higher expansion rate should cause larger displacement"
    
    print(f"   Slow expansion (H=0.01): Δx = {delta1:.6f}")
    print(f"   Fast expansion (H=0.10): Δx = {delta2:.6f}")
    print("   ✓ Dynamic expansion rate verified")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPANSION SYSTEM - PHASE 1 TESTS")
    print("=" * 60)
    
    test_no_expansion()
    test_linear_expansion()
    test_expansion_from_different_center()
    test_dynamic_expansion_rate()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 1 expansion tests passed!")
    print("=" * 60)
