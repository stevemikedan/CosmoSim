"""
Test substrate system - Phase 1.

Tests basic substrate functionality:
- VectorFieldSubstrate initialization
- Force calculation
- EnvironmentEngine integration
"""

import jax.numpy as jnp
import numpy as np
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_vector_field_substrate_basic():
    """Test basic vector field substrate functionality."""
    print("\n1. Test Vector Field Substrate - Basic:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=5.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,  # flat
        substrate="vector",
        substrate_params={
            "grid_size": (5, 5, 5),
            "amplitude": 1.0,
            "noise": False
        },
        bounds=5.0
    )
    
    env = EnvironmentEngine(cfg)
    
    # Test position at (1, 1, 1)
    pos = jnp.array([[1.0, 1.0, 1.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Force should be non-zero (random vector field)
    # Note: f2 includes original force (0) + substrate force
    force_mag = jnp.linalg.norm(f2)
    print(f"   Force magnitude: {force_mag:.6f}")
    
    assert force_mag > 0.0, "Vector field should produce non-zero force"
    
    print("   ✓ Vector field force verified")


def test_vector_field_interpolation():
    """Test that force varies with position (interpolation)."""
    print("\n2. Test Vector Field Interpolation:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=5.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        substrate="vector",
        substrate_params={
            "grid_size": (5, 5, 5),
            "amplitude": 1.0,
            "noise": False
        },
        bounds=5.0
    )
    
    env = EnvironmentEngine(cfg)
    
    # Two different positions
    pos1 = jnp.array([[1.0, 1.0, 1.0]])
    pos2 = jnp.array([[-1.0, -1.0, -1.0]])
    vel = jnp.zeros((1, 3))
    force = jnp.zeros((1, 3))
    
    state = initialize_state(cfg)
    
    _, _, f1 = env.apply_environment(pos1, vel, force, state)
    _, _, f2 = env.apply_environment(pos2, vel, force, state)
    
    # Forces should be different (random field)
    # (Probability of exact match is negligible)
    diff = jnp.linalg.norm(f1 - f2)
    print(f"   Force difference: {diff:.6f}")
    
    assert diff > 0.0, "Forces at different positions should differ"
    
    print("   ✓ Interpolation verified")


def test_null_substrate():
    """Test that null substrate does nothing."""
    print("\n3. Test Null Substrate:")
    
    cfg = UniverseConfig(
        physics_mode=0,
        radius=5.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=0,
        substrate="none",  # Null substrate
        bounds=5.0
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 1.0, 1.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    p2, v2, f2 = env.apply_environment(pos, vel, force, state)
    
    # Force should remain zero
    assert jnp.allclose(f2, 0.0), "Null substrate should not produce force"
    
    print("   ✓ Null substrate verified")


if __name__ == "__main__":
    print("=" * 60)
    print("SUBSTRATE SYSTEM - PHASE 1 TESTS")
    print("=" * 60)
    
    test_vector_field_substrate_basic()
    test_vector_field_interpolation()
    test_null_substrate()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 1 substrate tests passed!")
    print("=" * 60)
