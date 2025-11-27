"""
Test expansion system - Phase 3.

Tests scale factor expansion with cosmological a(t) dynamics:
- Linear mode: a(t) = 1 + H*t
- Matter-dominated: a(t) ∝ t^(2/3)
- Radiation-dominated: a(t) ∝ t^(1/2)
- Inflation: a(t) = exp(H*t)
"""

import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from environment import EnvironmentEngine


def test_scale_factor_inflation():
    """Test exponential inflation mode."""
    print("\n1. Test Scale Factor - Inflation Mode:")
    
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
        expansion_type="scale_factor",
        expansion_mode="inflation",
        H=2.0,
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    # Run expansion for several steps
    print(f"   Initial pos: {pos[0, 0]:.4f}")
    
    for i in range(5):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
        print(f"   Step {i+1}: pos={pos[0, 0]:.4f}, a={env.expansion.a:.4f}")
    
    # Inflation should dramatically push pos outward
    assert pos[0, 0] > 1.5, "Inflation mode should strongly expand space"
    
    print("   ✓ Inflation expansion verified")


def test_scale_factor_linear():
    """Test linear growth mode."""
    print("\n2. Test Scale Factor - Linear Mode:")
    
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
        expansion_type="scale_factor",
        expansion_mode="linear",
        H=0.10,
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    print(f"   Initial: a={env.expansion.a:.4f}, pos={pos[0, 0]:.4f}")
    
    # Run a few steps
    for i in range(3):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
        print(f"   Step {i+1}: a={env.expansion.a:.4f}, pos={pos[0, 0]:.4f}")
    
    # Linear: a = 1 + H*t, so after 3 steps at dt=0.1: a ≈ 1 + 0.10*0.3 = 1.03
    assert env.expansion.a > 1.0, "Scale factor should grow"
    assert pos[0, 0] > 1.0, "Position should expand"
    
    print("   ✓ Linear growth verified")


def test_scale_factor_matter():
    """Test matter-dominated mode."""
    print("\n3. Test Scale Factor - Matter-Dominated:")
    
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
        expansion_type="scale_factor",
        expansion_mode="matter",
        H=0.05,  # Not used in matter mode
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    print(f"   Initial: a={env.expansion.a:.4f}, t={env.expansion.t:.4f}")
    
    # Run several steps
    for i in range(5):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
    
    # Matter: a ∝ t^(2/3)
    print(f"   Final: a={env.expansion.a:.4f}, t={env.expansion.t:.4f}")
    print(f"   Final pos: {pos[0, 0]:.4f}")
    
    # Check that a grows as t^(2/3)
    expected_a = env.expansion.t ** (2.0/3.0)
    assert jnp.abs(env.expansion.a - expected_a) < 0.01, "a should follow t^(2/3)"
    
    print("   ✓ Matter-dominated verified")


def test_scale_factor_radiation():
    """Test radiation-dominated mode."""
    print("\n4. Test Scale Factor - Radiation-Dominated:")
    
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
        expansion_type="scale_factor",
        expansion_mode="radiation",
        H=0.05,  # Not used in radiation mode
    )
    
    env = EnvironmentEngine(cfg)
    
    pos = jnp.array([[1.0, 0.0, 0.0]])
    vel = jnp.zeros_like(pos)
    force = jnp.zeros_like(pos)
    
    state = initialize_state(cfg)
    
    print(f"   Initial: a={env.expansion.a:.4f}, t={env.expansion.t:.4f}")
    
    # Run several steps
    for i in range(5):
        pos, vel, force = env.apply_environment(pos, vel, force, state)
    
    # Radiation: a ∝ t^(1/2)
    print(f"   Final: a={env.expansion.a:.4f}, t={env.expansion.t:.4f}")
    print(f"   Final pos: {pos[0, 0]:.4f}")
    
    # Check that a grows as t^(1/2)
    expected_a = env.expansion.t ** 0.5
    assert jnp.abs(env.expansion.a - expected_a) < 0.01, "a should follow t^(1/2)"
    
    print("   ✓ Radiation-dominated verified")


def test_compare_expansion_modes():
    """Compare growth rates of different modes."""
    print("\n5. Test Compare Expansion Modes:")
    
    modes = ["linear", "matter", "radiation", "inflation"]
    results = {}
    
    for mode in modes:
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
            expansion_type="scale_factor",
            expansion_mode=mode,
            H=0.05,
        )
        
        env = EnvironmentEngine(cfg)
        
        pos = jnp.array([[1.0, 0.0, 0.0]])
        vel = jnp.zeros_like(pos)
        force = jnp.zeros_like(pos)
        state = initialize_state(cfg)
        
        # Run 10 steps
        for _ in range(10):
            pos, vel, force = env.apply_environment(pos, vel, force, state)
        
        results[mode] = {
            'a': env.expansion.a,
            'pos': pos[0, 0]
        }
    
    print(f"   After 10 steps:")
    for mode, res in results.items():
        print(f"     {mode:12s}: a={res['a']:.4f}, pos={res['pos']:.4f}")
    
    # Inflation should grow fastest
    assert results['inflation']['a'] > results['linear']['a'], "Inflation should grow fastest"
    assert results['inflation']['a'] > results['matter']['a'], "Inflation should outpace matter"
    assert results['inflation']['a'] > results['radiation']['a'], "Inflation should outpace radiation"
    
    print("   ✓ Expansion mode comparison verified")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPANSION SYSTEM - PHASE 3 TESTS")
    print("=" * 60)
    
    test_scale_factor_inflation()
    test_scale_factor_linear()
    test_scale_factor_matter()
    test_scale_factor_radiation()
    test_compare_expansion_modes()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 3 expansion tests passed!")
    print("=" * 60)
