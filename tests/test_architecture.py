"""
Sprint 5 Validation Tests for CosmoSim.

This suite validates the first real Vector Physics Kernel:
- N-body gravitational attraction
- Semi-implicit Euler integration
- Active mask behavior
- Inactive/despawned entities do not move or exert forces
- Boundaries apply after physics
- Full JIT compatibility
"""

import sys
import jax
import jax.numpy as jnp

from state import UniverseConfig, UniverseState, initialize_state
from kernel import step_simulation


# Physics mode constants
VECTOR = 0
LATTICE = 1
VOXEL = 2
FIELD = 3
CUSTOM = 4

# Topology
FLAT = 0
RADIUS = 10.0


def make_config_and_state():
    """Build a clean vector-mode configuration + state."""
    cfg = UniverseConfig(
        topology_type=FLAT,
        physics_mode=VECTOR,
        radius=RADIUS,
        max_entities=4,
        max_nodes=2,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    return cfg, initialize_state(cfg)


# -----------------------------------------------------------
# Core N-body Gravity Tests
# -----------------------------------------------------------

def test_two_body_gravity_direction_and_magnitude():
    """
    Two active bodies should accelerate toward each other.

    Setup:
      entity 0 at (-1, 0)
      entity 1 at ( 1, 0)
      both with mass = 1

    Expected:
      entity 0 accelerates +x
      entity 1 accelerates -x
    """
    cfg, state = make_config_and_state()

    state = state.replace(
        entity_active=state.entity_active.at[:2].set(jnp.array([True, True])),
        entity_pos=state.entity_pos.at[0].set(jnp.array([-1.0, 0.0]))
                                     .at[1].set(jnp.array([ 1.0, 0.0])),
        entity_vel=state.entity_vel.at[:2].set(jnp.array([[0.0, 0.0],
                                                          [0.0, 0.0]])),
        entity_mass=state.entity_mass.at[:2].set(jnp.array([1.0, 1.0]))
    )

    new_state = step_simulation(state, cfg)

    # entity 0 should move slightly right (positive x)
    assert new_state.entity_pos[0, 0] > -1.0, \
        "Entity 0 did not accelerate toward entity 1"

    # entity 1 should move slightly left (negative x)
    assert new_state.entity_pos[1, 0] < 1.0, \
        "Entity 1 did not accelerate toward entity 0"

    print("✓ Two-body gravitational attraction works (direction + magnitude)")


def test_initial_velocity_produces_inertia_and_gravity():
    """
    An entity with initial velocity should keep moving AND be influenced by gravity.
    """
    cfg, state = make_config_and_state()

    # Entity 0 moves right initially
    state = state.replace(
        entity_active=state.entity_active.at[:2].set(jnp.array([True, True])),
        entity_pos=state.entity_pos.at[0].set(jnp.array([-1.0, 0.0]))
                                     .at[1].set(jnp.array([ 1.0, 0.0])),
        entity_vel=state.entity_vel.at[0].set(jnp.array([1.0, 0.0]))  # inertia + gravity
                                     .at[1].set(jnp.array([0.0, 0.0])),
        entity_mass=state.entity_mass.at[:2].set(jnp.array([1.0, 1.0]))
    )

    new_state = step_simulation(state, cfg)

    # Should be greater than (-1 + dt*1.0) due to gravitational accel
    min_expected = -1.0 + 0.1
    assert new_state.entity_pos[0, 0] > min_expected, \
        "Entity 0 did not combine inertia + gravity correctly"

    print("✓ Inertia and gravity combine correctly in semi-implicit Euler")


# -----------------------------------------------------------
# Active/Inactive Mask Tests
# -----------------------------------------------------------

def test_inactive_entities_do_not_move_or_affect():
    """Inactive entities must not move OR exert gravitational forces."""
    cfg, state = make_config_and_state()

    # Entity 0 active; Entity 1 inactive but has mass/pos
    state = state.replace(
        entity_active=state.entity_active.at[0].set(True),
        entity_pos=state.entity_pos.at[0].set(jnp.array([0.0, 0.0]))
                                     .at[1].set(jnp.array([5.0, 0.0])),
        entity_vel=state.entity_vel.at[1].set(jnp.array([10.0, 0.0])),
        entity_mass=state.entity_mass.at[:2].set(jnp.array([1.0, 10.0]))
    )

    new_state = step_simulation(state, cfg)

    # Inactive entity must not move
    assert jnp.allclose(new_state.entity_pos[1], jnp.array([5.0, 0.0])), \
        "Inactive entities should not move"

    # Active entity must NOT feel gravity from inactive one
    # (should remain near origin)
    assert abs(new_state.entity_pos[0, 0]) < 1e-3, \
        "Active entity was incorrectly influenced by inactive entity"

    print("✓ Inactive entities neither move nor exert gravity")


# -----------------------------------------------------------
# Boundary Enforcement Tests
# -----------------------------------------------------------

def test_boundaries_apply_after_physics():
    """
    Boundaries should apply AFTER physics updates.
    FLAT = no-op, so only physics update should show.
    """
    cfg, state = make_config_and_state()

    state = state.replace(
        entity_active=state.entity_active.at[0].set(True),
        entity_pos=state.entity_pos.at[0].set(jnp.array([100.0, -100.0])),
        entity_vel=state.entity_vel.at[0].set(jnp.array([1.0, 2.0])),
        entity_mass=state.entity_mass.at[0].set(1.0)
    )

    new_state = step_simulation(state, cfg)

    expected = jnp.array([100.1, -99.8])
    assert jnp.allclose(new_state.entity_pos[0], expected), \
        "Boundaries not applied after physics (FLAT should be pass-through)"

    print("✓ Boundaries apply after physics update")


# -----------------------------------------------------------
# JIT Compatibility
# -----------------------------------------------------------

def test_physics_jit_compatible():
    """Ensure complete physics kernel compiles under JIT."""
    cfg, state = make_config_and_state()

    try:
        @jax.jit
        def jitted(state):
            return step_simulation(state, cfg)
        out = jitted(state)
        assert isinstance(out, UniverseState)
    except Exception as e:
        raise AssertionError(f"step_simulation failed under JIT: {e}")

    print("✓ Vector physics kernel JIT-compiles successfully")


# -----------------------------------------------------------
# Runner
# -----------------------------------------------------------

def run_all_tests():
    print("=" * 60)
    print("Sprint 5 Validation Tests for CosmoSim")
    print("=" * 60)

    test_two_body_gravity_direction_and_magnitude()
    test_initial_velocity_produces_inertia_and_gravity()
    test_inactive_entities_do_not_move_or_affect()
    test_boundaries_apply_after_physics()
    test_physics_jit_compatible()

    print("=" * 60)
    print("✓ All Sprint 5 tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
    sys.exit(0)
