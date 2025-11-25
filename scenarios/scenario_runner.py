import jax
import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation


def binary_star_system():
    """Two equal-mass stars orbiting each other."""
    cfg = UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=20.0,
        max_entities=4,
        max_nodes=1,
        dt=0.05,
        c=1.0,
        G=2.0,
    )

    state = initialize_state(cfg)

    # star masses
    m = 5.0

    # positions
    state = spawn_entity(state, jnp.array([-5.0, 0.0]), jnp.array([0.0, -0.5]), m, 1)
    state = spawn_entity(state, jnp.array([ 5.0, 0.0]), jnp.array([0.0,  0.5]), m, 1)

    return cfg, state


def mini_solar_system():
    """A sun and three planets."""
    cfg = UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=30.0,
        max_entities=10,
        max_nodes=1,
        dt=0.03,
        c=1.0,
        G=1.0,
    )

    state = initialize_state(cfg)

    # sun
    state = spawn_entity(state, jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]), 20.0, 1)

    # planets
    planets = [
        (jnp.array([ 4.0, 0.0]), jnp.array([0.0,  1.0]), 1.0),
        (jnp.array([ 8.0, 0.0]), jnp.array([0.0,  0.7]), 2.0),
        (jnp.array([12.0, 0.0]), jnp.array([0.0,  0.5]), 1.5),
    ]

    for pos, vel, mass in planets:
        state = spawn_entity(state, pos, vel, mass, 1)

    return cfg, state


def build_config() -> UniverseConfig:
    cfg, _ = binary_star_system()
    return cfg


def build_initial_state(config: UniverseConfig):
    # Note: We ignore the passed config here because binary_star_system 
    # creates its own config/state pair. In a more advanced version,
    # we might use the passed config to override parameters.
    _, state = binary_star_system()
    return state


def run(config: UniverseConfig, state):
    # Option A: No simulation loop, just return state.
    # This file is a scenario definition library, not a runner.
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
