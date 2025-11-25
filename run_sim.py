import jax
import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation

def build_config() -> UniverseConfig:
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    # Spawn two bodies
    state = spawn_entity(state, jnp.array([-1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    return state


def run(config: UniverseConfig, state):
    print("Initializing Standard Simulation (run_sim.py)...")
    print("Starting Loop (50 steps)...")
    for step in range(50):
        state = step_simulation(state, config)
        print(f"Step {step}:")
        print(f"  Pos: {state.entity_pos[:2]}")
        print(f"  Vel: {state.entity_vel[:2]}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
