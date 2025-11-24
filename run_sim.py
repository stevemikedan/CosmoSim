import jax
import jax.numpy as jnp
from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation

def run():
    print("Initializing Standard Simulation (run_sim.py)...")
    cfg = UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )

    state = initialize_state(cfg)

    # Spawn two bodies
    state = spawn_entity(state, jnp.array([-1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)

    print("Starting Loop (50 steps)...")
    for step in range(50):
        state = step_simulation(state, cfg)
        print(f"Step {step}:")
        print(f"  Pos: {state.entity_pos[:2]}")
        print(f"  Vel: {state.entity_vel[:2]}")

if __name__ == "__main__":
    run()
