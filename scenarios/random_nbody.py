import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation


def build_config() -> UniverseConfig:
    # Default values from original function
    radius = 10.0
    n = 25
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=radius,
        max_entities=n,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    
    radius = config.radius
    n = config.max_entities
    velocity_scale = 0.3

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.normal(k1, (n, 2)) * (radius * 0.5)
    velocities = jax.random.normal(k2, (n, 2)) * velocity_scale
    masses = jax.random.uniform(k3, (n,)) * 1.5 + 0.3

    # Spawn all bodies
    for i in range(n):
        state = spawn_entity(
            state,
            positions[i],
            velocities[i],
            masses[i],
            1,
        )
    return state


def run(config: UniverseConfig, state):
    steps = 400
    output = "random_nbody.png"
    n = config.max_entities
    
    print(f"Running Random N-Body Test with {n} bodies...")

    jit_step = jax.jit(step_simulation)

    traj = []

    for _ in range(steps):
        state = jit_step(state, config)
        active_mask = state.entity_active
        traj.append(state.entity_pos[active_mask])

    traj = jnp.stack(traj)  # (steps, n, 2)

    plt.figure(figsize=(8, 8))
    for i in range(n):
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]
        plt.plot(xs, ys, linewidth=1)

    plt.title(f"Random N-Body System ({n} entities)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)

    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
