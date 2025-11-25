"""
CosmoSim Trajectory Plot Script (Sprint 6.6)

Runs a CosmoSim simulation, records entity positions over time,
and saves a trajectory plot PNG under outputs/trajectories/.
"""

import os
import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from state import UniverseConfig, initialize_state
from entities import spawn_entity
from kernel import step_simulation


def build_config() -> UniverseConfig:
    return UniverseConfig(
        topology_type=0,   # FLAT
        physics_mode=0,    # VECTOR
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    # Spawn two symmetric entities
    state = spawn_entity(state, jnp.array([-1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    return state


def run(config: UniverseConfig, state):
    steps = 300
    output_dir = os.path.join("outputs", "trajectories")
    
    print("Initializing Trajectory Simulation...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # JIT the physics step for speed
    jit_step = jax.jit(step_simulation)

    # Record positions over time
    trajectory = []  # list of (n_entities, 2) arrays

    print(f"Running {steps} steps for trajectory capture...")
    for _ in range(steps):
        state = jit_step(state, config)
        active_mask = state.entity_active
        positions = state.entity_pos[active_mask]
        trajectory.append(positions)

    # Stack into (steps, n_entities, 2)
    traj_array = jnp.stack(trajectory)  # assumes constant n_active (true here)

    # Plot each entity's path
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    n_entities = traj_array.shape[1]
    for i in range(n_entities):
        xs = traj_array[:, i, 0]
        ys = traj_array[:, i, 1]
        color = colors[i % len(colors)]
        ax.plot(xs, ys, ".-", color=color, label=f"Entity {i}")
        ax.scatter(xs[0], ys[0], c=color, marker="o", s=80)   # start
        ax.scatter(xs[-1], ys[-1], c="k", marker="x", s=80)   # end

    ax.set_title(f"CosmoSim Trajectories over {steps} steps")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"trajectory_{timestamp}_steps{steps}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved output to {output_path}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
