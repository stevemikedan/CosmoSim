import os
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from kernel import step_simulation
import jax

def run_plot(config, state):
    """
    Original random_nbody run with internal simulation and plotting.
    NOT compatible with JSON export mode.
    """
    steps = 400
    n = config.max_entities
    
    print(f"Running Random N-Body Test with {n} bodies...")

    jit_step = jax.jit(step_simulation)

    traj = []

    for _ in range(steps):
        state = jit_step(state, config)
        active_mask = state.entity_active
        traj.append(state.entity_pos[active_mask])

    traj = jnp.stack(traj)  # (steps, n, dim)

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

    # Standardized output handling
    output_dir = os.path.join("outputs", "random_nbody")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output = os.path.join(output_dir, f"random_nbody_{timestamp}.png")

    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    return state
