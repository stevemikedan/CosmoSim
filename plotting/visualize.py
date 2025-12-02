"""
CosmoSim Visualization Script (Sprint 6.5)

This script runs the simulation and saves the final frame as an image.
Originally intended for real-time animation, it now supports headless execution
by saving the output to outputs/animations/ with timestamp.
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

# Configuration
FRAMES = 300
RADIUS = 10.0

def build_config() -> UniverseConfig:
    return UniverseConfig(
        topology_type=0,  # FLAT
        physics_mode=0,   # VECTOR
        radius=RADIUS,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    # Spawn two entities symmetrically
    state = spawn_entity(state, jnp.array([-1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    return state


def run(config: UniverseConfig, state):
    print("Initializing Visualization (Headless Mode)...")
    
    # We capture config in a closure to avoid hashing issues with static_argnums
    @jax.jit
    def jit_step(state):
        return step_simulation(state, config)
    
    print(f"Running simulation for {FRAMES} frames...")
    
    # Run loop
    for _ in range(FRAMES):
        state = jit_step(state)
        
    print("Plotting final frame...")
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"CosmoSim - Frame {FRAMES}")
    
    # Extract active positions
    active_mask = state.entity_active
    positions = state.entity_pos[active_mask]
    
    if len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, label='Active')
    else:
        ax.scatter([], [], c='blue', s=100, label='Active')
        
    ax.legend()
    
    # Output handling
    output_dir = os.path.join("outputs", "animations")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"animation_frame_{timestamp}_step{FRAMES}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path)
    print(f"Final frame saved to {output_path}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
