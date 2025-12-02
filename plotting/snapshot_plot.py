"""
CosmoSim Snapshot Plot Script (Sprint 6.5)

This script runs the simulation for a fixed number of steps and 
generates a static scatter plot of the final state.
Saves output to outputs/snapshots/ with timestamp.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )


def build_initial_state(config: UniverseConfig):
    state = initialize_state(config)
    # Spawn entities
    state = spawn_entity(state, jnp.array([-1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    return state

def run(config: UniverseConfig, state):
    print("Initializing Snapshot Simulation...")
    
    # JIT the physics step for speed
    # We capture config in a closure to avoid hashing issues with static_argnums
    @jax.jit
    def jit_step(state):
        return step_simulation(state, config)
    
    # Run 50 steps
    STEPS = 50
    print(f"Running {STEPS} steps...")
    for _ in range(STEPS):
        state = jit_step(state)
        
    print("Plotting snapshot...")
    
    # Extract active entities
    active_mask = state.entity_active
    positions = state.entity_pos[active_mask]
    
    plt.xlim(-config.radius, config.radius)
    plt.ylim(-config.radius, config.radius)
    plt.grid(True)
    plt.title(f"CosmoSim Snapshot - Step {STEPS}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    # Output handling
    output_dir = os.path.join("outputs", "snapshots")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"snapshot_{timestamp}_step{STEPS}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path)
    print(f"Snapshot saved to {output_path}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
