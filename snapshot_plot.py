"""
CosmoSim Snapshot Plot Script (Sprint 6.5)

This script runs the simulation for a fixed number of steps and 
generates a static scatter plot of the final state.
Saves output to outputs/snapshots/ with timestamp.
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

def run_snapshot():
    print("Initializing Snapshot Simulation...")
    
    cfg = UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    
    state = initialize_state(cfg)
    
    # Spawn entities
    state = spawn_entity(state, jnp.array([-1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0]), jnp.array([0.0, 0.0]), 1.0, 1)
    
    jit_step = jax.jit(step_simulation)
    
    # Run 50 steps
    STEPS = 50
    print(f"Running {STEPS} steps...")
    for _ in range(STEPS):
        state = jit_step(state, cfg)
        
    print("Plotting snapshot...")
    
    # Extract active entities
    active_mask = state.entity_active
    positions = state.entity_pos[active_mask]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, label='Entities')
    
    plt.xlim(-cfg.radius, cfg.radius)
    plt.ylim(-cfg.radius, cfg.radius)
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

if __name__ == "__main__":
    run_snapshot()
