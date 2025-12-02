"""
CosmoSim Energy Plot Script (Sprint 6.5)

This script tracks the total energy (Kinetic + Potential) of the system
over time to verify physics stability.
Saves output to outputs/energy/ with timestamp.
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

def compute_energy(state, config):
    """Compute total kinetic and potential energy."""
    # Kinetic Energy: 0.5 * m * v^2
    # Shape: (N,)
    v_sq = jnp.sum(state.entity_vel**2, axis=-1)
    ke = 0.5 * state.entity_mass * v_sq
    total_ke = jnp.sum(jnp.where(state.entity_active, ke, 0.0))
    
    # Potential Energy: -G * m_i * m_j / r
    # Pairwise calculation
    pos = state.entity_pos
    disp = pos[None, :, :] - pos[:, None, :]
    dist = jnp.sqrt(jnp.sum(disp**2, axis=-1) + 1e-6)
    
    # Mask self-interactions and inactive entities
    mass_prod = state.entity_mass[:, None] * state.entity_mass[None, :]
    
    # Mask: active_i AND active_j AND i != j
    active_mask = state.entity_active[:, None] & state.entity_active[None, :]
    eye_mask = ~jnp.eye(len(state.entity_mass), dtype=bool)
    
    full_mask = active_mask & eye_mask
    
    pe_matrix = -config.G * mass_prod / dist
    total_pe = jnp.sum(jnp.where(full_mask, pe_matrix, 0.0)) / 2.0
    
    return total_ke, total_pe

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
    # Spawn two entities
    state = spawn_entity(state, jnp.array([-1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    state = spawn_entity(state, jnp.array([ 1.0, 0.0] + [0.0] * (config.dim - 2)), jnp.zeros(config.dim), 1.0, 1)
    return state


def run(config: UniverseConfig, state):
    print("Initializing Energy Diagnostics...")
    
    # We capture config in closures to avoid hashing issues with static_argnums
    @jax.jit
    def jit_step(state):
        return step_simulation(state, config)
    
    @jax.jit
    def jit_energy(state):
        return compute_energy(state, config)
    
    steps = []
    ke_history = []
    pe_history = []
    total_history = []
    
    STEPS = 200
    print(f"Running simulation and tracking energy for {STEPS} steps...")
    for i in range(STEPS):
        state = jit_step(state)
        ke, pe = jit_energy(state)
        
        steps.append(i)
        ke_history.append(ke)
        pe_history.append(pe)
        total_history.append(ke + pe)
        
    print("Plotting energy...")
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ke_history, label='Kinetic Energy', color='green')
    plt.plot(steps, pe_history, label='Potential Energy', color='red')
    plt.plot(steps, total_history, label='Total Energy', color='blue', linewidth=2)
    
    plt.title("System Energy over Time")
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    
    # Output handling
    output_dir = os.path.join("outputs", "energy")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"energy_{timestamp}_steps{STEPS}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.savefig(output_path)
    print(f"Energy plot saved to {output_path}")
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
