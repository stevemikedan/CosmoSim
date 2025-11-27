"""
Stellar Trio – 3-body orbital dance.
Fully compatible with CosmoSim’s UniverseState and UniverseConfig.
"""

from __future__ import annotations

import jax.numpy as jnp

from state import UniverseConfig, UniverseState
from kernel import step_simulation


def build_config():
    """
    Create a 3-body configuration.
    """
    return UniverseConfig(
        physics_mode=0,        # 0 = VECTOR (Newtonian gravity)
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=2,
        topology_type=0,       # flat
        bounds=10.0,
    )


def build_initial_state(cfg: UniverseConfig) -> UniverseState:
    """
    Build the initial UniverseState with three orbiting stars.
    """
    from state import initialize_state
    from entities import spawn_entity
    
    state = initialize_state(cfg)
    
    # Orbital positions
    positions = [
        jnp.array([-3.0, 0.0]),   # Star A
        jnp.array([ 3.0, 0.0]),   # Star B
        jnp.array([ 0.0, 4.0]),   # Star C
    ]
    
    velocities = [
        jnp.array([0.0,  1.8]),   # A upward
        jnp.array([0.0, -1.8]),   # B downward
        jnp.array([1.2,  0.0]),   # C rightward
    ]
    
    masses = [4.0, 4.0, 4.0]
    
    # Spawn all three stars
    for i in range(3):
        state = spawn_entity(state, positions[i], velocities[i], masses[i], 1)
    
    state.scenario_name = "stellar_trio"
    return state


def run(cfg: UniverseConfig, state: UniverseState, steps: int = 300):
    """
    Standard simulation loop.
    """
    for _ in range(steps):
        state = step_simulation(state, cfg)

    return state
