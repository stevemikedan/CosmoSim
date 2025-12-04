"""
Stellar Trio – 3-body orbital dance.
Fully compatible with CosmoSim’s UniverseState and UniverseConfig.
"""

from __future__ import annotations

import jax.numpy as jnp

from state import UniverseConfig, UniverseState
from kernel import step_simulation

SCENARIO_PRESETS = {
    "wide-orbit": {
        "radius": 40.0,
        "dt": 0.02,
    },
    "tight-orbit": {
        "radius": 10.0,
        "dt": 0.01,
    },
    "high-energy": {
        "G": 12.0,
        "dt": 0.005,
    },
    "stable-ish": {
        "G": 2.0,
        "dt": 0.015,
    },
}

def build_config(params: dict | None = None):
    """
    Create a 3-body configuration.
    """
    p = params or {}
    radius = p.get('radius', 10.0)
    dt = p.get('dt', 0.1)
    G = p.get('G', 1.0)
    c = p.get('c', 1.0)
    topology_type = p.get('topology_type', 0)
    physics_mode = p.get('physics_mode', 0)

    return UniverseConfig(
        physics_mode=physics_mode,        # 0 = VECTOR (Newtonian gravity)
        radius=radius,
        max_entities=3,
        max_nodes=1,
        dt=dt,
        c=c,
        G=G,
        dim=2,
        topology_type=topology_type,       # flat
        bounds=radius,
    )


def build_initial_state(cfg: UniverseConfig, params: dict | None = None) -> UniverseState:
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
