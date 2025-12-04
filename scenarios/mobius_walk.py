import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity

SCENARIO_PARAMS = {
    "speed": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
    "steps": {"type": "int", "default": 100, "min": 10, "max": 1000}
}

SCENARIO_PRESETS = {
    "slow": {
        "dt": 0.1,
        "speed": 1.0,
    },
    "fast": {
        "dt": 0.02,
        "speed": 5.0,
    },
    "wrap-test": {
        "speed": 2.0,
        "steps": 200,
        "topology_type": 1 # Toroidal
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.1)
    topology_type = p.get('topology_type', 1) # Default to toroidal for "mobius" feel
    
    return UniverseConfig(
        topology_type=topology_type,
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=0.0, # No gravity, just walking
        dim=2,
        bounds=10.0
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    speed = p.get('speed', 1.0)
    
    # Single walker starting at center
    state = spawn_entity(
        state,
        jnp.array([0.0, 0.0]),
        jnp.array([speed, speed * 0.5]),
        1.0,
        1
    )
        
    state.scenario_name = "mobius_walk"
    return state

def run(config, state):
    return state
