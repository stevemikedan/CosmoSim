import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity

SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 100, "min": 10, "max": 5000},
    "viscosity": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0}, # Not used in physics yet but good for metadata
    "dim": {"type": "int", "default": 2, "allowed": [2]}
}

SCENARIO_PRESETS = {
    "laminar": {
        "dt": 0.01,
        "viscosity": 0.3,
    },
    "turbulent": {
        "dt": 0.002,
        "viscosity": 0.01,
    },
    "shock": {
        "dt": 0.0005,
        "viscosity": 0.001,
    },
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    n = p.get('max_entities', p.get('N', 100))
    dt = p.get('dt', 0.01)
    G = p.get('G', 1.0)
    c = p.get('c', 1.0)
    topology_type = p.get('topology_type', 0)
    physics_mode = p.get('physics_mode', 0)
    
    return UniverseConfig(
        topology_type=topology_type,
        physics_mode=physics_mode,
        radius=20.0,
        max_entities=n,
        max_nodes=1,
        dt=dt,
        c=c,
        G=G,
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    n = config.max_entities
    
    # Two parallel lines moving in opposite directions
    half_n = n // 2
    
    # Top line moving right
    pos_top_x = jnp.linspace(-10, 10, half_n)
    pos_top_y = jnp.ones(half_n) * 2.0
    vel_top_x = jnp.ones(half_n) * 1.0
    vel_top_y = jnp.zeros(half_n)
    
    # Bottom line moving left
    pos_bot_x = jnp.linspace(-10, 10, n - half_n)
    pos_bot_y = jnp.ones(n - half_n) * -2.0
    vel_bot_x = jnp.ones(n - half_n) * -1.0
    vel_bot_y = jnp.zeros(n - half_n)
    
    positions = jnp.concatenate([
        jnp.stack([pos_top_x, pos_top_y], axis=1),
        jnp.stack([pos_bot_x, pos_bot_y], axis=1)
    ])
    
    velocities = jnp.concatenate([
        jnp.stack([vel_top_x, vel_top_y], axis=1),
        jnp.stack([vel_bot_x, vel_bot_y], axis=1)
    ])
    
    masses = jnp.ones((n,))
    
    for i in range(n):
        state = spawn_entity(state, positions[i], velocities[i], masses[i], 1)
        
    state.scenario_name = "vortex_sheet"
    return state

def run(config, state):
    return state
