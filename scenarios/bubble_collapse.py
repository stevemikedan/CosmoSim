import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity

SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 100, "min": 10, "max": 5000},
    "radius": {"type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
    "initial_asymmetry": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
    "dim": {"type": "int", "default": 2, "allowed": [2]}
}

SCENARIO_PRESETS = {
    "slow": {
        "dt": 0.005,
        "G": 0.5,
    },
    "rapid": {
        "dt": 0.05,
        "G": 8.0,
    },
    "symmetric": {
        "initial_asymmetry": 0.0,
        "dt": 0.01,
    },
    "fragmented": {
        "initial_asymmetry": 0.3,
        "N": 200,
    },
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    radius = p.get('radius', 10.0)
    n = p.get('max_entities', p.get('N', 100))
    dt = p.get('dt', 0.01)
    G = p.get('G', 1.0)
    c = p.get('c', 1.0)
    topology_type = p.get('topology_type', 0)
    physics_mode = p.get('physics_mode', 0)
    
    return UniverseConfig(
        topology_type=topology_type,
        physics_mode=physics_mode,
        radius=radius,
        max_entities=n,
        max_nodes=1,
        dt=dt,
        c=c,
        G=G,
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    n = config.max_entities
    radius = config.radius
    asymmetry = p.get('initial_asymmetry', 0.0)
    
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key, 2)
    
    # Create a spherical shell
    # For 2D, this is a circle. For 3D, a sphere. Assuming 2D for now based on other scenarios.
    angles = jax.random.uniform(k1, (n,), minval=0, maxval=2*jnp.pi)
    
    # Add asymmetry to radius
    r_noise = jax.random.normal(k2, (n,)) * asymmetry * radius * 0.2
    r = radius + r_noise
    
    positions = jnp.stack([r * jnp.cos(angles), r * jnp.sin(angles)], axis=1)
    
    # Zero initial velocity for pure collapse
    velocities = jnp.zeros((n, 2))
    masses = jnp.ones((n,))
    
    for i in range(n):
        state = spawn_entity(state, positions[i], velocities[i], masses[i], 1)
        
    state.scenario_name = "bubble_collapse"
    return state

def run(config, state):
    return state
