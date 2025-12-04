import jax

from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity

# PSS - Parameterized Scenario System
SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 25, "min": 1, "max": 5000},
    "radius": {"type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
    "dim": {"type": "int", "default": 3, "allowed": [2, 3]}
}

SCENARIO_PRESETS = {
    "small_cluster": { "N": 200 },
    "medium_cluster": { "N": 1000 },
    "large_cluster": { "N": 2500 },
    "fast": {
        "dt": 0.05,
        "G": 1.0,
        "N": 40,
        "radius": 12.0,
    },
    "dense": {
        "N": 200,
        "radius": 5.0,
        "G": 5.0,
    },
    "chaotic": {
        "dt": 0.1,
        "G": 15.0,
        "N": 75,
    },
    "gentle": {
        "dt": 0.02,
        "G": 0.5,
        "radius": 20.0,
    },
}

def build_config(params: dict | None = None) -> UniverseConfig:
    """
    Build configuration for random N-body simulation.
    
    Args:
        params: Optional parameter dict from CLI/PSS
    """
    # Extract params with defaults
    p = params or {}
    radius = p.get('radius', 10.0)
    n = p.get('max_entities', p.get('N', 25))  # Support both names
    dt = p.get('dt', 0.2)
    G = p.get('G', 5.0)
    c = p.get('c', 1.0)
    topology_type = p.get('topology_type', 0)
    physics_mode = p.get('physics_mode', 0)
    dim = p.get('dim', 3)  # Default to 3D for random n-body
    
    return UniverseConfig(
        topology_type=topology_type,
        physics_mode=physics_mode,
        radius=radius,
        max_entities=n,
        max_nodes=1,
        dt=dt,
        c=c,
        G=G,
        dim=dim,
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    
    # Use params if provided, otherwise use config defaults
    if params:
        radius = params.get('radius', config.radius)
        n = params.get('N', config.max_entities)
    else:
        radius = config.radius
        n = config.max_entities
        
    velocity_scale = 0.01  # Very small initial velocities so gravity dominates

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.normal(k1, (n, config.dim)) * (radius * 0.1)  # Tighter cluster
    velocities = jax.random.normal(k2, (n, config.dim)) * velocity_scale
    masses = jax.random.uniform(k3, (n,)) * 1.5 + 0.3

    # Spawn all bodies
    for i in range(n):
        state = spawn_entity(
            state,
            positions[i],
            velocities[i],
            masses[i],
            1,
        )
    state.scenario_name = "random_nbody"
    return state


def run(config, state):
    """
    JSON-export-compatible random_nbody run.
    DO NOT run simulation physics here.
    The global export_simulation() loop will drive the updates.
    """
    return state


if __name__ == "__main__":
    cfg = build_config()
    state = build_initial_state(cfg)
    run(cfg, state)
