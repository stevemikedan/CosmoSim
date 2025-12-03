import jax

from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity



# PSS - Parameterized Scenario System
SCENARIO_PARAMS = {
    "N": {"type": "int", "default": 25, "min": 1, "max": 5000},
    "radius": {"type": "float", "default": 10.0, "min": 1.0, "max": 100.0}
}

SCENARIO_PRESETS = {
    "small_cluster": { "N": 200 },
    "medium_cluster": { "N": 1000 },
    "large_cluster": { "N": 2500 }
}


def build_config() -> UniverseConfig:
    # Default values from original function
    radius = 10.0
    n = 25
    return UniverseConfig(
        topology_type=0,
        physics_mode=0,
        radius=radius,
        max_entities=n,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
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
        
    velocity_scale = 0.3

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.normal(k1, (n, config.dim)) * (radius * 0.5)
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
