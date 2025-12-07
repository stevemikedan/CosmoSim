"""
Torus Wrapping Validation Scenario

Tests torus topology nearest-image displacement and periodic boundaries.
Developer scenario for PS2.4 validation.
"""
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation

DEVELOPER_SCENARIO = True

SCENARIO_PARAMS = {
    "domain_size": {"type": "float", "default": 10.0, "min": 5.0, "max": 50.0},
    "speed": {"type": "float", "default": 0.5, "min": 0.1, "max": 2.0},
    "dt": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.1)
    domain = p.get('domain_size', 10.0)
    
    return UniverseConfig(
        topology_type=1,  # Torus
        physics_mode=0,
        radius=domain / 2.0,  # Domain is [-R, R]
        max_entities=2,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=0.0,  # No gravity
        dim=2,
        enable_diagnostics=True,
        torus_size=domain,
        bounds=domain / 2.0,  # NEW - enforces consistent torus domain
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    speed = p.get('speed', 0.5)
    R = config.radius
    
    # Place particles on opposite sides of domain
    # They should attract through the periodic boundary
    pos1 = jnp.array([-R * 0.9, 0.0])  # Near left edge
    pos2 = jnp.array([R * 0.9, 0.0])   # Near right edge
    
    # Give them velocities that will make them wrap
    vel1 = jnp.array([speed, 0.0])     # Moving right
    vel2 = jnp.array([-speed, 0.0])    # Moving left
    
    state = spawn_entity(state, pos1, vel1, 1.0, 1)
    state = spawn_entity(state, pos2, vel2, 1.0, 1)
    
    state.scenario_name = "torus_wrap"
    return state

def run(config, state, steps=300):
    """Run torus wrapping test."""
    print(f"[TORUS_WRAP] Domain: [-{config.radius:.1f}, {config.radius:.1f}]")
    print(f"[TORUS_WRAP] Initial positions:")
    print(f"  P1: [{float(state.entity_pos[0, 0]):.2f}, {float(state.entity_pos[0, 1]):.2f}]")
    print(f"  P2: [{float(state.entity_pos[1, 0]):.2f}, {float(state.entity_pos[1, 1]):.2f}]")
    
    for i in range(steps):
        state = step_simulation(state, config)
        
        # Print positions every 50 steps to see wrapping
        if (i + 1) % 50 == 0:
            print(f"[TORUS_WRAP] Step {i+1}:")
            print(f"  P1: [{float(state.entity_pos[0, 0]):.2f}, {float(state.entity_pos[0, 1]):.2f}]")
            print(f"  P2: [{float(state.entity_pos[1, 0]):.2f}, {float(state.entity_pos[1, 1]):.2f}]")
    
    print(f"[TORUS_WRAP] Final positions:")
    print(f"  P1: [{float(state.entity_pos[0, 0]):.2f}, {float(state.entity_pos[0, 1]):.2f}]")
    print(f"  P2: [{float(state.entity_pos[1, 0]):.2f}, {float(state.entity_pos[1, 1]):.2f}]")
    
    return state
