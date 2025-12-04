import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation
from topologies.mobius_topology import MobiusTopology

# Möbius strip domain parameters
STRIP_LENGTH = 20.0  # L: u ∈ [0, L]
STRIP_WIDTH = 1.0    # W: v ∈ [-W, W] (Matches engine default)

SCENARIO_PARAMS = {
    "speed": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0}
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
    }
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.1)
    
    return UniverseConfig(
        topology_type=MobiusTopology.MOBIUS_TOPOLOGY,
        physics_mode=0,
        radius=STRIP_LENGTH,  # Radius acts as Length L in MobiusTopology
        max_entities=1,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=0.0,  # No gravity, just walking
        dim=2,
        bounds=STRIP_LENGTH
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    speed = p.get('speed', 1.0)
    
    # Initialize walker in (u, v) coordinates
    # Start at center of strip: u = L/2, v = 0
    initial_u = STRIP_LENGTH / 2.0
    initial_v = 0.0
    
    # Initial velocity: moving in +u direction
    state = spawn_entity(
        state,
        jnp.array([initial_u, initial_v]),
        jnp.array([speed, 0.0]),
        1.0,
        1
    )
        
    state.scenario_name = "mobius_walk"
    return state


def apply_mobius_motion(state, config, params):
    """
    Apply Möbius strip walker dynamics.
    
    Sets velocity to constant drift along u-axis.
    Engine handles integration and Möbius wrapping.
    """
    speed = params.get('speed', 1.0)
    
    active = state.entity_active
    
    # Enforce constant velocity along strip
    # u-velocity = speed
    # v-velocity = 0
    
    # Create velocity vector [speed, 0]
    vel = jnp.zeros_like(state.entity_vel)
    vel = vel.at[:, 0].set(speed)
    
    # Only apply to active entities
    new_vel = jnp.where(active[:, None] > 0, vel, state.entity_vel)
    
    # Return current pos (let engine integrate) and new vel
    return state.entity_pos, new_vel


def run(config, state, steps=300):
    """Run Möbius strip walk simulation with engine-native topology."""
    # Extract speed from initial velocity
    initial_speed = float(jnp.abs(state.entity_vel[0, 0]))
    params = {'speed': initial_speed if initial_speed > 0 else 1.0}
    
    for _ in range(steps):
        # Apply motion (velocity enforcement)
        new_pos, new_vel = apply_mobius_motion(state, config, params)
        state = state.replace(entity_pos=new_pos, entity_vel=new_vel)
        
        # Standard physics step (Integrates and Wraps via MobiusTopology)
        state = step_simulation(state, config)
    
    return state
