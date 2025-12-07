"""
Sphere Geodesic Validation Scenario

Tests sphere topology geodesic displacement calculations.
Developer scenario for PS2.4 validation.
"""
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from entities import spawn_entity
from kernel import step_simulation, compute_diagnostics

DEVELOPER_SCENARIO = True

SCENARIO_PARAMS = {
    "radius": {"type": "float", "default": 10.0, "min": 5.0, "max": 50.0},
    "N": {"type": "int", "default": 3, "min": 2, "max": 10},
    "G": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
    "dt": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.2},
}

def build_config(params: dict | None = None) -> UniverseConfig:
    p = params or {}
    dt = p.get('dt', 0.05)
    radius = p.get('radius', 10.0)
    G = p.get('G', 0.1)
    N = p.get('N', 3)
    
    return UniverseConfig(
        topology_type=2,  # Sphere
        physics_mode=0,
        radius=radius,
        max_entities=N,
        max_nodes=1,
        dt=dt,
        c=1.0,
        G=G,
        dim=3,
        enable_diagnostics=True,
        enforce_sphere_constraint=True,
    )

def build_initial_state(config: UniverseConfig, params: dict | None = None) -> UniverseState:
    state = initialize_state(config)
    p = params or {}
    
    N = config.max_entities
    R = config.radius
    
    # Place particles on sphere surface
    # North pole, equator points at 0°, 120°, 240°
    if N >= 1:
        # North pole
        pos = jnp.array([0.0, 0.0, R])
        vel = jnp.array([0.1, 0.0, 0.0])
        state = spawn_entity(state, pos, vel, 1.0, 1)
    
    if N >= 2:
        # Equator at 0°
        pos = jnp.array([R, 0.0, 0.0])
        vel = jnp.array([0.0, 0.1, 0.0])
        state = spawn_entity(state, pos, vel, 1.0, 1)
    
    if N >= 3:
        # Equator at 120°
        angle = 2.0 * jnp.pi / 3.0
        pos = jnp.array([R * jnp.cos(angle), R * jnp.sin(angle), 0.0])
        vel = jnp.array([0.0, 0.0, 0.1])
        state = spawn_entity(state, pos, vel, 1.0, 1)
    
    # Additional particles distributed around sphere
    for i in range(3, N):
        theta = 2.0 * jnp.pi * i / N
        phi = jnp.pi * (i % 3) / 3.0
        
        pos = jnp.array([
            R * jnp.sin(phi) * jnp.cos(theta),
            R * jnp.sin(phi) * jnp.sin(theta),
            R * jnp.cos(phi)
        ])
        vel = jnp.array([0.05, 0.05, 0.05])
        state = spawn_entity(state, pos, vel, 1.0, 1)
    
    state.scenario_name = "sphere_geodesic"
    return state

def run(config, state, steps=300):
    """Run sphere geodesic test."""
    print(f"[SPHERE_GEO] Sphere radius: {config.radius}")
    print(f"[SPHERE_GEO] {config.max_entities} particles on sphere surface")
    
    # Compute diagnostics for initial state
    if config.enable_diagnostics:
        state = compute_diagnostics(state, config)
        
    print(f"[SPHERE_GEO] Initial E = {float(state.total_energy):.6f}")
    
    for i in range(steps):
        state = step_simulation(state, config)
        
        # Check that particles stay on sphere surface
        radii = jnp.sqrt(jnp.sum(state.entity_pos**2, axis=1))
        
        if (i + 1) % 100 == 0:
            print(f"[SPHERE_GEO] Step {i+1}: E = {float(state.total_energy):.6f}")
            print(f"  Radii range: [{float(jnp.min(radii)):.2f}, {float(jnp.max(radii)):.2f}] (expect ~{config.radius})")
    
    print(f"[SPHERE_GEO] Final E = {float(state.total_energy):.6f}")
    return state
