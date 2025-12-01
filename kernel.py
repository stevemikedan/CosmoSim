"""Physics kernel implementations for CosmoSim."""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from entities import spawn_entity, despawn_entity
from topology import enforce_boundaries, apply_topology
from physics_utils import compute_gravity_forces, integrate_euler, integrate_leapfrog


def update_vector_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Update physics for VECTOR mode (trivial placeholder for Sprint 3).

    Returns:
        Updated universe state with positions moved by velocity * dt
    """
    # 1. Compute gravitational forces using softened gravity
    total_force = compute_gravity_forces(
        state.entity_pos,
        state.entity_mass,
        state.entity_active,
        config
    )
    
    # 2. Integrate using selected integrator
    if config.integrator == "leapfrog":
        new_pos, new_vel = integrate_leapfrog(
            state.entity_pos,
            state.entity_vel,
            total_force,
            state.entity_mass,
            state.entity_active,
            config.dt
        )
    else:  # Default to Euler
        new_pos, new_vel = integrate_euler(
            state.entity_pos,
            state.entity_vel,
            total_force,
            state.entity_mass,
            state.entity_active,
            config.dt
        )
    
    # 3. Apply topology to positions (and velocities if needed)
    final_pos, final_vel = apply_topology(new_pos, new_vel, config)
    
    return state.replace(entity_pos=final_pos, entity_vel=final_vel)

def update_lattice_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Update physics for LATTICE mode (no-op placeholder for Sprint 3)."""
    return state

def dispatch_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Dispatch to the appropriate physics kernel based on physics_mode."""
    def vector_branch(args):
        return update_vector_physics(args[0], args[1])

    def lattice_branch(args):
        return update_lattice_physics(args[0], args[1])

    def reserved_branch(args):
        # Reserved for future physics modes (VOXEL, FIELD, CUSTOM, etc.)
        return args[0]

    branches = [
        vector_branch,   # mode 0: VECTOR
        lattice_branch,  # mode 1: LATTICE
        reserved_branch, # mode 2+: RESERVED
    ]

    safe_mode = jnp.clip(config.physics_mode, 0, 2)
    return jax.lax.switch(safe_mode, branches, (state, config))

def step_simulation(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Execute one simulation timestep.

    This function updates global time, applies physics, enforces topology
    boundaries, and returns a new UniverseState. It is fully JIT‑compatible.
    """
    # 1. Update global time
    state = state.replace(time=state.time + config.dt)

    # 2. Apply physics update
    state = dispatch_physics(state, config)

    # 3. Enforce topology boundaries (for legacy behavior)
    new_entity_pos = enforce_boundaries(
        state.entity_pos,
        config.topology_type,
        config.radius,
    )
    state = state.replace(entity_pos=new_entity_pos)

    # 4. Return updated state
    return state
