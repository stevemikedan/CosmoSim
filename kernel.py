"""Physics kernel implementations for CosmoSim."""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from entities import spawn_entity, despawn_entity
from topology import enforce_boundaries


def update_vector_physics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Update physics for VECTOR mode (trivial placeholder for Sprint 3).

    Returns:
        Updated universe state with positions moved by velocity * dt
    """
    # 1. Compute pairwise displacement: r_j - r_i
    # Shape: (N, N, 2)
    # disp[i, j] is vector from i to j
    disp = state.entity_pos[None, :, :] - state.entity_pos[:, None, :]
    
    # 2. Compute distances with epsilon to avoid singularity
    # Shape: (N, N)
    dist_sq = jnp.sum(disp**2, axis=-1) + 1e-6
    dist = jnp.sqrt(dist_sq)
    
    # Mask mass of inactive entities so they don't exert gravity
    # Shape: (N,)
    active_mass = jnp.where(state.entity_active, state.entity_mass, 0.0)
    
    # 3. Compute gravitational force magnitudes
    # F = G * m_i * m_j / r^2
    # Use real mass for receiver (i) and active_mass for source (j)
    # This ensures inactive entities don't exert gravity.
    # Shape: (N, N)
    force_mag = config.G * state.entity_mass[:, None] * active_mass[None, :] / dist_sq
    
    # 4. Compute acceleration
    # acc[i] = sum_j (F_ij * disp_ij / dist_ij) / m_i
    # We need to be careful with broadcasting.
    # disp / dist[:, :, None] gives unit vectors (N, N, 2)
    # force_mag[:, :, None] scales them (N, N, 1)
    # Result is force vectors (N, N, 2)
    force_vec = disp * (force_mag / dist)[:, :, None]
    
    # Sum forces on each entity i (sum over j)
    # Shape: (N, 2)
    total_force = jnp.sum(force_vec, axis=1)
    
    # Acceleration = Force / Mass
    # Handle zero mass to avoid division by zero (though mass should be > 0)
    # Shape: (N, 2)
    acc = total_force / (state.entity_mass[:, None] + 1e-6)
    
    # 5. Semi-implicit Euler integration
    # vel_new = vel + acc * dt
    new_vel = state.entity_vel + acc * config.dt
    
    # pos_new = pos + vel_new * dt
    new_pos = state.entity_pos + new_vel * config.dt
    
    # 6. Apply active mask
    # Only update active entities. Inactive ones stay put.
    # Force calculation naturally handles inactive if their mass is 0,
    # but we explicitly mask to be safe and ensure they don't move.
    active_mask = state.entity_active[:, None]
    
    final_vel = jnp.where(active_mask, new_vel, state.entity_vel)
    final_pos = jnp.where(active_mask, new_pos, state.entity_pos)
    
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

    # 3. Enforce topology boundaries
    new_entity_pos = enforce_boundaries(
        state.entity_pos,
        config.topology_type,
        config.radius,
    )
    state = state.replace(entity_pos=new_entity_pos)

    # 4. Return updated state
    return state
