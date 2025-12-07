"""
Numerical integrator module.

Implements stable integration schemes (Velocity Verlet) and safety controls
for simulation stability.
"""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from typing import Callable

def velocity_verlet(
    state: UniverseState,
    config: UniverseConfig,
    force_fn: Callable[[UniverseState, UniverseConfig], jnp.ndarray]
) -> UniverseState:
    """
    Perform one step of Velocity Verlet integration.
    
    Algorithm:
    1. v_half = v + a * (dt / 2)
    2. r_new = r + v_half * dt
    3. a_new = force_fn(r_new)
    4. v_new = v_half + a_new * (dt / 2)
    
    Includes safety clamps for acceleration and velocity.
    
    Args:
        state: Current universe state
        config: Universe configuration
        force_fn: Function computing forces/accelerations from state
        
    Returns:
        Updated universe state
    """
    dt = config.dt
    
    # Current state
    pos = state.entity_pos
    vel = state.entity_vel
    mass = state.entity_mass
    active = state.entity_active
    
    # 0. Initial Acceleration (compute if not stored, but usually we'd store it)
    # For now, we assume we compute it from current state.
    # Optimization: We could store 'acc' in state to avoid recomputing, 
    # but for strict Velocity Verlet starting from pos/vel, we compute it.
    # Or we can assume the previous step's acceleration is needed.
    # Standard VV assumes we know a(t).
    acc = force_fn(state, config)
    
    # Safety: Clamp initial acceleration
    acc_norm = jnp.linalg.norm(acc, axis=-1, keepdims=True) + 1e-12
    max_accel = getattr(config, 'max_accel', 1e5)
    acc = jnp.where(
        acc_norm > max_accel,
        acc * (max_accel / acc_norm),
        acc
    )
    
    # 1. Half-step velocity
    v_half = vel + acc * (dt * 0.5)
    
    # 2. Update position
    pos_new = pos + v_half * dt
    
    # Create intermediate state for force calculation
    state_new_pos = state.replace(entity_pos=pos_new)
    
    # 3. New Forces / Acceleration
    acc_new = force_fn(state_new_pos, config)
    
    # Safety: Clamp new acceleration
    acc_new_norm = jnp.linalg.norm(acc_new, axis=-1, keepdims=True) + 1e-12
    acc_new = jnp.where(
        acc_new_norm > max_accel,
        acc_new * (max_accel / acc_new_norm),
        acc_new
    )
    
    # 4. Full-step velocity
    vel_new = v_half + acc_new * (dt * 0.5)
    
    # Safety: Clamp velocity
    vel_norm = jnp.linalg.norm(vel_new, axis=-1, keepdims=True) + 1e-12
    max_vel = getattr(config, 'max_vel', 1e4)
    vel_new = jnp.where(
        vel_norm > max_vel,
        vel_new * (max_vel / vel_norm),
        vel_new
    )
    
    # 5. NaN / Inf Rejection (per-particle)
    # Check for any non-finite values in active entities
    is_valid_pos = jnp.all(jnp.isfinite(pos_new), axis=-1) | (~active)
    is_valid_vel = jnp.all(jnp.isfinite(vel_new), axis=-1) | (~active)
    
    # Radius check
    radius_max = getattr(config, 'radius_max', 1e12)
    is_within_bounds = (jnp.linalg.norm(pos_new, axis=-1) <= radius_max) | (~active)
    
    # Per-particle validity (N,) boolean array
    is_particle_valid = is_valid_pos & is_valid_vel & is_within_bounds
    
    # Apply per-particle: if particle invalid OR inactive, keep old pos/vel
    # CRITICAL: Inactive entities must NEVER move
    should_update = is_particle_valid & active
    final_pos = jnp.where(should_update[:, None], pos_new, pos)
    final_vel = jnp.where(should_update[:, None], vel_new, vel)
    
    # Update state
    return state.replace(entity_pos=final_pos, entity_vel=final_vel)
