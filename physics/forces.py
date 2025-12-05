"""
Force calculation module.

Computes forces acting on entities using topology-aware mathematics.
Integrates gravity and other interactions.
"""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from environment.topology_math import compute_displacement, compute_distance

def compute_forces(state: UniverseState, config: UniverseConfig) -> jnp.ndarray:
    """
    Compute total acceleration on all entities.
    
    Includes:
    - Newtonian Gravity (topology-aware)
    - Softening (PS2.5 placeholder)
    
    Args:
        state: Current universe state
        config: Universe configuration
        
    Returns:
        Acceleration array (max_entities, dim)
    """
    pos = state.entity_pos
    mass = state.entity_mass
    active = state.entity_active
    
    # 1. Gravity
    # F_ij = -G * m_i * m_j * disp_ij / (dist_ij^3 + eps)
    # acc_i = sum_j (-G * m_j * disp_ij / ...)
    
    # We need pairwise displacements and distances
    # Shape: (N, N, dim) and (N, N)
    
    # Expand dims for broadcasting
    # p1: (N, 1, dim) - receiver
    # p2: (1, N, dim) - source
    p1 = pos[:, None, :]
    p2 = pos[None, :, :]
    
    # To ensure attractive gravity (Force pointing from Receiver P1 to Source P2),
    # we need the vector pointing from P1 to P2.
    # compute_displacement(A, B) returns B - A.
    # So we need compute_displacement(p2, p1) which equates to p1 - p2.
    # Then F = -G * m * (p1 - p2) / r^3  points towards P2.
    disp = compute_displacement(p2, p1, config.topology_type, config)
    
    # Compute distance (symmetric)
    dist = compute_distance(p1, p2, config.topology_type, config)
    
    # Softening
    inv_r3 = 1.0 / (dist**3 + 1e-12)
    
    # Mask self-interactions and inactive entities
    
    # Source mass (1, N)
    m_j = mass[None, :]
    
    # Active mask (1, N)
    active_j = active[None, :]
    
    # Force magnitude term (scalar part): G * m_j * inv_r3
    # We apply the negative sign in the final formula as requested.
    force_scalar = config.G * m_j * inv_r3
    
    # Apply mask: set force to 0 if source is inactive
    force_scalar = jnp.where(active_j, force_scalar, 0.0)
    
    # Calculate acceleration contributions
    # acc_i = sum(force_j_on_i) / m_i (we already divided by m_i implicit in LHS, RHS has m_j)
    # acc_contributions = -force_scalar * disp
    
    acc_contributions = -force_scalar[..., None] * disp
    
    # Sum over sources (axis 1)
    total_acc = jnp.sum(acc_contributions, axis=1)
    
    # Mask out inactive receivers
    total_acc = jnp.where(active[:, None], total_acc, 0.0)
    
    return total_acc
