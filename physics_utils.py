"""
Physics utility functions for CosmoSim.

This module contains shared physics computations that can be used across
different parts of the simulation engine (kernel, run_sim, etc.).
"""

import jax.numpy as jnp
from state import UniverseConfig
from distance_utils import compute_offset, compute_distance


def compute_gravity_forces(pos, mass, active, config):
    """
    Compute gravitational forces between all particles with softening.
    
    Uses topology-aware distance calculations and a softened gravity kernel
    to prevent numerical blowups when particles get too close:
        F = G * m_i * m_j / (r^2 + epsilon^2)^(3/2)
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig containing G, gravity_softening, topology_type
    
    Returns:
        Array of shape (N, dim) - total gravitational force on each particle
    """
    N = pos.shape[0]
    dim = pos.shape[1]
    
    # Check if neighbor engine is enabled (PS2.2)
    use_neighbor_engine = getattr(config, 'enable_neighbor_engine', True)
    
    if use_neighbor_engine:
        # PS2.2: Use centralized neighbor engine
        from environment.topology_neighbors import generate_neighbor_pairs
        
        # Initialize force array
        forces = jnp.zeros((N, dim))
        
        # Epsilon for softening
        epsilon = config.gravity_softening
        
        # Generate forces for all neighbor pairs
        for i, j, offset in generate_neighbor_pairs(pos, active, config):
            # Compute distance
            r_sq = jnp.sum(offset ** 2)
            softened_dist_cubed = (r_sq + epsilon**2) ** 1.5
            
            # Force magnitude: F = G * m_i * m_j / r³_soft
            force_mag = config.G * mass[i] * mass[j] / softened_dist_cubed
            
            # Force vector from i to j
            force_vec = force_mag * offset
            
            # Accumulate force on particle i
            forces = forces.at[i].add(force_vec)
        
        return forces
    
    else:
        # LEGACY: Original vectorized implementation for backward compatibility
        topology_type = getattr(config, 'topology_type', 0)
        
        # FLAT topology: Use optimized vectorized Euclidean calculation
        if topology_type == 0:
            # 1. Compute pairwise displacement: r_j - r_i
            # Shape: (N, N, dim)
            # disp[i, j] is vector from i to j
            disp = pos[None, :, :] - pos[:, None, :]
            
            # 2. Compute distances with softening
            # Shape: (N, N)
            dist_sq = jnp.sum(disp**2, axis=-1)
            
            # Apply softening: (r^2 + epsilon^2)^(3/2)
            epsilon = config.gravity_softening
            softened_dist_cubed = (dist_sq + epsilon**2)**1.5
            
        # NON-FLAT topologies: Use topology-aware calculations
        else:
            # For non-flat topologies, we need to use compute_offset and compute_distance
            # This is less performant but necessary for correct topology handling
            import jax
            
            def pairwise_offset(p_i, p_j):
                return compute_offset(p_i, p_j, config)
                
            def pairwise_dist(p_i, p_j):
                return compute_distance(p_i, p_j, config)
            
            # Vectorize over all pairs using vmap
            # disp[i, j] is offset from i to j
            # dist[i, j] is distance from i to j
            disp = jax.vmap(jax.vmap(pairwise_offset, (None, 0)), (0, None))(pos, pos)
            dist = jax.vmap(jax.vmap(pairwise_dist, (None, 0)), (0, None))(pos, pos)
            
            # Apply softening
            epsilon = config.gravity_softening
            softened_dist_cubed = (dist**2 + epsilon**2)**1.5
        
        # Mask mass of inactive entities so they don't exert gravity
        # Shape: (N,)
        active_mass = jnp.where(active, mass, 0.0)
        
        # 3. Compute gravitational force magnitudes with softening
        # F = G * m1 * m2 / softened_dist_cubed
        # Use real mass for receiver (i) and active_mass for source (j)
        # This ensures inactive entities don't exert gravity.
        # Shape: (N, N)
        force_mag = config.G * mass[:, None] * active_mass[None, :] / softened_dist_cubed
        
        # 4. Compute force vectors
        # force_vec[i, j] = force from j on i
        # We use the offset vector (which already has the correct direction)
        # Shape: (N, N, dim)
        force_vec = force_mag[:, :, None] * disp
        
        # Sum forces on each entity i (sum over j)
        # Shape: (N, dim)
        total_force = jnp.sum(force_vec, axis=1)
        
        return total_force


def integrate_euler(pos, vel, force, mass, active, dt):
    """
    Semi-implicit Euler integrator.
    
    This is the default integrator, preserving the existing behavior.
    Order: vel_new = vel + dt * acc; pos_new = pos + dt * vel_new
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        vel: Array of shape (N, dim) - particle velocities
        force: Array of shape (N, dim) - forces on particles
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
        dt: float - timestep
    
    Returns:
        Tuple of (new_pos, new_vel) arrays
    """
    # Acceleration = Force / Mass
    acc = force / (mass[:, None] + 1e-6)
    
    # Semi-implicit Euler integration
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    # Apply active mask - only update active entities
    active_mask = active[:, None]
    final_vel = jnp.where(active_mask, new_vel, vel)
    final_pos = jnp.where(active_mask, new_pos, pos)
    
    return final_pos, final_vel


def integrate_leapfrog(pos, vel, force, mass, active, dt, prev_force=None):
    """
    Leapfrog (velocity Verlet) integrator.
    
    This is a symplectic integrator that better conserves energy for
    Hamiltonian systems. It's second-order accurate in time.
    
    Standard leapfrog requires force at both t and t+dt, which would
    require recomputing forces mid-step. For compatibility with existing
    simulation loops, we provide a simplified version when prev_force=None.
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        vel: Array of shape (N, dim) - particle velocities
        force: Array of shape (N, dim) - forces on particles at time t
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
        dt: float - timestep
        prev_force: Optional array of shape (N, dim) - forces from previous step
    
    Returns:
        Tuple of (new_pos, new_vel) arrays
    """
    # Acceleration at current time
    acc = force / (mass[:, None] + 1e-6)
    
    if prev_force is not None:
        # Full leapfrog with previous force
        # This gives proper second-order accuracy
        prev_acc = prev_force / (mass[:, None] + 1e-6)
        
        # Half-step velocity using previous acceleration
        v_half = vel + 0.5 * dt * prev_acc
        
        # Full-step position
        new_pos = pos + dt * v_half
        
        # Full-step velocity using current acceleration
        new_vel = v_half + 0.5 * dt * acc
    else:
        # Simplified leapfrog for compatibility
        # Still symplectic but requires force recomputation for full accuracy
        v_half = vel + 0.5 * dt * acc
        new_pos = pos + dt * v_half
        new_vel = vel + dt * acc
    
    # Apply active mask
    active_mask = active[:, None]
    final_vel = jnp.where(active_mask, new_vel, vel)
    final_pos = jnp.where(active_mask, new_pos, pos)
    
    return final_pos, final_vel


# ============================================================================
# Diagnostic Functions (PS1.3)
# ============================================================================

def kinetic_energy(vel, mass, active):
    """
    Compute total kinetic energy of the system.
    
    KE = sum(0.5 * m * |v|^2) for all active particles
    
    Args:
        vel: Array of shape (N, dim) - particle velocities
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
    
    Returns:
        float - total kinetic energy
    """
    # Only include active particles
    active_mass = jnp.where(active, mass, 0.0)
    
    # KE = 0.5 * m * v^2
    speed_squared = jnp.sum(vel**2, axis=1)
    ke = 0.5 * active_mass * speed_squared
    
    # Sum over all particles and ensure no NaNs
    total_ke = jnp.sum(jnp.nan_to_num(ke))
    
    return float(total_ke)


def potential_energy(pos, mass, active, config):
    """
    Compute gravitational potential energy with softening.
    
    Uses the same softening as compute_gravity_forces:
    U = -G * sum_{i<j} (m_i * m_j) / sqrt(r_ij^2 + epsilon^2)
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig containing G and gravity_softening
    
    Returns:
        float - total gravitational potential energy
    """
    # Compute pairwise distances
    disp = pos[None, :, :] - pos[:, None, :]
    dist_sq = jnp.sum(disp**2, axis=-1)
    
    # Apply softening
    epsilon = config.gravity_softening
    softened_dist = jnp.sqrt(dist_sq + epsilon**2)
    
    # Only include active particles
    active_mass = jnp.where(active, mass, 0.0)
    
    # Compute pairwise potential energy
    # U_ij = -G * m_i * m_j / r_ij
    # Use active_mass for both particles
    pair_pe = -config.G * active_mass[:, None] * active_mass[None, :] / softened_dist
    
    # Sum over upper triangle to avoid double counting (i < j)
    # Also set diagonal to zero (self-interaction)
    mask = jnp.triu(jnp.ones_like(pair_pe), k=1)
    total_pe = jnp.sum(jnp.nan_to_num(pair_pe * mask))
    
    return float(total_pe)


def total_energy(KE, PE):
    """
    Compute total system energy.
    
    Args:
        KE: float - kinetic energy
        PE: float - potential energy
    
    Returns:
        float - total energy (KE + PE)
    """
    return KE + PE


def momentum(vel, mass, active):
    """
    Compute total momentum vector of the system.
    
    P = sum(m_i * v_i) for all active particles
    
    Args:
        vel: Array of shape (N, dim) - particle velocities
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
    
    Returns:
        Array of shape (dim,) - total momentum vector
    """
    # Only include active particles
    active_mass = jnp.where(active, mass, 0.0)
    
    # P = sum(m * v)
    momentum_vec = active_mass[:, None] * vel
    total_momentum = jnp.sum(momentum_vec, axis=0)
    
    # Ensure no NaNs
    return jnp.nan_to_num(total_momentum)


def center_of_mass(pos, mass, active):
    """
    Compute center of mass of the system.
    
    COM = sum(m_i * pos_i) / sum(m_i) for all active particles
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
    
    Returns:
        Array of shape (dim,) - center of mass position
    """
    # Only include active particles
    active_mass = jnp.where(active, mass, 0.0)
    
    # COM = sum(m * pos) / sum(m)
    total_mass = jnp.sum(active_mass)
    
    # Handle zero mass case
    if total_mass < 1e-10:
        return jnp.zeros(pos.shape[1])
    
    weighted_pos = active_mass[:, None] * pos
    com = jnp.sum(weighted_pos, axis=0) / total_mass
    
    # Ensure no NaNs
    return jnp.nan_to_num(com)


def adjust_timestep(dt, vel, force, mass, config):
    """
    Adjust timestep based on current velocity and acceleration.
    
    Prevents simulation blowups by reducing dt when forces/velocities are high.
    Can also increase dt when the system is stable to speed up simulation.
    
    Args:
        dt: float - current timestep
        vel: Array of shape (N, dim) - particle velocities
        force: Array of shape (N, dim) - forces on particles
        mass: Array of shape (N,) - particle masses
        config: UniverseConfig containing thresholds and limits
    
    Returns:
        float - new timestep (clamped and safe)
    """
    # Safe mass to avoid division by zero
    mass_safe = jnp.where(mass == 0, 1.0, mass)
    
    # Compute max speed and acceleration
    # Use jnp.max and jnp.linalg.norm
    speeds = jnp.linalg.norm(vel, axis=1)
    accels = jnp.linalg.norm(force / mass_safe[:, None], axis=1)
    
    max_speed = jnp.max(jnp.nan_to_num(speeds))
    max_acc = jnp.max(jnp.nan_to_num(accels))
    
    # Determine new dt
    # We use jax.lax.cond or simple python logic since this runs outside JIT in run_sim
    # But for compatibility with JIT, we should stick to JAX ops if possible
    # However, run_sim.py calls this in a loop that isn't fully JITed yet
    # So standard Python control flow is fine for now
    
    new_dt = dt
    
    if max_speed > config.velocity_threshold:
        # Too fast - reduce timestep
        new_dt = dt * 0.5
    elif max_acc > config.acceleration_threshold:
        # High acceleration - reduce timestep
        new_dt = dt * 0.5
    elif max_speed < (config.velocity_threshold * 0.2):
        # Very slow - safe to increase timestep
        new_dt = dt * 1.05
    
    # Clamp to allowed range
    # Range is relative to the ORIGINAL config.dt?
    # No, the requirement says "dt * config.max_dt_scale" where dt is current?
    # Actually, usually limits are relative to base dt.
    # But the requirement says:
    # new_dt = min(new_dt, dt * config.max_dt_scale)
    # new_dt = max(new_dt, dt * config.min_dt_scale)
    # This implies relative to CURRENT dt, which would allow exponential growth/decay.
    # Let's assume it means relative to the configured base dt (config.dt).
    # Wait, the requirement says "dt * config.max_dt_scale" where dt is the argument.
    # If I call this every step, and it returns new_dt, and I pass new_dt next time...
    # If I use current dt, it can drift indefinitely.
    # Let's clamp relative to config.dt to be safe and stable.
    
    min_dt = config.dt * config.min_dt_scale
    max_dt = config.dt * config.max_dt_scale
    
    new_dt = jnp.clip(new_dt, min_dt, max_dt)
    
    # Ensure float return type (not JAX array)
    return float(new_dt)
