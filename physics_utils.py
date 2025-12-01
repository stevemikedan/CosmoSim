"""
Physics utility functions for CosmoSim.

This module contains shared physics computations that can be used across
different parts of the simulation engine (kernel, run_sim, etc.).
"""

import jax.numpy as jnp
from state import UniverseConfig


def compute_gravity_forces(pos, mass, active, config):
    """
    Compute gravitational forces between all particles with softening.
    
    Uses a softened gravity kernel to prevent numerical blowups when
    particles get too close:
        F = G * m_i * m_j / (r^2 + epsilon^2)^(3/2)
    
    Args:
        pos: Array of shape (N, dim) - particle positions
        mass: Array of shape (N,) - particle masses
        active: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig containing G and gravity_softening
    
    Returns:
        Array of shape (N, dim) - total gravitational force on each particle
    """
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
    # We need the direction (unit vector) from i to j
    # For softened gravity, we use: F_vec = F_mag * displacement
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
