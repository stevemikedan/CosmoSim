"""
Universe topology definitions for CosmoSim.

This module defines different universe topologies (e.g., toroidal, spherical,
flat) and their associated boundary conditions and neighbor calculations.
Topologies are polymorphic but use fixed-shape arrays.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Callable, Protocol


def compute_distance(p1: jnp.ndarray, p2: jnp.ndarray, topology_type: int, radius: float) -> jnp.ndarray:
    """Compute distance between two points based on topology type.
    
    Args:
        p1: Point 1, shape (2,)
        p2: Point 2, shape (2,)
        topology_type: 0=FLAT, 1=SPHERE, 2=TORUS, >=3=RESERVED
        radius: Radius parameter for the topology
        
    Returns:
        Distance as a scalar JAX array
    """
    
    def flat_distance(p1, p2, radius):
        """Euclidean distance on flat plane."""
        diff = p2 - p1
        return jnp.sqrt(jnp.sum(diff * diff))
    
    def sphere_distance(p1, p2, radius):
        """Great-circle distance on sphere using angular coordinates."""
        theta1, phi1 = p1
        theta2, phi2 = p2
        
        # Spherical law of cosines
        cos_ang = (jnp.sin(phi1) * jnp.sin(phi2) + 
                   jnp.cos(phi1) * jnp.cos(phi2) * jnp.cos(theta2 - theta1))
        
        # Clamp to avoid numerical issues with arccos
        cos_ang = jnp.clip(cos_ang, -1.0, 1.0)
        ang = jnp.arccos(cos_ang)
        
        return radius * ang
    
    def torus_distance(p1, p2, radius):
        """Distance on rectangular torus with wrap-around."""
        L = 2.0 * radius
        dx = p2 - p1
        # Wrap to nearest image
        dx = (dx + L/2.0) % L - L/2.0
        return jnp.sqrt(jnp.sum(dx * dx))
    
    def reserved_distance(p1, p2, radius):
        """Placeholder for future custom topologies."""
        return jnp.nan
    
    # Dispatch based on topology_type using jax.lax.switch
    branches = [
        lambda args: flat_distance(args[0], args[1], args[2]),
        lambda args: sphere_distance(args[0], args[1], args[2]),
        lambda args: torus_distance(args[0], args[1], args[2]),
        lambda args: reserved_distance(args[0], args[1], args[2]),
    ]
    
    # Clamp topology_type to valid range [0, 3]
    safe_type = jnp.clip(topology_type, 0, 3)
    
    return jax.lax.switch(safe_type, branches, (p1, p2, radius))


def enforce_boundaries(positions: jnp.ndarray, topology_type: int, radius: float) -> jnp.ndarray:
    """Enforce boundary conditions based on topology type.
    
    Args:
        positions: Array of positions, shape (N, 2)
        topology_type: 0=FLAT, 1=SPHERE, 2=TORUS, >=3=RESERVED
        radius: Radius parameter for the topology
        
    Returns:
        Positions with boundaries enforced, shape (N, 2)
    """
    
    def flat_boundaries(positions, radius):
        """Flat topology has no boundaries - return unchanged."""
        return positions
    
    def sphere_boundaries(positions, radius):
        """Wrap angular coordinates for sphere topology."""
        theta = positions[:, 0]
        phi = positions[:, 1]
        
        # Wrap theta to [-pi, pi]
        theta = ((theta + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
        
        # Wrap phi to [-pi/2, pi/2]
        phi = ((phi + jnp.pi/2.0) % jnp.pi) - jnp.pi/2.0
        
        return jnp.stack([theta, phi], axis=1)
    
    def torus_boundaries(positions, radius):
        """Wrap positions to torus domain [0, L] x [0, L]."""
        L = 2.0 * radius
        return positions % L
    
    def reserved_boundaries(positions, radius):
        """Placeholder for future custom topologies."""
        return positions
    
    # Dispatch based on topology_type using jax.lax.switch
    branches = [
        lambda args: flat_boundaries(args[0], args[1]),
        lambda args: sphere_boundaries(args[0], args[1]),
        lambda args: torus_boundaries(args[0], args[1]),
        lambda args: reserved_boundaries(args[0], args[1]),
    ]
    
    # Clamp topology_type to valid range [0, 3]
    safe_type = jnp.clip(topology_type, 0, 3)
    
    return jax.lax.switch(safe_type, branches, (positions, radius))
