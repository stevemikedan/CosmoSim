"""
Utility functions for environment calculations.

Helper math for expansion, coordinate transforms, and field operations.
"""

from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple


def compute_hubble_flow(
    pos: jnp.ndarray,
    H: float,
    center: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Hubble expansion velocity at given positions.
    
    v = H * (r - r_center)
    
    Args:
        pos: Positions of shape (N, dim)
        H: Hubble parameter (expansion rate)
        center: Expansion center of shape (dim,)
    
    Returns:
        Velocities of shape (N, dim)
    """
    offset = pos - center
    return H * offset


def scale_factor_derivative(a: float, H: float) -> float:
    """
    Compute time derivative of scale factor.
    
    da/dt = H * a
    
    Args:
        a: Scale factor
        H: Hubble parameter
    
    Returns:
        Rate of change of scale factor
    """
    return H * a


def comoving_to_physical(
    pos_comoving: jnp.ndarray,
    a: float
) -> jnp.ndarray:
    """
    Convert comoving coordinates to physical coordinates.
    
    r_physical = a * r_comoving
    
    Args:
        pos_comoving: Comoving positions
        a: Scale factor
    
    Returns:
        Physical positions
    """
    return a * pos_comoving


def physical_to_comoving(
    pos_physical: jnp.ndarray,
    a: float
) -> jnp.ndarray:
    """
    Convert physical coordinates to comoving coordinates.
    
    r_comoving = r_physical / a
    
    Args:
        pos_physical: Physical positions
        a: Scale factor
    
    Returns:
        Comoving positions
    """
    return pos_physical / a


def radial_distance(
    pos: jnp.ndarray,
    center: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute radial distance from center.
    
    Args:
        pos: Positions of shape (N, dim)
        center: Center point of shape (dim,)
    
    Returns:
        Radial distances of shape (N,)
    """
    offset = pos - center
    return jnp.sqrt(jnp.sum(offset ** 2, axis=-1))
