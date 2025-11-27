"""
Torus topology implementation.

Periodic boundary conditions in all dimensions - wrap-around space.
"""

from __future__ import annotations
import jax.numpy as jnp
from .base_topology import Topology


class TorusTopology(Topology):
    """
    Toroidal (periodic) topology.
    
    Implements periodic boundary conditions where particles that exit
    one side of the simulation box re-enter from the opposite side.
    Classic "Pac-Man" topology extended to all dimensions.
    """
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply periodic wrapping to positions.
        
        Wraps positions modulo the box size so particles moving off
        one edge reappear on the opposite edge.
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Wrapped position within [-bounds, bounds]
        """
        # Use bounds if available, otherwise radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            bounds = float(self.config.bounds)
        else:
            bounds = float(self.config.radius)
        
        # Periodic wrapping: shift to [0, 2*bounds], modulo, shift back
        wrapped = ((pos + bounds) % (2 * bounds)) - bounds
        
        return wrapped
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute minimum image distance in periodic space.
        
        Finds the shortest distance between two points considering
        all periodic images.
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Minimum image distance
        """
        # Use bounds
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            bounds = float(self.config.bounds)
        else:
            bounds = float(self.config.radius)
        
        box_size = 2 * bounds
        
        # Compute raw difference
        diff = p2 - p1
        
        # Apply minimum image convention
        # Wrap differences to [-box_size/2, box_size/2]
        diff = diff - box_size * jnp.round(diff / box_size)
        
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    
    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Identity transformation (torus is rendered in embedded space).
        
        Args:
            pos: Position array
        
        Returns:
            Unchanged position
        """
        return pos
