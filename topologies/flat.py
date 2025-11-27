"""
Flat (Euclidean) topology implementation.

Standard Euclidean space with optional hard boundaries.
"""

from __future__ import annotations
import jax.numpy as jnp
from .base_topology import Topology


class FlatTopology(Topology):
    """
    Flat Euclidean space topology.
    
    Standard 3D Euclidean geometry with optional boundary enforcement.
    No wrapping, no curvature - just regular space.
    """
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply boundary constraints if configured.
        
        If bounds are set, clips positions to stay within bounds.
        Otherwise, positions are unconstrained.
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Constrained position (clipped to bounds if applicable)
        """
        # If bounds are configured, clip to bounds
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            bounds = float(self.config.bounds)
            pos = jnp.clip(pos, -bounds, bounds)
        
        return pos
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Euclidean distance.
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Euclidean distance
        """
        diff = p2 - p1
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    
    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Identity transformation for flat space.
        
        Args:
            pos: Position array
        
        Returns:
            Unchanged position
        """
        return pos
