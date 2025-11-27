"""
Spherical topology implementation.

Positions constrained to the surface of a sphere.
"""

from __future__ import annotations
import jax.numpy as jnp
from .base_topology import Topology


class SphereTopology(Topology):
    """
    Spherical (S²) topology.
    
    Constrains all positions to lie on the surface of a sphere
    of radius R. Useful for simulations on curved manifolds.
    """
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Project positions onto sphere surface.
        
        Normalizes position vectors to have magnitude equal to
        the sphere radius.
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Positions normalized to sphere surface
        """
        # Get sphere radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            radius = float(self.config.bounds)
        else:
            radius = float(self.config.radius)
        
        # Compute current radii
        r = jnp.sqrt(jnp.sum(pos ** 2, axis=-1, keepdims=True))
        
        # Avoid division by zero
        r = jnp.where(r < 1e-10, 1e-10, r)
        
        # Normalize to sphere radius
        normalized = pos * (radius / r)
        
        return normalized
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute geodesic (great circle) distance on sphere.
        
        Uses the haversine formula to find the shortest path along
        the sphere surface.
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Geodesic distance along sphere surface
        """
        # Get sphere radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            radius = float(self.config.bounds)
        else:
            radius = float(self.config.radius)
        
        # Normalize positions to unit sphere
        r1 = jnp.sqrt(jnp.sum(p1 ** 2, axis=-1, keepdims=True))
        r2 = jnp.sqrt(jnp.sum(p2 ** 2, axis=-1, keepdims=True))
        
        n1 = p1 / jnp.where(r1 < 1e-10, 1e-10, r1)
        n2 = p2 / jnp.where(r2 < 1e-10, 1e-10, r2)
        
        # Compute dot product (cosine of angle)
        cos_angle = jnp.sum(n1 * n2, axis=-1)
        cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
        
        # Geodesic distance = R * angle
        angle = jnp.arccos(cos_angle)
        
        return radius * angle
    
    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Optional stereographic projection for 2D visualization.
        
        For now, returns positions as-is in embedded 3D space.
        
        Args:
            pos: Position array
        
        Returns:
            Position (unchanged for now)
        """
        return pos
