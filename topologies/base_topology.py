"""
Base topology class for CosmoSim.

All topology implementations inherit from this abstract base class.
"""

from __future__ import annotations
import jax.numpy as jnp
from typing import Any
from state import UniverseConfig


class Topology:
    """
    Abstract base class for spacetime topology implementations.
    
    Topologies define how space is structured (flat, curved, periodic, etc.)
    and provide methods for position wrapping, distance computation, and
    rendering transformations.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize topology with universe configuration.
        
        Args:
            config: UniverseConfig containing simulation parameters
        """
        self.config = config
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply topology constraints to position.
        
        This method enforces the geometric structure of the space,
        such as periodic boundary conditions or spherical constraints.
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Wrapped position array of same shape
        """
        raise NotImplementedError("Subclasses must implement wrap_position()")
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute distance between two points in this topology.
        
        Different topologies may have different distance metrics
        (Euclidean, geodesic, etc.).
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Distance scalar or array
        """
        raise NotImplementedError("Subclasses must implement distance()")
    
    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Transform position for visualization.
        
        Optional method that can map positions to a different coordinate
        system for rendering (e.g., stereographic projection).
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Transformed position for rendering (default: unchanged)
        """
        return pos
