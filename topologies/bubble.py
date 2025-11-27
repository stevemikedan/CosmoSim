"""
Bubble universe topology implementation.

Inflationary bubble universe with curved interior metric and
optional expansion dynamics.
"""

from __future__ import annotations
import jax.numpy as jnp
from .base_topology import Topology


class BubbleTopology(Topology):
    """
    Bubble universe topology with curved metric.
    
    Represents a finite inflationary bubble with:
    - Radial symmetry from bubble center
    - Curved interior metric: ds² = a(t)² (dr²/(1-kr²) + r² dΩ²)
    - Optional expansion factor a(t)
    - Bubble wall at radius R
    
    Simplified for simulation efficiency while preserving key features.
    """
    
    def __init__(self, config, curvature_k: float = 0.1, expansion_rate: float = 0.0):
        """
        Initialize bubble topology.
        
        Args:
            config: UniverseConfig
            curvature_k: Spatial curvature parameter (default: 0.1, positive = closed)
            expansion_rate: Hubble-like expansion rate da/dt (default: 0.0)
        """
        super().__init__(config)
        self.curvature_k = curvature_k
        self.expansion_rate = expansion_rate
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Constrain positions to bubble interior.
        
        Particles are reflected/constrained at the bubble wall.
        Optionally applies metric scaling based on radial position.
        
        Args:
            pos: Position array of shape (..., dim)
        
        Returns:
            Constrained position within bubble
        """
        # Bubble wall radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            R_bubble = float(self.config.bounds)
        else:
            R_bubble = float(self.config.radius)
        
        # Compute radial distance from origin
        r = jnp.sqrt(jnp.sum(pos ** 2, axis=-1, keepdims=True))
        
        # Avoid division by zero
        r_safe = jnp.where(r < 1e-10, 1e-10, r)
        
        # If outside bubble, reflect back inside
        # Simple reflection: if r > R, set r = R - small epsilon
        r_constrained = jnp.where(
            r > R_bubble,
            R_bubble * 0.99,  # Just inside the wall
            r
        )
        
        # Normalize and scale
        direction = pos / r_safe
        constrained_pos = direction * r_constrained
        
        return constrained_pos
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute distance in curved bubble metric.
        
        Approximates the metric distance accounting for spatial curvature.
        For efficiency, uses a perturbed Euclidean distance.
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Metric distance in curved space
        """
        # Euclidean difference
        diff = p2 - p1
        euclidean_dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
        
        # Get bubble radius for normalization
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            R = float(self.config.bounds)
        else:
            R = float(self.config.radius)
        
        # Compute average radial position
        r_avg = 0.5 * (
            jnp.sqrt(jnp.sum(p1 ** 2, axis=-1)) +
            jnp.sqrt(jnp.sum(p2 ** 2, axis=-1))
        )
        
        # Metric correction factor from curvature
        # ds² ≈ (1 + k*r²/R²) * dx²
        # This is a simplified approximation
        r_normalized = r_avg / R
        metric_factor = jnp.sqrt(1.0 + self.curvature_k * r_normalized ** 2)
        
        return euclidean_dist * metric_factor
    
    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply expansion scaling for visualization.
        
        If expansion is active, shows the current size of the bubble.
        
        Args:
            pos: Position array
        
        Returns:
            Scaled position (for expanding universe)
        """
        # For now, return unchanged
        # Future: could apply scale factor a(t)
        return pos
