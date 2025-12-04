"""
Möbius strip topology implementation.

Non-orientable surface with a single boundary inversion twist.
"""

from __future__ import annotations
import jax.numpy as jnp
from .base_topology import Topology


class MobiusTopology(Topology):
    """
    Möbius strip topology.
    
    Implements a non-orientable surface where crossing the u-boundary
    inverts the v-coordinate.
    
    Coordinate model:
    u ∈ [0, L]   (Length along the strip)
    v ∈ [-W, W]  (Width across the strip)
    
    Wrapping Rule:
    if u > L:    u = u - L;   v = -v
    if u < 0:    u = u + L;   v = -v
    """
    
    MOBIUS_TOPOLOGY = 5
    
    def wrap_position(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Apply Möbius wrapping to positions.
        
        Args:
            pos: Position array of shape (..., 2) [u, v]
        
        Returns:
            Wrapped position
        """
        L = float(self.config.radius)
        # Use width if available, else default to 1.0 (or radius/4 if we wanted, but 1.0 is safer default)
        W = getattr(self.config, 'width', 1.0)
        
        u = pos[..., 0]
        v = pos[..., 1]
        
        # Check wrap right (u > L)
        wrap_right = u > L
        u = jnp.where(wrap_right, u - L, u)
        v = jnp.where(wrap_right, -v, v)
        
        # Check wrap left (u < 0)
        wrap_left = u < 0
        u = jnp.where(wrap_left, u + L, u)
        v = jnp.where(wrap_left, -v, v)
        
        # Clamp v to [-W, W]
        v = jnp.clip(v, -W, W)
        
        return jnp.stack([u, v], axis=-1)
    
    def distance(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute minimum distance on Möbius strip.
        
        Considers direct path and path through the twist.
        """
        L = float(self.config.radius)
        
        # Direct difference
        diff1 = p2 - p1
        dist1 = jnp.sqrt(jnp.sum(diff1**2, axis=-1))
        
        # Inverted difference (wrapping through boundary)
        # p2' = (p2.u ± L, -p2.v)
        
        # Try wrapping p2 to the right (u + L) and inverting v
        p2_right = jnp.stack([p2[..., 0] + L, -p2[..., 1]], axis=-1)
        diff2 = p2_right - p1
        dist2 = jnp.sqrt(jnp.sum(diff2**2, axis=-1))
        
        # Try wrapping p2 to the left (u - L) and inverting v
        p2_left = jnp.stack([p2[..., 0] - L, -p2[..., 1]], axis=-1)
        diff3 = p2_left - p1
        dist3 = jnp.sqrt(jnp.sum(diff3**2, axis=-1))
        
        # Return minimum
        return jnp.minimum(dist1, jnp.minimum(dist2, dist3))

    def difference_vector(self, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute minimum offset vector (p2 - p1) on Möbius strip.
        
        Returns the vector Δ such that |Δ| is minimized.
        """
        L = float(self.config.radius)
        
        # Direct difference
        diff1 = p2 - p1
        dist1_sq = jnp.sum(diff1**2, axis=-1)
        
        # Inverted difference (wrapping through boundary)
        # p2' = (p2.u ± L, -p2.v)
        
        # Try wrapping p2 to the right (u + L) and inverting v
        p2_right = jnp.stack([p2[..., 0] + L, -p2[..., 1]], axis=-1)
        diff2 = p2_right - p1
        dist2_sq = jnp.sum(diff2**2, axis=-1)
        
        # Try wrapping p2 to the left (u - L) and inverting v
        p2_left = jnp.stack([p2[..., 0] - L, -p2[..., 1]], axis=-1)
        diff3 = p2_left - p1
        dist3_sq = jnp.sum(diff3**2, axis=-1)
        
        # Select minimum distance vector
        # We need to select per-entity.
        # Create masks for where dist2 or dist3 is smaller than dist1
        
        # Expand dims for broadcasting if needed, but jnp.where handles it
        
        # Compare dist1 and dist2
        use_2 = dist2_sq < dist1_sq
        best_diff = jnp.where(use_2[..., None], diff2, diff1)
        best_dist_sq = jnp.where(use_2, dist2_sq, dist1_sq)
        
        # Compare result with dist3
        use_3 = dist3_sq < best_dist_sq
        best_diff = jnp.where(use_3[..., None], diff3, best_diff)
        
        return best_diff

    def project_for_render(self, pos: jnp.ndarray) -> jnp.ndarray:
        """
        Identity projection for now (2D view).
        """
        return pos
