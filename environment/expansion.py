"""
Expansion models for CosmoSim.

Provides different expansion dynamics including:
- Linear (Hubble-like) expansion
- Future: Anisotropic, scale-factor, bubble-aware expansions
"""

from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple, Optional
from state import UniverseConfig


class BaseExpansion:
    """
    Abstract base class for all expansion models.
    
    Expansion models modify positions and velocities to simulate
    expanding space (cosmological expansion, inflation, etc.).
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize expansion model.
        
        Args:
            config: UniverseConfig containing expansion parameters
        """
        self.config = config
    
    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply expansion to positions and velocities.
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (expanded_pos, expanded_vel)
        """
        raise NotImplementedError("Subclasses must implement apply()")


class LinearExpansion(BaseExpansion):
    """
    Simple linear (Hubble-like) expansion.
    
    Implements uniform radial expansion from a center point:
        v_expansion = H * (r - r_center)
    
    where H is the Hubble parameter (expansion rate).
    
    This creates a velocity field proportional to distance from
    the expansion center, mimicking cosmological expansion.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize linear expansion.
        
        Args:
            config: UniverseConfig with expansion_rate and expansion_center
        """
        super().__init__(config)
        
        # Get expansion rate (Hubble parameter)
        self.rate = float(getattr(config, "expansion_rate", 0.0))
        
        # Get expansion center (defaults to origin)
        center_tuple = getattr(config, "expansion_center", (0.0, 0.0, 0.0))
        # Convert to JAX array, handling different input dimensions
        if isinstance(center_tuple, (tuple, list)):
            # Pad or truncate to match config.dim
            center_list = list(center_tuple)
            while len(center_list) < config.dim:
                center_list.append(0.0)
            center_list = center_list[:config.dim]
            self.center = jnp.array(center_list, dtype=float)
        else:
            self.center = jnp.array(center_tuple, dtype=float)
    
    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply linear expansion.
        
        Updates positions and velocities according to Hubble flow:
        - pos' = pos + H * (pos - center) * dt
        - vel' = vel + H * (pos - center)
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (expanded_pos, expanded_vel)
        """
        # If no expansion, return unchanged
        if self.rate == 0.0:
            return pos, vel
        
        # Compute displacement from expansion center
        offset = pos - self.center
        
        # Expansion velocity: v = H * r
        expansion_vel = offset * self.rate
        
        # Update positions: r' = r + v * dt
        pos_expanded = pos + expansion_vel * dt
        
        # Update velocities: v' = v + expansion_acceleration
        vel_expanded = vel + expansion_vel
        
        return pos_expanded, vel_expanded


class AnisotropicExpansion(BaseExpansion):
    """
    Direction-dependent (anisotropic) expansion.
    
    Implements axis-wise expansion with different rates for each direction:
        v_expansion = H_vec ⊙ (r - r_center)
    
    where H_vec = [Hx, Hy, Hz] and ⊙ is element-wise multiplication.
    
    This allows space to expand at different rates along different axes,
    enabling stretching, compression, or shearing effects. Useful for
    simulating non-uniform cosmic expansion or anisotropic inflation.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize anisotropic expansion.
        
        Args:
            config: UniverseConfig with expansion_axes and expansion_center
        """
        super().__init__(config)
        
        # Get axis-wise expansion rates (Hx, Hy, Hz)
        axes_tuple = getattr(config, "expansion_axes", (0.0, 0.0, 0.0))
        
        # Convert to JAX array, handling different dimensions
        if isinstance(axes_tuple, (tuple, list)):
            # Pad or truncate to match config.dim
            axes_list = list(axes_tuple)
            while len(axes_list) < config.dim:
                axes_list.append(0.0)
            axes_list = axes_list[:config.dim]
            self.rates = jnp.array(axes_list, dtype=float)
        else:
            self.rates = jnp.array(axes_tuple, dtype=float)
        
        # Get expansion center (defaults to origin)
        center_tuple = getattr(config, "expansion_center", (0.0, 0.0, 0.0))
        # Convert to JAX array
        if isinstance(center_tuple, (tuple, list)):
            center_list = list(center_tuple)
            while len(center_list) < config.dim:
                center_list.append(0.0)
            center_list = center_list[:config.dim]
            self.center = jnp.array(center_list, dtype=float)
        else:
            self.center = jnp.array(center_tuple, dtype=float)
    
    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply anisotropic expansion.
        
        Updates positions and velocities with axis-dependent expansion:
        - pos' = pos + H_vec ⊙ (pos - center) * dt
        - vel' = vel + H_vec ⊙ (pos - center)
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (expanded_pos, expanded_vel)
        """
        # If all rates are zero, return unchanged
        if jnp.allclose(self.rates, 0.0):
            return pos, vel
        
        # Compute displacement from expansion center
        offset = pos - self.center
        
        # Expansion velocity: element-wise multiply by axis rates
        # rates is shape (dim,), offset is (N, dim) → broadcasting works
        expansion_vel = offset * self.rates
        
        # Update positions
        pos_expanded = pos + expansion_vel * dt
        
        # Update velocities
        vel_expanded = vel + expansion_vel
        
        return pos_expanded, vel_expanded


def get_expansion_handler(config: UniverseConfig) -> Optional[BaseExpansion]:
    """
    Factory function for expansion models.
    
    Phase 2 supports:
    - "none": No expansion
    - "linear": Linear/Hubble expansion
    - "anisotropic": Direction-dependent expansion
    
    Args:
        config: UniverseConfig containing expansion_type
    
    Returns:
        Expansion handler instance, or None if expansion_type is "none"
    
    Raises:
        ValueError: If expansion_type is unknown
    
    Example:
        >>> expansion = get_expansion_handler(config)
        >>> if expansion:
        >>>     pos, vel = expansion.apply(pos, vel, dt)
    """
    expansion_type = getattr(config, "expansion_type", "none")
    
    if expansion_type == "none" or expansion_type is None:
        return None
    
    expansion_type_lower = expansion_type.lower()
    
    if expansion_type_lower == "linear":
        return LinearExpansion(config)
    
    if expansion_type_lower == "anisotropic":
        return AnisotropicExpansion(config)
    
    raise ValueError(f"Unknown expansion type: {expansion_type}")
