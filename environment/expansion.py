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


class ScaleFactorExpansion(BaseExpansion):
    """
    Cosmological expansion using time-evolving scale factor a(t).
    
    Implements different cosmic expansion epochs:
    - Linear: a(t) = 1 + H*t (simple linear growth)
    - Matter-dominated: a(t) ∝ t^(2/3) (matter era)
    - Radiation-dominated: a(t) ∝ t^(1/2) (early universe)
    - Inflation: a(t) = exp(H*t) (exponential expansion)
    
    Positions scale with a(t), and velocities are updated to reflect
    the changing expansion rate.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize scale factor expansion.
        
        Args:
            config: UniverseConfig with expansion_mode and H
        """
        super().__init__(config)
        
        # Expansion mode: "linear", "matter", "radiation", "inflation"
        self.mode = getattr(config, "expansion_mode", "linear")
        
        # Hubble-like constant
        self.H = float(getattr(config, "H", 0.01))
        
        # Internal state: scale factor and cosmic time
        self.a = 1.0  # Initial scale factor
        self.t = 0.0  # Cosmic time
    
    def _update_scale_factor(self, dt: float):
        """
        Update scale factor based on cosmic time and expansion mode.
        
        Args:
            dt: Timestep
        """
        self.t += dt
        
        if self.mode == "linear":
            # Linear growth: a = 1 + H*t
            self.a = 1.0 + self.H * self.t
        
        elif self.mode == "matter":
            # Matter-dominated: a ∝ t^(2/3)
            if self.t > 0:
                self.a = float(max(self.t ** (2.0/3.0), 1e-6))
            else:
                self.a = 1.0
        
        elif self.mode == "radiation":
            # Radiation-dominated: a ∝ t^(1/2)
            if self.t > 0:
                self.a = float(max(self.t ** 0.5, 1e-6))
            else:
                self.a = 1.0
        
        elif self.mode == "inflation":
            # Exponential inflation: a = exp(H*t)
            self.a = float(jnp.exp(self.H * self.t))
        
        else:
            raise ValueError(f"Unknown expansion_mode: {self.mode}")
    
    def apply(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply scale factor expansion.
        
        Positions are scaled by a(t), and velocities are updated
        to reflect the expansion.
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (scaled_pos, updated_vel)
        """
        # Update scale factor
        old_a = self.a
        self._update_scale_factor(dt)
        
        # Scale positions by the ratio of new to old scale factor
        # This gives incremental scaling: pos_new = pos_old * (a_new / a_old)
        if old_a > 1e-9:
            scale_ratio = self.a / old_a
            pos_scaled = pos * scale_ratio
        else:
            pos_scaled = pos
        
        # Update velocity to account for expansion
        # v_new = v_old + H_effective * pos
        # where H_effective = (a_new - a_old) / (a_old * dt)
        if dt > 1e-9 and old_a > 1e-9:
            H_effective = (self.a - old_a) / (old_a * dt)
            vel_expansion = H_effective * pos
            vel_updated = vel + vel_expansion
        else:
            vel_updated = vel
        
        return pos_scaled, vel_updated


class BubbleExpansion(BaseExpansion):
    """
    Expansion model for bubble universes with curved interior geometry.
    
    Features:
    - Radial expansion: v ~ rate * r
    - Optional bubble radius growth (inflating bubble wall)
    - Curvature-aware scaling (metric effects)
    - Designed for compatibility with BubbleTopology
    
    The bubble radius is stored internally and updated each step.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize bubble expansion.
        
        Args:
            config: UniverseConfig with bubble parameters
        """
        super().__init__(config)
        
        # Base scalar expansion rate (for radial expansion)
        self.rate = float(getattr(config, "expansion_rate", 0.01))
        
        # Initial bubble radius (usually config.radius)
        self.bubble_radius = float(
            getattr(config, "bubble_radius", getattr(config, "radius", 10.0))
        )
        
        # Whether the bubble itself expands (inflation)
        self.expand_bubble = bool(getattr(config, "bubble_expand", False))
        
        # Curvature parameter (0 = flat, + = closed)
        self.curvature_k = float(getattr(config, "curvature_k", 0.0))
        
        # Center point of bubble
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
        Apply bubble expansion.
        
        Radial expansion is applied as:
            v_exp = rate * r * f(curvature)
        then positions updated:
            pos_new = pos + dt * v_exp
            
        Bubble radius may grow each step if bubble_expand is True.
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (expanded_pos, expanded_vel)
        """
        # Offset from bubble center
        offset = pos - self.center
        
        # Compute radial distance
        r = jnp.linalg.norm(offset, axis=1, keepdims=True)
        
        # Avoid division by zero when normalizing
        r_safe = jnp.maximum(r, 1e-9)
        direction = offset / r_safe
        
        # Curvature factor (simple approximation):
        # f = sqrt(1 - k r^2) for positive curvature
        # This models the metric effect where expansion is slower near the edge
        # of a positively curved space (like a hypersphere projection)
        if self.curvature_k != 0.0:
            # Clip to avoid sqrt of negative number
            curvature_term = 1.0 - self.curvature_k * r * r
            curvature_factor = jnp.sqrt(jnp.clip(curvature_term, 1e-9, 1.0))
        else:
            curvature_factor = 1.0
        
        # Expansion velocity: v = rate * r * curvature_factor
        expansion_vel = direction * (self.rate * r * curvature_factor)
        
        # Update positions
        pos_expanded = pos + expansion_vel * dt
        
        # Update velocities
        vel_expanded = vel + expansion_vel
        
        # Expand bubble radius itself (inflation)
        if self.expand_bubble:
            self.bubble_radius = self.bubble_radius * (1.0 + self.rate * dt)
        
        return pos_expanded, vel_expanded


def get_expansion_handler(config: UniverseConfig) -> Optional[BaseExpansion]:
    """
    Factory function for expansion models.
    
    Phase 4 supports:
    - "none": No expansion
    - "linear": Linear/Hubble expansion
    - "anisotropic": Direction-dependent expansion
    - "scale_factor": Cosmological a(t) expansion
    - "bubble": Topology-aware bubble expansion
    
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
    
    if expansion_type_lower == "scale_factor":
        return ScaleFactorExpansion(config)
    
    if expansion_type_lower == "bubble":
        return BubbleExpansion(config)
    
    raise ValueError(f"Unknown expansion type: {expansion_type}")
