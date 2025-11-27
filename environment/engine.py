"""
Environment engine core implementation.

Coordinates all environment-level effects including topology constraints,
substrate fields, expansion dynamics, and global forces.
"""

from __future__ import annotations
import jax.numpy as jnp
from typing import Optional, Tuple, Any
from state import UniverseConfig, UniverseState
from topologies import get_topology_handler
from topologies.base_topology import Topology


class EnvironmentEngine:
    """
    Central manager coordinating topology, substrates, expansion,
    and environment-level effects.
    
    The EnvironmentEngine applies environmental effects in a consistent order:
    1. Update substrate fields (fluid evolution, noise, etc.)
    2. Apply substrate forces to particles
    3. Apply expansion effects (Hubble flow)
    4. Apply topology constraints (wrapping, projection)
    
    This ensures proper interaction between different environmental systems.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize environment engine with configuration.
        
        Args:
            config: UniverseConfig containing simulation parameters
        """
        self.config = config
        
        # Load topology handler
        topology_type = getattr(config, "topology_type", 0)
        topology_map = {
            0: "flat",
            1: "torus",
            2: "sphere",
            3: "bubble",
        }
        topology_name = topology_map.get(topology_type, "flat")
        self.topology: Topology = get_topology_handler(topology_name, config)
        
        # Substrate placeholder (future implementation)
        # For now, substrate support is disabled
        self.substrate = None
        
        # Expansion parameters
        self.expansion_rate = 0.0
        if hasattr(config, "expansion_factor"):
            # Derive expansion rate from expansion_factor if available
            # H = (da/dt) / a ≈ expansion_rate
            self.expansion_rate = 0.0  # Placeholder for now
        
        # Expansion center (defaults to origin)
        self.expansion_center = jnp.zeros(config.dim)
    
    def apply_environment(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        force: jnp.ndarray,
        state: UniverseState
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply all environmental effects to positions, velocities, and forces.
        
        This method orchestrates the complete environment pipeline:
        1. Substrate field updates
        2. Substrate force application
        3. Expansion dynamics
        4. Topology constraints
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            force: Force array of shape (N, dim)
            state: Current UniverseState
        
        Returns:
            Tuple of (updated_pos, updated_vel, updated_force)
        """
        dt = self.config.dt
        
        # 1. Update substrate fields (if substrate is active)
        if self.substrate is not None:
            self.substrate.update(state, dt)
        
        # 2. Apply substrate forces (if substrate is active)
        if self.substrate is not None:
            substrate_force = self.substrate.force_at(pos, vel)
            force = force + substrate_force
        
        # 3. Apply expansion effects (if expansion is active)
        if self.expansion_rate != 0.0:
            pos, vel = self._apply_expansion(pos, vel, dt)
        
        # 4. Apply topology constraints (always applied)
        pos = self.topology.wrap_position(pos)
        
        return pos, vel, force
    
    def _apply_expansion(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply Hubble-like expansion to positions and velocities.
        
        Implements radial expansion from a center point:
        - Position: r' = r + (r - r_center) * H * dt
        - Velocity: v' = v + (r - r_center) * H
        
        where H is the expansion rate (Hubble parameter).
        
        Args:
            pos: Position array of shape (N, dim)
            vel: Velocity array of shape (N, dim)
            dt: Timestep
        
        Returns:
            Tuple of (expanded_pos, expanded_vel)
        """
        # Compute offset from expansion center
        offset = pos - self.expansion_center
        
        # Radial expansion velocity: v_expansion = H * r
        expansion_velocity = offset * self.expansion_rate
        
        # Update positions
        pos_expanded = pos + expansion_velocity * dt
        
        # Update velocities (add Hubble flow)
        vel_expanded = vel + expansion_velocity
        
        return pos_expanded, vel_expanded
    
    def compute_distance(
        self,
        p1: jnp.ndarray,
        p2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute topology-aware distance between points.
        
        Uses the topology's distance metric, which may differ from
        Euclidean distance (e.g., geodesic on sphere, minimum image
        for torus, curved metric for bubble).
        
        Args:
            p1: First position of shape (..., dim)
            p2: Second position of shape (..., dim)
        
        Returns:
            Distance scalar or array
        """
        return self.topology.distance(p1, p2)
    
    def set_expansion_rate(self, rate: float):
        """
        Set the expansion rate (Hubble parameter).
        
        Args:
            rate: Expansion rate H (typical units: 1/time)
        """
        self.expansion_rate = rate
    
    def set_expansion_center(self, center: jnp.ndarray):
        """
        Set the center point for expansion.
        
        Args:
            center: Center position of shape (dim,)
        """
        self.expansion_center = center
