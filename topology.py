"""
Universe topology definitions for CosmoSim.

This module defines different universe topologies (e.g., toroidal, flat)
and their associated boundary conditions and neighbor calculations.
Topologies are polymorphic but use fixed-shape arrays.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Callable, Protocol

# Topology Constants
TOPOLOGY_FLAT = 0
TOPOLOGY_TORUS = 1
TOPOLOGY_SPHERE = 2
TOPOLOGY_BUBBLE = 3


def compute_distance(p1: jnp.ndarray, p2: jnp.ndarray, topology_type: int, radius: float) -> jnp.ndarray:
    """Compute distance between two points based on topology type.

    Args:
        p1: Point 1, shape (dim,)
        p2: Point 2, shape (dim,)
        topology_type: 0=FLAT, 1=TORUS, >1=RESERVED
        radius: Radius parameter for the topology (used for torus)

    Returns:
        Distance as a scalar JAX array
    """
    # Flat distance works for any dimensionality
    def flat_distance(p1, p2, radius):
        diff = p2 - p1
        return jnp.sqrt(jnp.sum(diff * diff))

    def torus_distance(p1, p2, radius):
        # Wrap each coordinate independently within [-radius, +radius]
        width = 2.0 * radius
        # Shift to [0, width] range, apply modulo, then shift back
        wrapped = (p2 + radius) % width - radius - (p1 + radius) % width + radius
        diff = wrapped
        return jnp.sqrt(jnp.sum(diff * diff))

    def reserved_distance(p1, p2, radius):
        return jnp.nan

    branches = [
        lambda args: flat_distance(args[0], args[1], args[2]),
        lambda args: torus_distance(args[0], args[1], args[2]),
        lambda args: reserved_distance(args[0], args[1], args[2]),
    ]

    safe_type = jnp.clip(topology_type, 0, 2)
    return jax.lax.switch(safe_type, branches, (p1, p2, radius))


def enforce_boundaries(positions: jnp.ndarray, topology_type: int, radius: float) -> jnp.ndarray:
    """Enforce boundary conditions based on topology type.

    Args:
        positions: Array of positions, shape (N, dim)
        topology_type: 0=FLAT, 1=TORUS, >1=RESERVED
        radius: Radius parameter for the topology (used for torus)

    Returns:
        Positions with boundaries enforced, shape (N, dim)
    """
    def flat_boundaries(positions, radius):
        return positions

    def torus_boundaries(positions, radius):
        width = 2.0 * radius
        # Wrap each coordinate to [ -radius, +radius )
        return positions - width * jnp.floor((positions + radius) / width)

    def reserved_boundaries(positions, radius):
        return positions

    branches = [
        lambda args: flat_boundaries(args[0], args[1]),
        lambda args: torus_boundaries(args[0], args[1]),
        lambda args: reserved_boundaries(args[0], args[1]),
    ]

    safe_type = jnp.clip(topology_type, 0, 2)
    return jax.lax.switch(safe_type, branches, (positions, radius))


def apply_topology(pos: jnp.ndarray, vel: jnp.ndarray, config) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply topology to positions (and optionally velocities).

    Currently supports:
    - Flat (type 0): identity.
    - Toroidal (type 1): wrap positions within bounds if bounds is set.

    Args:
        pos: Positions array, shape (N, dim)
        vel: Velocities array, shape (N, dim)
        config: UniverseConfig instance containing topology_type and bounds.

    Returns:
        (new_pos, new_vel) after applying topology.
    """
    # Flat topology – identity
    def flat(pos, vel, bounds):
        return pos, vel

    def torus(pos, vel, bounds):
        # If bounds is None or zero, treat as flat
        def wrap(p):
            width = 2.0 * bounds
            return p - width * jnp.floor((p + bounds) / width)
        new_pos = wrap(pos)
        return new_pos, vel

    def sphere(pos, vel, bounds):
        # If no constraint requested, behave as identity
        if not getattr(config, "enforce_sphere_constraint", False):
            return pos, vel
        
        R = getattr(config, "radius", None)
        if R is None or R <= 0:
            return pos, vel
        
        # Project positions back to radius R
        norms = jnp.linalg.norm(pos, axis=1, keepdims=True)
        # Avoid divide-by-zero
        norms = jnp.where(norms == 0, 1e-12, norms)
        new_pos = pos * (R / norms)
        
        # Project velocities onto tangent plane
        # v_tangent = v - (v·n) n
        n = new_pos / R
        v_dot_n = jnp.sum(vel * n, axis=1, keepdims=True)
        new_vel = vel - v_dot_n * n
        
        return new_pos, new_vel

    def reserved(pos, vel, bounds):
        return pos, vel

    branches = [
        lambda args: flat(args[0], args[1], args[2]),      # 0 = flat
        lambda args: torus(args[0], args[1], args[2]),     # 1 = torus
        lambda args: sphere(args[0], args[1], args[2]),    # 2 = sphere
        lambda args: reserved(args[0], args[1], args[2]),  # 3+ = reserved
    ]

    # Unified torus bounds derivation
    if config.topology_type == TOPOLOGY_TORUS:
        # Prefer torus_size if defined
        if hasattr(config, "torus_size") and config.torus_size is not None:
            effective_width = config.torus_size
        elif config.radius is not None:
            # fallback for legacy configs
            effective_width = 2.0 * config.radius
        else:
            raise ValueError("Torus topology requires torus_size or radius > 0.")
        
        # Prevent zero or negative domain
        if effective_width <= 0:
            raise ValueError(f"Invalid torus width {effective_width}. Must be > 0.")
        
        bounds = effective_width / 2.0
    else:
        # Non-torus fallback (retain existing behavior)
        bounds = config.radius if config.bounds is None else config.bounds
        if bounds is None:
            bounds = 0.0
    
    # Early validation guard for torus
    if config.topology_type == TOPOLOGY_TORUS and bounds <= 0:
        raise ValueError("Torus topology requires bounds > 0 (derived from torus_size or radius).")
    
    safe_type = jnp.clip(config.topology_type, 0, 2)
    return jax.lax.switch(safe_type, branches, (pos, vel, bounds))
