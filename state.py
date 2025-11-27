"""State management for CosmoSim universe simulations.

This module defines the core simulation state structures using JAX-compatible
dataclasses, including universe state, particle states, and field configurations.
All state structures use fixed-shape JAX arrays for JIT compilation.
"""

import jax
import jax.numpy as jnp
from chex import dataclass

@dataclass
class UniverseConfig:
    """Static configuration parameters for a universe simulation."""
    # Required fields (no defaults)
    physics_mode: int   # 0 = VECTOR, 1 = LATTICE
    radius: float
    max_entities: int
    max_nodes: int
    dt: float           # timestep
    c: float            # speed of light-like constant
    G: float            # gravitational constant-like parameter
    # Optional fields with defaults – placed after required fields
    dim: int = 2        # Spatial dimensionality (2, 3, etc.)
    topology_type: int = 0  # 0 = FLAT, 1 = TORUS, 2 = SPHERE, 3 = BUBBLE
    bounds: float | None = None  # None means infinite flat space
    # Expansion parameters
    expansion_type: str = "none"  # "none", "linear", "anisotropic", "scale_factor"
    expansion_rate: float = 0.0   # Hubble parameter H (for linear expansion)
    expansion_center: tuple = (0.0, 0.0, 0.0)  # Center of expansion
    expansion_axes: tuple = (0.0, 0.0, 0.0)  # Axis-wise rates (Hx, Hy, Hz) for anisotropic
    expansion_mode: str = "linear"  # "linear", "matter", "radiation", "inflation" for scale_factor
    H: float = 0.01  # Hubble-like constant for scale_factor expansion
    # Bubble expansion parameters
    bubble_radius: float = 10.0  # Initial bubble size
    bubble_expand: bool = False  # Should the bubble itself inflate?
    curvature_k: float = 0.0  # Curvature parameter for interior metric

@dataclass
class UniverseState:
    """Preallocated JAX arrays representing the complete simulation state."""
    # Global scalars (stored as JAX arrays)
    time: jnp.ndarray
    expansion_factor: jnp.ndarray
    curvature_k: jnp.ndarray

    # Entity arrays (always present)
    entity_active: jnp.ndarray  # shape (max_entities,) boolean mask
    entity_pos: jnp.ndarray     # shape (max_entities, dim)
    entity_vel: jnp.ndarray     # shape (max_entities, dim)
    entity_mass: jnp.ndarray    # shape (max_entities,)
    entity_type: jnp.ndarray    # shape (max_entities,)

    # Lattice arrays (always present, used in LATTICE mode)
    node_active: jnp.ndarray    # shape (max_nodes,)
    node_pos: jnp.ndarray       # shape (max_nodes, 2)
    edge_active: jnp.ndarray    # shape (max_nodes, max_nodes)
    edge_indices: jnp.ndarray   # shape (max_nodes, max_nodes, 2)

    # Optional topology fields – placed at the end to keep .replace() compatibility
    topology_type: int = 0
    bounds: float = 0.0

def initialize_state(config: UniverseConfig) -> UniverseState:
    """Initialize a UniverseState from configuration parameters.

    Args:
        config: Configuration parameters for the universe

    Returns:
        A fully initialized UniverseState with preallocated arrays
    """
    return UniverseState(
        # Global scalars
        time=jnp.array(0.0),
        expansion_factor=jnp.array(1.0),
        curvature_k=jnp.array(0.0),

        # Entity arrays
        entity_active=jnp.zeros(config.max_entities, dtype=bool),
        entity_pos=jnp.zeros((config.max_entities, config.dim)),
        entity_vel=jnp.zeros((config.max_entities, config.dim)),
        entity_mass=jnp.zeros(config.max_entities),
        entity_type=jnp.zeros(config.max_entities),

        # Lattice arrays
        node_active=jnp.zeros(config.max_nodes, dtype=bool),
        node_pos=jnp.zeros((config.max_nodes, 2)),
        edge_active=jnp.zeros((config.max_nodes, config.max_nodes), dtype=bool),
        edge_indices=jnp.zeros((config.max_nodes, config.max_nodes, 2), dtype=jnp.int32),

        # Topology fields
        topology_type=config.topology_type,
        bounds=0.0 if config.bounds is None else config.bounds,
    )
