"""
Topologies package for CosmoSim.

Provides different spacetime topology implementations including:
- Flat (Euclidean)
- Torus (periodic boundary conditions)
- Sphere (spherical geometry)
- Bubble (inflationary bubble universe)
- Hyperbolic (future implementation)
"""

from .factory import get_topology_handler
from .base_topology import Topology

__all__ = ["get_topology_handler", "Topology"]
