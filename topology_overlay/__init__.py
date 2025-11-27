"""
Topology overlay system for CosmoSim visualizations.

This package provides modular overlay generators that create visual
layers (grids, bounds, manifolds, etc.) for Three.js rendering.
"""

from .factory import get_topology
from .base import TopologyOverlay

__all__ = ["get_topology", "TopologyOverlay"]
