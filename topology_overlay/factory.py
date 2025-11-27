"""
Topology overlay factory.

Provides a unified interface for creating topology overlays by name.
"""

from __future__ import annotations
from typing import Optional
from state import UniverseConfig
from .base import TopologyOverlay
from .grid import GridOverlay
from .bounds import BoundsOverlay
from .curvature import CurvatureOverlay


def get_topology(name: Optional[str], config: UniverseConfig, **kwargs) -> Optional[TopologyOverlay]:
    """
    Get a topology overlay generator by name.
    
    Args:
        name: Overlay type name ("grid", "bounds", "curvature", or None)
        config: UniverseConfig for the simulation
        **kwargs: Additional parameters passed to the overlay constructor
    
    Returns:
        TopologyOverlay instance, or None if name is None
    
    Raises:
        ValueError: If overlay name is unknown
    
    Example:
        >>> overlay = get_topology("grid", config, divisions=20)
        >>> overlay_data = overlay.generate()
    """
    if name is None or name == "none":
        return None
    
    name_lower = name.lower()
    
    if name_lower == "grid":
        return GridOverlay(config, **kwargs)
    elif name_lower == "bounds":
        return BoundsOverlay(config, **kwargs)
    elif name_lower == "curvature":
        return CurvatureOverlay(config, **kwargs)
    else:
        raise ValueError(f"Unknown topology overlay: {name}")
