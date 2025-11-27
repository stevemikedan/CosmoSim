"""
Base class for topology overlays.

All overlay generators inherit from TopologyOverlay and implement
the generate() method to produce JSON-serializable geometry.
"""

from __future__ import annotations
from typing import Dict, Any
from state import UniverseConfig


class TopologyOverlay:
    """
    Abstract base class for topology overlay generators.
    
    Overlays generate visual geometry (grids, bounds, manifolds) that
    can be rendered in Three.js alongside particle frames.
    """
    
    def __init__(self, config: UniverseConfig):
        """
        Initialize overlay with universe configuration.
        
        Args:
            config: UniverseConfig containing simulation parameters
        """
        self.config = config
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate overlay geometry as JSON-serializable dict.
        
        Returns:
            Dictionary containing overlay type, geometry data, and
            rendering parameters (color, opacity, etc.)
            
        Example:
            {
                "type": "grid",
                "lines": [ [[x1,y1,z1], [x2,y2,z2]], ... ],
                "color": "#FFFFFF",
                "opacity": 0.2
            }
        """
        raise NotImplementedError("Subclasses must implement generate()")
