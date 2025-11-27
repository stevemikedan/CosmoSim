"""
Bounds overlay generator.

Creates a visual boundary box/circle showing simulation limits.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any
from .base import TopologyOverlay


class BoundsOverlay(TopologyOverlay):
    """
    Generates a boundary visualization.
    
    Shows the limits of the simulation space as a box (3D) or
    square (2D) outline.
    """
    
    def __init__(self, config, color: str = "#FF4444", opacity: float = 0.5):
        """
        Initialize bounds overlay.
        
        Args:
            config: UniverseConfig
            color: Hex color string (default: "#FF4444" - red)
            opacity: Bounds opacity 0.0-1.0 (default: 0.5)
        """
        super().__init__(config)
        self.color = color
        self.opacity = opacity
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate boundary box/square geometry.
        
        Returns:
            Dictionary with boundary lines, color, and opacity
        """
        # Use bounds if available, otherwise fallback to radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            r = float(self.config.bounds)
        else:
            r = float(self.config.radius)
        
        lines = []
        
        if self.config.dim == 2:
            # 2D square boundary in XY plane
            corners = [
                [-r, -r, 0.0],  # bottom-left
                [ r, -r, 0.0],  # bottom-right
                [ r,  r, 0.0],  # top-right
                [-r,  r, 0.0],  # top-left
            ]
            
            # Connect corners in a square
            for i in range(4):
                lines.append([corners[i], corners[(i + 1) % 4]])
        
        else:
            # 3D box boundary
            # Generate 8 corners of the cube
            corners = []
            for x in [-r, r]:
                for y in [-r, r]:
                    for z in [-r, r]:
                        corners.append([x, y, z])
            
            # Generate 12 edges of the cube
            # Bottom face (y = -r)
            lines.append([corners[0], corners[1]])  # -x-y-z to +x-y-z
            lines.append([corners[1], corners[3]])  # +x-y-z to +x-y+z
            lines.append([corners[3], corners[2]])  # +x-y+z to -x-y+z
            lines.append([corners[2], corners[0]])  # -x-y+z to -x-y-z
            
            # Top face (y = +r)
            lines.append([corners[4], corners[5]])  # -x+y-z to +x+y-z
            lines.append([corners[5], corners[7]])  # +x+y-z to +x+y+z
            lines.append([corners[7], corners[6]])  # +x+y+z to -x+y+z
            lines.append([corners[6], corners[4]])  # -x+y+z to -x+y-z
            
            # Vertical edges
            lines.append([corners[0], corners[4]])  # -x-y-z to -x+y-z
            lines.append([corners[1], corners[5]])  # +x-y-z to +x+y-z
            lines.append([corners[2], corners[6]])  # -x-y+z to -x+y+z
            lines.append([corners[3], corners[7]])  # +x-y+z to +x+y+z
        
        return {
            "type": "bounds",
            "lines": lines,
            "color": self.color,
            "opacity": self.opacity,
        }
