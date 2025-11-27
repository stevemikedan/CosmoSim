"""
Grid overlay generator.

Creates a visual grid for spatial orientation in the simulation.
Supports 2D and 3D grids scaled to universe radius.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from .base import TopologyOverlay


class GridOverlay(TopologyOverlay):
    """
    Generates a grid overlay for spatial reference.
    
    The grid is centered at the origin and scaled to the universe
    radius. Grid spacing is automatically determined based on the
    simulation bounds.
    """
    
    def __init__(self, config, divisions: int = 10, color: str = "#888888", opacity: float = 0.3):
        """
        Initialize grid overlay.
        
        Args:
            config: UniverseConfig
            divisions: Number of grid divisions (default: 10)
            color: Hex color string (default: "#888888")
            opacity: Grid opacity 0.0-1.0 (default: 0.3)
        """
        super().__init__(config)
        self.divisions = divisions
        self.color = color
        self.opacity = opacity
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate grid line geometry.
        
        Returns:
            Dictionary with grid lines, color, and opacity
        """
        # Use bounds if available, otherwise fallback to radius
        if hasattr(self.config, 'bounds') and self.config.bounds is not None and self.config.bounds > 0:
            r = float(self.config.bounds)
        else:
            r = float(self.config.radius)
        
        step = (2 * r) / self.divisions
        
        lines = []
        
        # Generate grid based on dimensionality
        if self.config.dim == 2:
            # 2D grid in XY plane
            for i in range(self.divisions + 1):
                x = -r + i * step
                z = -r + i * step
                
                # Vertical lines (parallel to Y axis) - but in XY, so parallel to X
                lines.append([[-r, z, 0.0], [r, z, 0.0]])
                
                # Horizontal lines (parallel to X axis) - but in XY, so parallel to Y
                lines.append([[x, -r, 0.0], [x, r, 0.0]])
        
        else:
            # 3D grid in XZ plane (y=0)
            for i in range(self.divisions + 1):
                x = -r + i * step
                z = -r + i * step
                
                # Lines parallel to X axis
                lines.append([[-r, 0.0, z], [r, 0.0, z]])
                
                # Lines parallel to Z axis
                lines.append([[x, 0.0, -r], [x, 0.0, r]])
        
        return {
            "type": "grid",
            "lines": lines,
            "color": self.color,
            "opacity": self.opacity,
        }
