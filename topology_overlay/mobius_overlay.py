"""
Möbius topology overlay generator.

Visualizes the Möbius strip topology as a 2D rectangle with
boundary indicators showing the twist (seam inversion).
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from .base import TopologyOverlay


class MobiusOverlay(TopologyOverlay):
    """
    Generates a Möbius strip topology visualization.
    
    Visualizes the (u, v) domain as a rectangle.
    - u-axis: Along the strip length [0, L]
    - v-axis: Across the strip width [-W, W]
    
    Draws arrows on the left (u=0) and right (u=L) boundaries
    indicating the orientation flip (up vs down).
    """
    
    def __init__(self, config, color: str = "#00FFFF", opacity: float = 0.5):
        """
        Initialize Möbius overlay.
        
        Args:
            config: UniverseConfig
            color: Hex color string (default: "#00FFFF" - cyan)
            opacity: Overlay opacity 0.0-1.0 (default: 0.5)
        """
        super().__init__(config)
        self.color = color
        self.opacity = opacity
        
    def generate(self) -> Dict[str, Any]:
        """
        Generate geometry for web viewer (Three.js).
        
        Returns:
            Dictionary with lines for rectangle and arrows.
        """
        L = float(self.config.radius)
        W = getattr(self.config, 'width', 1.0)
        
        lines = []
        
        # 1. Main Rectangle Boundary
        # (0, -W) -> (L, -W) -> (L, W) -> (0, W) -> (0, -W)
        corners = [
            [0.0, -W, 0.0],
            [L,   -W, 0.0],
            [L,    W, 0.0],
            [0.0,  W, 0.0]
        ]
        
        for i in range(4):
            lines.append([corners[i], corners[(i + 1) % 4]])
            
        # 2. Dashed Midline (v=0)
        # Represented as segments
        segments = 10
        step = L / segments
        for i in range(0, segments, 2):
            x1 = i * step
            x2 = (i + 1) * step
            lines.append([[x1, 0.0, 0.0], [x2, 0.0, 0.0]])
            
        # 3. Arrows
        # Left boundary (u=0): Upward arrows
        # Arrow shaft
        lines.append([[0.0, -W * 0.8, 0.0], [0.0, W * 0.8, 0.0]])
        # Arrow head (at top)
        head_size = W * 0.2
        lines.append([[0.0, W * 0.8, 0.0], [-head_size * 0.5, W * 0.8 - head_size, 0.0]])
        lines.append([[0.0, W * 0.8, 0.0], [ head_size * 0.5, W * 0.8 - head_size, 0.0]])
        
        # Right boundary (u=L): Downward arrows (Inverted)
        # Arrow shaft
        lines.append([[L, -W * 0.8, 0.0], [L, W * 0.8, 0.0]])
        # Arrow head (at bottom)
        lines.append([[L, -W * 0.8, 0.0], [L - head_size * 0.5, -W * 0.8 + head_size, 0.0]])
        lines.append([[L, -W * 0.8, 0.0], [L + head_size * 0.5, -W * 0.8 + head_size, 0.0]])
        
        return {
            "type": "mobius",
            "lines": lines,
            "color": self.color,
            "opacity": self.opacity,
        }

    def draw(self, ax, config):
        """
        Draw overlay on matplotlib axes.
        
        Args:
            ax: Matplotlib axes
            config: UniverseConfig
        """
        L = float(config.radius)
        W = getattr(config, 'width', 1.0)
        
        # 1. Main Rectangle
        rect_x = [0, L, L, 0, 0]
        rect_y = [-W, -W, W, W, -W]
        ax.plot(rect_x, rect_y, color=self.color, alpha=self.opacity, linestyle='-')
        
        # 2. Dashed Midline
        ax.plot([0, L], [0, 0], color=self.color, alpha=self.opacity * 0.5, linestyle='--')
        
        # 3. Arrows
        # Left boundary (u=0): Upward
        ax.arrow(0, -W * 0.5, 0, W, head_width=W*0.1, head_length=W*0.1, 
                 fc=self.color, ec=self.color, alpha=self.opacity, length_includes_head=True)
        
        # Right boundary (u=L): Downward
        ax.arrow(L, W * 0.5, 0, -W, head_width=W*0.1, head_length=W*0.1, 
                 fc=self.color, ec=self.color, alpha=self.opacity, length_includes_head=True)
        
        # 4. Labels (Optional but helpful)
        ax.text(0, -W * 1.1, "u=0", color=self.color, ha='center', alpha=self.opacity)
        ax.text(L, -W * 1.1, "u=L", color=self.color, ha='center', alpha=self.opacity)
