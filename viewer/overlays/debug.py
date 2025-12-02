"""Debug HUD overlay."""
from .base import Overlay
import jax.numpy as jnp

class DebugOverlay(Overlay):
    """Displays simulation statistics and debug info."""
    
    def apply(self, state, viewer, ax):
        if not viewer.show_debug:
            return
            
        # Calculate energy
        # Note: This duplicates some logic from energy_plot.py but is needed for real-time display
        # For performance, we might want to skip this every frame if it's slow
        
        # Simple stats
        t = float(state.time)
        dt = viewer.config.dt # Use config dt, or maybe we should track actual dt if adaptive
        active_count = int(jnp.sum(state.entity_active))
        
        info_text = (
            f"Time: {t:.3f}\n"
            f"dt: {dt:.4f}\n"
            f"Active: {active_count}\n"
            f"Color Mode: {viewer.color_mode}\n"
            f"Radius Mode: {viewer.render_radius_mode}\n"
            f"Topology: {viewer.config.topology_type}"
        )
        
        # Render text in top-left corner
        # transform=ax.transAxes ensures it stays fixed relative to the window
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
