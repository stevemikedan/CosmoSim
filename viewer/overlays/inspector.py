"""Inspector overlay for selected entities."""
from .base import Overlay
import matplotlib.patches as patches
import jax.numpy as jnp

class InspectorOverlay(Overlay):
    """Highlights selected entity and displays details."""
    
    def apply(self, state, viewer, ax):
        if viewer.selected_entity_idx is None:
            return
            
        idx = viewer.selected_entity_idx
        
        # Check if index is valid and entity is active
        if idx >= len(state.entity_active) or not state.entity_active[idx]:
            viewer.selected_entity_idx = None
            return
            
        # Highlight Entity
        pos = state.entity_pos[idx]
        radius = state.entity_radius[idx]
        
        # Draw selection ring (slightly larger than entity)
        # Scale radius for visibility if needed, or use fixed size
        display_radius = radius * 1.5 if viewer.render_radius_mode == "scaled" else (viewer.get_sizes() / 100)**0.5
        
        # Use a fixed size marker for the ring if constant mode
        if viewer.render_radius_mode == "constant":
             ax.plot(pos[0], pos[1], 'o', ms=15, mfc='none', mec='yellow', mew=2)
        else:
             circle = patches.Circle((pos[0], pos[1]), display_radius, 
                                    fill=False, edgecolor='yellow', linewidth=2)
             ax.add_patch(circle)
        
        # Display Info Panel
        if viewer.show_inspector:
            vel = state.entity_vel[idx]
            speed = jnp.linalg.norm(vel)
            mass = state.entity_mass[idx]
            etype = state.entity_type[idx]
            
            info = (
                f"ID: {idx}\n"
                f"Type: {int(etype)}\n"
                f"Pos: ({pos[0]:.2f}, {pos[1]:.2f})\n"
                f"Vel: ({vel[0]:.2f}, {vel[1]:.2f})\n"
                f"Speed: {speed:.3f}\n"
                f"Mass: {mass:.2f}\n"
                f"Radius: {radius:.2f}"
            )
            
            # Bottom-left corner
            ax.text(0.02, 0.02, info, transform=ax.transAxes,
                    verticalalignment='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='yellow', ec='yellow'),
                    color='white')
