"""Vector field overlays."""
from .base import Overlay
import jax.numpy as jnp

class VectorOverlay(Overlay):
    """Displays velocity or acceleration vectors."""
    
    def __init__(self, vector_type="velocity"):
        self.vector_type = vector_type # "velocity" or "acceleration"
        
    def apply(self, state, viewer, ax):
        show = False
        if self.vector_type == "velocity" and viewer.show_velocity_vectors:
            show = True
            vectors = state.entity_vel
            color = 'cyan'
            scale = 1.0
        elif self.vector_type == "acceleration" and viewer.show_acceleration_vectors:
            show = True
            # Acceleration is not directly in state, would need to compute it or rely on viewer to provide it
            # For now, we skip acceleration as it requires a physics step call or storage
            # Placeholder:
            return 
            
        if not show:
            return
            
        active_mask = state.entity_active
        pos = state.entity_pos[active_mask]
        vec = vectors[active_mask]
        
        if len(pos) > 0:
            ax.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], 
                     color=color, alpha=0.6, scale=None, scale_units='xy')
