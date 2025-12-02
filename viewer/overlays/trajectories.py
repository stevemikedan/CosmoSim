"""Trajectory trail overlay."""
from .base import Overlay
from collections import deque
import jax.numpy as jnp

class TrajectoryOverlay(Overlay):
    """Displays trails of recent entity positions."""
    
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.history = {} # Dict[entity_idx, deque]
        
    def apply(self, state, viewer, ax):
        if not viewer.show_trajectories:
            self.history.clear()
            return
            
        active_indices = jnp.where(state.entity_active)[0]
        
        # Update history
        # Note: This runs every frame, so trails might be short if dt is small
        # Could sample less frequently
        for idx in active_indices:
            idx = int(idx)
            if idx not in self.history:
                self.history[idx] = deque(maxlen=self.max_length)
            
            pos = state.entity_pos[idx]
            self.history[idx].append((float(pos[0]), float(pos[1])))
            
        # Render trails
        for idx in active_indices:
            idx = int(idx)
            if idx in self.history and len(self.history[idx]) > 1:
                points = list(self.history[idx])
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                ax.plot(xs, ys, '-', color='white', alpha=0.3, linewidth=1)
