"""Interactive Viewer for CosmoSim."""
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path for imports if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import UniverseConfig, UniverseState
from kernel import step_simulation
from viewer.overlays.debug import DebugOverlay
from viewer.overlays.inspector import InspectorOverlay
from viewer.overlays.vectors import VectorOverlay
from viewer.overlays.trajectories import TrajectoryOverlay

class Viewer:
    """Interactive viewer for CosmoSim simulations."""
    
    def __init__(self, config: UniverseConfig, state: UniverseState):
        self.initial_config = config
        self.config = config
        self.initial_state = state
        self.state = state
        
        # UI State
        self.paused = False
        self.show_debug = True
        self.show_inspector = False
        self.show_velocity_vectors = False
        self.show_acceleration_vectors = False
        self.show_trajectories = False
        
        self.color_mode = "type"  # "type", "constant", "velocity"
        self.render_radius_mode = "constant"  # "constant", "scaled"
        self.scale_factor = 1.0
        self.speed_multiplier = 1.0
        
        self.selected_entity_idx = None
        
        # Overlays
        self.overlays = [
            TrajectoryOverlay(), # Draw trails first (behind)
            VectorOverlay("velocity"),
            InspectorOverlay(),
            DebugOverlay() # Draw text last (on top)
        ]
        
        # Metrics Log
        self.metrics_log = []
        
        # Setup Plot
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        # Event Connections
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def setup_plot(self):
        """Initialize plot settings."""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('black') # Dark background for better visibility
        self.fig.patch.set_facecolor('#111111')
        
        # Set limits based on config
        r = self.config.radius
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        
    def on_key_press(self, event):
        """Handle keyboard events."""
        key = event.key
        
        # Playback
        if key == ' ':
            self.paused = not self.paused
        elif key == 'right':
            self.step_once()
        elif key == 'up':
            self.speed_multiplier *= 1.5
        elif key == 'down':
            self.speed_multiplier /= 1.5
            
        # Modes
        elif key == 'c':
            modes = ["type", "constant", "velocity"]
            idx = (modes.index(self.color_mode) + 1) % len(modes)
            self.color_mode = modes[idx]
        elif key == 'r':
            modes = ["constant", "scaled"]
            idx = (modes.index(self.render_radius_mode) + 1) % len(modes)
            self.render_radius_mode = modes[idx]
        elif key == 'd':
            self.show_debug = not self.show_debug
        elif key == 'v':
            self.show_velocity_vectors = not self.show_velocity_vectors
        elif key == 't':
            self.show_trajectories = not self.show_trajectories
        elif key == 'i':
            self.show_inspector = not self.show_inspector
            
        # Reset
        elif key == 'R':
            self.reset()
            
    def on_click(self, event):
        """Handle mouse clicks for selection."""
        if event.inaxes != self.ax:
            return
            
        # Find nearest entity
        click_pos = jnp.array([event.xdata, event.ydata])
        active_mask = self.state.entity_active
        positions = self.state.entity_pos
        
        # Compute distances (only for active entities)
        # We use a large number for inactive ones to ignore them
        diff = positions - click_pos
        dist_sq = jnp.sum(diff**2, axis=-1)
        dist_sq = jnp.where(active_mask, dist_sq, jnp.inf)
        
        nearest_idx = int(jnp.argmin(dist_sq))
        min_dist = float(jnp.sqrt(dist_sq[nearest_idx]))
        
        # Selection threshold (visual radius or fixed)
        threshold = max(self.config.radius * 0.05, 1.0) 
        
        if min_dist < threshold:
            self.selected_entity_idx = nearest_idx
            self.show_inspector = True # Auto-show inspector
        else:
            self.selected_entity_idx = None
            
    def reset(self):
        """Reset simulation to initial state."""
        self.state = self.initial_state
        self.metrics_log.clear()
        print("Simulation reset.")
        
    def step_once(self):
        """Advance simulation by one step."""
        # Apply speed multiplier to dt
        current_dt = self.config.dt * self.speed_multiplier
        step_config = self.config.replace(dt=current_dt)
        
        self.state = step_simulation(self.state, step_config)
        
        # Logging
        self.metrics_log.append({
            "t": float(self.state.time),
            "dt": float(current_dt),
            "active_count": int(jnp.sum(self.state.entity_active))
        })

    def get_colors(self):
        """Generate colors based on current mode."""
        active_mask = self.state.entity_active
        if self.color_mode == "constant":
            return 'cyan'
        elif self.color_mode == "type":
            # Map types to colors
            types = self.state.entity_type[active_mask]
            colors = []
            for t in types:
                if t == 0: colors.append('#3498db') # Blue
                elif t == 1: colors.append('#e74c3c') # Red
                elif t == 2: colors.append('#2ecc71') # Green
                else: colors.append('#f1c40f') # Yellow
            return colors
        elif self.color_mode == "velocity":
            # Map velocity magnitude to colormap
            vel = self.state.entity_vel[active_mask]
            speed = jnp.linalg.norm(vel, axis=-1)
            norm_speed = np.clip(speed / 5.0, 0, 1)
            return plt.cm.plasma(norm_speed)
        return 'cyan'
        
    def get_sizes(self):
        """Generate point sizes based on radius mode."""
        active_mask = self.state.entity_active
        if self.render_radius_mode == "constant":
            return 30 * self.scale_factor
        elif self.render_radius_mode == "scaled":
            radii = self.state.entity_radius[active_mask]
            # Scale for visibility
            return (radii * 200 * self.scale_factor) ** 2
        return 30
        
    def update(self):
        """Update simulation state."""
        if not self.paused:
            self.step_once()
            
    def render(self):
        """Render current frame."""
        self.ax.clear()
        self.setup_plot() # Reset limits/grid
        
        # Draw Entities
        active_mask = self.state.entity_active
        positions = self.state.entity_pos[active_mask]
        
        if len(positions) > 0:
            colors = self.get_colors()
            sizes = self.get_sizes()
            self.ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=sizes, edgecolors='white', linewidth=0.5)
            
        # Apply Overlays
        for overlay in self.overlays:
            overlay.apply(self.state, self, self.ax)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def run(self):
        """Main loop."""
        print("Starting Viewer...")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                start_time = time.time()
                
                self.update()
                self.render()
                
                # Cap framerate roughly
                elapsed = time.time() - start_time
                delay = max(0.001, 0.016 - elapsed) # ~60 FPS cap
                # plt.pause handles the GUI event loop
                plt.pause(delay)
                
                if not plt.fignum_exists(self.fig.number):
                    print("Window closed.")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            plt.close()

# Simple test runner
if __name__ == "__main__":
    from state import initialize_state
    from entities import spawn_entity
    
    cfg = UniverseConfig(
        physics_mode=0, radius=10.0, max_entities=100, max_nodes=10,
        dt=0.05, c=1.0, G=1.0, topology_type=0
    )
    st = initialize_state(cfg)
    # Spawn some particles
    st = spawn_entity(st, jnp.array([-2.0, 0.0]), jnp.array([0.0, 1.0]), 1.0, 0)
    st = spawn_entity(st, jnp.array([ 2.0, 0.0]), jnp.array([0.0, -1.0]), 1.0, 1)
    
    viewer = Viewer(cfg, st)
    viewer.run()
