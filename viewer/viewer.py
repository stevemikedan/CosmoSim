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
from viewer.metrics import MetricsEngine
from viewer.params_panel import ParameterPanel
from dataclasses import asdict

class Viewer:
    """Interactive viewer for CosmoSim simulations."""
    
    def __init__(self, config: UniverseConfig, state: UniverseState, scenario_name: str = "Unknown", pss_params: dict = None):
        self.initial_config = config
        self.config = config
        self.initial_state = state
        self.state = state
        self.scenario_name = scenario_name
        self.pss_params_dict = pss_params if pss_params else {}
        self.config_as_dict = asdict(config)
        
        # UI State
        self.paused = False
        self.show_debug = True
        self.show_inspector = False
        self.show_velocity_vectors = False
        self.show_acceleration_vectors = False
        self.show_trajectories = False
        self.show_diagnostics_panel = False
        self.show_params_panel = False
        self.frame_counter = 0
        
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
        
        # Metrics Engine
        self.metrics = MetricsEngine()
        self.diagnostics_fig = None
        self.diagnostics_axes = None
        self.frame_idx = 0
        
        # Setup Plot
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
        # Initialize Parameter Panel
        self.params_panel = ParameterPanel(self.fig)
        
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
            
        # Diagnostics
        elif key == 'D':  # Shift+D for diagnostics panel
            self.show_diagnostics_panel = not self.show_diagnostics_panel
            if not self.show_diagnostics_panel and self.diagnostics_fig:
                plt.close(self.diagnostics_fig)
                self.diagnostics_fig = None
        elif key == 'e':
            self.metrics.toggle('energy')
        elif key == 'm':
            self.metrics.toggle('momentum')
        elif key == 'C':  # Shift+C for COM (c is used for color)
            self.metrics.toggle('com')
        elif key == 'h':
            self.metrics.toggle('velocity_hist')
        elif key == 'u':
            self.metrics.toggle('substrate_diag')
        elif key == 'P':  # Shift+P for params panel
            self.show_params_panel = not self.show_params_panel
            self.params_panel.set_visible(self.show_params_panel)
            
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
        
        # Update metrics engine
        self.frame_idx += 1
        state_dict = {
            'pos': np.array(self.state.entity_pos),
            'vel': np.array(self.state.entity_vel),
            'mass': np.array(self.state.entity_mass),
            'active': np.array(self.state.entity_active),
            'diagnostics': {}  # Could be populated from state.extra if available
        }
        self.metrics.update(self.frame_idx, state_dict)
        
        # Logging
        self.metrics_log.append({
            "t": float(self.state.time),
            "dt": float(current_dt),
            "active_count": int(jnp.sum(self.state.entity_active))
        })
        
        # Update Parameter Panel State
        self.params_panel.update({
            "scenario_name": self.scenario_name,
            "universe_config": self.config_as_dict,
            "pss_params": self.pss_params_dict,
            "flags": {
                "diagnostics": self.show_diagnostics_panel,
                "neighbor_engine": getattr(self.config, 'enable_neighbor_engine', False),
                "spatial_partition": getattr(self.config, 'enable_spatial_partition', False)
            },
            "current_frame": self.frame_idx,
            "current_time": float(self.state.time)
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
        
        # Render diagnostics panel (separate figure)
        if self.show_diagnostics_panel and self.frame_idx % 10 == 0:  # Update at ~4-5 FPS (every 10 frames)
            self.render_diagnostics_panel()
            
        # Render params panel (throttled)
        self.frame_counter += 1
        if self.show_params_panel and self.frame_counter % 10 == 0:
            self.params_panel.render()
            
    def render_diagnostics_panel(self):
        """Render metrics diagnostics in separate figure."""
        # Create figure if needed
        if self.diagnostics_fig is None:
            self.diagnostics_fig, self.diagnostics_axes = plt.subplots(2, 2, figsize=(12, 10))
            self.diagnostics_fig.suptitle('CosmoSim Diagnostics', fontsize=14, color='white')
            self.diagnostics_fig.patch.set_facecolor('#111111')
            plt.ion()
        
        # Clear all axes
        for ax_row in self.diagnostics_axes:
            for ax in ax_row:
                ax.clear()
                ax.set_facecolor('#222222')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
        
        # Plot Energy (top-left)
        if self.metrics.enabled.get('energy') and len(self.metrics.energy_history) > 0:
            history = np.array(self.metrics.energy_history)
            frames = history[:, 0]
            ke = history[:, 1]
            pe = history[:, 2]
            te = history[:, 3]
            
            ax = self.diagnostics_axes[0, 0]
            ax.plot(frames, ke, 'b-', label='KE', alpha=0.8)
            if not np.all(np.isnan(pe)):
                ax.plot(frames, pe, 'r-', label='PE', alpha=0.8)
            if not np.all(np.isnan(te)):
                ax.plot(frames, te, 'g-', label='Total', alpha=0.8)
            ax.set_xlabel('Frame', color='white')
            ax.set_ylabel('Energy', color='white')
            ax.set_title('Energy', color='white')
            ax.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, color='white')
        
        # Plot Momentum (top-right)
        if self.metrics.enabled.get('momentum') and len(self.metrics.momentum_history) > 0:
            history = np.array(self.metrics.momentum_history)
            frames = history[:, 0]
            px = history[:, 1]
            py = history[:, 2]
            pz = history[:, 3]
            
            ax = self.diagnostics_axes[0, 1]
            ax.plot(frames, px, 'r-', label='px', alpha=0.8)
            ax.plot(frames, py, 'g-', label='py', alpha=0.8)
            if not np.all(pz == 0):
                ax.plot(frames, pz, 'b-', label='pz', alpha=0.8)
            ax.set_xlabel('Frame', color='white')
            ax.set_ylabel('Momentum', color='white')
            ax.set_title('Momentum', color='white')
            ax.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, color='white')
        
        # Plot COM (bottom-left)
        if self.metrics.enabled.get('com') and len(self.metrics.com_history) > 0:
            history = np.array(self.metrics.com_history)
            frames = history[:, 0]
            cx = history[:, 1]
            cy = history[:, 2]
            cz = history[:, 3]
            
            ax = self.diagnostics_axes[1, 0]
            ax.plot(frames, cx, 'r-', label='cx', alpha=0.8)
            ax.plot(frames, cy, 'g-', label='cy', alpha=0.8)
            if not np.all(cz == 0):
                ax.plot(frames, cz, 'b-', label='cz', alpha=0.8)
            ax.set_xlabel('Frame', color='white')
            ax.set_ylabel('Position', color='white')
            ax.set_title('Center of Mass', color='white')
            ax.legend(facecolor='#333333', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, color='white')
        
        # Plot Velocity Histogram (bottom-right)
        if self.metrics.enabled.get('velocity_hist') and self.metrics.vel_hist_data:
            counts, bin_edges = self.metrics.vel_hist_data
            ax = self.diagnostics_axes[1, 1]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, counts, width=np.diff(bin_edges), color='cyan', alpha=0.7, edgecolor='white')
            ax.set_xlabel('|Velocity|', color='white')
            ax.set_ylabel('Count', color='white')
            ax.set_title('Velocity Distribution', color='white')
            ax.grid(True, alpha=0.2, color='white')
        
        # Update canvas
        self.diagnostics_fig.canvas.draw()
        self.diagnostics_fig.canvas.flush_events()
        
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
                
                # Handle window events
                try:
                    if not plt.fignum_exists(self.fig.number):
                        print("Window closed.")
                        break
                    
                    # plt.pause handles the GUI event loop
                    plt.pause(delay)
                except Exception:
                    # Catch TclError or other GUI issues on exit
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            try:
                plt.close('all')
            except:
                pass

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
