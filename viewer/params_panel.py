"""
Parameter Panel for CosmoSim Viewer (Phase E1.5).

This module provides a lightweight, read-only side panel that displays
scenario metadata, configuration, and runtime flags.
"""

import matplotlib.pyplot as plt
from dataclasses import asdict

class ParameterPanel:
    def __init__(self, fig):
        """
        Initialize the parameter panel.
        
        Args:
            fig: The matplotlib figure to attach the panel to.
        """
        self.fig = fig
        self.text_str = ""
        self.visible = False
        
        # Create a dedicated axes for the panel
        # Position: Right side, taking up 25% width
        # [left, bottom, width, height] in figure coordinates
        self.ax = self.fig.add_axes([0.75, 0.1, 0.24, 0.8])
        self.ax.set_axis_off()  # Hide axis lines/ticks
        self.ax.set_visible(False) # Hidden by default
        
        # Initialize text object
        self.text_obj = self.ax.text(
            0.05, 0.95, 
            "", 
            transform=self.ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            color='white',
            fontsize=8,
            family='monospace',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
        )

    def update(self, viewer_state: dict):
        """
        Update internal text buffer from viewer state.
        
        Args:
            viewer_state: Dictionary containing:
                - scenario_name: str
                - universe_config: dict
                - pss_params: dict
                - flags: dict
                - current_frame: int
                - current_time: float
        """
        scenario = viewer_state.get("scenario_name", "Unknown")
        config = viewer_state.get("universe_config", {})
        pss = viewer_state.get("pss_params", {})
        flags = viewer_state.get("flags", {})
        frame = viewer_state.get("current_frame", 0)
        time_val = viewer_state.get("current_time", 0.0)

        # Format UniverseConfig
        # We handle both dict and dataclass (just in case, though viewer passes dict)
        if not isinstance(config, dict):
            try:
                config = asdict(config)
            except:
                config = {}

        # Helper to get config val
        def get_cfg(key):
            return config.get(key, "N/A")
            
        # Topology mapping
        topo_map = {0: "FLAT", 1: "TORUS", 2: "SPHERE", 3: "BUBBLE"}
        topo_val = get_cfg("topology_type")
        topo_str = topo_map.get(topo_val, str(topo_val))

        lines = [
            "-------------------------------",
            f"Scenario: {scenario}",
            f"Topology: {topo_str}",
            f"Substrate: {get_cfg('substrate_type')}",
            f"Expansion: {get_cfg('expansion_type')}",
            f"dt: {get_cfg('dt')}",
            f"Entities: {get_cfg('max_entities')}",
            f"Radius: {get_cfg('radius')}",
            "",
            "PSS Parameters:"
        ]

        if pss:
            for k, v in pss.items():
                lines.append(f"  {k} = {v}")
        else:
            lines.append("  (None)")

        lines.append("")
        lines.append("Flags:")
        for k, v in flags.items():
            lines.append(f"  {k}: {v}")

        lines.append("")
        lines.append(f"Frame: {frame}")
        lines.append(f"Time: {time_val:.3f}")
        lines.append("-------------------------------")

        self.text_str = "\n".join(lines)

    def render(self):
        """
        Draw text into the panel.
        Non-blocking.
        """
        if not self.visible:
            if self.ax.get_visible():
                self.ax.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        if not self.ax.get_visible():
            self.ax.set_visible(True)

        self.text_obj.set_text(self.text_str)
        
        # Use draw_idle for non-blocking update
        self.fig.canvas.draw_idle()

    def set_visible(self, visible: bool):
        """Set visibility state."""
        self.visible = visible
        if not visible:
            self.ax.set_visible(False)
            self.fig.canvas.draw_idle()
        else:
            self.ax.set_visible(True)
            self.render()
