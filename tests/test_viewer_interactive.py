"""Test script for interactive viewer logic."""
import sys
import os
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from state import UniverseConfig, initialize_state
from entities import spawn_entity
from viewer.viewer import Viewer

def test_viewer_logic():
    print("Initializing Viewer Test...")
    
    # Setup Config
    cfg = UniverseConfig(
        physics_mode=0, radius=10.0, max_entities=100, max_nodes=10,
        dt=0.1, c=1.0, G=1.0, topology_type=0
    )
    st = initialize_state(cfg)
    
    # Spawn entities
    st = spawn_entity(st, jnp.array([-2.0, 0.0]), jnp.array([0.0, 1.0]), 1.0, 0)
    st = spawn_entity(st, jnp.array([ 2.0, 0.0]), jnp.array([0.0, -1.0]), 1.0, 1)
    
    # Initialize Viewer
    # Use non-interactive backend for testing to avoid window popup
    plt.switch_backend('Agg') 
    viewer = Viewer(cfg, st)
    
    print("Testing Update Loop...")
    # Test Update
    initial_time = float(viewer.state.time)
    viewer.update()
    new_time = float(viewer.state.time)
    assert new_time > initial_time, "Simulation did not advance"
    print(f"Time advanced: {initial_time} -> {new_time}")
    
    print("Testing Render...")
    # Test Render (should not crash)
    viewer.render()
    
    print("Testing Controls...")
    # Test Pause
    viewer.on_key_press(type('Event', (object,), {'key': ' '}))
    assert viewer.paused, "Pause toggle failed"
    
    viewer.update()
    paused_time = float(viewer.state.time)
    assert paused_time == new_time, "Simulation advanced while paused"
    
    # Test Step
    viewer.on_key_press(type('Event', (object,), {'key': 'right'}))
    stepped_time = float(viewer.state.time)
    assert stepped_time > paused_time, "Step failed"
    
    # Test Selection
    print("Testing Selection...")
    # Click near entity 0 at (-2, 0)
    # We need to mock the event with xdata, ydata and inaxes check
    class MockEvent:
        def __init__(self, x, y, ax):
            self.xdata = x
            self.ydata = y
            self.inaxes = ax
            
    event = MockEvent(-2.05, 0.05, viewer.ax)
    viewer.on_click(event)
    
    # Entity 0 is at roughly (-2, 0) after small steps
    # Should be selected
    assert viewer.selected_entity_idx is not None, "Selection failed"
    print(f"Selected Entity: {viewer.selected_entity_idx}")
    
    # Test Inspector Toggle
    assert viewer.show_inspector, "Inspector did not auto-show"
    
    print("Viewer Logic Verified Successfully!")

if __name__ == "__main__":
    test_viewer_logic()
