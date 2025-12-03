"""
Tests for Viewer Parameter Panel (Phase E1.5).
"""

import pytest
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch
from viewer.params_panel import ParameterPanel
from viewer.viewer import Viewer
from state import UniverseConfig, UniverseState
import jax.numpy as jnp

@pytest.fixture
def mock_fig():
    """Create a mock matplotlib figure."""
    fig = MagicMock()
    # Mock add_axes to return a mock axes
    mock_ax = MagicMock()
    # Mock text object returned by ax.text
    mock_text = MagicMock()
    mock_ax.text.return_value = mock_text
    fig.add_axes.return_value = mock_ax
    return fig

@pytest.fixture
def sample_state():
    """Create a sample viewer state dictionary."""
    return {
        "scenario_name": "test_scenario",
        "universe_config": {
            "topology_type": 1,
            "substrate_type": "vector",
            "expansion_type": "none",
            "dt": 0.1,
            "max_entities": 100,
            "radius": 10.0
        },
        "pss_params": {"N": 100, "radius": 10.0},
        "flags": {
            "diagnostics": True,
            "neighbor_engine": False,
            "spatial_partition": True
        },
        "current_frame": 42,
        "current_time": 4.2
    }

def test_params_panel_initialization(mock_fig):
    """Test that panel initializes correctly."""
    panel = ParameterPanel(mock_fig)
    
    # Check axes creation
    mock_fig.add_axes.assert_called_once()
    assert panel.ax is not None
    
    # Check text object creation
    panel.ax.text.assert_called_once()
    assert panel.text_obj is not None
    
    # Check initial visibility
    assert panel.visible is False
    panel.ax.set_visible.assert_called_with(False)

def test_params_panel_update_accepts_all_fields(mock_fig, sample_state):
    """Test that update method processes all fields correctly."""
    panel = ParameterPanel(mock_fig)
    panel.update(sample_state)
    
    # Check that text string was updated
    assert panel.text_str != ""
    assert "Scenario: test_scenario" in panel.text_str
    assert "Topology: TORUS" in panel.text_str  # 1 -> TORUS
    assert "N = 100" in panel.text_str
    assert "diagnostics: True" in panel.text_str
    assert "Frame: 42" in panel.text_str
    assert "Time: 4.200" in panel.text_str

def test_params_panel_render_nonblocking(mock_fig, sample_state):
    """Test that render uses non-blocking draw_idle."""
    panel = ParameterPanel(mock_fig)
    panel.update(sample_state)
    
    # Make visible first
    panel.set_visible(True)
    
    # Render
    panel.render()
    
    # Check text update
    panel.text_obj.set_text.assert_called_with(panel.text_str)
    
    # Check non-blocking draw
    mock_fig.canvas.draw_idle.assert_called()
    # Ensure plt.show() was NOT called (we can't easily mock plt.show globally here 
    # without patching, but we verify we called the right canvas method)

def test_params_panel_toggle(mock_fig):
    """Test visibility toggling."""
    panel = ParameterPanel(mock_fig)
    
    # Toggle On
    panel.set_visible(True)
    assert panel.visible is True
    panel.ax.set_visible.assert_called_with(True)
    mock_fig.canvas.draw_idle.assert_called()
    
    # Reset mocks
    panel.ax.set_visible.reset_mock()
    mock_fig.canvas.draw_idle.reset_mock()
    
    # Toggle Off
    panel.set_visible(False)
    assert panel.visible is False
    panel.ax.set_visible.assert_called_with(False)
    mock_fig.canvas.draw_idle.assert_called()

def test_params_panel_works_with_empty_params(mock_fig):
    """Test robustness against missing data."""
    panel = ParameterPanel(mock_fig)
    empty_state = {}
    
    # Should not raise exception
    panel.update(empty_state)
    
    assert "Scenario: Unknown" in panel.text_str
    assert "Frame: 0" in panel.text_str

@patch('matplotlib.pyplot.subplots')
def test_params_panel_integration_with_viewer(mock_subplots):
    """Test integration within Viewer class."""
    # Mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    # Create dummy config/state
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=100,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    state = UniverseState(
        time=jnp.array(0.0),
        expansion_factor=jnp.array(1.0),
        curvature_k=jnp.array(0.0),
        entity_pos=jnp.zeros((1, 2)),
        entity_vel=jnp.zeros((1, 2)),
        entity_mass=jnp.zeros((1,)),
        entity_radius=jnp.zeros((1,)),
        entity_active=jnp.array([True], dtype=bool),
        entity_type=jnp.zeros((1,), dtype=int),
        node_active=jnp.zeros((1,), dtype=bool),
        node_pos=jnp.zeros((1, 2)),
        edge_active=jnp.zeros((1, 1), dtype=bool),
        edge_indices=jnp.zeros((1, 1, 2), dtype=int)
    )
    
    # Initialize Viewer
    viewer = Viewer(config, state, scenario_name="test", pss_params={"a": 1})
    
    # Check panel exists
    assert hasattr(viewer, 'params_panel')
    assert isinstance(viewer.params_panel, ParameterPanel)
    
    # Check initial state
    assert viewer.show_params_panel is False
    
    # Simulate key press (Shift+P)
    mock_event = MagicMock()
    mock_event.key = 'P'
    viewer.on_key_press(mock_event)
    
    assert viewer.show_params_panel is True
    assert viewer.params_panel.visible is True
    
    # Simulate step_once (should update panel)
    with patch('viewer.viewer.step_simulation') as mock_step:
        mock_step.return_value = state # Return same state
        viewer.step_once()
        
        # Check update called with correct data
        assert "Scenario: test" in viewer.params_panel.text_str
        assert "a = 1" in viewer.params_panel.text_str

def test_params_panel_throttle_behavior(mock_fig):
    """Test that panel rendering is throttled in Viewer."""
    # We need to mock Viewer's render method logic essentially
    # But simpler to test the logic directly if we can access Viewer
    pass # Logic is simple: if frame % 10 == 0. 
    # We verified integration above. 
    # Let's verify the Viewer.render calls panel.render only periodically.

@patch('matplotlib.pyplot.subplots')
def test_viewer_throttling(mock_subplots):
    """Verify throttling logic in Viewer.render."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=100,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    state = UniverseState(
        time=jnp.array(0.0),
        expansion_factor=jnp.array(1.0),
        curvature_k=jnp.array(0.0),
        entity_pos=jnp.zeros((1, 2)),
        entity_vel=jnp.zeros((1, 2)),
        entity_mass=jnp.zeros((1,)),
        entity_radius=jnp.zeros((1,)),
        entity_active=jnp.array([True], dtype=bool),
        entity_type=jnp.zeros((1,), dtype=int),
        node_active=jnp.zeros((1,), dtype=bool),
        node_pos=jnp.zeros((1, 2)),
        edge_active=jnp.zeros((1, 1), dtype=bool),
        edge_indices=jnp.zeros((1, 1, 2), dtype=int)
    )
    
    viewer = Viewer(config, state)
    viewer.show_params_panel = True
    viewer.params_panel = MagicMock()
    
    # Frame 1 (counter=1) -> No render
    viewer.render()
    viewer.params_panel.render.assert_not_called()
    
    # Advance counter to 9 (next render makes it 10)
    viewer.frame_counter = 9
    viewer.render()
    viewer.params_panel.render.assert_called_once()
