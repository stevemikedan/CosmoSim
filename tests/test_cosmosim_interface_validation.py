"""
Tests for cosmosim.py interface validation functionality.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path for cosmosim import
sys.path.insert(0, str(Path(__file__).parent.parent))

import cosmosim


def test_validate_interface_accepts_valid_module():
    """Verify validate_interface returns True for a properly-structured module."""
    # Create a fake module with all required functions
    fake_module = SimpleNamespace(
        __name__="test_module",
        build_config=lambda: None,
        build_initial_state=lambda cfg: None,
        run=lambda cfg, state: None,
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is True


def test_validate_interface_rejects_missing_build_config(capsys):
    """Verify validate_interface returns False when build_config is missing."""
    fake_module = SimpleNamespace(
        __name__="incomplete_module",
        build_initial_state=lambda cfg: None,
        run=lambda cfg, state: None,
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is False
    
    captured = capsys.readouterr()
    assert "build_config" in captured.err
    assert "incomplete_module" in captured.err


def test_validate_interface_rejects_missing_build_initial_state(capsys):
    """Verify validate_interface returns False when build_initial_state is missing."""
    fake_module = SimpleNamespace(
        __name__="incomplete_module",
        build_config=lambda: None,
        run=lambda cfg, state: None,
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is False
    
    captured = capsys.readouterr()
    assert "build_initial_state" in captured.err


def test_validate_interface_rejects_missing_run(capsys):
    """Verify validate_interface returns False when run is missing."""
    fake_module = SimpleNamespace(
        __name__="incomplete_module",
        build_config=lambda: None,
        build_initial_state=lambda cfg: None,
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is False
    
    captured = capsys.readouterr()
    assert "run" in captured.err


def test_validate_interface_rejects_non_callable_methods(capsys):
    """Verify validate_interface returns False when methods exist but aren't callable."""
    fake_module = SimpleNamespace(
        __name__="bad_module",
        build_config="not a function",
        build_initial_state=lambda cfg: None,
        run=lambda cfg, state: None,
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is False
    
    captured = capsys.readouterr()
    assert "build_config" in captured.err


def test_validate_interface_lists_all_missing_functions(capsys):
    """Verify that all missing functions are reported together."""
    fake_module = SimpleNamespace(
        __name__="empty_module",
    )
    
    result = cosmosim.validate_interface(fake_module)
    assert result is False
    
    captured = capsys.readouterr()
    assert "build_config" in captured.err
    assert "build_initial_state" in captured.err
    assert "run" in captured.err
