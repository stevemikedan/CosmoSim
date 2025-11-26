"""Minimal test to debug conftest fixture error"""
import pytest

def test_basic():
    """Most basic test possible"""
    assert 1 + 1 == 2
    print("Basic test passed!")

def test_import_module():
    """Test that we can import a simulation module"""
    import run_sim
    cfg = run_sim.build_config()
    assert cfg is not None
    print("Module import test passed!")
