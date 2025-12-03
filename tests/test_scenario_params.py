"""
Tests for PSS (Parameterized Scenario System) functionality.

Tests parameter parsing, schema loading, type conversion, and integration.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cosmosim import parse_param_string, load_scenario_schema, merge_params


# =====================================================================
# PSS0.x Tests: Parameter Parsing
# =====================================================================

def test_parse_params_basic():
    """Test basic parameter string parsing."""
    result = parse_param_string("N=100,radius=5.5,active=true")
    
    assert result == {"N": "100", "radius": "5.5", "active": "true"}


def test_parse_params_empty():
    """Test parsing empty or None parameter string."""
    assert parse_param_string(None) == {}
    assert parse_param_string("") == {}


def test_parse_params_whitespace():
    """Test parsing with whitespace around keys and values."""
    result = parse_param_string(" N = 100 , radius = 5.5 ")
    
    assert result == {"N": "100", "radius": "5.5"}


def test_parse_params_malformed():
    """Test parsing malformed input (missing equals)."""
    result = parse_param_string("N=100,invalid,radius=5.5")
    
    # Should skip 'invalid' and parse the rest
    assert result == {"N": "100", "radius": "5.5"}


def test_parse_params_single_value():
    """Test parsing single key=value pair."""
    result = parse_param_string("N=500")
    
    assert result == {"N": "500"}


# =====================================================================
# PSS1.0/1.1 Tests: Schema Loading
# =====================================================================

def test_schema_load_valid():
    """Test loading valid schema from module."""
    import scenarios.bulk_ring as bulk_ring
    
    schema = load_scenario_schema(bulk_ring)
    
    assert schema is not None
    assert "N" in schema
    assert "radius" in schema
    assert schema["N"]["type"] == "int"
    assert schema["N"]["default"] == 64


def test_schema_load_missing():
    """Test loading schema from module without SCENARIO_PARAMS."""
    import scenarios.manual_run as manual_run
    
    schema = load_scenario_schema(manual_run)
    
    assert schema is None


class MockModuleNoSchema:
    """Mock module without SCENARIO_PARAMS."""
    pass


def test_schema_load_no_attribute():
    """Test schema loading when module has no SCENARIO_PARAMS attribute."""
    schema = load_scenario_schema(MockModuleNoSchema())
    
    assert schema is None


# =====================================================================
# PSS1.1 Tests: Parameter Merging & Type Conversion
# =====================================================================

def test_merge_params_basic():
    """Test basic parameter merging with defaults."""
    schema = {
        "N": {"type": "int", "default": 100},
        "radius": {"type": "float", "default": 5.0},
    }
    cli_params = {}
    
    result = merge_params(schema, cli_params)
    
    assert result == {"N": 100, "radius": 5.0}


def test_merge_params_type_conversion_int():
    """Test type conversion for int parameters."""
    schema = {
        "N": {"type": "int", "default": 100},
    }
    cli_params = {"N": "500"}
    
    result = merge_params(schema, cli_params)
    
    assert result["N"] == 500
    assert isinstance(result["N"], int)


def test_merge_params_type_conversion_float():
    """Test type conversion for float parameters."""
    schema = {
        "radius": {"type": "float", "default": 5.0},
    }
    cli_params = {"radius": "10.5"}
    
    result = merge_params(schema, cli_params)
    
    assert result["radius"] == 10.5
    assert isinstance(result["radius"], float)


def test_merge_params_type_conversion_bool():
    """Test type conversion for bool parameters."""
    schema = {
        "active": {"type": "bool", "default": False},
    }
    
    # Test various true values
    for value in ["true", "True", "TRUE", "yes", "y", "1"]:
        result = merge_params(schema, {"active": value})
        assert result["active"] is True
    
    # Test false values
    for value in ["false", "False", "no", "0"]:
        result = merge_params(schema, {"active": value})
        assert result["active"] is False


def test_merge_params_type_conversion_str():
    """Test type conversion for str parameters."""
    schema = {
        "name": {"type": "str", "default": "default"},
    }
    cli_params = {"name": "custom"}
    
    result = merge_params(schema, cli_params)
    
    assert result["name"] == "custom"
    assert isinstance(result["name"], str)


def test_merge_params_invalid_type(capsys):
    """Test handling of invalid type conversion."""
    schema = {
        "N": {"type": "int", "default": 100},
    }
    cli_params = {"N": "not_a_number"}
    
    result = merge_params(schema, cli_params)
    
    # Should use default when conversion fails
    assert result["N"] == 100
    
    # Should print warning
    captured = capsys.readouterr()
    assert "Invalid type for parameter 'N'" in captured.out


def test_merge_params_unknown_param(capsys):
    """Test handling of unknown parameters."""
    schema = {
        "N": {"type": "int", "default": 100},
    }
    cli_params = {"N": "500", "unknown": "value"}
    
    result = merge_params(schema, cli_params)
    
    # Should ignore unknown parameter
    assert result == {"N": 500}
    
    # Should print warning
    captured = capsys.readouterr()
    assert "Unknown parameter 'unknown' ignored" in captured.out


def test_merge_params_bounds_clamping_min(capsys):
    """Test bounds checking - minimum clamping."""
    schema = {
        "N": {"type": "int", "default": 100, "min": 10},
    }
    cli_params = {"N": "5"}
    
    result = merge_params(schema, cli_params)
    
    # Should clamp to minimum
    assert result["N"] == 10
    
    # Should print warning
    captured = capsys.readouterr()
    assert "below min=10, clamping" in captured.out


def test_merge_params_bounds_clamping_max(capsys):
    """Test bounds checking - maximum clamping."""
    schema = {
        "radius": {"type": "float", "default": 5.0, "max": 50.0},
    }
    cli_params = {"radius": "100.0"}
    
    result = merge_params(schema, cli_params)
    
    # Should clamp to maximum
    assert result["radius"] == 50.0
    
    # Should print warning
    captured = capsys.readouterr()
    assert "above max=50.0, clamping" in captured.out


def test_merge_params_missing_schema(capsys):
    """Test merging when schema is None."""
    result = merge_params(None, {})
    
    assert result == {}
    
    # Should not print warning when no CLI params
    captured = capsys.readouterr()
    assert "Warning" not in captured.out


def test_merge_params_missing_schema_with_cli(capsys):
    """Test merging when schema is None but CLI params provided."""
    result = merge_params(None, {"N": "100"})
    
    assert result == {}
    
    # Should NOT print warning (warning moved to run_scenario)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_merge_params_multiple_overrides():
    """Test merging with multiple CLI overrides."""
    schema = {
        "N": {"type": "int", "default": 100},
        "radius": {"type": "float", "default": 5.0},
        "speed": {"type": "float", "default": 1.0},
        "mass": {"type": "float", "default": 1.0},
    }
    cli_params = {"N": "200", "radius": "10.0"}
    
    result = merge_params(schema, cli_params)
    
    # Overridden values
    assert result["N"] == 200
    assert result["radius"] == 10.0
    
    # Default values
    assert result["speed"] == 1.0
    assert result["mass"] == 1.0


# =====================================================================
# Integration Tests
# =====================================================================

def test_scenario_runs_without_schema():
    """Test that scenario without schema runs normally."""
    import scenarios.manual_run as manual_run
    from state import UniverseConfig
    
    # Should work without params parameter
    cfg = manual_run.build_config()
    state = manual_run.build_initial_state(cfg)
    
    assert state is not None


def test_scenario_runs_with_schema_no_params():
    """Test that scenario with schema runs with defaults."""
    import scenarios.bulk_ring as bulk_ring
    from state import UniverseConfig
    
    cfg = bulk_ring.build_config()
    state = bulk_ring.build_initial_state(cfg)
    
    assert state is not None
    # Should use default N=64 from schema
    assert state.entity_active.sum() == 64


def test_scenario_runs_with_params():
    """Test that scenario with params works correctly."""
    import scenarios.bulk_ring as bulk_ring
    from state import UniverseConfig
    
    cfg = bulk_ring.build_config()
    params = {"N": 32, "radius": 10.0, "speed": 1.5, "mass": 2.0}
    state = bulk_ring.build_initial_state(cfg, params)
    
    assert state is not None
    # Should use param N=32
    assert state.entity_active.sum() == 32


def test_end_to_end_param_pipeline():
    """Test complete pipeline from CLI string to scenario execution."""
    import scenarios.bulk_ring as bulk_ring
    
    # Simulate CLI input
    param_str = "N=50,radius=12.0"
    
    # Parse
    cli_params = parse_param_string(param_str)
    
    # Load schema
    schema = load_scenario_schema(bulk_ring)
    
    # Merge
    merged = merge_params(schema, cli_params)
    
    # Use in scenario
    cfg = bulk_ring.build_config()
    state = bulk_ring.build_initial_state(cfg, merged)
    
    assert state is not None
    assert state.entity_active.sum() == 50
