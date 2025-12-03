import pytest
from cosmosim import validate_schema, merge_params, safe_convert_type

# =============================================================================
# 1. Test Bounds Clamping
# =============================================================================
def test_clamp_min_max(capsys):
    schema = {
        "radius": {"type": "float", "default": 10.0, "min": 5.0, "max": 20.0},
        "count": {"type": "int", "default": 10, "min": 1, "max": 100}
    }
    
    # Test below min
    params_low = {"radius": "2.0", "count": "-5"}
    merged_low = merge_params(schema, params_low)
    captured_low = capsys.readouterr()
    
    assert merged_low["radius"] == 5.0
    assert merged_low["count"] == 1
    assert "[PSS WARNING] radius=2.0 below min=5.0, clamping" in captured_low.out
    assert "[PSS WARNING] count=-5 below min=1, clamping" in captured_low.out
    
    # Test above max
    params_high = {"radius": "50.0", "count": "200"}
    merged_high = merge_params(schema, params_high)
    captured_high = capsys.readouterr()
    
    assert merged_high["radius"] == 20.0
    assert merged_high["count"] == 100
    assert "[PSS WARNING] radius=50.0 above max=20.0, clamping" in captured_high.out
    assert "[PSS WARNING] count=200 above max=100, clamping" in captured_high.out

# =============================================================================
# 2. Test Invalid Type Fallback
# =============================================================================
def test_invalid_type_falls_back_to_default(capsys):
    schema = {
        "speed": {"type": "float", "default": 1.0}
    }
    
    params = {"speed": "fast"}
    merged = merge_params(schema, params)
    captured = capsys.readouterr()
    
    assert merged["speed"] == 1.0
    assert "[PSS WARNING] Invalid type for parameter 'speed'; using default" in captured.out

# =============================================================================
# 3. Test Unknown Parameter Warning
# =============================================================================
def test_unknown_param_produces_warning(capsys):
    schema = {
        "known": {"type": "int", "default": 1}
    }
    
    params = {"unknown": "123"}
    merged = merge_params(schema, params)
    captured = capsys.readouterr()
    
    assert "unknown" not in merged
    assert "[PSS WARNING] Unknown parameter 'unknown' ignored" in captured.out

# =============================================================================
# 4. Test Allowed Values Enforced
# =============================================================================
def test_allowed_values_enforced(capsys):
    schema = {
        "mode": {"type": "str", "default": "A", "allowed": ["A", "B", "C"]},
        "level": {"type": "int", "default": 1, "allowed": [1, 2, 3]}
    }
    
    # Invalid values
    params = {"mode": "D", "level": "5"}
    merged = merge_params(schema, params)
    captured = capsys.readouterr()
    
    assert merged["mode"] == "A"
    assert merged["level"] == 1
    assert "[PSS WARNING] Value 'D' not allowed for 'mode'; using default" in captured.out
    assert "[PSS WARNING] Value '5' not allowed for 'level'; using default" in captured.out
    
    # Valid values
    params_valid = {"mode": "B", "level": "3"}
    merged_valid = merge_params(schema, params_valid)
    assert merged_valid["mode"] == "B"
    assert merged_valid["level"] == 3

# =============================================================================
# 5. Test Required Parameter
# =============================================================================
def test_required_param_warns_and_defaults(capsys):
    schema = {
        "req_with_default": {"type": "int", "default": 10, "required": True},
        "req_no_default": {"type": "int", "required": True}
    }
    
    # Missing required params
    merged = merge_params(schema, {})
    captured = capsys.readouterr()
    
    assert merged["req_with_default"] == 10
    assert "req_no_default" not in merged
    assert "[PSS WARNING] Required param 'req_with_default' missing; using default" in captured.out
    assert "[PSS WARNING] Required param 'req_no_default' missing and no default provided" in captured.out

# =============================================================================
# 6. Test Schema Validation Warnings
# =============================================================================
def test_schema_validation_warnings():
    # Invalid schema: min > max
    schema_bad_range = {
        "bad_range": {"type": "int", "default": 5, "min": 10, "max": 0}
    }
    warnings = validate_schema(schema_bad_range)
    assert any("min > max" in w for w in warnings)
    
    # Invalid schema: default type mismatch
    schema_bad_type = {
        "bad_type": {"type": "int", "default": "string"}
    }
    warnings = validate_schema(schema_bad_type)
    assert any("default string is not int" in w for w in warnings)
    
    # Invalid schema: default out of bounds
    schema_bad_default = {
        "bad_default": {"type": "int", "default": 100, "max": 50}
    }
    warnings = validate_schema(schema_bad_default)
    assert any("default 100 > max 50" in w for w in warnings)

# =============================================================================
# 7. Test Bool Parsing Variants
# =============================================================================
def test_bool_parsing_variants():
    assert safe_convert_type("true", "bool") is True
    assert safe_convert_type("True", "bool") is True
    assert safe_convert_type("TRUE", "bool") is True
    assert safe_convert_type("yes", "bool") is True
    assert safe_convert_type("y", "bool") is True
    assert safe_convert_type("1", "bool") is True
    
    assert safe_convert_type("false", "bool") is False
    assert safe_convert_type("False", "bool") is False
    assert safe_convert_type("FALSE", "bool") is False
    assert safe_convert_type("no", "bool") is False
    assert safe_convert_type("n", "bool") is False
    assert safe_convert_type("0", "bool") is False
    
    with pytest.raises(ValueError):
        safe_convert_type("maybe", "bool")

# =============================================================================
# 8. Test Partial Override
# =============================================================================
def test_merge_params_partial_override():
    schema = {
        "p1": {"type": "int", "default": 1},
        "p2": {"type": "int", "default": 2}
    }
    
    params = {"p1": "10"}
    merged = merge_params(schema, params)
    
    assert merged["p1"] == 10
    assert merged["p2"] == 2

# =============================================================================
# 9. Test No Schema
# =============================================================================
def test_merge_params_no_schema():
    merged = merge_params(None, {"p1": "10"})
    assert merged == {}

# =============================================================================
# 10. Test Empty Params
# =============================================================================
def test_merge_params_empty_params():
    schema = {
        "p1": {"type": "int", "default": 1}
    }
    
    merged = merge_params(schema, {})
    assert merged["p1"] == 1
