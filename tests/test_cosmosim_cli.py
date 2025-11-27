"""
Tests for cosmosim.py CLI functionality.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call
import inspect
from pathlib import Path

# Add parent directory to path for cosmosim import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cosmosim


def test_list_flag_displays_scenarios(capsys):
    """Verify --list displays available scenarios and exits with 0."""
    fake_scenarios = {
        "run_sim": "run_sim",
        "manual_run": "scenarios.manual_run",
        "visualize": "visualize",
    }
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        exit_code = cosmosim.main(["--list"])
    
    assert exit_code == 0
    
    captured = capsys.readouterr()
    assert "run_sim" in captured.out
    assert "manual_run" in captured.out
    assert "visualize" in captured.out
    assert "scenarios.manual_run" in captured.out


def test_missing_scenario_argument_returns_error():
    """Verify missing --scenario returns non-zero exit code."""
    with patch("cosmosim.load_scenarios", return_value={}):
        exit_code = cosmosim.main([])
    
    assert exit_code == 1


def test_valid_scenario_execution_pipeline():
    """Verify complete execution pipeline with valid scenario."""
    # Create fake module
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "test_scenario"])
    
    assert exit_code == 0
    fake_module.build_config.assert_called_once()
    fake_module.build_initial_state.assert_called_once_with("CONFIG")
    fake_module.run.assert_called_once_with("CONFIG", "STATE")


def test_steps_override_when_supported():
    """Verify --steps passes override when run() accepts it."""
    # Create run function that accepts steps parameter
    def mock_run(config, state, steps=None):
        return "FINAL"
    
    # We need to manually attach the signature because MagicMock(side_effect=...)
    # doesn't automatically copy the signature for inspect.signature()
    run_mock = MagicMock(side_effect=mock_run)
    run_mock.__signature__ = inspect.signature(mock_run)
    
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=run_mock,
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "test_scenario", "--steps", "42"])
    
    assert exit_code == 0
    # Verify run was called with steps=42
    call_args = fake_module.run.call_args
    assert call_args[0] == ("CONFIG", "STATE")
    assert call_args[1] == {"steps": 42}


def test_steps_override_when_not_supported(capsys):
    """Verify warning printed when --steps used but run() doesn't support it."""
    # Create run function without steps parameter
    def mock_run(config, state):
        return "FINAL"
    
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(side_effect=mock_run),
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "test_scenario", "--steps", "42"])
    
    assert exit_code == 0
    
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    
    # Verify run was called WITHOUT steps parameter
    call_args = fake_module.run.call_args
    assert call_args[0] == ("CONFIG", "STATE")
    assert call_args[1] == {}  # No kwargs


def test_headless_flag_sets_environment_variable():
    """Verify --headless sets COSMOSIM_HEADLESS environment variable."""
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch.dict(os.environ, {}, clear=True):
        with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
            with patch("cosmosim.importlib.import_module", return_value=fake_module):
                with patch.object(cosmosim, "validate_interface", return_value=True):
                    exit_code = cosmosim.main(["--scenario", "test_scenario", "--headless"])
        
        assert exit_code == 0
        assert os.environ.get("COSMOSIM_HEADLESS") == "1"


def test_output_dir_creates_directory_and_sets_env_var(tmp_path):
    """Verify --output-dir creates directory and sets COSMOSIM_OUTPUT_DIR."""
    output_dir = tmp_path / "custom_output"
    
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch.dict(os.environ, {}, clear=True):
        with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
            with patch("cosmosim.importlib.import_module", return_value=fake_module):
                with patch.object(cosmosim, "validate_interface", return_value=True):
                    exit_code = cosmosim.main([
                        "--scenario", "test_scenario",
                        "--output-dir", str(output_dir)
                    ])
        
        assert exit_code == 0
        assert output_dir.exists()
        assert os.environ.get("COSMOSIM_OUTPUT_DIR") == str(output_dir.absolute())


def test_unknown_scenario_produces_error(capsys):
    """Verify unknown scenario name produces clean error message."""
    fake_scenarios = {"run_sim": "run_sim"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        exit_code = cosmosim.main(["--scenario", "nonexistent_scenario"])
    
    assert exit_code == 1
    
    captured = capsys.readouterr()
    assert "Unknown scenario" in captured.err
    assert "--list" in captured.err


def test_import_failure_produces_clean_error(capsys):
    """Verify ImportError during module import produces clean error."""
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", side_effect=ImportError("Module not found")):
            exit_code = cosmosim.main(["--scenario", "test_scenario"])
    
    assert exit_code == 1
    
    captured = capsys.readouterr()
    assert "Failed to import" in captured.err
    assert "test_module" in captured.err


def test_interface_validation_failure_propagates():
    """Verify interface validation failure causes exit with code 1."""
    fake_module = SimpleNamespace(__name__="bad_module")
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=False):
                exit_code = cosmosim.main(["--scenario", "test_scenario"])
    
    assert exit_code == 1


def test_config_dump_flag_prints_configuration(capsys):
    """Verify --config-dump prints the configuration before running."""
    fake_config = "TEST_CONFIG_OBJECT"
    
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value=fake_config),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    fake_scenarios = {"test_scenario": "test_module"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "test_scenario", "--config-dump"])
    
    assert exit_code == 0
    
    captured = capsys.readouterr()
    assert "Configuration:" in captured.out
    assert "TEST_CONFIG_OBJECT" in captured.out


def test_full_module_path_as_scenario_name():
    """Verify that full module paths (with dots) are accepted as scenario names."""
    fake_module = SimpleNamespace(
        __name__="scenarios.custom",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    # Provide empty scenarios dict so the full path is used
    with patch("cosmosim.load_scenarios", return_value={}):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "scenarios.custom"])
    
    assert exit_code == 0


def test_scenario_display_name_uses_short_name():
    """Verify that scenario display name uses the short name provided by user."""
    fake_module = SimpleNamespace(
        __name__="scenarios.manual_run",
        build_config=MagicMock(return_value="CONFIG"),
        build_initial_state=MagicMock(return_value="STATE"),
        run=MagicMock(return_value="FINAL"),
    )
    
    fake_scenarios = {"manual_run": "scenarios.manual_run"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                exit_code = cosmosim.main(["--scenario", "manual_run"])
    
    assert exit_code == 0


@patch("cosmosim.export_simulation")
@patch("os.environ", new_callable=dict)
@patch("datetime.datetime")
def test_run_scenario_export_json_default_dir(mock_datetime, mock_environ, mock_export):
    """Test JSON export with timestamped directory in frames/."""
    # Mock timestamp
    mock_now = MagicMock()
    mock_now.strftime.return_value = "2025-11-26T23-15-19"
    mock_datetime.now.return_value = mock_now
    
    # Setup mock scenario
    fake_cfg = object()
    fake_state = object()
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value=fake_cfg),
        build_initial_state=MagicMock(return_value=fake_state),
        run=MagicMock(),
    )
    
    # Setup args
    args = SimpleNamespace(
        steps=50,
        headless=False,
        output_dir=None,
        export_json=True,
        config_dump=False,
    )
    
    # Run
    cosmosim.run_scenario(fake_module, args, "test_scenario")
    
    # Verify export_simulation called with timestamped directory
    expected_dir = str(Path("frames") / "test_scenario_50_steps_2025-11-26T23-15-19")
    mock_export.assert_called_once_with(
        fake_cfg, fake_state, steps=50, output_dir=expected_dir
    )
    
    # Verify environment variable set
    expected_full_path = str(Path(expected_dir).resolve())
    assert mock_environ.get("COSMOSIM_EXPORT_JSON_DIR") == expected_full_path



@patch("cosmosim.export_simulation")
@patch("datetime.datetime")
def test_run_scenario_export_json_steps_default(mock_datetime, mock_export):
    """Test JSON export uses default steps if not provided."""
    # Mock timestamp
    mock_now = MagicMock()
    mock_now.strftime.return_value = "2025-11-26T23-15-19"
    mock_datetime.now.return_value = mock_now
    
    # Setup mock scenario
    fake_cfg = object()
    fake_state = object()
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value=fake_cfg),
        build_initial_state=MagicMock(return_value=fake_state),
        run=MagicMock(),
    )
    
    # Setup args
    args = SimpleNamespace(
        steps=None,
        headless=False,
        output_dir=None,
        export_json=True,
        config_dump=False,
    )
    
    # Run
    cosmosim.run_scenario(fake_module, args, "test_scenario")
    
    # Verify export_simulation called with default steps (100) and timestamped directory
    expected_dir = str(Path("frames") / "test_scenario_100_steps_2025-11-26T23-15-19")
    mock_export.assert_called_once_with(
        fake_cfg, fake_state, steps=100, output_dir=expected_dir
    )


@patch("kernel.step_simulation")
@patch("exporters.json_export.export_frame")
def test_export_simulation_integration(mock_export_frame, mock_step):
    """Test export_simulation logic with physics mocking."""
    # Mock physics to be a no-op
    mock_step.side_effect = lambda state, cfg: state
    
    # Import here to avoid circular imports or early binding issues
    from exporters.json_export import export_simulation
    
    cfg = MagicMock()
    state = MagicMock()
    steps = 5
    output_dir = "test_out"
    
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        export_simulation(cfg, state, steps=steps, output_dir=output_dir)
        
        # Verify mkdir called
        mock_mkdir.assert_called()
        
        # Verify export_frame called 'steps' times
        assert mock_export_frame.call_count == steps
        
        # Verify step_simulation called 'steps' times
        assert mock_step.call_count == steps


def test_run_scenario_normal_mode():
    """Test that normal run mode is preserved (no export)."""
    fake_cfg = object()
    fake_state = object()
    fake_module = SimpleNamespace(
        __name__="fake_scenario",
        build_config=MagicMock(return_value=fake_cfg),
        build_initial_state=MagicMock(return_value=fake_state),
        run=MagicMock(return_value=object()),
    )
    
    args = SimpleNamespace(
        steps=None,
        headless=False,
        output_dir=None,
        export_json=False,  # normal mode
        export_json_dir=None,
        config_dump=False,
    )
    
    with patch("cosmosim.export_simulation") as mock_export:
        cosmosim.run_scenario(fake_module, args, "test_scenario")
        
        # Verify module.run is called
        fake_module.run.assert_called_once()
        
        # Verify export_simulation is NOT called
        mock_export.assert_not_called()


def test_main_wires_export_json_flags_into_run_scenario():
    """Verify main() correctly forwards export flags to run_scenario."""
    fake_module = SimpleNamespace(
        __name__="manual_run",
        build_config=MagicMock(),
        build_initial_state=MagicMock(),
        run=MagicMock(),
    )
    
    fake_scenarios = {"manual_run": "scenarios.manual_run"}
    
    with patch("cosmosim.load_scenarios", return_value=fake_scenarios):
        with patch("cosmosim.importlib.import_module", return_value=fake_module):
            with patch.object(cosmosim, "validate_interface", return_value=True):
                with patch.object(cosmosim, "run_scenario") as mock_run_scenario:
                    exit_code = cosmosim.main([
                        "--scenario", "manual_run",
                        "--export-json",
                        "--steps", "123"
                    ])
    
    assert exit_code == 0
    
    # Verify run_scenario was called once
    mock_run_scenario.assert_called_once()
    
    # Extract the args passed to run_scenario
    call_args = mock_run_scenario.call_args
    module_arg, args_arg, scenario_name_arg = call_args[0]
    
    # Verify the args have correct export flags
    assert args_arg.export_json == True
    assert args_arg.steps == 123
