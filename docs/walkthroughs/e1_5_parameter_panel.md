# E1.5: Scenario Parameter Readout Panel

## Overview
Implemented a toggleable parameter panel in the Python viewer that displays scenario metadata, universe configuration, PSS parameters, engine flags, and simulation progress.

**Status**: ✅ Complete

## Changes Made

### 1. New Module: `viewer/params_panel.py`
Created `ParameterPanel` class with:
- Matplotlib text rendering in dedicated axes
- Non-blocking updates using `draw_idle()`
- Visibility toggling
- Formatted display of config and parameters

#### Display Sections
- Scenario name
- Topology type
- UniverseConfig values (dt, entities, radius, etc.)
- PSS parameters (merged from schema/preset/CLI)
- Engine flags (diagnostics, neighbor_engine, spatial_partition)
- Current frame and simulation time

### 2. Modified: `viewer/viewer.py`
Integrated panel into Viewer class:
- Added `__init__` parameters: `scenario_name`, `pss_params`
- Initialize `ParameterPanel` in `__init__`
- Update panel state in `step_once()`
- Render panel (throttled) in `render()`
- Added keyboard toggle: **Shift+P**

### 3. Modified: `cosmosim.py`
Pass scenario data to viewer:
```python
viewer = Viewer(cfg, state, 
                scenario_name=scenario_name, 
                pss_params=merged_params)
```

### 4. Test Suite: `tests/test_viewer_params_panel.py`
Created comprehensive tests:
- Panel initialization
- Update accepts all fields
- Non-blocking render
- Visibility toggling
- Empty parameter robustness
- Integration with Viewer
- Throttling behavior

### 5. Fixed: `viewer/test.html`
Restored corrupted HTML header and structure.

## Usage

**Keyboard Shortcut**: Press **Shift+P** to toggle the parameter panel on/off.

The panel displays in real-time on the right side of the viewer window with:
- Black background (alpha 0.7)
- White monospace text
- Auto-updates every 10 frames (throttled for performance)

## Implementation Details

### Throttling
Panel renders every 10 frames to avoid performance impact:
```python
self.frame_counter += 1
if self.show_params_panel and self.frame_counter % 10 == 0:
    self.params_panel.render()
```

### Panel Position
Fixed axes at `[0.75, 0.1, 0.24, 0.8]` (right 25% of figure).

### Data Flow
1. `cosmosim.py` passes `scenario_name` and `pss_params`
2. `Viewer` stores as instance variables
3. `step_once()` updates panel with current state
4. `render()` draws panel if visible

## Validation
✅ All 8 tests pass  
✅ Panel toggles correctly  
✅ Non-blocking rendering  
✅ Displays all required information  
✅ Backward compatible (optional parameters)

## Future Enhancements
- Configurable position/size
- Custom color schemes
- Export panel content to file
- Interactive parameter editing
