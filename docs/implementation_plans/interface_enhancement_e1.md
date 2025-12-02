# Interface Enhancement Phase E1.1 + E1.2 Implementation Plan

## Goal
Implement a Python-based interactive viewer for CosmoSim to enable real-time debugging and visualization. This replaces the need for headless plotting scripts for interactive tasks.

## Architecture

### 1. Viewer Core (`viewer/viewer.py`)
A `Viewer` class that manages the simulation loop and rendering.

- **Dependencies**: `matplotlib`, `jax`, `numpy`
- **Key Components**:
  - `fig`, `ax`: Matplotlib figure and axis
  - `config`: `UniverseConfig`
  - `state`: `UniverseState`
  - `overlays`: List of `Overlay` objects
  - **State Flags**: `paused`, `show_debug`, `color_mode`, `render_radius_mode`, `show_vectors`, `show_trajectories`

### 2. State Updates (`state.py`)
- Add `entity_radius` (float array) to `UniverseState`.
- Update `initialize_state` to set default radius (e.g., 0.1 or config-based).

### 3. Overlay System (`viewer/overlays/`)
- `base.py`: Abstract base class `Overlay` with `apply(state, ax)` method.
- `debug.py`: `DebugOverlay` for HUD text.
- `vectors.py`: `VelocityOverlay`, `AccelerationOverlay`.
- `trajectories.py`: `TrajectoryOverlay` (manages history buffer).
- `inspector.py`: `InspectorOverlay` (handles selection highlight).

## Phase E1.1: UI Foundation

### [MODIFY] `state.py`
- Add `entity_radius: jnp.ndarray` to `UniverseState`.
- Initialize with default value.

### [NEW] `viewer/viewer.py`
- Implement `Viewer` class.
- **Render Loop**:
  1. Clear axis.
  2. Draw entities (scatter plot).
     - Handle `color_mode` ("type", "constant", "velocity").
     - Handle `render_radius_mode` (scale `s` parameter).
  3. Apply overlays.
  4. `plt.pause(0.001)`.

### [NEW] `viewer/overlays/base.py`
- Define `Overlay` interface.

### [NEW] `viewer/overlays/debug.py`
- Implement HUD using `ax.text`.

## Phase E1.2: Interactive Tools

### [MODIFY] `viewer/viewer.py`
- **Event Handling**:
  - `mpl_connect('key_press_event', ...)`
  - `mpl_connect('button_press_event', ...)`
- **Controls**:
  - Space: Pause/Resume
  - Right Arrow: Step
  - Up/Down: Speed control
  - Click: Select entity (find nearest within radius)

### [NEW] `viewer/overlays/inspector.py`
- Draw selection ring around `selected_entity_idx`.
- Display info panel (text box) if entity selected.

### [NEW] `viewer/overlays/vectors.py`
- Use `ax.quiver` for velocity/acceleration.

### [NEW] `viewer/overlays/trajectories.py`
- Store `deque` of positions for each entity.
- Draw lines using `ax.plot` (with alpha fade).

## Execution Strategy
1. **E1.1**: Update state, build basic viewer, add debug HUD. Verify.
2. **E1.2**: Add interactivity, selection, and advanced overlays. Verify.
