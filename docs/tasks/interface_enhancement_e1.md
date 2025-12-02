# Interface Enhancement Phase E1.1 + E1.2 - Task Checklist

## Phase E1.1: UI Foundation (Execute FIRST)

- [x] **E1.1-TASK 1: Viewer Color Mode**
  - [x] Implement `viewer.color_mode` ("type", "constant", "velocity")
  - [x] Implement color generation logic

- [x] **E1.1-TASK 2: Radius-Aware Rendering**
  - [x] Add `entity_radius` to `UniverseState` in `state.py`
  - [x] Update `initialize_state` to set default radius
  - [x] Implement `viewer.render_radius_mode` ("constant", "scaled")

- [x] **E1.1-TASK 3: Debug HUD**
  - [x] Implement `viewer.show_debug` toggle
  - [x] Render text overlay (time, dt, energy, active count, modes)

- [x] **E1.1-TASK 4: Overlay Pipeline**
  - [x] Create `viewer/overlays/` directory
  - [x] Define `Overlay` base class
  - [x] Implement overlay application loop in `Viewer`

- [x] **E1.1-TASK 5: Metrics Logging**
  - [x] Add `metrics_log` to simulation runtime
  - [x] Implement logging hook in simulation loop

## Phase E1.2: Interactive Debug Tools (Execute SECOND)

- [x] **E1.2-TASK 1: Playback Controls**
  - [x] Pause/Resume toggle
  - [x] Single-step advance
  - [x] Speed multiplier control
  - [x] Keyboard shortcuts (Space, Right Arrow, Up/Down)

- [x] **E1.2-TASK 2: Entity Selection & Inspector**
  - [x] Implement click detection (picking)
  - [x] Display inspector panel for selected entity
  - [x] Show ID, pos, vel, mass, radius, type

- [x] **E1.2-TASK 3: Vector Fields**
  - [x] Toggle `show_velocity_vectors`
  - [x] Toggle `show_acceleration_vectors`
  - [x] Render arrows scaled by magnitude

- [x] **E1.2-TASK 4: Trajectories**
  - [x] Implement position history buffer
  - [x] Render faded trail lines

- [x] **E1.2-TASK 5: Mode Switching**
  - [x] Keyboard shortcuts for toggling all modes

- [x] **E1.2-TASK 6: Reset/Reload**
  - [x] Implement simulation reset
  - [x] Implement scenario reload

- [x] **E1.2-TASK 7: Inspector Overlay**
  - [x] Highlight selected entity (ring/outline)

## Verification
- [x] Verify E1.1 features (color, radius, HUD)
- [x] Verify E1.2 features (interaction, selection, vectors)
