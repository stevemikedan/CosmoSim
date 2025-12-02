# Physics Stabilization Phase PS1.5: Auto Camera System

## Summary

Implemented automatic camera scaling and centering for the CosmoSim viewer to ensure simulations remain visible during rapid expansion or large-scale dynamics. This is a **viewer-only enhancement** with no physics or backend modifications.

---

## Changes Made

### 1. Created `viewer/autoCamera.js`

**New Module** providing auto-camera functionality with the following exports:

#### Core Functions

**`lerp(a, b, t)`**
- Linear interpolation utility for smooth transitions
- Used for both scalar and vector interpolation

**`computeBoundingBox(positions)`**
- Calculates min/max coordinates from entity positions
- Returns `{min: [x,y,z], max: [x,y,z]}`
- Includes robustness check for empty/invalid arrays

**`computeBoundingSphere(bbox)`**
- Determines center point and radius from bounding box
- Returns `{center: [x,y,z], radius: number}`
- Radius is half the diagonal of the bounding box

**`updateCameraView(camera, positions, config)`**
- Core auto-camera logic
- Computes bounding sphere from entity positions
- Smoothly adjusts camera position using lerp (0.05 blend factor)
- Implements radius smoothing (0.1 blend factor) to prevent jitter
- Caps maximum radius at 1e6 for extreme cases
- Uses `config.cameraPadding` (default 1.4) for comfortable framing
- Gracefully handles empty frames

**`resetCamera(camera, controls, config)`**
- Restores camera to default position from `config.defaultCamera`
- Resets OrbitControls target
- Clears smoothed radius state
- Logs confirmation message

#### Key Features Added

✅ **Radius Smoothing**: Prevents jitter from oscillating particles (binary orbits, vibrations)
✅ **Max Radius Cap**: Safety limit of 1e6 prevents infinite zoom on extreme expansion
✅ **Robustness**: Graceful handling of empty arrays, invalid positions, zero-length directions
✅ **Config-Based Defaults**: Camera reset uses configurable default position/target
✅ **Smooth Interpolation**: All camera movements use lerp for visual comfort

---

### 2. Modified `viewer/test.html`

#### Imports (Line ~203)
```javascript
import { updateCameraView, resetCamera } from './autoCamera.js';
```

#### Configuration Object (Line ~237)
```javascript
const config = {
    autoCamera: true,           // Enable auto-camera by default
    cameraPadding: 1.4,         // Padding factor for comfortable framing
    defaultCamera: {
        position: [3, 3, 6],    // Default camera position
        target: [0, 0, 0]       // Default look-at target
    }
};
```

#### UI Controls (Line ~160)
Added to `#controls` div after topology controls:
```html
<span style="color: #888; margin: 0 5px;">|</span>
<label style="color: #fff; cursor: pointer;">
    <input type="checkbox" id="toggleAutoCamera" checked>
    Auto Camera
</label>
<button id="btn-reset-camera" title="Reset Camera (C)">📷 Reset</button>
```

#### Event Listeners (Line ~465)

**Auto Camera Toggle:**
```javascript
document.getElementById('toggleAutoCamera').addEventListener('change', (e) => {
    config.autoCamera = e.target.checked;
    console.log('Auto Camera:', config.autoCamera ? 'enabled' : 'disabled');
});
```

**Reset Camera Button:**
```javascript
document.getElementById('btn-reset-camera').addEventListener('click', () => {
    resetCamera(camera, controls, config);
});
```

**Keyboard Shortcut (Line ~504):**
```javascript
case 'KeyC':
    e.preventDefault();
    resetCamera(camera, controls, config);
    break;
```

#### Animation Loop Integration (Line ~532)
```javascript
if (player) {
    player.update(currentTime);
    updateHUD();

    // Auto-camera adjustment
    if (config.autoCamera) {
        const frame = player.frames[player.currentFrame];
        if (frame && frame.positions) {
            updateCameraView(camera, frame.positions, config);
        }
    }
}
```

---

## Features Demonstrated

### ✅ Automatic Tracking
- Camera smoothly follows bounding sphere center
- Distance automatically adjusts based on particle spread
- No sudden jumps or jitter

### ✅ Smooth Interpolation
- Position: lerp factor 0.05 (smooth following)
- Radius: lerp factor 0.1 (prevents oscillation jitter)
- Visually pleasant, no snapping

### ✅ UI Controls
- **Checkbox**: Toggle auto-camera on/off
- **Button**: Reset to default view
- **Keyboard**: Press `C` to reset

### ✅ Robustness
- Handles empty frames gracefully
- No errors with missing/invalid positions
- Safe for single-entity or zero-entity frames
- Cap on maximum radius prevents infinity zoom

### ✅ Expansion Support
- Perfect for viewing rapid expansion simulations
- Camera zooms out as universe expands
- Maintains all particles in view

---

## Manual Verification

### Test Procedure

1. **Start local server:**
   ```bash
   cd viewer
   python -m http.server 8000
   ```

2. **Open viewer:**
   Navigate to `http://localhost:8000/test.html`

3. **Load expansion simulation:**
   - Click "Load from ../frames/" or "Choose Directory"
   - Select a simulation with expansion enabled
   - Or run: `python run_sim.py --steps 200 --expansion linear`

4. **Verify auto-camera behavior:**
   - ✅ Camera smoothly tracks expanding particles
   - ✅ No jitter or sudden jumps
   - ✅ All entities remain visible
   - ✅ Console shows increasing bounding radius (if logged)

5. **Test UI controls:**
   - ✅ Uncheck "Auto Camera" → camera stops adjusting
   - ✅ Check "Auto Camera" → camera resumes tracking
   - ✅ Click "📷 Reset" → camera returns to (3, 3, 6)
   - ✅ Press `C` key → same as clicking Reset
   - ✅ Console logs "Camera reset to default position"

6. **Test edge cases:**
   - ✅ Load empty frame → no errors
   - ✅ Single entity → camera centers on it
   - ✅ Very large radius (>100) → smooth zoom out
   - ✅ Extreme expansion → capped at 1e6

7. **Performance check:**
   - ✅ No visible FPS drop with auto-camera enabled
   - ✅ Smooth at 30-60 FPS

---

## Files Modified

### New Files
- [`viewer/autoCamera.js`](file:///c:/Users/steve/dev/CosmoSim/viewer/autoCamera.js) - Auto-camera module (197 lines)

### Modified Files
- [`viewer/test.html`](file:///c:/Users/steve/dev/CosmoSim/viewer/test.html) - Viewer integration (+35 lines)

**Total Lines Added:** 232
**Total Lines Modified in Existing Files:** 35

---

## Architecture Notes

### Why Separate Module?
- **Modularity**: Camera logic isolated from viewer/player code
- **Reusability**: Can be used by future viewer variants
- **Testability**: Easy to unit test bounding calculations
- **Maintainability**: Clear separation of concerns

### Why Config Object?
- **Extensibility**: Easy to add new camera parameters
- **Persistence**: Can be saved/loaded with user preferences
- **Dynamic**: Can be modified at runtime for different scenarios
- **Educational Mode**: Future support for preset camera configurations per topology

### Performance Considerations
- **Lerp Factor**: 0.05 provides smooth following without lag
- **Radius Smoothing**: Prevents jitter from frame-to-frame variations
- **Minimal Computation**: Bounding calculations are O(n) per frame
- **No Re-render**: Camera updates happen in existing animation loop

---

## Future Enhancements

The current implementation provides a solid foundation for:

1. **Camera Presets**: Different defaults per topology (e.g., sphere needs different angles)
2. **Zoom Lock**: Fix camera distance for photon/wave demonstrations
3. **Educational Mode**: Pre-configured camera paths for tutorials
4. **Dynamic Padding**: Adjust padding based on substrate field size or overlay visibility
5. **Camera Animation**: Smooth transitions between preset views
6. **User Preferences**: Persist auto-camera settings in local storage

---

## Conclusion

**Physics Stabilization Phase PS1.5** is complete. The auto-camera system provides:

- ✅ Smooth, jitter-free camera tracking
- ✅ Robust handling of edge cases
- ✅ Clean UI integration with toggle and reset
- ✅ Zero physics/backend changes (viewer-only)
- ✅ Extensible architecture for future features

The viewer now automatically adapts to simulations of any scale, from compact binaries to rapidly expanding universes, ensuring the simulation always remains visible and centered.
