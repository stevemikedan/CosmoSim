"""JSON Exporter for CosmoSim simulation frames.

This module provides utilities to export the simulation state to a series of
JSON files that can be consumed by a Three.js visualizer. It is deliberately
self‑contained and does not modify any simulation logic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp

# Import the core simulation step – this is the same function used by the
# scenario modules, ensuring no duplication of physics.
import kernel

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_python_list(array: jnp.ndarray) -> List[Any]:
    """Convert a JAX array to a plain Python list.

    JAX arrays expose ``tolist()`` which returns nested Python lists. This
    helper exists to keep the conversion logic in one place.
    """
    return array.tolist()


def _sanitize_number(value: float) -> float | None:
    """Replace NaN/inf with ``None`` for JSON serialisation.

    JSON does not support ``NaN`` or ``Infinity``. Returning ``None`` makes the
    value explicit and safe for downstream consumers. Also converts JAX/NumPy
    scalars to Python primitives.
    """
    # Convert JAX/NumPy scalars to Python primitives first
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    
    if isinstance(value, (float, int)) and not math.isfinite(value):
        return None
    return float(value) if not isinstance(value, (bool, int)) else value


def _sanitize_nested(data: List[Any]) -> List[Any]:
    """Recursively sanitize a nested list to ensure JSON compatibility.

    Converts JAX/NumPy types to Python primitives and replaces non-finite
    numbers with ``None``.
    """
    sanitized: List[Any] = []
    for item in data:
        if isinstance(item, list):
            sanitized.append(_sanitize_nested(item))
        elif isinstance(item, bool):
            # Handle bool before numbers since bool is subclass of int
            sanitized.append(bool(item))
        elif isinstance(item, (int, float)):
            sanitized.append(_sanitize_number(item))
        else:
            # Handle JAX/NumPy types by converting to Python primitives
            try:
                # Try to convert to Python scalar
                sanitized.append(item.item() if hasattr(item, 'item') else item)
            except (AttributeError, ValueError):
                # Fallback to direct conversion
                sanitized.append(_sanitize_number(float(item)) if item is not None else None)
    return sanitized



def _topology_mode_name(mode: int) -> str:
    """Map topology mode integer to a human‑readable name.

    Currently only ``0`` (flat) and ``1`` (toroidal) are defined.
    """
    if mode == 0:
        return "flat"
    if mode == 1:
        return "toroidal"
    return "unknown"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_frame_dict(state: Any) -> Dict[str, Any]:
    """Convert a :class:`UniverseState` into a JSON‑serialisable dictionary.

    The function extracts the most relevant fields for visualisation and pads
    2‑D data with a ``z`` coordinate of ``0.0`` so that Three.js can always
    consume a three‑component vector.
    """
    # Positions and velocities are JAX arrays of shape (max_entities, dim).
    positions = _to_python_list(state.entity_pos)
    velocities = _to_python_list(state.entity_vel)

    # Pad to 3‑D if necessary.
    dim = len(positions[0]) if positions else 0
    if dim == 2:
        positions = [pos + [0.0] for pos in positions]
        velocities = [vel + [0.0] for vel in velocities]

    # Sanitize numeric values.
    positions = _sanitize_nested(positions)
    velocities = _sanitize_nested(velocities)

    masses = _sanitize_nested(_to_python_list(state.entity_mass))
    types = _sanitize_nested(_to_python_list(state.entity_type))

    topology = {
        "mode": _topology_mode_name(state.topology_type),
        "params": {"bounds": _sanitize_number(state.bounds)},
    }

    # Helper for safe scalar conversion
    def _safe_float(val):
        return _sanitize_number(float(val))

    # Helper for safe vector conversion
    def _safe_vector(val):
        return _sanitize_nested(_to_python_list(val))

    return {
        "time": _safe_float(state.time),
        "step_count": int(state.step_count),
        "expansion_factor": _safe_float(state.expansion_factor),
        "dt_actual": _safe_float(state.dt_actual),
        
        # Diagnostics
        "kinetic_energy": _safe_float(state.kinetic_energy),
        "potential_energy": _safe_float(state.potential_energy),
        "total_energy": _safe_float(state.total_energy),
        "energy_drift": _safe_float(state.energy_drift),
        "momentum": _safe_vector(state.momentum),
        "center_of_mass": _safe_vector(state.center_of_mass),
        
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "types": types,
        "active": _sanitize_nested(_to_python_list(state.entity_active)),
        "topology": topology,
    }


def export_frame(state: Any, frame_index: int, output_dir: str | Path) -> None:
    """Write a single simulation frame to ``frame_{index:05}.json``.

    Args:
        state: The current :class:`UniverseState`.
        frame_index: Zero‑based index of the frame.
        output_dir: Directory where the JSON file will be written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_dict = get_frame_dict(state)
    # Overwrite the placeholder ``frame`` value with the actual index.
    frame_dict["frame"] = frame_index

    filename = f"frame_{frame_index:05d}.json"
    with open(out_path / filename, "w") as f:
        json.dump(frame_dict, f, indent=2)

def export_simulation(cfg: Any, state: Any, *, steps: int, output_dir: str | Path) -> Any:
    """Run a simulation for ``steps`` frames, exporting each to JSON.

    This helper mirrors the typical ``run`` pattern used in scenario modules
    but adds a JSON export step before each physics update.

    Args:
        cfg: :class:`UniverseConfig` used for the simulation.
        state: Initial :class:`UniverseState`.
        steps: Number of simulation steps / frames to export.
        output_dir: Destination directory for the JSON files.

    Returns:
        The final :class:`UniverseState` after ``steps`` updates.
    """
    # Use the provided output directory directly
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i in range(steps):
        export_frame(state, i, out_path)
        state = kernel.step_simulation(state, cfg)
    return state
