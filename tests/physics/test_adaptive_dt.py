"""
Tests for adaptive timestep controller (Phase PS1.4).

Verifies that:
- Timestep shrinks when velocities or accelerations are high
- Timestep grows when system is stable
- Adaptive behavior can be disabled
- Timestep stays within configured bounds
- No NaNs or invalid values are produced
"""

import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from physics_utils import adjust_timestep


def test_dt_shrinks_with_high_velocity():
    """Verify dt decreases when velocity exceeds threshold."""
    # Velocity (10, 0, 0) exceeds threshold 5.0
    vel = jnp.array([[10.0, 0.0, 0.0]])
    force = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        enable_adaptive_dt=True,
        velocity_threshold=5.0,
        acceleration_threshold=10.0
    )
    
    new_dt = adjust_timestep(config.dt, vel, force, mass, config)
    
    # Should reduce by half
    assert new_dt < config.dt, "dt should decrease for high velocity"
    assert jnp.isclose(new_dt, 0.05), f"Expected dt=0.05, got {new_dt}"


def test_dt_shrinks_with_high_acceleration():
    """Verify dt decreases when acceleration exceeds threshold."""
    # Force 20.0 / Mass 1.0 = Acc 20.0 exceeds threshold 10.0
    vel = jnp.array([[0.0, 0.0, 0.0]])
    force = jnp.array([[20.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        enable_adaptive_dt=True,
        velocity_threshold=5.0,
        acceleration_threshold=10.0
    )
    
    new_dt = adjust_timestep(config.dt, vel, force, mass, config)
    
    # Should reduce by half
    assert new_dt < config.dt, "dt should decrease for high acceleration"
    assert jnp.isclose(new_dt, 0.05), f"Expected dt=0.05, got {new_dt}"


def test_dt_grows_when_stable():
    """Verify dt increases when system is slow and stable."""
    # Velocity 0.5 is < 20% of threshold 5.0 (which is 1.0)
    vel = jnp.array([[0.5, 0.0, 0.0]])
    force = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        enable_adaptive_dt=True,
        velocity_threshold=5.0,
        acceleration_threshold=10.0
    )
    
    new_dt = adjust_timestep(config.dt, vel, force, mass, config)
    
    # Should increase by 5%
    assert new_dt > config.dt, "dt should increase for stable system"
    assert jnp.isclose(new_dt, 0.105), f"Expected dt=0.105, got {new_dt}"


def test_dt_clamped_to_bounds():
    """Verify dt stays within min/max scaling limits."""
    vel = jnp.array([[0.0, 0.0, 0.0]])
    force = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        enable_adaptive_dt=True,
        max_dt_scale=2.0,
        min_dt_scale=0.5
    )
    
    # Try to increase beyond max
    # Pass a large dt as input
    large_dt = 0.5  # 5x base dt
    new_dt = adjust_timestep(large_dt, vel, force, mass, config)
    
    # Should clamp to max_dt_scale * base_dt = 2.0 * 0.1 = 0.2
    assert new_dt <= 0.200001, f"dt should be clamped to max (0.2), got {new_dt}"
    
    # Try to decrease below min
    # Pass a small dt as input
    small_dt = 0.01 # 0.1x base dt
    new_dt = adjust_timestep(small_dt, vel, force, mass, config)
    
    # Should clamp to min_dt_scale * base_dt = 0.5 * 0.1 = 0.05
    assert new_dt >= 0.049999, f"dt should be clamped to min (0.05), got {new_dt}"


def test_no_nans_with_empty_system():
    """Verify no NaNs produced for empty/zero inputs."""
    vel = jnp.array([[0.0, 0.0, 0.0]])
    force = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([0.0]) # Zero mass
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    
    new_dt = adjust_timestep(config.dt, vel, force, mass, config)
    
    assert jnp.isfinite(new_dt), "Result should be finite"
    assert new_dt > 0, "Result should be positive"


def test_integration_with_adaptive_dt():
    """Verify integration works with varying dt."""
    # This is more of a sanity check that the function returns a usable float
    vel = jnp.array([[10.0, 0.0, 0.0]])
    force = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        enable_adaptive_dt=True
    )
    
    new_dt = adjust_timestep(config.dt, vel, force, mass, config)
    
    # Verify return type is standard float (for JSON serialization compatibility)
    assert isinstance(new_dt, float), "Should return standard python float"
