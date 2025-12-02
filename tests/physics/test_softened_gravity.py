"""
Tests for softened gravity implementation (Phase PS1.1).

Verifies that the softened gravity kernel prevents numerical blowups
when particles get too close.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from physics_utils import compute_gravity_forces


def test_finite_forces_at_small_separation():
    """Verify that forces remain finite even at extremely small separations."""
    # Create two particles very close together
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0]  # Very close - would cause blowup without softening
    ])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.05
    )
    
    forces = compute_gravity_forces(pos, mass, active, config)
    
    # Forces should be finite (not NaN or Inf)
    assert jnp.all(jnp.isfinite(forces)), "Forces contain NaN or Inf values"
    
    # Forces should be non-zero (particles are attracting)
    force_magnitude = jnp.linalg.norm(forces[0])
    assert force_magnitude > 0, "Force magnitude should be non-zero"
    
    # Force should be reasonable (not astronomically large)
    assert force_magnitude < 1000, f"Force magnitude too large: {force_magnitude}"


def test_epsilon_decreases_force_magnitude():
    """Verify that increasing epsilon decreases force magnitude at close range."""
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0]  # Close separation
    ])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    # Test with small epsilon
    config_small = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.01
    )
    
    forces_small = compute_gravity_forces(pos, mass, active, config_small)
    force_mag_small = jnp.linalg.norm(forces_small[0])
    
    # Test with large epsilon
    config_large = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.1
    )
    
    forces_large = compute_gravity_forces(pos, mass, active, config_large)
    force_mag_large = jnp.linalg.norm(forces_large[0])
    
    # Larger epsilon should produce smaller force
    assert force_mag_large < force_mag_small, \
        f"Larger epsilon should reduce force: {force_mag_large} >= {force_mag_small}"


def test_backward_compatibility_with_zero_softening():
    """Verify that epsilon=0 approximates original unsoftened behavior at safe distances."""
    # Use a safe separation where softening doesn't matter much
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]  # Safe separation
    ])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    # Config with softening
    config_soft = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.05
    )
    
    # Config with minimal softening (approximates zero)
    config_minimal = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=1e-10  # Very small, approximates zero
    )
    
    forces_soft = compute_gravity_forces(pos, mass, active, config_soft)
    forces_minimal = compute_gravity_forces(pos, mass, active, config_minimal)
    
    force_mag_soft = jnp.linalg.norm(forces_soft[0])
    force_mag_minimal = jnp.linalg.norm(forces_minimal[0])
    
    # At safe distances, softening should have minimal effect
    # Allow 10% difference
    relative_diff = abs(force_mag_soft - force_mag_minimal) / force_mag_minimal
    assert relative_diff < 0.1, \
        f"Forces differ too much at safe distance: {relative_diff:.2%}"


def test_inactive_particles_produce_no_force():
    """Verify that inactive particles don't exert gravitational force."""
    pos = jnp.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0]
    ])
    mass = jnp.array([1.0, 1.0, 1.0])
    active = jnp.array([True, False, True])  # Middle particle inactive
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.05
    )
    
    forces = compute_gravity_forces(pos, mass, active, config)
    
    # Inactive particle (index 1) should experience no force
    force_on_inactive = jnp.linalg.norm(forces[1])
    assert force_on_inactive == 0.0, "Inactive particle should have zero force"
    
    # Active particles should still experience forces
    force_on_active_0 = jnp.linalg.norm(forces[0])
    force_on_active_2 = jnp.linalg.norm(forces[2])
    assert force_on_active_0 > 0, "Active particle 0 should experience force"
    assert force_on_active_2 > 0, "Active particle 2 should experience force"


def test_force_direction_is_correct():
    """Verify that gravitational force points in the correct direction."""
    # Two particles along x-axis
    pos = jnp.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        gravity_softening=0.05
    )
    
    forces = compute_gravity_forces(pos, mass, active, config)
    
    # Force on particle 0 should point toward particle 1 (positive x direction)
    assert forces[0, 0] > 0, "Force on particle 0 should point in +x direction"
    assert abs(forces[0, 1]) < 1e-10, "Force should be purely in x direction"
    assert abs(forces[0, 2]) < 1e-10, "Force should be purely in x direction"
    
    # Force on particle 1 should point toward particle 0 (negative x direction)
    assert forces[1, 0] < 0, "Force on particle 1 should point in -x direction"
    
    # Forces should be equal and opposite (Newton's third law)
    assert jnp.allclose(forces[0], -forces[1]), "Forces should be equal and opposite"
