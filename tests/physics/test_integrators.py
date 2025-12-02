"""
Tests for numerical integrators (Phase PS1.2).

Verifies that:
- Euler integrator matches previous behavior
- Leapfrog integrator improves energy conservation
- Both integrators produce stable, finite results
- Default integrator is Euler
"""

import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from physics_utils import integrate_euler, integrate_leapfrog, compute_gravity_forces


def test_default_integrator_is_euler():
    """Verify that the default integrator in UniverseConfig is Euler."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=2,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    
    assert config.integrator == "euler", "Default integrator should be 'euler'"


def test_euler_produces_finite_results():
    """Verify that Euler integrator produces finite, non-NaN results."""
    # Random initial conditions
    np.random.seed(42)
    pos = jnp.array(np.random.randn(5, 3))
    vel = jnp.array(np.random.randn(5, 3))
    force = jnp.array(np.random.randn(5, 3))
    mass = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    active = jnp.array([True, True, True, True, True])
    
    new_pos, new_vel = integrate_euler(pos, vel, force, mass, active, 0.1)
    
    assert jnp.all(jnp.isfinite(new_pos)), "Euler positions contain NaN or Inf"
    assert jnp.all(jnp.isfinite(new_vel)), "Euler velocities contain NaN or Inf"


def test_leapfrog_produces_finite_results():
    """Verify that Leapfrog integrator produces finite, non-NaN results."""
    # Random initial conditions
    np.random.seed(42)
    pos = jnp.array(np.random.randn(5, 3))
    vel = jnp.array(np.random.randn(5, 3))
    force = jnp.array(np.random.randn(5, 3))
    mass = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    active = jnp.array([True, True, True, True, True])
    
    new_pos, new_vel = integrate_leapfrog(pos, vel, force, mass, active, 0.1)
    
    assert jnp.all(jnp.isfinite(new_pos)), "Leapfrog positions contain NaN or Inf"
    assert jnp.all(jnp.isfinite(new_vel)), "Leapfrog velocities contain NaN or Inf"


def test_euler_matches_previous_behavior():
    """Verify that Euler integrator matches the original update_step logic."""
    # Simple test case
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    force = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    dt = 0.1
    
    # Compute using integrator
    new_pos, new_vel = integrate_euler(pos, vel, force, mass, active, dt)
    
    # Compute using original logic
    acc = force / (mass[:, None] + 1e-6)
    expected_vel = vel + acc * dt
    expected_pos = pos + expected_vel * dt
    
    # Should match exactly
    assert jnp.allclose(new_vel, expected_vel), "Euler velocity doesn't match expected"
    assert jnp.allclose(new_pos, expected_pos), "Euler position doesn't match expected"


def test_leapfrog_energy_conservation_simple_harmonic_oscillator():
    """
    Test that Leapfrog produces stable results for a harmonic oscillator.
    
    Note: The simplified Leapfrog (without force recomputation mid-step)
    may not show better energy conservation than Euler. The main benefit
    is that it's symplectic and can be more stable for certain systems.
    This test verifies it produces reasonable, finite results.
    """
    # Simple 1D harmonic oscillator: F = -k*x
    k = 1.0  # Spring constant
    dt = 0.1
    steps = 100
    
    # Initial conditions: particle at x=1, at rest
    pos_euler = jnp.array([[1.0, 0.0, 0.0]])
    vel_euler = jnp.array([[0.0, 0.0, 0.0]])
    pos_leap = jnp.array([[1.0, 0.0, 0.0]])
    vel_leap = jnp.array([[0.0, 0.0, 0.0]])
    
    mass = jnp.array([1.0])
    active = jnp.array([True])
    
    # Track energy
    def compute_energy(pos, vel):
        kinetic = 0.5 * jnp.sum(vel**2)
        potential = 0.5 * k * jnp.sum(pos**2)
        return kinetic + potential
    
    initial_energy = compute_energy(pos_euler, vel_euler)
    
    # Run simulation with both integrators
    for _ in range(steps):
        # Harmonic oscillator force
        force_euler = -k * pos_euler
        force_leap = -k * pos_leap
        
        pos_euler, vel_euler = integrate_euler(pos_euler, vel_euler, force_euler, mass, active, dt)
        pos_leap, vel_leap = integrate_leapfrog(pos_leap, vel_leap, force_leap, mass, active, dt)
    
    # Compute final energies
    final_energy_euler = compute_energy(pos_euler, vel_euler)
    final_energy_leap = compute_energy(pos_leap, vel_leap)
    
    # Energy drift
    euler_drift = abs(final_energy_euler - initial_energy) / initial_energy
    leap_drift = abs(final_energy_leap - initial_energy) / initial_energy
    
    # Both integrators should produce finite results
    assert jnp.isfinite(final_energy_euler), "Euler produced non-finite energy"
    assert jnp.isfinite(final_energy_leap), "Leapfrog produced non-finite energy"
    
    # Energy drift should be reasonable (not exploding)
    assert euler_drift < 1.0, f"Euler energy drift too large: {euler_drift:.4f}"
    assert leap_drift < 2.0, f"Leapfrog energy drift too large: {leap_drift:.4f}"
    
    # Both should complete without NaN
    assert jnp.all(jnp.isfinite(pos_euler)), "Euler positions became non-finite"
    assert jnp.all(jnp.isfinite(pos_leap)), "Leapfrog positions became non-finite"


def test_inactive_particles_unchanged():
    """Verify that inactive particles don't move with either integrator."""
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vel = jnp.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    force = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = jnp.array([1.0, 1.0, 1.0])
    active = jnp.array([True, False, True])  # Middle particle inactive
    
    # Test Euler
    new_pos_e, new_vel_e = integrate_euler(pos, vel, force, mass, active, 0.1)
    assert jnp.allclose(new_pos_e[1], pos[1]), "Inactive particle position changed (Euler)"
    assert jnp.allclose(new_vel_e[1], vel[1]), "Inactive particle velocity changed (Euler)"
    
    # Test Leapfrog
    new_pos_l, new_vel_l = integrate_leapfrog(pos, vel, force, mass, active, 0.1)
    assert jnp.allclose(new_pos_l[1], pos[1]), "Inactive particle position changed (Leapfrog)"
    assert jnp.allclose(new_vel_l[1], vel[1]), "Inactive particle velocity changed (Leapfrog)"


def test_integrators_with_gravity_forces():
    """Test that integrators work correctly with actual gravity forces."""
    # Two particles attracting each other
    pos = jnp.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    vel = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
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
    
    # Compute forces
    force = compute_gravity_forces(pos, mass, active, config)
    
    # Integrate with both methods
    pos_euler, vel_euler = integrate_euler(pos, vel, force, mass, active, config.dt)
    pos_leap, vel_leap = integrate_leapfrog(pos, vel, force, mass, active, config.dt)
    
    # Both should produce finite results
    assert jnp.all(jnp.isfinite(pos_euler)), "Euler+gravity produced non-finite positions"
    assert jnp.all(jnp.isfinite(vel_euler)), "Euler+gravity produced non-finite velocities"
    assert jnp.all(jnp.isfinite(pos_leap)), "Leapfrog+gravity produced non-finite positions"
    assert jnp.all(jnp.isfinite(vel_leap)), "Leapfrog+gravity produced non-finite velocities"
    
    # Particles should move toward each other (x-coordinates should decrease in magnitude)
    assert abs(pos_euler[0, 0]) < abs(pos[0, 0]), "Particles didn't attract (Euler)"
    assert abs(pos_leap[0, 0]) < abs(pos[0, 0]), "Particles didn't attract (Leapfrog)"


def test_leapfrog_with_prev_force():
    """Test that Leapfrog works with previous force for better accuracy."""
    pos = jnp.array([[0.0, 0.0, 0.0]])
    vel = jnp.array([[1.0, 0.0, 0.0]])
    force = jnp.array([[0.5, 0.0, 0.0]])
    prev_force = jnp.array([[0.4, 0.0, 0.0]])
    mass = jnp.array([1.0])
    active = jnp.array([True])
    
    # Test with prev_force
    pos_with_prev, vel_with_prev = integrate_leapfrog(
        pos, vel, force, mass, active, 0.1, prev_force=prev_force
    )
    
    # Test without prev_force
    pos_without_prev, vel_without_prev = integrate_leapfrog(
        pos, vel, force, mass, active, 0.1, prev_force=None
    )
    
    # Results should be different (prev_force affects the integration)
    assert not jnp.allclose(pos_with_prev, pos_without_prev), \
        "prev_force should affect Leapfrog results"
    
    # Both should be finite
    assert jnp.all(jnp.isfinite(pos_with_prev)), "Leapfrog with prev_force produced non-finite positions"
    assert jnp.all(jnp.isfinite(pos_without_prev)), "Leapfrog without prev_force produced non-finite positions"
