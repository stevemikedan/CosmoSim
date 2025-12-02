"""
Tests for energy and momentum diagnostics (Phase PS1.3).

Verifies that:
- Kinetic energy is correct for known velocities
- Potential energy matches hand-calculated values
- Total energy is the sum of KE and PE
- Momentum is conserved (zero for stationary system)
- Center of mass is correct
- Diagnostics handle inactive particles correctly
"""

import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from physics_utils import (
    kinetic_energy,
    potential_energy,
    total_energy,
    momentum,
    center_of_mass
)


def test_kinetic_energy_simple():
    """Verify kinetic energy calculation for simple case."""
    # Two particles with mass 2.0 and velocity (1, 0, 0)
    vel = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = jnp.array([2.0, 2.0])
    active = jnp.array([True, True])
    
    # KE = 0.5 * m * v^2
    # Per particle: 0.5 * 2.0 * 1.0^2 = 1.0
    # Total: 2.0
    ke = kinetic_energy(vel, mass, active)
    
    assert jnp.isclose(ke, 2.0), f"Expected KE=2.0, got {ke}"


def test_kinetic_energy_inactive():
    """Verify inactive particles don't contribute to KE."""
    vel = jnp.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    mass = jnp.array([2.0, 2.0])
    active = jnp.array([True, False])  # Second particle inactive
    
    # Only first particle contributes: 1.0
    ke = kinetic_energy(vel, mass, active)
    
    assert jnp.isclose(ke, 1.0), f"Expected KE=1.0, got {ke}"


def test_potential_energy_two_body():
    """Verify potential energy matches hand calculation for 2 bodies."""
    # Two particles at distance r=2.0
    pos = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
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
        gravity_softening=0.0  # Zero softening for simple check
    )
    
    # U = -G * m1 * m2 / r
    # U = -1.0 * 1.0 * 1.0 / 2.0 = -0.5
    pe = potential_energy(pos, mass, active, config)
    
    assert jnp.isclose(pe, -0.5), f"Expected PE=-0.5, got {pe}"


def test_potential_energy_softened():
    """Verify potential energy with softening."""
    # Two particles at distance r=0 (overlapping)
    pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
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
        gravity_softening=1.0  # epsilon = 1.0
    )
    
    # U = -G * m1 * m2 / sqrt(r^2 + eps^2)
    # U = -1.0 * 1.0 * 1.0 / sqrt(0 + 1.0) = -1.0
    pe = potential_energy(pos, mass, active, config)
    
    assert jnp.isclose(pe, -1.0), f"Expected PE=-1.0, got {pe}"


def test_total_energy_sum():
    """Verify total energy is sum of KE and PE."""
    ke = 10.0
    pe = -5.0
    e = total_energy(ke, pe)
    assert e == 5.0


def test_momentum_stationary():
    """Verify momentum of stationary system is zero."""
    vel = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    p = momentum(vel, mass, active)
    
    assert jnp.allclose(p, 0.0), f"Expected zero momentum, got {p}"


def test_momentum_moving():
    """Verify momentum calculation for moving system."""
    # Particle 1: mass 1, vel (1, 0, 0) -> p=(1, 0, 0)
    # Particle 2: mass 2, vel (-0.5, 0, 0) -> p=(-1, 0, 0)
    # Total momentum should be zero
    vel = jnp.array([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    mass = jnp.array([1.0, 2.0])
    active = jnp.array([True, True])
    
    p = momentum(vel, mass, active)
    
    assert jnp.allclose(p, 0.0), f"Expected zero momentum, got {p}"


def test_center_of_mass_symmetric():
    """Verify center of mass for symmetric configuration."""
    # Two equal masses at (-1, 0, 0) and (1, 0, 0)
    pos = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    com = center_of_mass(pos, mass, active)
    
    assert jnp.allclose(com, 0.0), f"Expected COM at origin, got {com}"


def test_center_of_mass_weighted():
    """Verify center of mass for unequal masses."""
    # Mass 1 at origin, Mass 3 at (4, 0, 0)
    pos = jnp.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mass = jnp.array([1.0, 3.0])
    active = jnp.array([True, True])
    
    # COM = (1*0 + 3*4) / (1+3) = 12 / 4 = 3
    com = center_of_mass(pos, mass, active)
    
    expected = jnp.array([3.0, 0.0, 0.0])
    assert jnp.allclose(com, expected), f"Expected COM={expected}, got {com}"


def test_diagnostics_no_nans():
    """Verify diagnostics don't produce NaNs even with empty/zero inputs."""
    # Zero particles (all inactive)
    pos = jnp.array([[0.0, 0.0, 0.0]])
    vel = jnp.array([[0.0, 0.0, 0.0]])
    mass = jnp.array([1.0])
    active = jnp.array([False])
    
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=1,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0
    )
    
    ke = kinetic_energy(vel, mass, active)
    pe = potential_energy(pos, mass, active, config)
    p = momentum(vel, mass, active)
    com = center_of_mass(pos, mass, active)
    
    assert jnp.isfinite(ke)
    assert jnp.isfinite(pe)
    assert jnp.all(jnp.isfinite(p))
    assert jnp.all(jnp.isfinite(com))
