import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from environment.topology_neighbors import (
    compute_topology_offset,
    generate_neighbor_pairs
)
from physics_utils import compute_gravity_forces


# =============================================================================
# 1. Test Flat Topology Offsets
# =============================================================================
def test_flat_topology_offsets_basic():
    """Test that flat topology returns standard Euclidean offsets."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0  # FLAT
    )
    
    pos_i = jnp.array([1.0, 2.0])
    pos_j = jnp.array([4.0, 6.0])
    
    offset = compute_topology_offset(pos_i, pos_j, config)
    expected = jnp.array([3.0, 4.0])
    
    assert jnp.allclose(offset, expected), f"Expected {expected}, got {offset}"


# =============================================================================
# 2. Test Torus Wraparound Offsets
# =============================================================================
def test_torus_wraparound_offsets():
    """Test that torus topology applies minimum image convention."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=1,  # TORUS
        torus_size=10.0
    )
    
    # Test wraparound: particles at opposite edges should be closer through wrap
    pos_i = jnp.array([1.0, 1.0])
    pos_j = jnp.array([9.0, 9.0])
    
    offset = compute_topology_offset(pos_i, pos_j, config)
    
    # Minimum image: should wrap to [-5, 5] range
    # dx = [8, 8] -> round(8/10)*10 = [10, 10] -> dx = [8-10, 8-10] = [-2, -2]
    expected = jnp.array([-2.0, -2.0])
    
    assert jnp.allclose(offset, expected, atol=1e-5), f"Expected {expected}, got {offset}"


# =============================================================================
# 3. Test Sphere Offset Directionality
# =============================================================================
def test_sphere_offset_directionality():
    """Test that sphere topology returns great-circle tangent offsets."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        dim=3,
        topology_type=2  # SPHERE
    )
    
    # Two points on sphere at (R, 0, 0) and (0, R, 0)
    R = 10.0
    pos_i = jnp.array([R, 0.0, 0.0])
    pos_j = jnp.array([0.0, R, 0.0])
    
    offset = compute_topology_offset(pos_i, pos_j, config)
    
    # Offset should be tangent to sphere at pos_i
    # Should be perpendicular to pos_i (dot product = 0)
    dot_product = jnp.dot(offset, pos_i)
    assert abs(dot_product) < 1e-5, f"Offset not tangent to sphere: dot={dot_product}"
    
    # Magnitude should be arc length: R * theta
    # Angle between (R,0,0) and (0,R,0) is 90° = π/2
    expected_magnitude = R * (jnp.pi / 2)
    actual_magnitude = jnp.linalg.norm(offset)
    assert jnp.allclose(actual_magnitude, expected_magnitude, rtol=0.01), \
        f"Expected magnitude {expected_magnitude}, got {actual_magnitude}"


# =============================================================================
# 4. Test Bubble Offset Curvature Behavior
# =============================================================================
def test_bubble_offset_curvature_behavior():
    """Test that bubble topology returns Euclidean offset (curvature in distance)."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=3,  # BUBBLE
        bubble_radius=10.0,
        bubble_curvature=0.1
    )
    
    pos_i = jnp.array([1.0, 2.0])
    pos_j = jnp.array([4.0, 6.0])
    
    offset = compute_topology_offset(pos_i, pos_j, config)
    expected = jnp.array([3.0, 4.0])  # Euclidean offset
    
    assert jnp.allclose(offset, expected), f"Expected {expected}, got {offset}"


# =============================================================================
# 5. Test Symmetric Pairs
# =============================================================================
def test_symmetric_pairs():
    """Test that generate_neighbor_pairs returns both (i,j) and (j,i)."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0  # FLAT
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    active_mask = jnp.array([True, True, True])
    
    pairs = list(generate_neighbor_pairs(positions, active_mask, config))
    
    # Extract (i, j) pairs without offsets
    pair_indices = [(i, j) for i, j, _ in pairs]
    
    # Check that both (0, 1) and (1, 0) exist
    assert (0, 1) in pair_indices
    assert (1, 0) in pair_indices
    
    # Check that both (0, 2) and (2, 0) exist
    assert (0, 2) in pair_indices
    assert (2, 0) in pair_indices


# =============================================================================
# 6. Test No Self-Pairs
# =============================================================================
def test_no_self_pairs():
    """Test that generate_neighbor_pairs skips i == j."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0  # FLAT
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    active_mask = jnp.array([True, True, True])
    
    pairs = list(generate_neighbor_pairs(positions, active_mask, config))
    
    # Check that no self-pairs (i, i) exist
    for i, j, _ in pairs:
        assert i != j, f"Found self-pair: ({i}, {i})"


# =============================================================================
# 7. Test Active Mask Respected
# =============================================================================
def test_active_mask_respected():
    """Test that generate_neighbor_pairs only includes active particles."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0  # FLAT
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    active_mask = jnp.array([True, False, True])  # Particle 1 is inactive
    
    pairs = list(generate_neighbor_pairs(positions, active_mask, config))
    
    # Check that no pairs involve particle 1
    for i, j, _ in pairs:
        assert i != 1, f"Found pair with inactive particle: ({i}, {j})"
        assert j != 1, f"Found pair with inactive particle: ({i}, {j})"
    
    # Check that only (0, 2) and (2, 0) exist
    pair_indices = [(i, j) for i, j, _ in pairs]
    assert len(pair_indices) == 2
    assert (0, 2) in pair_indices
    assert (2, 0) in pair_indices


# =============================================================================
# 8. Test Neighbor Engine Flag Off Restores Original Behavior
# =============================================================================
def test_neighbor_engine_flag_off_restores_original_behavior():
    """Test that enable_neighbor_engine=False uses legacy code path."""
    # Note: enable_neighbor_engine defaults to False due to JIT incompatibility
    # This test verifies the neighbor engine produces identical results when enabled
    config_legacy = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,  # FLAT
        enable_neighbor_engine=False  # Legacy (default)
    )
    
    config_new = config_legacy.replace(enable_neighbor_engine=True)  # Enable new engine
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0])
    active = jnp.array([True, True, True])
    
    # Compute forces with legacy engine (default)
    forces_legacy = compute_gravity_forces(positions, masses, active, config_legacy)
    
    # Compute forces with new engine (explicitly enabled, non-JIT)
    forces_new = compute_gravity_forces(positions, masses, active, config_new)
    
    # Forces should be identical
    assert jnp.allclose(forces_new, forces_legacy, atol=1e-6), \
        f"Forces differ:\nNew: {forces_new}\nLegacy: {forces_legacy}"
