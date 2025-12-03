import pytest
import jax.numpy as jnp
import numpy as np
from state import UniverseConfig
from environment.spatial_partition import (
    compute_cell_index,
    build_spatial_grid,
    get_neighbor_cells
)
from environment.topology_neighbors import (
    start_spatial_partition,
    generate_partitioned_pairs,
    generate_neighbor_pairs,
    spatial_partition_debug_info
)
from physics_utils import compute_gravity_forces


# =============================================================================
# 1. Test Partition Basic Flat
# =============================================================================
def test_partition_basic_flat():
    """Test basic spatial partitioning in flat topology."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,  # FLAT
        spatial_cell_size=5.0
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [3.0, 3.0],
        [10.0, 10.0],
        [12.0, 12.0]
    ])
    active_mask = jnp.array([True, True, True, True])
    
    partition = start_spatial_partition(positions, active_mask, config)
    
    assert partition is not None
    assert 'grid' in partition
    assert 'cell_size' in partition
    assert partition['cell_size'] == 5.0
    
    # Verify particles are assigned to cells
    grid = partition['grid']
    assert len(grid) > 0
    
    # Particles 0 and 1 should be in the same or neighboring cells
    # Particles 2 and 3 should be in the same or neighboring cells


# =============================================================================
# 2. Test Partition Respects Active Mask
# =============================================================================
def test_partition_respects_active_mask():
    """Test that inactive particles are excluded from partition."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        spatial_cell_size=5.0
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [3.0, 3.0],
        [10.0, 10.0]
    ])
    active_mask = jnp.array([True, False, True])  # Particle 1 is inactive
    
    partition = start_spatial_partition(positions, active_mask, config)
    grid = partition['grid']
    
    # Count total particles in grid
    total_particles = sum(len(indices) for indices in grid.values())
    assert total_particles == 2  # Only active particles (0 and 2)
    
    # Verify particle 1 is not in the grid
    all_indices = [idx for indices in grid.values() for idx in indices]
    assert 1 not in all_indices


# =============================================================================
# 3. Test Neighbor Cell Wrapping Torus
# =============================================================================
def test_neighbor_cell_wrapping_torus():
    """Test that neighbor cells wrap correctly in torus topology."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=1,  # TORUS
        torus_size=20.0,
        spatial_cell_size=5.0
    )
    
    # Test cell index wrapping
    pos_near_edge = np.array([19.0, 19.0])
    cell_idx = compute_cell_index(pos_near_edge, 5.0, config)
    
    # Get neighbor cells (should wrap around)
    num_cells = int(np.ceil(20.0 / 5.0))  # 4 cells per axis
    neighbors = get_neighbor_cells(cell_idx, topology_type=1, num_cells_per_axis=num_cells)
    
    # Verify wrapping occurs
    assert len(neighbors) == 9  # 3x3 for 2D
    
    # Check that wrapped indices are within bounds [0, num_cells)
    for neighbor in neighbors:
        for coord in neighbor:
            assert 0 <= coord < num_cells


# =============================================================================
# 4. Test Partition vs Non-Partition Equivalence (Small N)
# =============================================================================
def test_partition_vs_nonpartition_equivalence_smallN():
    """Test that partitioned and non-partitioned approaches give identical results."""
    config_partition = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,  # FLAT
        enable_spatial_partition=True,
        spatial_cell_size=3.0
    )
    
    config_vectorized = config_partition.replace(enable_spatial_partition=False)
    
    positions = jnp.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [5.0, 5.0],
        [7.0, 7.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0])
    active = jnp.array([True, True, True, True, True])
    
    # Compute forces with partition
    forces_partition = compute_gravity_forces(positions, masses, active, config_partition)
    
    # Compute forces with vectorized
    forces_vectorized = compute_gravity_forces(positions, masses, active, config_vectorized)
    
    # Forces should be identical (within floating point tolerance)
    assert jnp.allclose(forces_partition, forces_vectorized, atol=1e-6), \
        f"Forces differ:\nPartition: {forces_partition}\nVectorized: {forces_vectorized}"


# =============================================================================
# 5. Test Partition Determinism
# =============================================================================
def test_partition_determinism():
    """Test that partition produces deterministic results."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        spatial_cell_size=3.0
    )
    
    positions = jnp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [5.0, 5.0],
        [8.0, 8.0],
        [10.0, 10.0]
    ])
    active_mask = jnp.array([True, True, True, True, True])
    
    # Build partition twice
    partition1 = start_spatial_partition(positions, active_mask, config)
    partition2 = start_spatial_partition(positions, active_mask, config)
    
    # Grids should be identical
    assert partition1['grid'] == partition2['grid']
    
    # Collect pairs from both partitions
    pairs1 = list(generate_partitioned_pairs(positions, active_mask, partition1, config))
    pairs2 = list(generate_partitioned_pairs(positions, active_mask, partition2, config))
    
    # Extract pair indices (ignoring offsets)
    pair_indices1 = [(i, j) for i, j, _ in pairs1]
    pair_indices2 = [(i, j) for i, j, _ in pairs2]
    
    # Should be identical
    assert pair_indices1 == pair_indices2


# =============================================================================
# 6. Test Partition Flag Off Restores PS2.2 Behavior
# =============================================================================
def test_partition_flag_off_restores_ps2_2_behavior():
    """Test that disabling partition uses vectorized fallback."""
    config_off = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=3,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        enable_spatial_partition=False  # Disabled
    )
    
    config_on = config_off.replace(enable_spatial_partition=True)
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0])
    active = jnp.array([True, True, True])
    
    # Both should give same forces
    forces_off = compute_gravity_forces(positions, masses, active, config_off)
    forces_on = compute_gravity_forces(positions, masses, active, config_on)
    
    assert jnp.allclose(forces_off, forces_on, atol=1e-6), \
        f"Forces differ:\nOff: {forces_off}\nOn: {forces_on}"


# =============================================================================
# 7. Test Partition No Duplicate Pairs
# =============================================================================
def test_partition_no_duplicate_pairs():
    """Test that partition doesn't generate duplicate pairs."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        spatial_cell_size=2.0
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    active_mask = jnp.array([True, True, True, True])
    
    partition = start_spatial_partition(positions, active_mask, config)
    pairs = list(generate_partitioned_pairs(positions, active_mask, partition, config))
    
    # Extract pair indices
    pair_indices = [(i, j) for i, j, _ in pairs]
    
    # Check for duplicates
    assert len(pair_indices) == len(set(pair_indices)), \
        f"Found duplicate pairs: {pair_indices}"


# =============================================================================
# 8. Test Partition No Self-Pairs
# =============================================================================
def test_partition_no_self_pairs():
    """Test that partition doesn't generate self-pairs (i, i)."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=5,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        spatial_cell_size=2.0
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0]
    ])
    active_mask = jnp.array([True, True, True])
    
    partition = start_spatial_partition(positions, active_mask, config)
    pairs = list(generate_partitioned_pairs(positions, active_mask, partition, config))
    
    # Check for self-pairs
    for i, j, _ in pairs:
        assert i != j, f"Found self-pair: ({i}, {i})"


# =============================================================================
# Additional Test: Partition Debug Info
# =============================================================================
def test_partition_debug_info():
    """Test spatial_partition_debug_info returns correct information."""
    config = UniverseConfig(
        physics_mode=0,
        radius=10.0,
        max_entities=10,
        max_nodes=1,
        dt=0.1,
        c=1.0,
        G=1.0,
        topology_type=0,
        spatial_cell_size=5.0
    )
    
    positions = jnp.array([
        [0.0, 0.0],
        [3.0, 3.0],
        [10.0, 10.0]
    ])
    active_mask = jnp.array([True, True, True])
    
    partition = start_spatial_partition(positions, active_mask, config)
    info = spatial_partition_debug_info(partition)
    
    assert info['status'] == 'active'
    assert info['num_particles'] == 3
    assert info['num_cells'] > 0
    assert info['cell_size'] == 5.0
    assert 'avg_particles_per_cell' in info
