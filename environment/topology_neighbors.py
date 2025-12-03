"""
Topology-Aware Neighbor Query System (PS2.2 + PS2.3)

This module provides a centralized neighbor engine that:
1. Computes topology-correct offset vectors between particles
2. Generates neighbor pairs for force calculations
3. Implements spatial partitioning for efficient neighbor lookup

TOPOLOGIES SUPPORTED:
- FLAT (0): Standard Euclidean geometry
- TORUS (1): Periodic boundaries with minimum image convention
- SPHERE (2): Great-circle geodesic on sphere surface
- BUBBLE (3): Curved radial metric in embedded space
"""

import jax.numpy as jnp


def compute_topology_offset(pos_i, pos_j, config):
    """
    Compute separation vector from pos_i to pos_j respecting topology.
    
    This is the core function that handles topology-aware offset calculations.
    It does NOT normalize or modify input positions, only computes the offset vector.
    
    Args:
        pos_i: Position of first particle (shape: [dim])
        pos_j: Position of second particle (shape: [dim])
        config: UniverseConfig containing topology information
        
    Returns:
        Offset vector Δx from pos_i to pos_j (shape: [dim])
    """
    topology_type = getattr(config, 'topology_type', 0)
    
    # FLAT topology: Standard Euclidean offset
    if topology_type == 0:
        return pos_j - pos_i
    
    # TORUS topology: Minimum image convention
    elif topology_type == 1:
        # Periodic box size
        L = getattr(config, 'torus_size', None)
        if L is None:
            L = config.radius * 2.0
        
        # Compute raw offset
        dx = pos_j - pos_i
        
        # Apply minimum image: wrap to [-L/2, L/2]
        dx = dx - jnp.round(dx / L) * L
        
        return dx
    
    # SPHERE topology: Tangent-plane offset along great circle
    elif topology_type == 2:
        R = config.radius
        
        # Check for invalid radius
        if R <= 0:
            print("[PS2.2 WARNING] Invalid topology parameters: sphere radius <= 0")
            return pos_j - pos_i
        
        # Normalize to unit sphere
        u = pos_i / R
        v = pos_j / R
        
        # Compute angle between points
        cosθ = jnp.clip(jnp.dot(u, v), -1.0, 1.0)
        θ = jnp.arccos(cosθ)
        
        # Small angle: use direct difference as fallback
        small_eps = 1e-8
        if θ < small_eps:
            return pos_j - pos_i
        
        # Compute great-circle axis and tangent direction
        axis = jnp.cross(u, v)
        axis_norm = jnp.linalg.norm(axis)
        
        # Degenerate case (antipodal or coincident): return any perpendicular
        if axis_norm < small_eps:
            # Find arbitrary perpendicular vector
            if jnp.abs(u[0]) < 0.9:
                perp = jnp.array([1.0, 0.0, 0.0])
            else:
                perp = jnp.array([0.0, 1.0, 0.0])
            tangent = perp - jnp.dot(perp, u) * u
            tangent = tangent / jnp.linalg.norm(tangent)
        else:
            # Tangent direction at pos_i pointing toward pos_j
            tangent = jnp.cross(axis / axis_norm, u)
            tangent = tangent / jnp.linalg.norm(tangent)
        
        # Offset is tangent direction scaled by arc distance
        offset = tangent * (R * θ)
        
        return offset
    
    # BUBBLE topology: Euclidean offset (curvature handled in distance metric)
    elif topology_type == 3:
        bubble_radius = getattr(config, 'bubble_radius', 10.0)
        if bubble_radius <= 0:
            print("[PS2.2 WARNING] Invalid topology parameters: bubble_radius <= 0")
        
        return pos_j - pos_i
    
    # Fallback: Euclidean
    else:
        return pos_j - pos_i


def generate_neighbor_pairs(positions, active_mask, config):
    """
    Generate all neighbor pairs (i, j) with topology-aware offsets.
    
    This is an O(N²) implementation suitable for PS2.2. PS2.3 will replace
    this with spatial partitioning for better performance.
    
    Args:
        positions: Array of shape (N, dim) - particle positions
        active_mask: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig containing topology information
        
    Yields:
        Tuples of (i, j, offset_ij) where:
        - i: Index of first particle
        - j: Index of second particle
        - offset_ij: Topology-aware offset vector from i to j
    
    Notes:
        - Skips pairs where i == j
        - Returns symmetric pairs: both (i,j) and (j,i)
        - Only returns pairs where both particles are active
    """
    N = positions.shape[0]
    
    for i in range(N):
        if not active_mask[i]:
            continue
            
        for j in range(N):
            if not active_mask[j]:
                continue
                
            # Skip self-pairs
            if i == j:
                continue
            
            # Compute topology-aware offset
            offset = compute_topology_offset(positions[i], positions[j], config)
            
            yield i, j, offset


# =============================================================================
# PS2.3 SPATIAL PARTITIONING FUNCTIONS
# =============================================================================

def start_spatial_partition(positions, active_mask, config):
    """
    Initialize spatial partitioning grid for neighbor queries.
    
    Args:
        positions: Array of shape (N, dim) - particle positions
        active_mask: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig instance
        
    Returns:
        Partition structure or None on failure
    """
    from environment.spatial_partition import start_spatial_partition as build_partition
    return build_partition(positions, active_mask, config)


def get_partition_neighbors(partition, pos, active, config):
    """
    Get neighbors using spatial partitioning for O(N log N) or O(N) performance.
    
    This is now implemented via generate_partitioned_pairs().
    """
    return generate_partitioned_pairs(pos, active, partition, config)


def spatial_partition_debug_info(partition):
    """
    Return diagnostic information about spatial partition state.
    
    Args:
        partition: Partition structure from start_spatial_partition()
        
    Returns:
        dict with diagnostic info
    """
    if partition is None:
        return {'status': 'failed', 'num_cells': 0, 'num_particles': 0}
    
    grid = partition.get('grid', {})
    num_cells = len(grid)
    num_particles = sum(len(indices) for indices in grid.values())
    cell_size = partition.get('cell_size', 0.0)
    
    return {
        'status': 'active',
        'num_cells': num_cells,
        'num_particles': num_particles,
        'cell_size': cell_size,
        'avg_particles_per_cell': num_particles / num_cells if num_cells > 0 else 0
    }


def generate_partitioned_pairs(positions, active_mask, partition, config):
    """
    Generate neighbor pairs using spatial partitioning.
    
    Uses PS2.2's compute_topology_offset() but limits search space to
    neighboring cells only, reducing complexity from O(N²) to O(N).
    
    Args:
        positions: Array of shape (N, dim) - particle positions
        active_mask: Array of shape (N,) -boolean mask for active particles
        partition: Partition structure from start_spatial_partition()
        config: UniverseConfig containing topology information
        
    Yields:
        Tuples of (i, j, offset_ij) where:
        - i: Index of first particle
        - j: Index of second particle
        - offset_ij: Topology-aware offset vector from i to j
    
    Notes:
        - Skips pairs where i == j
        - Returns symmetric pairs: both (i,j) and (j,i)
        - Only returns pairs where both particles are active
        - Deterministic iteration order (sorted cells, sorted indices)
    """
    if partition is None:
        # Fallback to non-partitioned pairs
        print("[PS2.3 WARNING] Partition is None, falling back to non-partitioned pairs")
        yield from generate_neighbor_pairs(positions, active_mask, config)
        return
  
    grid = partition['grid']
    cell_size = partition['cell_size']
    num_cells = partition.get('num_cells')
    topology_type = partition.get('topology_type', 0)
    
    # Import neighbor cell lookup
    from environment.spatial_partition import get_neighbor_cells, compute_cell_index
    
    # Convert to numpy for indexing
    import numpy as np
    positions_np = np.array(positions)
    
    # Track pairs we've already yielded to avoid duplicates
    yielded_pairs = set()
    
    # Iterate over all cells in sorted order (determinism)
    for cell_idx in sorted(grid.keys()):
        particles_in_cell = grid[cell_idx]
        
        # Get neighboring cells
        neighbor_cells = get_neighbor_cells(cell_idx, topology_type, num_cells)
        
        # For each particle in this cell
        for i in particles_in_cell:
            if not active_mask[i]:
                continue
            
            # Check all particles in neighboring cells
            for neighbor_cell in neighbor_cells:
                if neighbor_cell not in grid:
                    continue
                
                for j in grid[neighbor_cell]:
                    if not active_mask[j]:
                        continue
                    
                    # Skip self-pairs
                    if i == j:
                        continue
                    
                    # Check if we've already yielded this pair
                    # Note: We want BOTH (i,j) and (j,i) for symmetric forces
                    pair_key = (i, j)
                    if pair_key in yielded_pairs:
                        continue
                    
                    yielded_pairs.add(pair_key)
                    
                    # Compute topology-aware offset
                    offset = compute_topology_offset(positions_np[i], positions_np[j], config)
                    
                    yield i, j, offset
