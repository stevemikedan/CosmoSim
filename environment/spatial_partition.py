"""
Spatial Partitioning Engine (PS2.3)

This module provides efficient neighbor lookup using spatial hashing.
Reduces complexity from O(N²) to O(N) for uniformly distributed particles.

DESIGN:
- Cell-based spatial hash grid
- Deterministic iteration (sorted keys, sorted indices)
- Topology-aware (flat, torus, sphere, bubble)
- Python data structures (no JAX compilation)
"""

import jax.numpy as jnp
import numpy as np
from collections import defaultdict


def compute_cell_index(pos, cell_size, config):
    """
    Convert position to integer cell index.
    
    Args:
        pos: Position vector (dim,)
        cell_size: Size of each cell
        config: UniverseConfig containing topology information
        
    Returns:
        Tuple of cell indices (cx, cy, cz) or (cx, cy) for 2D
    """
    topology_type = getattr(config, 'topology_type', 0)
    dim = len(pos)
    
    # Convert JAX array to numpy for integer operations
    pos_np = np.array(pos)
    
    # Compute raw cell indices via floor division
    cell_coords = np.floor(pos_np / cell_size).astype(int)
    
    # TORUS topology: wrap cell indices
    if topology_type == 1:
        # Get torus size
        L = getattr(config, 'torus_size', None)
        if L is None:
            L = config.radius * 2.0
            
        # Number of cells along each axis
        num_cells = int(np.ceil(L / cell_size))
        
        # Wrap indices modulo num_cells
        cell_coords = cell_coords % num_cells
    
    # SPHERE/BUBBLE topology: Use tangent approximation (no wrapping)
    # For PS2.3, we treat sphere/bubble as locally flat
    # This is acceptable as particles near each other are in local tangent space
    
    return tuple(cell_coords)


def get_neighbor_cells(cell_index, topology_type=0, num_cells_per_axis=None):
    """
    Get all 27 neighboring cells (3x3x3 stencil) around the given cell.
    
    Args:
        cell_index: Tuple of cell coordinates (cx, cy, cz) or (cx, cy)
        topology_type: Topology type (0=flat, 1=torus, etc.)
        num_cells_per_axis: For torus, number of cells per axis (for wrapping)
        
    Returns:
        List of neighboring cell indices
    """
    dim = len(cell_index)
    neighbors = []
    
    # Generate all offsets in 3x3x3 (or 3x3 for 2D) stencil
    if dim == 2:
        offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
    else:  # dim == 3
        offsets = [(dx, dy, dz) 
                   for dx in [-1, 0, 1] 
                   for dy in [-1, 0, 1] 
                   for dz in [-1, 0, 1]]
    
    for offset in offsets:
        neighbor = tuple(cell_index[i] + offset[i] for i in range(dim))
        
        # TORUS topology: wrap neighbor indices
        if topology_type == 1 and num_cells_per_axis is not None:
            neighbor = tuple(n % num_cells_per_axis for n in neighbor)
        
        neighbors.append(neighbor)
    
    return neighbors


def build_spatial_grid(positions, active_mask, config):
    """
    Build spatial hash grid for efficient neighbor lookup.
    
    Args:
        positions: Array of shape (N, dim) - particle positions
        active_mask: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig containing partition parameters
        
    Returns:
        Dictionary with:
        - 'grid': dict mapping cell_index -> sorted list of particle indices
        - 'cell_size': float - size of each cell
        - 'num_cells': int - number of cells per axis (for torus)
    """
    # Get partition parameters
    cell_size = getattr(config, 'spatial_cell_size', None)
    if cell_size is None:
        # Auto-compute: radius / 10
        cell_size = config.radius / 10.0
    
    topology_type = getattr(config, 'topology_type', 0)
    
    # Compute num_cells for torus
    num_cells = None
    if topology_type == 1:
        L = getattr(config, 'torus_size', None)
        if L is None:
            L = config.radius * 2.0
        num_cells = int(np.ceil(L / cell_size))
    
    # Build grid: cell_index -> list of particle indices
    grid = defaultdict(list)
    
    # Convert to numpy for easier indexing
    positions_np = np.array(positions)
    active_np = np.array(active_mask)
    
    for i in range(len(positions_np)):
        if not active_np[i]:
            continue
        
        # Compute cell index for this particle
        try:
            cell_idx = compute_cell_index(positions_np[i], cell_size, config)
            grid[cell_idx].append(i)
        except Exception as e:
            # Fallback: skip this particle if cell computation fails
            print(f"[PS2.3 WARNING] Failed to compute cell for particle {i}: {e}")
            continue
    
    # Sort particle indices within each cell for determinism
    for cell_idx in grid:
        grid[cell_idx].sort()
    
    return {
        'grid': dict(grid),  # Convert defaultdict to regular dict
        'cell_size': cell_size,
        'num_cells': num_cells,
        'topology_type': topology_type
    }


def start_spatial_partition(positions, active_mask, config):
    """
    PS2.2 placeholder implementation (now functional in PS2.3).
    
    Initialize spatial partitioning grid for neighbor queries.
    
    Args:
        positions: Array of shape (N, dim) - particle positions
        active_mask: Array of shape (N,) - boolean mask for active particles
        config: UniverseConfig instance
        
    Returns:
        Partition structure (dict) or None on failure
    """
    try:
        return build_spatial_grid(positions, active_mask, config)
    except Exception as e:
        print(f"[PS2.3 WARNING] Spatial partition failed: {e}")
        return None
