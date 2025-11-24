"""
Entity management for CosmoSim.

This module handles the lifecycle of entities (particles, massive bodies)
in the simulation. It provides JAX-compatible functions to spawn and
despawn entities within the fixed-size arrays of UniverseState.
"""

import jax
import jax.numpy as jnp
from state import UniverseState


def spawn_entity(state: UniverseState, position: jnp.ndarray, velocity: jnp.ndarray, 
                 mass: float, ent_type: int) -> UniverseState:
    """Spawn a new entity in the first available slot.
    
    Args:
        state: Current universe state
        position: Position vector (2,)
        velocity: Velocity vector (2,)
        mass: Mass scalar
        ent_type: Entity type integer
        
    Returns:
        Updated state with new entity, or original state if full.
    """
    # Find the first inactive index
    # argmin on boolean array returns index of first False (0)
    # If all are True (1), it returns 0. So we must check if any are False.
    idx = jnp.argmin(state.entity_active)
    is_full = jnp.all(state.entity_active)
    
    # Create the updated state eagerly (JAX lazy evaluation handles this efficiently)
    # We use .at[idx].set() for functional updates
    new_active = state.entity_active.at[idx].set(True)
    new_pos = state.entity_pos.at[idx].set(position)
    new_vel = state.entity_vel.at[idx].set(velocity)
    new_mass = state.entity_mass.at[idx].set(mass)
    new_type = state.entity_type.at[idx].set(ent_type)
    
    new_state = state.replace(
        entity_active=new_active,
        entity_pos=new_pos,
        entity_vel=new_vel,
        entity_mass=new_mass,
        entity_type=new_type
    )
    
    # Return new_state if not full, otherwise return original state
    return jax.lax.cond(
        is_full,
        lambda _: state,
        lambda _: new_state,
        operand=None
    )


def despawn_entity(state: UniverseState, index: int) -> UniverseState:
    """Despawn an entity by marking it as inactive.
    
    Args:
        state: Current universe state
        index: Index of the entity to despawn
        
    Returns:
        Updated state with entity_active[index] = False
    """
    # Simply set active to False. Data remains but is ignored by physics.
    new_active = state.entity_active.at[index].set(False)
    return state.replace(entity_active=new_active)
