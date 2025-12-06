from __future__ import annotations
import jax.numpy as jnp
from environment.topology_math import compute_distance

def compute_diagnostics(state, config):
    """
    PS2.4 placeholder.
    This file will be expanded in later steps.
    """
    mask = (state.entity_active == 1)

    pos  = state.entity_pos[mask]
    vel  = state.entity_vel[mask]
    mass = state.entity_mass[mask]

    speed2 = jnp.sum(vel * vel, axis=-1)
    KE = 0.5 * jnp.sum(mass * speed2)

    state = state.replace(kinetic_energy=KE)

    # --- Topology-aware Potential Energy (PE) ---

    # Use masked active positions and masses defined earlier:
    # pos: (N, dim)
    # mass: (N,)

    N = pos.shape[0]

    # If fewer than 2 active bodies, PE = 0
    # --- JAX-safe Potential Energy computation block ---

    import jax

    # Unique unordered index pairs (i < j)
    idx_i, idx_j = jnp.triu_indices(N, k=1)

    # Define PE computation for the non-empty case
    def _compute_pe(_):
        dist = compute_distance(
            pos[idx_i], pos[idx_j],
            config.topology_type, config
        )
        dist = jnp.maximum(dist, 1e-12)

        m_i = mass[idx_i]
        m_j = mass[idx_j]
        G   = config.G
        return -jnp.sum(G * m_i * m_j / dist)

    # Use JAX conditional to produce either 0.0 or computed PE
    PE = jax.lax.cond(
        idx_i.size == 0,
        lambda _: jnp.array(0.0),
        _compute_pe,
        operand=None
    )

    state = state.replace(potential_energy=PE)

    # END OF PE ADDITION

    # === PS2.4 Diagnostics: Total Energy, Initial Baseline, Drift ===

    # KE and PE must already exist at this point:
    #   KE: scalar kinetic energy
    #   PE: scalar potential energy

    # Total mechanical energy
    total_energy = KE + PE

    # Establish the baseline initial_energy at step 0
    initial_energy = jnp.where(
        state.step_count == 0,
        total_energy,
        state.initial_energy
    )

    # Compute drift relative to initial baseline
    energy_drift = (total_energy - initial_energy) / (initial_energy + 1e-12)

    # Store new diagnostic values
    state = state.replace(
        total_energy=total_energy,
        initial_energy=initial_energy,
        energy_drift=energy_drift,
    )

    # === END TOTAL ENERGY / DRIFT ADDITION ===

    # === PS2.4 Diagnostics: Momentum and Center of Mass ===

    # Momentum: sum_i (m_i * v_i)
    # mass: (N,)
    # vel:  (N, dim)
    momentum = jnp.sum(mass[:, None] * vel, axis=0)

    # Center of mass: sum_i (m_i * x_i) / sum_i m_i
    # pos: (N, dim)
    total_mass = jnp.sum(mass)
    center_of_mass = jnp.sum(mass[:, None] * pos, axis=0) / (total_mass + 1e-12)

    # Store momentum and COM in state
    state = state.replace(
        momentum=momentum,
        center_of_mass=center_of_mass,
    )

    # === END MOMENTUM / COM ADDITION ===

    return state
