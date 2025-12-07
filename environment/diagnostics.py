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
    # Drift Stabilization: Handle initial_energy close to zero (avoid divide-by-zero)
    initial_safe = jnp.where(jnp.abs(initial_energy) < 1e-12, 1.0, initial_energy)
    energy_drift = (total_energy - initial_energy) / initial_safe

    # === PS2.4 Diagnostics: Momentum and Center of Mass ===

    # Momentum: sum_i (m_i * v_i)
    momentum = jnp.sum(mass[:, None] * vel, axis=0)

    # Center of mass: sum_i (m_i * x_i) / sum_i m_i
    total_mass = jnp.sum(mass)
    center_of_mass = jnp.sum(mass[:, None] * pos, axis=0) / (total_mass + 1e-12)

    # === Final Scalar Normalization & NaN Safety ===
    
    # Ensure scalars are truly scalar shape () for JIT consistency
    KE = KE.reshape(())
    PE = PE.reshape(())
    total_energy = total_energy.reshape(())
    energy_drift = energy_drift.reshape(())
    initial_energy = initial_energy.reshape(())
    dt_actual = state.dt_actual.reshape(()) # Preserve existing dt_actual

    # Apply NaN Safety (replace NaNs with 0.0)
    KE = jnp.nan_to_num(KE)
    PE = jnp.nan_to_num(PE)
    total_energy = jnp.nan_to_num(total_energy)
    energy_drift = jnp.nan_to_num(energy_drift)
    momentum = jnp.nan_to_num(momentum)
    center_of_mass = jnp.nan_to_num(center_of_mass)

    # Store new diagnostic values
    state = state.replace(
        kinetic_energy=KE,
        potential_energy=PE,
        total_energy=total_energy,
        initial_energy=initial_energy,
        energy_drift=energy_drift,
        momentum=momentum,
        center_of_mass=center_of_mass,
        dt_actual=dt_actual, 
    )

    return state
