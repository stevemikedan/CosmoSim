"""Physics kernel implementations for CosmoSim."""

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState
from entities import spawn_entity, despawn_entity
from topology import enforce_boundaries, apply_topology
from physics.integrator import velocity_verlet
from physics.forces import compute_forces
from environment.topology_math import compute_distance

def compute_diagnostics(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """
    Compute and update diagnostics fields in the state.
    
    Calculates:
    - Kinetic Energy: 0.5 * m * v^2
    - Potential Energy: -0.5 * sum(G * m_i * m_j / r_ij)
    - Total Energy: KE + PE
    - Momentum: sum(m * v)
    - Center of Mass: sum(m * r) / sum(m)
    - Energy Drift: (E - E0) / E0 (requires storing E0, for now just store E)
    """
    pos = state.entity_pos
    vel = state.entity_vel
    mass = state.entity_mass
    active = state.entity_active
    
    # Mask inactive entities
    # active is boolean, cast to float for multiplication
    active_f = active.astype(jnp.float32)
    mass_active = mass * active_f
    
    # 1. Kinetic Energy
    # KE = 0.5 * sum(m * |v|^2)
    v_sq = jnp.sum(vel**2, axis=-1)
    ke = 0.5 * jnp.sum(mass_active * v_sq)
    
    # 2. Potential Energy
    # PE = -0.5 * sum_{i!=j} (G * m_i * m_j / r_ij)
    # We need pairwise distances again.
    # Note: This duplicates distance calc from forces. 
    # In a highly optimized engine, we'd return PE from compute_forces or share the dist matrix.
    # For now, recomputing is cleaner for modularity.
    # Note: Using top-level import now.
    # from environment.topology_math import compute_distance
    
    p1 = pos[:, None, :]
    p2 = pos[None, :, :]
    dist = compute_distance(p1, p2, config.topology_type, config)
    
    # Avoid division by zero
    eps = getattr(config, 'gravity_softening', 1e-12)
    inv_dist = 1.0 / (dist + eps)
    
    # Pairwise potential
    # U_ij = -G * m_i * m_j / r_ij
    m_i = mass_active[:, None]
    m_j = mass_active[None, :]
    
    pot_matrix = -config.G * m_i * m_j * inv_dist
    
    # Mask self-interactions (diagonal)
    # dist is 0 on diagonal, but eps handles singularity.
    # However, self-potential should be 0.
    # We can mask diagonal.
    eye_mask = jnp.eye(config.max_entities, dtype=bool)
    pot_matrix = jnp.where(eye_mask, 0.0, pot_matrix)
    
    # Sum and divide by 2 (double counting)
    pe = 0.5 * jnp.sum(pot_matrix)
    
    # 3. Total Energy
    total_e = ke + pe
    
    # 4. Momentum
    # P = sum(m * v)
    mom = jnp.sum(mass_active[:, None] * vel, axis=0)
    
    # 5. Center of Mass
    # R_cm = sum(m * r) / sum(m)
    total_mass = jnp.sum(mass_active) + 1e-12
    com = jnp.sum(mass_active[:, None] * pos, axis=0) / total_mass
    
    # 6. Energy Drift
    # Update baseline if step 0
    # Requires state.initial_energy and state.step_count from state.py
    current_initial = state.initial_energy
    
    # If step_count is 0, we adopt total_e as baseline.
    initial_e = jnp.where(state.step_count == 0, total_e, current_initial)
    
    # Drift
    # Avoid div by zero
    denom = initial_e + 1e-12
    drift = (total_e - initial_e) / denom
    
    return state.replace(
        kinetic_energy=jnp.array(ke),
        potential_energy=jnp.array(pe),
        total_energy=jnp.array(total_e),
        energy_drift=jnp.array(drift),
        initial_energy=jnp.array(initial_e),
        momentum=mom,
        center_of_mass=com,
        dt_actual=jnp.array(config.dt)
    )

def step_simulation(state: UniverseState, config: UniverseConfig) -> UniverseState:
    """Execute one simulation timestep."""
    # 1. Update global time
    state = state.replace(time=state.time + config.dt)

    # Initial Force Computation (Step 0 Logic)
    # Ensure acceleration is initialized before the first integration step.
    # Note: Requires step_count and entity_acc in UniverseState (Step 3).
    # Using jax.lax.cond for JIT compatibility if needed, but python control flow for now 
    # as step_count is likely a tracer or scalar.
    # Assuming step_count is a JAX array, we should technically use where or cond, 
    # but for "Initialize" logic which happens once, 'if' might be tricky under JIT 
    # unless step_count is static. 
    # However, for now, we follow the requested structure.
    # We will use a JAX-safe conditional update if possible, or standard python if not JIT-ed.
    # Given requirements "If state.acc is None or state.step == 0", we implement:
    # (Since we can't check 'is None' on JAX array easily in JIT, we rely on step check).
    
    # We use jax.lax.cond to be safe for JIT, or jnp.where.
    # entity_acc update:
    initial_acc = compute_forces(state, config)
    
    # If step_count == 0, use initial_acc, else keep existing entity_acc.
    # As 'state.entity_acc' might not exist yet, this line assumes Step 3 fixes state.py.
    # But for this step "Modify ONLY kernel.py", we write the logic.
    
    # To properly implement "If ... compute", we only want to compute it if needed?
    # Actually, compute_forces is expensive. We should avoid it if step > 0.
    # But inside JIT, we might have to compute it or use lax.cond.
    # The prompt implies a Python-side check? "If state.acc is None".
    # But 'state' is a JAX PyTree.
    # Let's write it as a direct conditional assignment knowing it will be refined.
    
    def _init_forces(s):
        return s.replace(entity_acc=compute_forces(s, config))
        
    def _no_op(s):
        return s
        
    # We assume 'step_count' is available.
    pred = (state.step_count == 0)
    state = jax.lax.cond(pred, _init_forces, _no_op, state)

    # 2. Apply physics update (Integration + Wrapping)
    # Replaces legacy dispatch_physics/update_vector_physics
    state = velocity_verlet(state, config, compute_forces)
    
    # Apply topology boundaries (wrapping)
    new_pos = state.entity_pos
    new_vel = state.entity_vel
    final_pos, final_vel = apply_topology(new_pos, new_vel, config)
    state = state.replace(entity_pos=final_pos, entity_vel=final_vel)

    # Post-Integration Force Update
    # Update acceleration based on final positions (for next step & diagnostics)
    new_acc = compute_forces(state, config)
    state = state.replace(entity_acc=new_acc)

    # 3. Compute Diagnostics (Post-step)
    # Always run diagnostics as required for Step 2 prompt
    state = compute_diagnostics(state, config)

    # 4. Increment Step Count
    state = state.replace(step_count=state.step_count + 1)

    # 5. Return updated state
    return state
