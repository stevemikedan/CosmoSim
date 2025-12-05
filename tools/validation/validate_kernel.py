
import sys
import os

# Add project root to path (../../ from tools/validation)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from state import UniverseConfig, UniverseState, initialize_state
from kernel import step_simulation

def validate_kernel():
    print("Initializing state...")
    config = UniverseConfig(
        physics_mode=0,
        radius=100.0,
        max_entities=2,
        max_nodes=1,
        dt=0.01,
        c=1.0,
        G=1.0,
        dim=2,
        topology_type=0,
        enable_diagnostics=True
    )
    state = initialize_state(config)
    
    # Set up 2-body system
    pos = jnp.array([[-1.0, 0.0], [1.0, 0.0]])
    vel = jnp.array([[0.0, -0.5], [0.0, 0.5]])
    mass = jnp.array([1.0, 1.0])
    active = jnp.array([True, True])
    
    state = state.replace(
        entity_pos=pos,
        entity_vel=vel,
        entity_mass=mass,
        entity_active=active
    )
    
    print(f"Step 0: step_count={state.step_count}, acc_set={jnp.any(state.entity_acc != 0)}")
    assert state.step_count == 0
    # Acc should be zero initially
    
    # Step 1
    print("Running Step 1...")
    state = step_simulation(state, config)
    
    print(f"Step 1: step_count={state.step_count}")
    assert state.step_count == 1
    
    # Check Acc
    acc_mag = jnp.linalg.norm(state.entity_acc, axis=-1)
    print(f"Acc: {state.entity_acc}")
    print(f"Acc Mag: {acc_mag}")
    assert jnp.all(acc_mag > 0), "Acceleration should be set after step 1"
    
    # Check Diagnostics
    print(f"KE: {state.kinetic_energy}")
    print(f"PE: {state.potential_energy}")
    print(f"Total E: {state.total_energy}")
    print(f"Initial E: {state.initial_energy}")
    print(f"Drift: {state.energy_drift}")
    
    assert state.kinetic_energy > 0
    assert state.initial_energy != 0
    assert state.total_energy != 0
    
    # Step 2
    print("Running Step 2...")
    state = step_simulation(state, config)
    print(f"Step 2: step_count={state.step_count}")
    print(f"Drift Step 2: {state.energy_drift}")
    
    assert state.step_count == 2
    
    print("Validation passed!")

if __name__ == "__main__":
    validate_kernel()
