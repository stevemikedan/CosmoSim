"""
CLI Scenario Runner for CosmoSim.

Allows testing substrates, expansion, and topology from the command line.
"""

import argparse
import json
import time
import numpy as np
import jax.numpy as jnp
import jax
# STABILIZATION PATCH IMPORTS
import os


from state import UniverseConfig, UniverseState, initialize_state
from environment.engine import EnvironmentEngine
from entities import allocate_entities
from physics_utils import (
    compute_gravity_forces,
    integrate_euler,
    integrate_leapfrog,
    kinetic_energy,
    potential_energy,
    total_energy,
    momentum,
    center_of_mass,
    adjust_timestep
)


def run_sim(config, steps=200, gravity=True, seed=42, save_every=1):
    """Run the simulation loop."""
    print(f"Initializing simulation with seed {seed}...")
    np.random.seed(seed)
    
    # Initialize environment
    env = EnvironmentEngine(config)
    
    # Initialize state (for compatibility)
    state = initialize_state(config)
    
    # Allocate and initialize entities
    # We use numpy for initialization to respect the seed easily
    N = config.max_entities
    dim = config.dim
    
    # Random positions within radius/2
    pos_np = np.random.uniform(-config.radius/2, config.radius/2, size=(N, dim))
    # Random velocities
    vel_np = np.random.uniform(-0.1, 0.1, size=(N, dim))
    # Random masses
    mass_np = np.random.uniform(0.1, 1.0, size=(N,))
    # All active
    active_np = np.ones((N,), dtype=bool)
    
    # Convert to JAX arrays
    pos = jnp.array(pos_np)
    vel = jnp.array(vel_np)
    mass = jnp.array(mass_np)
    active = jnp.array(active_np)
    
    frames = []
    diagnostics = []  # PS1.3: Energy and momentum diagnostics
    
    print(f"Running {steps} steps...")
    
    for step in range(steps):
        # 1. Compute Forces
        if gravity:
            force = compute_gravity_forces(pos, mass, active, config)
        else:
            force = jnp.zeros_like(pos)
            
        # 2. Apply Environment (Substrate, Expansion, Topology)
        # We pass 'state' but update it partially to be safe
        # EnvironmentEngine expects a UniverseState for substrate.update
        # We'll update the state object with current pos/vel
        current_state = state.replace(
            entity_pos=pos,
            entity_vel=vel,
            entity_mass=mass,
            entity_active=active,
            time=step * config.dt
        )
        
        pos, vel, force = env.apply_environment(pos, vel, force, current_state)
        
        # 3. Adaptive Timestep (PS1.4)
        current_dt = config.dt
        if config.enable_adaptive_dt:
            current_dt = adjust_timestep(current_dt, vel, force, mass, config)
            
        # 4. Update Physics (Integration)
        if config.integrator == "leapfrog":
            pos, vel = integrate_leapfrog(pos, vel, force, mass, active, current_dt)
        else:  # Default to Euler
            pos, vel = integrate_euler(pos, vel, force, mass, active, current_dt)
        
        # --- STABILIZATION PATCH: clamp + sanitize -------------------------
        pos_np = np.array(pos)
        vel_np = np.array(vel)

        # Patch 1. Clamp velocities
        max_speed = 5.0
        speed = np.linalg.norm(vel_np, axis=1)
        too_fast = speed > max_speed
        if np.any(too_fast):
            vel_np[too_fast] = (
                vel_np[too_fast] / speed[too_fast][:, None] * max_speed
            )

        # Patch 2. Clamp radius (limit expansion blow-up)
        max_radius = config.radius * 5.0
        r = np.linalg.norm(pos_np, axis=1)
        too_far = r > max_radius
        if np.any(too_far):
            pos_np[too_far] = (
                pos_np[too_far] / r[too_far][:, None] * max_radius
            )

        # Patch 3. Replace NaN/Inf for JSON safety
        pos_np = np.nan_to_num(pos_np, nan=0.0, posinf=1e6, neginf=-1e6)
        vel_np = np.nan_to_num(vel_np, nan=0.0, posinf=1e6, neginf=-1e6)

        # Patch 4. Recast back to JAX for next step
        pos = jnp.array(pos_np)
        vel = jnp.array(vel_np)
        # -------------------------------------------------------------------

        # 4. Compute Diagnostics (PS1.3)
        if config.enable_diagnostics and (step % save_every == 0):
            KE = kinetic_energy(vel, mass, active)
            PE = potential_energy(pos, mass, active, config)
            E = total_energy(KE, PE)
            P = momentum(vel, mass, active)
            COM = center_of_mass(pos, mass, active)
            
            diagnostics.append({
                "step": step,
                "KE": float(KE),
                "PE": float(PE),
                "E": float(E),
                "momentum": P.tolist(),
                "COM": COM.tolist()
            })

        # 5. Save Frame
        if step % save_every == 0:
            # Convert to list for JSON serialization
            frames.append({
                "step": step,
                "pos": pos_np.tolist(),
                "vel": vel_np.tolist()
            })         
    return frames, diagnostics


def main():
    parser = argparse.ArgumentParser(description="CosmoSim Scenario Runner")
    
    # Simulation Config
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    parser.add_argument("--outfile", type=str, default=None, help="Output JSON file")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=1, help="Save interval")
    parser.add_argument("--gravity", type=str, default="on", choices=["on", "off"], help="Enable gravity")
    
    # Environment Config
    parser.add_argument("--substrate", type=str, default="none", help="Substrate type")
    parser.add_argument("--topology", type=str, default="flat", help="Topology type")
    parser.add_argument("--expansion", type=str, default="none", help="Expansion type")
    
    # Advanced Config (optional overrides)
    parser.add_argument("--radius", type=float, default=10.0, help="Universe radius")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--entities", type=int, default=50, help="Number of entities")
    
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # AUTO-NAMING LOGIC (Topology + Expansion + Substrate + Steps + Timestamp)
    #
    # Produces filenames like:
    #   bubbleTop_scale_factorExp_vectorSub_300steps_20251127_163552.json
    #
    # This avoids overwrites, encodes full simulation identity,
    # and works even if some CLI args are omitted (defaults fill in).
    # ----------------------------------------------------------------------

    def build_auto_filename(args):
        # Normalize values (fall back to defaults if missing)
        top = args.topology if args.topology else "flat"
        exp = args.expansion if args.expansion else "none"
        sub = args.substrate if args.substrate else "none"

        # Steps matter scientifically and visually
        steps = args.steps

        # Timestamp (YYYYMMDD_HHMMSS)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Construct final name
        filename = f"{top}Top_{exp}Exp_{sub}Sub_{steps}steps_{timestamp}.json"

        return f"sim_output/{filename}"

    # Ensure sim_output directory exists
    os.makedirs("sim_output", exist_ok=True)

    # Determine final outfile path
    outfile = args.outfile if args.outfile else build_auto_filename(args)

    print(f"[CosmoSim] Saving output to: {outfile}")
    # ----------------------------------------------------------------------
    
    # Map topology string to int type
    topology_map = {
        "flat": 0,
        "torus": 1,
        "sphere": 2,
        "bubble": 3
    }
    topo_type = topology_map.get(args.topology, 0)
    
    # Construct Config
    config = UniverseConfig(
        physics_mode=0,
        radius=args.radius,
        max_entities=args.entities,
        max_nodes=1,
        dt=args.dt,
        c=1.0,
        G=0.1,
        dim=3,
        topology_type=topo_type,
        
        # Expansion
        expansion_type=args.expansion,
        expansion_rate=0.05, # Default rate
        expansion_mode="inflation" if args.expansion == "scale_factor" else "linear",
        H=0.1, # Default H
        bubble_expand=True, # Default for bubble test
        bubble_radius=args.radius,
        
        # Substrate
        substrate=args.substrate,
        substrate_params={
            "grid_size": (10, 10, 10),
            "amplitude": 0.5,
            "noise": False
        }
    )
    
    # Run Simulation
    frames, diagnostics = run_sim(
        config, 
        steps=args.steps, 
        gravity=(args.gravity == "on"),
        seed=args.seed,
        save_every=args.save_every
    )
    
    # Save Output
    output_data = {
        "frames": frames,
        "diagnostics": diagnostics
    }
    
    with open(outfile, "w") as f:
        json.dump(output_data, f)
        
    print(f"Saved {len(frames)} frames and {len(diagnostics)} diagnostic records to {outfile}")


if __name__ == "__main__":
    main()
