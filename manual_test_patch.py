"""
Manual test to verify conftest patches work
"""
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Test that the patch gets applied when using pytest's monkeypatch
def test_patch_works():
    # Manually apply patches like pytest would
    import importlib
    from tests.conftest import _capped_range
    
    # Import a module
    import run_sim
    
    # Manually patch its range
    run_sim.range = _capped_range
    
    # Now test if run_sim's loop is capped
    print("\n=== Testing run_sim with patched range ===")
    cfg = run_sim.build_config()
    state = run_sim.build_initial_state(cfg)
    
    # Mock the step_simulation to avoid heavy computation
    def fake_step(state, cfg):
        return state
    
    import kernel
    original_step = kernel.step_simulation
    kernel.step_simulation = fake_step
    
    # Run should only do 2 iterations now  
    print("Running run_sim.run() - should only print 2 steps...")
    run_sim.run(cfg, state)
    print("Done!")
    
    kernel.step_simulation = original_step

if __name__ == "__main__":
    test_patch_works()
