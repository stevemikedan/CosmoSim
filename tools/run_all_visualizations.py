"""
Run all CosmoSim visualization scripts sequentially.
Useful for manual QA and sprint demonstrations.
"""

import subprocess

scripts = [
    ("trajectory_plot.py", "Trajectory Plot"),
    ("snapshot_plot.py", "Snapshot Plot"),
    ("energy_plot.py", "Energy Diagnostics"),
    ("visualize.py", "Real-Time Animation (final frame saved)"),
]

for filename, label in scripts:
    print(f"\n=== Running {label} ===")
    try:
        subprocess.run(["python", filename], check=True)
    except Exception as e:
        print(f"❌ {label} failed: {e}")
    else:
        print(f"✔ {label} completed successfully!")
