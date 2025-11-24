"""
Cleans all PNG output files in outputs/ subdirectories.
"""

import os
from pathlib import Path

root = Path("outputs")

if not root.exists():
    print("No outputs/ directory found.")
    exit()

count = 0

for folder in root.glob("**/*"):
    if folder.is_file() and folder.suffix == ".png":
        folder.unlink()
        count += 1

print(f"Removed {count} PNG files from outputs/")
