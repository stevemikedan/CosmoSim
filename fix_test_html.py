#!/usr/bin/env python3
"""
Fix test.html to use the correct JSON property names.
Changes:
- frame.pos -> frame.positions
- frame.vel -> frame.velocities  
- frame.mass -> frame.masses
- sim.topology -> frame.topology
"""

import re

# Read the file
with open('viewer/test.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the specific lines in the frame mapping section
old_pattern = r'''const frames = sim\.frames\.map\(\(frame, idx\) => \(\{
                    positions: frame\.pos,
                    velocities: frame\.vel,
                    masses: frame\.mass \|\| \[\],
                    active: frame\.active \|\| \[\],
                    topology: sim\.topology \|\| null,
                    index: idx
                \}\)\);'''

new_pattern = '''const frames = sim.frames.map((frame, idx) => ({
                    positions: frame.positions,
                    velocities: frame.velocities,
                    masses: frame.masses || [],
                    active: frame.active || [],
                    topology: frame.topology || null,
                    index: idx
                }));'''

# Replace (handle whitespace variations)
if 'frame.pos' in content:
    content = content.replace('frame.pos', 'frame.positions')
    content = content.replace('frame.vel', 'frame.velocities')
    content = content.replace('frame.mass ||', 'frame.masses ||')
    content = content.replace('sim.topology ||', 'frame.topology ||')
    
    # Write back
    with open('viewer/test.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fixed property names in test.html")
    print("  - frame.pos → frame.positions")
    print("  - frame.vel → frame.velocities")
    print("  - frame.mass → frame.masses")
    print("  - sim.topology → frame.topology")
else:
    print("✓ File already fixed or pattern not found")
