"""
Environment engine package for CosmoSim.

Provides centralized coordination of environmental effects including:
- Topology (space geometry)
- Substrates (fluid, lattice, noise fields)
- Expansion dynamics
- Environment-level forces
"""

from .engine import EnvironmentEngine

__all__ = ["EnvironmentEngine"]
