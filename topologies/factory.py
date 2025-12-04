"""
Topology factory for creating topology handlers.
"""

from __future__ import annotations
from typing import Optional
from state import UniverseConfig
from .base_topology import Topology
from .flat import FlatTopology
from .torus import TorusTopology
from .sphere import SphereTopology
from .bubble import BubbleTopology
from .hyperbolic import HyperbolicTopology
from .mobius_topology import MobiusTopology


def get_topology_handler(
    topology_type: Optional[str],
    config: UniverseConfig,
    **kwargs
) -> Topology:
    """
    Get a topology handler by type name.
    
    Args:
        topology_type: Type of topology ("flat", "torus", "sphere", "bubble", "hyperbolic", or None)
        config: UniverseConfig for the simulation
        **kwargs: Additional parameters passed to topology constructor
    
    Returns:
        Topology instance (defaults to FlatTopology if None)
    
    Raises:
        ValueError: If topology type is unknown
    
    Example:
        >>> topology = get_topology_handler("torus", config)
        >>> wrapped_pos = topology.wrap_position(positions)
    """
    if config.topology_type == MobiusTopology.MOBIUS_TOPOLOGY:
        return MobiusTopology(config, **kwargs)

    if topology_type is None or topology_type == "flat":
        return FlatTopology(config, **kwargs)
    
    topology_lower = topology_type.lower()
    
    if topology_lower == "flat":
        return FlatTopology(config, **kwargs)
    elif topology_lower == "torus" or topology_lower == "periodic":
        return TorusTopology(config, **kwargs)
    elif topology_lower == "sphere" or topology_lower == "spherical":
        return SphereTopology(config, **kwargs)
    elif topology_lower == "bubble":
        return BubbleTopology(config, **kwargs)
    elif topology_lower == "hyperbolic" or topology_lower == "h3":
        return HyperbolicTopology(config, **kwargs)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
