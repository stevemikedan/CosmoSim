"""Base class for viewer overlays."""
from abc import ABC, abstractmethod
from typing import Any

class Overlay(ABC):
    """Abstract base class for all viewer overlays."""
    
    @abstractmethod
    def apply(self, state, viewer, ax):
        """Apply the overlay to the current frame.
        
        Args:
            state: Current UniverseState
            viewer: The Viewer instance
            ax: Matplotlib axis
        """
        pass
