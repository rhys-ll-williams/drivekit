"""Simulator - the abstract base class for all traffic simulators"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


# ---------- Simulator abstraction (Adapter target) ----------

class Simulator(ABC):
    """Simulator - the abstract base class for all traffic simulators"""
    @abstractmethod
    def reset(self) -> Any:
        """Reset the simulator and return initial state."""

    @abstractmethod
    def step(self):
        """Advance the simulation by a single timestep"""

    @abstractmethod
    def reward(self) -> Tuple[Any, float, bool, Dict]:
        """Evaluate the current state, return (next_state, reward, done, info)."""

    @abstractmethod
    def has_ended(self):
        """Return True if the ego AV has left the simulation"""

    @abstractmethod
    def close(self) -> None:
        """Clean up simulator resources."""

    #############################################
    # Vehicle access
    #############################################
    @abstractmethod
    def get_speed(self, vid: str) -> float:
        """Return the current speed of the identified vehicle"""

    @abstractmethod
    def get_max_speed(self, vid: str) -> float:
        """Return the current maximum allowed speed of the identified vehicle"""

    @abstractmethod
    def set_speed(self, vid: str, spd: float):
        """Set the speed of the identified vehicle"""

    @abstractmethod
    def get_lane(self, vid: str):
        """Get the current laneof the identified vehicle"""

    @abstractmethod
    def get_nlanes(self, vid: str):
        """Get the number of available lanes for the identified vehicle"""

    @abstractmethod
    def change_lane(self, vid: str, target: int, duration: float):
        """Change the lane of the current vehicle to the target lane"""
