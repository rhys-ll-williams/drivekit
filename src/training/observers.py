"""Observer pattern for monitoring the RLTrainingLoop"""
from abc import ABC
from typing import Any, Dict

# ---------- Observer pattern for training events ----------

class TrainingObserver(ABC):
    """A basic observer that monitors training"""
    def __init__(self):
        self.stored_episodes = {}
        self.stored_steps = {}

    def on_episode_start(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """The episode has started"""
        self.stored_episodes[episode_idx] = info

    def on_step(self, step_idx: int, transition: Dict[str, Any]) -> None:
        """A simulation timestep has completed"""
        self.stored_steps[step_idx] = transition

    def on_episode_end(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """The episode has ended"""
        self.stored_episodes[episode_idx] = info


class ConsoleLoggerObserver(TrainingObserver):
    """A basic observer that prints to the console"""
    def on_episode_start(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """The episode has started"""
        print(f"[Episode {episode_idx}] start: {info}")

    def on_step(self, step_idx: int, transition: Dict[str, Any]) -> None:
        """A simulation timestep has completed"""
        print(f"Transision : {transition}")

    def on_episode_end(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """The episode has ended"""
        print(f"[Episode {episode_idx}] end: {info}")
