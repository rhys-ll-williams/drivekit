"""The abstract AcionCommand and the concrete
   commands for AVs to interact with the
   SUMO simulator"""
from abc import ABC, abstractmethod

from simulator.base import Simulator


# ---------- Command pattern for actions ----------
class ActionCommand(ABC):
    """The abstract ActionCommand"""
    @abstractmethod
    def execute(self, simulator: Simulator) -> None:
        """Apply this action to the simulator."""

    @abstractmethod
    def can(self, simulator: Simulator) -> bool:
        """Can this action be taken in this context - for guardrails"""
        return False

class MaintainCommand(ActionCommand):
    """A null implementation so that the agent can choose
       to continue driving in the same lane at the current speed"""
    def __init__(self):
        pass

    def execute(self, simulator: Simulator) -> None:
        pass

    def can(self, simulator: Simulator) -> bool:
        return True

class AccelerateCommand(ActionCommand):
    """Allow the AV to increase or decrease its speed"""
    def __init__(self, vehicle_id: str, sim_dt: float, factor: float, minimize=True):
        self.ego_id = vehicle_id
        self.sim_dt = sim_dt
        self.factor = factor
        self.minimize = minimize

    def execute(self, simulator: Simulator) -> None:
        # get the current speed and set the new speed - linear acceleration
        spd = simulator.get_speed(self.ego_id)
        max_spd = simulator.get_max_speed(self.ego_id)

        set_speed = min(spd + self.factor * self.sim_dt, max_spd)
        if not self.minimize:
            set_speed = max(0.0, spd + self.factor * self.sim_dt)
        simulator.set_speed(self.ego_id, set_speed)

    def can(self, simulator: Simulator) -> bool:
        spd = simulator.get_speed(self.ego_id)
        max_spd = simulator.get_max_speed(self.ego_id)
        if not self.minimize:
            if spd + (self.factor * self.sim_dt) < 0:
                return False #stopped
        else:
            if spd + self.factor * self.sim_dt > max_spd:
                return False # avoid speeding
        return True

class LaneChangeLeftCommand(ActionCommand):
    """Allow the agent to make a lane change to the left"""
    def __init__(self, vehicle_id: str):
        self.ego_id = vehicle_id

    def execute(self, simulator: Simulator) -> None:
        # simulator.change_lane(self.vehicle_id, self.target_lane)
        lane_index = simulator.get_lane(self.ego_id)
        target = max(0, lane_index - 1) # guardrail prevents exist
        simulator.change_lane(self.ego_id, target, 0.1)

    def can(self, simulator: Simulator) -> bool:
        lane_index = simulator.get_lane(self.ego_id)
        return lane_index > 0

class LaneChangeRightCommand(ActionCommand):
    """Allow the agent to make a lane change to the right"""
    def __init__(self, vehicle_id: str):
        self.ego_id = vehicle_id

    def execute(self, simulator: Simulator) -> None:
        lane_index = simulator.get_lane(self.ego_id)
        nlanes = simulator.get_nlanes(self.ego_id)
        target = min(nlanes - 1, lane_index + 1)
        simulator.change_lane(self.ego_id, target, 50.0)

    def can(self, simulator: Simulator) -> bool:
        lane_index = simulator.get_lane(self.ego_id)
        nlanes = simulator.get_nlanes(self.ego_id)
        return lane_index < (nlanes - 1)
