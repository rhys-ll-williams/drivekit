"""RuleBasedAgent - a rules-based autonomous vehicle
   decision-maker"""
import random
from typing import Any, Dict
from dataclasses import dataclass

from agents.base import AgentStrategy


# the RULES based agent ...
ACTION_ACCEL = 0
ACTION_STRONG_BRAKE = 1
ACTION_BRAKE = 2
ACTION_LANE_LEFT = 3
ACTION_LANE_RIGHT = 4
ACTION_MAINTAIN = 5

@dataclass
class SpeedControlConfig:
    """A data class tha captures all the variable parameters for the RuleBasedAgent"""
    target_speed_m_s: float = 14.6
    close_factor: float = 0.6
    slow_threshold: float = -1.5
    accel_threshold: float = 0.0

class RuleBasedAgent(AgentStrategy):
    """To avoid going off the road,
       the rules use can_change_right() and can_change_left()
       as checks.
       This approach can be guardrails on a deployed RL"""
    def __init__(self, action_list,
                 simulator,
                 config: SpeedControlConfig = SpeedControlConfig()):
        self.target_speed = config.target_speed_m_s
        self.close_factor = config.close_factor
        self.slow_threshold = config.slow_threshold
        self.accel_threshold = config.accel_threshold
        self.action_list = action_list
        self.n_actions = len(action_list)
        self.simulator = simulator

    def explore(self,state):
        action = random.randrange(self.n_actions)
        return action

    # override pylint because exploit in a rules-based system
    # will have a separate return path for every action
    # pylint: disable=too-many-return-statements
    def exploit(self, state):
        """
        Improved rule-based overtaking decision tree.
        obs = [ego_speed, rel_dist, rel_speed, lane_index, lane_pos_norm]
        """
        ego_speed, rel_dist, rel_speed = map(float, state[:3])

        # --- Derived quantities ---
        desired_gap = 2.0 + ego_speed * 0.1
        closing = rel_speed < 0.0
        too_close = rel_dist < desired_gap
        severely_close = rel_dist < desired_gap * self.close_factor

        # --- 1. Safety overrides everything ---
        if severely_close or rel_dist < 0.001:
            return ACTION_STRONG_BRAKE

        # --- 2. If the front car is pulling away ---
        if rel_speed > 0:
            if ego_speed < self.target_speed * 1.05:
                return ACTION_ACCEL
            return ACTION_MAINTAIN

        # --- 3. We are closing in on the front car ---
        if closing and too_close:
            # 3a. Try left-lane overtake if safe
            if self.action_list[ACTION_LANE_LEFT].can(self.simulator):
                return ACTION_LANE_LEFT

            # 3b. Try right-lane return (if we are already overtaking)
            if self.action_list[ACTION_LANE_RIGHT].can(self.simulator):
                return ACTION_LANE_RIGHT

            # 3c. Otherwise slow down
            return ACTION_BRAKE

        # --- 4. General speed control ---
        if ego_speed < self.target_speed:
            return ACTION_ACCEL

        return ACTION_MAINTAIN


    def learn(self, transition: Dict[str, Any]) -> None:
        # Rule-based: no learning
        pass

    def save(self) -> None:
        # rules are hard-coded
        pass

    def load(self, path: str) -> None:
        # rules are hard-coded
        pass

    def report(self) -> Any:
        return 1.0
