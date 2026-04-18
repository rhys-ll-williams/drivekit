import pytest
import numpy as np
from agents.rule_based import (
    RuleBasedAgent,
    SpeedControlConfig,
    ACTION_ACCEL,
    ACTION_STRONG_BRAKE,
    ACTION_BRAKE,
    ACTION_LANE_LEFT,
    ACTION_LANE_RIGHT,
    ACTION_MAINTAIN,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

class MockAction:
    """Mock action with a .can(simulator) method."""
    def __init__(self, can_return=True):
        self._can = can_return

    def can(self, simulator):
        return self._can


class MockSimulator:
    pass


@pytest.fixture
def action_list():
    # 6 actions in the correct order
    return [
        MockAction(),  # ACCEL
        MockAction(),  # STRONG_BRAKE
        MockAction(),  # BRAKE
        MockAction(),  # LANE_LEFT
        MockAction(),  # LANE_RIGHT
        MockAction(),  # MAINTAIN
    ]


@pytest.fixture
def agent(action_list):
    sim = MockSimulator()
    return RuleBasedAgent(action_list, sim, SpeedControlConfig())


# ---------------------------------------------------------
# Construction
# ---------------------------------------------------------

def test_agent_initialises(agent):
    assert agent.n_actions == 6
    assert agent.target_speed == pytest.approx(14.6)
    assert agent.close_factor == pytest.approx(0.6)


# ---------------------------------------------------------
# explore()
# ---------------------------------------------------------

def test_explore_returns_valid_action(agent):
    for _ in range(20):
        a = agent.explore(np.zeros(5))
        assert 0 <= a < agent.n_actions


# ---------------------------------------------------------
# exploit(): braking logic
# ---------------------------------------------------------

def test_exploit_strong_brake_when_too_close(agent):
    # rel_dist < close_factor * gap
    # ego_speed = 10 → gap = 2 + 1 = 3
    # close_factor * gap = 1.8
    state = np.array([10.0, 1.0, 0.0, 0, 0])  # rel_dist = 1.0 < 1.8
    a = agent.exploit(state)
    assert a == ACTION_STRONG_BRAKE


# ---------------------------------------------------------
# exploit(): speed control
# ---------------------------------------------------------

def test_exploit_accelerate_when_under_speed_and_clear(agent):
    # ego_speed < target, large gap
    state = np.array([10.0, 50.0, 0.0, 0, 0])
    a = agent.exploit(state)
    assert a == ACTION_ACCEL


# ---------------------------------------------------------
# exploit(): lane change logic
# ---------------------------------------------------------

def test_exploit_lane_left_if_slow_leader_and_left_available(action_list):
    # Make left lane available, right unavailable
    action_list[ACTION_LANE_LEFT] = MockAction(can_return=True)
    action_list[ACTION_LANE_RIGHT] = MockAction(can_return=False)

    agent = RuleBasedAgent(action_list, MockSimulator(), SpeedControlConfig())

    # rel_speed < slow_threshold (-1.5)
    state = np.array([10.0, 2.0, -3.0, 0, 0])
    a = agent.exploit(state)
    assert a == ACTION_LANE_LEFT


def test_exploit_lane_right_if_left_unavailable(action_list):
    action_list[ACTION_LANE_LEFT] = MockAction(can_return=False)
    action_list[ACTION_LANE_RIGHT] = MockAction(can_return=True)

    agent = RuleBasedAgent(action_list, MockSimulator(), SpeedControlConfig())

    state = np.array([10.0, 2.0, -3.0, 0, 0])
    a = agent.exploit(state)
    assert a == ACTION_LANE_RIGHT


def test_exploit_maintain_when_no_conditions_met(agent):
    # ego_speed near target, large gap, no slow leader
    state = np.array([ 1.4612264e+01,1.1051495e+03,-1.2646377e+01,
                       0.0000000e+00,3.7107188e-01])
    a = agent.exploit(state)
    assert a == ACTION_MAINTAIN


# ---------------------------------------------------------
# learn(), save(), load() are no-ops
# ---------------------------------------------------------

def test_learn_is_noop(agent):
    assert agent.learn({"transition": (1,2,3)}) is None


def test_save_is_noop(agent):
    assert agent.save() is None


def test_load_is_noop(agent):
    assert agent.load("path") is None
