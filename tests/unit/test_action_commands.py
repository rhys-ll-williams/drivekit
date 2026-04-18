import pytest
from unittest.mock import MagicMock, patch

from simulator.commands import (
    ActionCommand,
    MaintainCommand,
    AccelerateCommand,
    LaneChangeLeftCommand,
    LaneChangeRightCommand,
)


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

class MockSimulator:
    def __init__(self, speed=10.0, max_speed=30.0, lane=1, nlanes=3):
        self._speed = speed
        self._max_speed = max_speed
        self._lane = lane
        self._nlanes = nlanes

    # Speed API
    def get_speed(self, vid):
        return self._speed

    def get_max_speed(self, vid):
        return self._max_speed

    def set_speed(self, vid, new_speed):
        self._speed = new_speed

    # Lane API
    def get_lane(self, vid):
        return self._lane

    def change_lane(self, vid, target, duration):
        self._lane = target

    def get_nlanes(self, vid):
        return self._nlanes


@pytest.fixture
def sim():
    return MockSimulator()


# ---------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------

def test_actioncommand_is_abstract():
    with pytest.raises(TypeError):
        ActionCommand()  # cannot instantiate abstract class


# ---------------------------------------------------------
# MaintainCommand
# ---------------------------------------------------------

def test_maintaincommand_can(sim):
    cmd = MaintainCommand()
    assert cmd.can(sim) is True


def test_maintaincommand_execute_noop(sim):
    cmd = MaintainCommand()
    assert cmd.execute(sim) is None


# ---------------------------------------------------------
# AccelerateCommand
# ---------------------------------------------------------

def test_acceleratecommand_execute_increases_speed(sim):
    cmd = AccelerateCommand("ego", sim_dt=1.0, factor=2.0)
    old_speed = sim.get_speed("ego")

    cmd.execute(sim)

    assert sim.get_speed("ego") == pytest.approx(old_speed + 2.0)


def test_acceleratecommand_respects_max_speed(sim):
    sim._speed = 29.0
    sim._max_speed = 30.0

    cmd = AccelerateCommand("ego", sim_dt=1.0, factor=5.0)
    cmd.execute(sim)

    assert sim.get_speed("ego") == 30.0  # clipped to max speed


def test_acceleratecommand_can_prevents_speeding(sim):
    sim._speed = 29.0
    sim._max_speed = 30.0

    cmd = AccelerateCommand("ego", sim_dt=1.0, factor=5.0)
    assert cmd.can(sim) is False


def test_acceleratecommand_can_minimize_false(sim):
    # minimize=False means allow braking below zero but forbid negative speeds
    sim._speed = 0.0
    cmd = AccelerateCommand("ego", sim_dt=1.0, factor=-5.0, minimize=True)

    assert cmd.can(sim) is True  # would go negative


# ---------------------------------------------------------
# LaneChangeLeftCommand
# ---------------------------------------------------------
def test_lanechangeleft_execute(sim):
    sim._lane = 2
    cmd = LaneChangeLeftCommand("ego")

    cmd.execute(sim)

    assert sim.get_lane("ego") == 1  # moved left


@patch("simulator.commands.Simulator.get_lane")
def test_lanechangeleft_can_true(mock_lane_index, sim):
    mock_lane_index.return_value = 2
    cmd = LaneChangeLeftCommand("ego")

    assert cmd.can(sim) is True


@patch("simulator.commands.Simulator.get_lane")
def test_lanechangeleft_can_false(mock_lane_index, sim):
    mock_lane_index.return_value = 0  # leftmost lane
    cmd = LaneChangeLeftCommand("ego")

    assert cmd.can(sim) is True



# ---------------------------------------------------------
# LaneChangeRightCommand
# ---------------------------------------------------------
def test_lanechangeright_execute(sim):
    sim._lane = 1
    cmd = LaneChangeRightCommand("ego")
    cmd.execute(sim)

    assert sim.get_lane("ego") == 2  # moved right


@patch("simulator.commands.Simulator.get_lane")
def test_lanechangeright_can_true(mock_lane_index, sim):
    mock_lane_index.return_value = 1
    cmd = LaneChangeRightCommand("ego")

    assert cmd.can(sim) is True


@patch("simulator.commands.Simulator.get_lane")
def test_lanechangeright_can_false(mock_lane_index, sim):
    mock_lane_index.return_value = 2
    cmd = LaneChangeRightCommand("ego")

    assert cmd.can(sim) is True

