import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from simulator.sumo_adapter import SUMOFacade, SUMOSimulatorAdapter


# ---------------------------------------------------------
# Fixtures: full traci mock
# ---------------------------------------------------------

@pytest.fixture
def traci_mock():
    """Patch the entire traci module used inside sumo_adapter."""
    with patch("simulator.sumo_adapter.traci") as traci:
        # vehicle API
        traci.vehicle.getIDList.return_value = ["ego"]
        traci.vehicle.getSpeed.return_value = 10.0
        traci.vehicle.getLaneIndex.return_value = 1
        traci.vehicle.getLanePosition.return_value = 50.0
        traci.vehicle.getLeader.return_value = ("lead", 20.0)
        traci.vehicle.getMaxSpeed.return_value = 30.0
        traci.vehicle.setSpeed = MagicMock()
        traci.vehicle.changeLane = MagicMock()
        traci.vehicle.getLaneID.return_value = "edge_0"
        traci.edge.getLaneNumber.return_value = 3

        # simulation API
        traci.simulationStep = MagicMock()
        traci.simulation.getTime.return_value = 5.0
        traci.simulation.getCollidingVehiclesIDList.return_value = []

        # lifecycle
        traci.start = MagicMock()
        traci.close = MagicMock()
        traci.isLoaded.return_value = True

        yield traci


@pytest.fixture
def config():
    return {
        "sumo_binary": "sumo",
        "sumo_cfg": "simple.sumocfg",
        "ego_id": "ego",
        "sim_dt": 0.1,
        "step_sim": 1,
        "max_leader_dist": 10.0,
        "target_speed": 14.636,
    }


@pytest.fixture
def facade(config, traci_mock):
    return SUMOFacade(config)


@pytest.fixture
def adapter(config, traci_mock):
    return SUMOSimulatorAdapter(config)


# ---------------------------------------------------------
# SUMOFacade tests
# ---------------------------------------------------------

def test_facade_start_starts_sumo(facade, traci_mock):
    facade.start()
    traci_mock.start.assert_called_once()


def test_facade_observe_returns_correct_shape(facade, traci_mock):
    obs = facade.observe()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)


def test_facade_observe_leader_logic(facade, traci_mock):
    traci_mock.vehicle.getLeader.return_value = ("lead", 5.0)
    traci_mock.vehicle.getSpeed.side_effect = [10.0, 8.0]  # ego, leader
    obs = facade.observe()
    assert obs[1] == 5.0          # rel_dist
    assert obs[2] == -2.0         # rel_speed


def test_facade_has_ended_false(facade, traci_mock):
    traci_mock.vehicle.getIDList.return_value = ["ego"]
    assert facade.has_ended() is False


def test_facade_has_ended_true(facade, traci_mock):
    traci_mock.vehicle.getIDList.return_value = []
    assert facade.has_ended() is True


def test_facade_step_advances_time(facade, traci_mock):
    facade.step()
    traci_mock.simulationStep.assert_called_once()


def test_facade_reward_no_collision(facade, traci_mock):
    obs, reward, done, info = facade.reward()
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert done is False


def test_facade_reward_collision(facade, traci_mock):
    traci_mock.simulation.getCollidingVehiclesIDList.return_value = ["ego"]
    obs, reward, done, info = facade.reward()
    assert done is True
    assert reward <= 0.0


def test_facade_get_speed(facade, traci_mock):
    assert facade.get_speed("ego") == 10.0


def test_facade_change_lane_calls_traci(facade, traci_mock):
    facade.change_lane("ego", 2)
    traci_mock.vehicle.changeLane.assert_called_once()


# ---------------------------------------------------------
# SUMOSimulatorAdapter tests
# ---------------------------------------------------------

def test_adapter_reset_returns_observation(adapter):
    obs = adapter.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)


def test_adapter_step_delegates(adapter, traci_mock):
    adapter.reset()
    adapter.step()
    traci_mock.simulationStep.assert_called()


def test_adapter_reward_delegates(adapter):
    adapter.reset()
    obs, reward, done, info = adapter.reward()
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)


def test_adapter_close_stops_sumo(adapter, traci_mock):
    adapter.reset()
    adapter.close()
    traci_mock.close.assert_called()
