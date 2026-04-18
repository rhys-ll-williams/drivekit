import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from simulator.sumo_adapter import SUMOSimulatorAdapter
from agents.dqn import DQNAgent
from simulator.commands import MaintainCommand

from training.loop import RLTrainingLoop


def test_training_loop_integration():
    # A Mock SUMO
    with patch("simulator.sumo_adapter.traci") as traci:
        traci.vehicle.getIDList.return_value = ["ego"]
        traci.vehicle.getSpeed.return_value = 10.0
        traci.vehicle.getLaneIndex.return_value = 1
        traci.vehicle.getLanePosition.return_value = 50.0
        traci.vehicle.getLeader.return_value = ("lead", 20.0)
        traci.vehicle.getMaxSpeed.return_value = 30.0
        traci.simulation.getCollidingVehiclesIDList.return_value = []
        traci.simulationStep = MagicMock()

        # Build stack
        sim = SUMOSimulatorAdapter({"ego_id": "ego"})
        agent = DQNAgent(obs_dim=5, nactions=1)  # trivial
        actions = [MaintainCommand()]
        loop = RLTrainingLoop(sim, actions, agent, max_steps=5)

        # Run
        loop.run(1)
        state, reward, done, _ = sim.reward()

        # Assertions
        
        assert agent.exploit(state) is 0
        assert sim._started is True
