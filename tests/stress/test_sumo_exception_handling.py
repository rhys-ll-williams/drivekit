import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from simulator.sumo_adapter import SUMOFacade, SUMOSimulatorAdapter


# ---------------------------------------------------------
# Helper: configure traci to fail everywhere
# ---------------------------------------------------------

def configure_traci_to_fail(traci):
    class DummyException(Exception):
        pass

    traci.TraCIException = DummyException
    traci.FatalTraCIError = DummyException

    # vehicle API
    traci.vehicle.getIDList.side_effect = DummyException()
    traci.vehicle.getSpeed.side_effect = DummyException()
    traci.vehicle.getLaneIndex.side_effect = DummyException()
    traci.vehicle.getLanePosition.side_effect = DummyException()
    traci.vehicle.getLeader.side_effect = DummyException()
    traci.vehicle.getMaxSpeed.side_effect = DummyException()
    traci.vehicle.setSpeed.side_effect = DummyException()
    traci.vehicle.changeLane.side_effect = DummyException()
    traci.vehicle.getLaneID.side_effect = DummyException()

    # edge API
    traci.edge.getLaneNumber.side_effect = DummyException()

    # simulation API
    traci.simulationStep.side_effect = DummyException()
    traci.simulation.getTime.side_effect = DummyException()
    traci.simulation.getCollidingVehiclesIDList.side_effect = DummyException()

    # lifecycle
    traci.start.side_effect = DummyException()
    traci.close.side_effect = DummyException()
    traci.isLoaded.side_effect = DummyException()

    return DummyException


# ---------------------------------------------------------
# SUMOFacade.start() SHOULD raise
# ---------------------------------------------------------

def test_sumo_facade_start_raises_on_failure():
    with patch("simulator.sumo_adapter.traci") as traci:
        traci.vehicle = MagicMock()
        traci.edge = MagicMock()
        traci.simulation = MagicMock()

        DummyException = configure_traci_to_fail(traci)

        facade = SUMOFacade({"ego_id": "ego"})

        with pytest.raises(DummyException):
            facade.start()


# ---------------------------------------------------------
# SUMOFacade methods that SHOULD swallow exceptions
# ---------------------------------------------------------

def test_sumo_facade_methods_that_should_not_raise():
    with patch("simulator.sumo_adapter.traci") as traci:
        traci.vehicle = MagicMock()
        traci.edge = MagicMock()
        traci.simulation = MagicMock()

        DummyException = configure_traci_to_fail(traci)

        facade = SUMOFacade({"ego_id": "ego"})

        # stop() is wrapped
        facade.stop()

        # get_nlanes() is wrapped
        assert facade.get_nlanes("ego") == 0


# ---------------------------------------------------------
# SUMOSimulatorAdapter: methods that SHOULD swallow exceptions
# ---------------------------------------------------------

def test_sumo_adapter_reset_survives_exceptions():
    """reset() must raise because SUMOFacade.start() raises when traci.start() fails."""
    with patch("simulator.sumo_adapter.traci") as traci:
        traci.vehicle = MagicMock()
        traci.edge = MagicMock()
        traci.simulation = MagicMock()

        DummyException = configure_traci_to_fail(traci)

        sim = SUMOSimulatorAdapter({"ego_id": "ego"})

        # EXPECTED: reset() should raise because start() raises
        with pytest.raises(DummyException):
            sim.reset()
            
