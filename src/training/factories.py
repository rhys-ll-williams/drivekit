"""Factories - a number of Factory Pattern implementations
   to handle creation of Commands and Simulators


   NOTE: the Factory Pattern has a single public method create()
   since pylint expects a minimum of 2 public methods per class,
   # pylint: disable=too-few-public-methods appears in the
   class definitions to highlight that pylint should ignore that
   rule here where it was a deliberate choice to have just one
   public method"""
from __future__ import annotations

from typing import Any, Dict

from simulator.base import Simulator
from simulator.sumo_adapter import SUMOSimulatorAdapter

from simulator.commands import ActionCommand, AccelerateCommand
from simulator.commands import LaneChangeLeftCommand, LaneChangeRightCommand
from simulator.commands import MaintainCommand

# ---------- Factories (Factory / Abstract Factory) ----------
class SimulatorFactory:  # pylint: disable=too-few-public-methods
    """A factory that can generate the different simulators"""
    @staticmethod
    def create(sim_type: str, config: Dict[str, Any]) -> Simulator:
        """The create method
            agent_type :  a string name
            config     : a dictionary with configuration options"""
        if sim_type == "sumo":
            return SUMOSimulatorAdapter(config)
        # if sim_type == "carla": ...
        raise ValueError(f"Unknown simulator type: {sim_type}")

class CommandFactory:  # pylint: disable=too-few-public-methods
    """A factory that can generate the different commands"""
    @staticmethod
    def create(command_type: str, config: Dict[str, Any]) -> ActionCommand:
        """The create method
            agent_type :  a string name
            config     : a dictionary with configuration options"""
        if command_type == "accelerate":
            return AccelerateCommand(config['vehicle_id'],
                                     config['sim_dt'],
                                     config['factor'],
                                     config.get('minimize',False))
        if command_type=="lane_change":
            direction = config.get('change_direction',"left")
            if direction=="left":
                return LaneChangeLeftCommand(config['vehicle_id'])
            if direction=="right":
                return LaneChangeRightCommand(config['vehicle_id'])
        if command_type=="maintain":
            return MaintainCommand()
        raise ValueError(f"Unknown command type: {command_type}")
