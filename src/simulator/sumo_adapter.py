"""sumo_adapter - contains both a SUMOFacade that inerfaces directly
   with the SUMO simulator, and a SUMOSimulator that implements the
   Simulator abstract class"""
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import traci

from simulator.base import Simulator

# ==============================================================================================
# sumo_adapter
# netgenerate.exe --grid --grid.x-number 2
#                 --grid.y-number 1 --default.lanenumber 2
#                 --default.speed 13.9 --output-file simple.net.xml
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# Ensure SUMO traci is importable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    raise EnvironmentError("""Please set SUMO_HOME environment
variable pointing to your SUMO installation.""")

# -------------- Hyperparams --------------
SUMO_CONFIG = "simple.sumocfg"   # the sumo configuration file (must exist)
SUMO_BINARY = "sumo"             # or "sumo-gui"
EGO_ID = "ego"

# number of SUMO simulation steps per environment step
# (use step-length in sumo config as 0.1)
STEP_SIM = 1
SIM_DT = 0.1  # SUMO step-length (should match sumocfg)



# ---------- SUMO facade (hides raw API) ----------
# pylint: disable=too-many-instance-attributes
class SUMOFacade:
    """SUMOFacade - encapsulates calls to the SUMO simulator"""
    def __init__(self,config: Dict[str, Any]) -> None:
        self.sumo_binary = config.get('sumo_binary', SUMO_BINARY)
        self.sumo_cfg = config.get('sumo_cfg', SUMO_CONFIG)
        self.ego_id = config.get('ego_id', EGO_ID)
        self.sim_dt = config.get('sim_dt', SIM_DT)
        self.step_sim = config.get('step_sim',STEP_SIM)
        self.max_leader_dist = config.get("max_leader_dist", 10.0)
        self.target_speed = config.get("target_speed", 14.636)
        self.t = 0

    def start(self):
        """Start the SUMO simulator in the background"""
        try:
            traci.close(False)
        except traci.FatalTraCIError:
            # if the simulator was not running...
            pass

        # Start SUMO
        cmd = [self.sumo_binary, "-c", self.sumo_cfg, "--step-length", str(self.sim_dt)]
        traci.start(cmd)

    def _wait_for_ego_vehicle(self):
        """PRIVATE: Wait until the ego vehicle is inserted into the simulation."""
        #print("Waiting for ego vehicle to appear in SUMO...")
        while True:
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()
            if self.ego_id in vehicles:
                break
            time.sleep(0.05)  # avoid CPU busy-wait

    def observe(self):
        """Get the current state of the simulation"""
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(5, dtype=np.float32)

        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        lane_index = traci.vehicle.getLaneIndex(self.ego_id)
        lane_pos = traci.vehicle.getLanePosition(self.ego_id)
        lane_length = 1000.0
        lane_pos_norm = lane_pos / (lane_length + 1e-6)

        leader = traci.vehicle.getLeader(self.ego_id, self.max_leader_dist)
        if leader is None:
            rel_dist = self.max_leader_dist
            rel_speed = 0.0
        else:
            leader_id, dist = leader
            rel_dist = dist
            rel_speed = traci.vehicle.getSpeed(leader_id) - ego_speed

        return np.array([ego_speed, rel_dist, rel_speed, float(lane_index),
                         np.clip(lane_pos_norm, 0.0, 1.0)], dtype=np.float32)


    def stop(self) -> None:
        """Stop the SUMO simulation if it is running"""
        try:
            if traci.isLoaded():
                traci.close(False)
        except traci.TraCIException:
            pass

    def reset(self) -> Any:
        """Reset scenario and return initial state representation."""
        self.start()
        speed = np.random.uniform(0.5, self.target_speed)

        self._wait_for_ego_vehicle()
        self.t = 0

        # 1. Randomize ego starting speed slightly (0.5 m/s to target speed)
        try:
            traci.vehicle.setSpeed(self.ego_id, speed)
        except traci.TraCIException:
            pass

        # 2. Randomize speeds of other vehicles slightly (±10%)
        for vid in traci.vehicle.getIDList():
            if vid != self.ego_id:
                try:
                    cur_speed = traci.vehicle.getSpeed(vid)
                    speed = max(0.0, cur_speed * np.random.uniform(0.9, 1.1))
                    traci.vehicle.setSpeed(vid, max(0.0, speed))
                except traci.TraCIException:
                    pass

        # Return initial observation
        obs = self.observe()
        return obs

    def has_ended(self):
        """Check whether the ego AV is still in the simulation"""
        return self.ego_id not in traci.vehicle.getIDList()

    def get_time(self):
        """Get the simulaion time from SUMO"""
        return traci.simulation.getTime()

    def step(self) -> None:
        """Advance simulation by one tick."""
        for _ in range(self.step_sim):
            traci.simulationStep()
            self.t += 1

    def reward(self):
        """Returns the current state, the reward, and
           whether or not the simulation has ended"""
        obs = self.observe()
        ego_speed = obs[0]
        rel_dist = obs[1]
        #rel_speed = obs[2]
        done, reward = False, 0.0
        reward = ego_speed # reward making progress
        speed_err = abs(ego_speed - self.target_speed)
        reward -= 0.5 * speed_err
        desired_gap = 2.0 + ego_speed * 1.0
        if rel_dist < desired_gap:
            reward -= 2.0 * (desired_gap - rel_dist)
        else:
            reward += 0.01 * min(rel_dist, self.max_leader_dist)

        # --- Collision check ---
        collisions = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in collisions:
            done = True
            reward -= 100.0  # heavy penalty
            print(f"Collision detected at t={self.get_time():.1f}s")

        # Check if ego left the simulation
        if self.ego_id not in traci.vehicle.getIDList():
            done = True

        reward = float(np.clip(reward, -100.0, 100.0))/100.0
        return obs, reward, done, {}

    def get_speed(self, vid):
        """get the speed of a specific vehicle"""
        return traci.vehicle.getSpeed(vid)

    def get_max_speed(self, vid):
        """get the maximum speed of a specific vehicle"""
        return traci.vehicle.getMaxSpeed(vid)

    def set_speed(self, vid, spd):
        """set the speed of a specific vehicle"""
        traci.vehicle.setSpeed(vid, spd)

    def get_lane(self, vid):
        """get the current lane of a specific vehicle"""
        lane_index = 0
        try:
            lane_index = traci.vehicle.getLaneIndex(vid)
        except traci.TraCIException:
            print("FAILED - get_lane() returned 0 instead")
        return lane_index

    def change_lane(self, vid, target, duration=50.0):
        """change the lane of a specific vehicle over the given duration"""
        try:
            traci.vehicle.changeLane(vid, target, duration)
        except traci.TraCIException:
            print(f"FAILED - lane change {vid} {target} {duration}")

    def get_nlanes(self,vid):
        """get the number of available lanes"""
        nlanes=0
        try:
            lane_id = traci.vehicle.getLaneID(self.ego_id)
            edge_id = lane_id.split("_")[0]        # get edge ID from lane ID
            nlanes = traci.edge.getLaneNumber(edge_id)  # <- correct API
        except traci.TraCIException:
            print(f"FAILED - can't determine number of lanes for {vid}")
        return nlanes


# ---------- SUMO adapter (implements Simulator) ----------
class SUMOSimulatorAdapter(Simulator):
    """An implementation of Simulator that uses the SUMOFacade"""
    def __init__(self, config: Dict[str, Any]):
        self._facade = SUMOFacade(config)
        self._config = config
        self._started = False
        self.sim_dt = config.get('sim_dt', SIM_DT)

    def _ensure_started(self) -> None:
        if not self._started:
            self._facade.start() # self._config)
            self._started = True

    ########################################
    # Vehicle access
    ########################################
    def get_speed(self, vid):
        return self._facade.get_speed(vid)

    def get_max_speed(self, vid):
        return self._facade.get_max_speed(vid)

    def set_speed(self, vid, spd):
        return self._facade.set_speed(vid, spd)

    def get_lane(self, vid):
        return self._facade.get_lane(vid)

    def get_nlanes(self, vid):
        return self._facade.get_nlanes(vid)

    def change_lane(self, vid, target, duration=50.0):
        return self._facade.change_lane(vid, target, duration)

    ########################################
    # simulator controls
    ########################################
    def reset(self) -> Any:
        self._ensure_started()
        return self._facade.reset()

    def step(self):
        return self._facade.step()

    def reward(self) -> Tuple[Any, float, bool, Dict]:
        return self._facade.reward()

    def has_ended(self):
        return self._facade.has_ended()

    def close(self) -> None:
        if self._started:
            self._facade.stop()
            self._started = False
