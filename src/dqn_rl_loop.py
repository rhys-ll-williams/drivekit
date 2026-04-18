"""
train_sumo_dqn.py

Simple DQN training loop interacting with SUMO via TraCI.
Agent controls vehicle with id "ego" to:
 - keep a target speed while maintaining safe distance to leader (radar)
 - optionally change lanes left/right

This is intentionally simple. Software architecture and engineering are neded.
"""
import random
import numpy as np

import torch

import utils
from simulator.sumo_adapter import SUMOSimulatorAdapter

from agents.dqn import DQNAgent

from training.loop import RLTrainingLoop

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    # Training - the balance between exploration and exploitation
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 25000  # steps
    EPISODES = 500

    # CREATE A SIMULATOR.....the SUMO simulator could be replaced by any other
    env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/simple/simple.sumocfg"})

    action_list = utils.get_action_list(env.sim_dt,'action_config.json')

    # CREATE A LEARNER....
    NACTIONS = len(action_list)
    OBS_DIM = 5
    learner = DQNAgent(OBS_DIM,NACTIONS)

    HOT_START = False
    if HOT_START is True:
        learner.load('policy_final_n32.pth')

    # TRAIN
    train = RLTrainingLoop(env,action_list,learner,max_steps=800)
    train.set_epsilon(start=EPS_START,end=EPS_END,decay=EPS_DECAY)
    train.run(EPISODES)

    # Shut down the simulator
    env.close()

    # Save the trained policy agent
    learner.save()
    print("Training finished, policy saved to policy_final.pth")
