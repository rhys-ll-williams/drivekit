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
from agents.dqn import DQNAgent

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Training - the balance between exploration and exploitation
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 25000  # steps


if __name__ == "__main__":
    SIM_DT = 0.1
    EPISODES = 5000
    MAX_STEPS = 800

    action_list = utils.get_action_list(SIM_DT,'action_config.json')

    # CREATE A LEARNER....
    NACTIONS = len(action_list)
    OBS_DIM = 5
    learner = DQNAgent(OBS_DIM,NACTIONS)
    learner.load(None) # load all shards available in ReplayBuffer

    # TRAIN
    total_steps = 0
    TARGET_UPDATE = 1000
    for ep in range(EPISODES):
        for step in range(MAX_STEPS):
            learner.learn(None) # just learn from what is already available...
            # periodically update target
            if total_steps % TARGET_UPDATE == 0:
                learner.update()
            total_steps+=1
        learner.save()
        if ep % 1 == 0:
            print(f"COMPLETED {ep} of {EPISODES} with {learner.report()}")

    # Save the trained policy agent
    learner.save()
    print("Training finished, policy saved to policy_final.pth")
