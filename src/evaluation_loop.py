"""
train_sumo_dqn.py

Simple DQN training loop interacting with SUMO via TraCI.
Agent controls vehicle with id "ego" to:
 - keep a target speed while maintaining safe distance to leader (radar)
 - optionally change lanes left/right

This is intentionally simple. Software architecture and engineering are neded.
"""
import random
import json
import numpy as np

import torch

from simulator.sumo_adapter import SUMOSimulatorAdapter

from agents.dqn import DQNAgent
from agents.rule_based import RuleBasedAgent, SpeedControlConfig

from training.factories import CommandFactory

SEED = 1111
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# action definitions - these are the actions the AV can take.
with open('action_config.json', encoding='utf-8') as f:
    action_config = json.load(f)


if __name__ == "__main__":
    EPISODES = 100     # 100 for the simple tests
                     #  10 for the variable_speed_signs,
                     #   1 for the full-scale traffic tests
    MAX_STEPS = 800 # 800 for the simple tests
                     # 8000 for the full-scale traffic tests

    # CREATE A SIMULATOR.....the SUMO simulator could be replaced by any other
    # Comment out as needed
    #env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/variable_speed_signs/test.sumocfg"})
    #env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/1_mh2gs/1_mh2gs.sumocfg"})
    #env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/greenwich/greenwich.sumocfg"})
    #env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/2_ts2qs/2_ts2qs.sumocfg"})
    #env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/8_rh2bps/8_rh2bps.sumocfg"})
    env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/simple/simple.sumocfg"})

    # CREATE A LIST OF ACTIONS....
    action_list = []
    for name, command in action_config.items():
        # information from simulation that every action might need
        command["spec"]["sim_dt"]= env.sim_dt
        action_list.append(CommandFactory.create(command["action"],command["spec"]))

    # CREATE A LEARNER....
    NACTIONS = len(action_list)
    OBS_DIM = 5

    best_params = {
        "close_factor": 0.0,
        "slow_threshold": 0.001,
        "accel_threshold": 0.0,
        "target_speed": 14.0,
        "estimated_reward": 0.0
    }
    config = SpeedControlConfig()
    config.target_speed = best_params['target_speed']
    config.close_factor = best_params['close_factor']
    config.slow_threshold = best_params['slow_threshold']
    config.accel_threshold = best_params['accel_threshold']

    env.target_speed = config.target_speed

    # Agents - comment as needed.
    learner = RuleBasedAgent(action_list,env, config)
    learner = DQNAgent(OBS_DIM,NACTIONS)
    #learner.load('policy_final_hotstart_v0.pth')
    learner.load('policy_final_coldstart_v1.pth')

    # Evaluate
    for ep in range(EPISODES):
        accepted = [0]*NACTIONS
        rejected = [0]*NACTIONS

        state = env.reset()
        total_steps = 0
        ep_reward = 0.0
        done = False
        failed = 0
        for step in range(MAX_STEPS):
            action = learner.exploit(state)

            next_state = np.zeros(5, dtype=np.float32)
            reward = -50.0
            done= True

            if not env.has_ended():
                if action_list[action].can(env): # pass the guardrail
                    accepted[action]+=1
                    action_list[action].execute(env) # do the proposed command...
                else:
                    rejected[action]+=1
                    failed += 1
                env.step()
                next_state, reward, done, _ = env.reward()
            else:
                print("EGO EXITED")

            state = next_state
            ep_reward += reward
            total_steps += 1

            if done:
                break
        loss = learner.report()
        print(
            f"Episode {ep} "
            f"loss : {loss} "
            f"reward : {ep_reward:.2f} "
            f"steps : {step+1} "
            f"accepted : {accepted} "
            f"rejected : {rejected} "
            f"total_steps : {total_steps}"
        )

    # Shut down the simulator
    env.close()

    # Save the trained policy agent
    # learner.save()
    print("Training finished, policy saved to policy_final.pth")
