# sumo_rules_generator.py

#!/usr/bin/env python3
"""
sumo_rules_generator.py

Generates a ReplayBuffer based on the RulesBasedAgent

"""
from statistics import mean # tracking the mean reward for information only

import utils
from simulator.sumo_adapter import SUMOSimulatorAdapter
from agents.base import ReplayBuffer
from agents.rule_based import RuleBasedAgent, SpeedControlConfig

# --------------- Evaluation function ---------------

def generate_trainer(agent, action_list,
                     simulator,
                     episodes, max_steps):
    """Runs a dummy training loop, accumulating the
       transitions that the rules-based AV makes in the
       ReplayBuffer - generating a labelled dataset"""
    simulator.reset()
    total_rewards = []

    replay = ReplayBuffer(1000000)
    for ep in range(episodes):
        action_used = set()
        ep_reward = 0.0

        state = simulator.reset()
        for _ in range(max_steps):
            if not simulator.has_ended():
                action = agent.exploit(state)
                if action>= len(action_list):
                    print(f"{action=} but {len(action_list)=}")
                if action_list[action].can(simulator):
                    action_list[action].execute(simulator)
                action_used.add(action)
                simulator.step()
                next_state, reward, done, _ = simulator.reward()
                replay.push(state, action, next_state, reward, done)
                state=next_state
                ep_reward += reward

        total_rewards.append(ep_reward)
        print(
            f"{ep} of {episodes}"
            f" used {action_used}"
            f" mean reward so far {mean(total_rewards)}"
            )
    simulator.close()

if __name__ == "__main__":
    eps = 10000    # run this many episodes per parameter set to reduce noise
    eps = 50
    MSTEPS = 800

    # Generate cases from the rules-based simulation for use in hot-starting the DQN

    # CREATE A SIMULATOR.....the SUMO simulator could be replaced by any other
    env = SUMOSimulatorAdapter({ "sumo_cfg" : "../datasets/simple/simple.sumocfg"})


    actions = utils.get_action_list(env.sim_dt,'action_config.json')

    config = SpeedControlConfig()
    config.target_speed = 14.0
    config.close_factor = 0.0
    config.slow_threshold = 0.001
    config.accel_threshold = 0.0
    env.target_speed = config.target_speed

    ag = RuleBasedAgent(actions, env, config)

    generate_trainer(ag, actions, env, eps, MSTEPS)
