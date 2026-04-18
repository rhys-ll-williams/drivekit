"""
utils.py

Convenience functions for building pipelines
"""
import json
from training.factories import CommandFactory


def get_action_list(sim_dt, cfg="action_config.json"):
    """Build the list of actions for the AV using the Factory"""
    # action definitions - these are the actions the AV can take.
    with open(cfg, encoding='utf-8') as f:
        action_config = json.load(f)

    # CREATE A LIST OF ACTIONS....
    action_list = []
    for _, command in action_config.items():
        # information from simulation that every action might need
        command["spec"]["sim_dt"]= sim_dt
        action_list.append(CommandFactory.create(command["action"],command["spec"]))
    return action_list
