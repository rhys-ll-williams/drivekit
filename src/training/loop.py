"""RLTrainingLoop - the reinforcement learning training loop"""
from abc import ABC
from typing import Any, Dict, List

import math
import random
import numpy as np

from simulator.base import Simulator
from agents.base import AgentStrategy
from training.observers import TrainingObserver

# ---------- Template Method for training loop ----------
# DESIGN DECISION: lots of attributes and arguments is acceptable for the RLTrainingLoop
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
class RLTrainingLoop(ABC):
    """RLTrainingLoop - the reinforcement learning training loop"""
    def __init__(self,
                 simulator: Simulator,
                 action_list: List,
                 learner: AgentStrategy,
                 observers: List[TrainingObserver] | None = None,
                 max_steps: int = 800,
                 target_update: int = 1000):
        self.simulator = simulator
        self.action_list = action_list
        self.learner = learner
        self.observers = observers or []
        self.max_steps = max_steps
        self.total_steps = 0
        self.target_update = target_update
        # defaults
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 25000


    def set_epsilon(self,start = 1.0,end = 0.05,decay = 25000):
        """Sets the epsilon scale and its decay rate, this helps stabilize learning"""
        self.eps_start = start
        self.eps_end = end
        self.eps_decay = decay

    def _run_single_episode(self, episode_idx):
        state = self.simulator.reset()
        # notify any observers that we are staring an episode
        self.notify_episode_start(episode_idx, {"initial_state": state})

        ep_reward = 0.0
        # the traffic simulator is run until either MAX_STEPS is reached OR
        # the 'ego' AV has left the simulation region - which will give done = True
        done = False

        for step in range(self.max_steps):
            self.total_steps += 1
            # epsilon decay
            epsilon = self.eps_end + (self.eps_start - self.eps_end) \
                      * math.exp(-1.0 * self.total_steps / self.eps_decay)

            # select action
            if random.random() < epsilon:  #EXPLORATION
                action= self.learner.explore(state)
            else:                          #EXPLOITATION
                action = self.learner.exploit(state)

            # guardrails during training (due to random initialization of model
            # this avoids wandering around requesting actions that are not allowed
            # such as changing lanes into the verges
            # This is a "safe" exploration - simply try up to 50 times to find a valid action
            its = 0
            while (not self.action_list[action].can(self.simulator)) and (its<50):
                action= self.learner.explore(state)
                its += 1

            next_state = np.zeros(5, dtype=np.float32)
            reward = -50.0
            done= True
            if not self.simulator.has_ended():
                if self.action_list[action].can(self.simulator): # pass the guardrail
                    self.action_list[action].execute(self.simulator) # do the proposed command...
                else:
                    print(f"FAILED: {action}")
                self.simulator.step()
                next_state, reward, done, _ = self.simulator.reward()
            else:
                print("EGO EXITED")

            # scale rewards
            # reward /= 1000.0

            transition = (state, action, next_state, reward, done)
            self.learner.learn({ "transition" : transition})
            # notify the observers that a learning step is happening
            self.notify_step(step, transition)

            state = next_state
            ep_reward += reward

            # periodically update target
            if self.total_steps % self.target_update == 0:
                self.learner.update()

            # if the 'ego' AV has left the simulation zone then no more can
            # be learned for this case
            if done:
                break

        loss = self.learner.report()
        print(
            f"Episode {episode_idx} "
            f"loss={loss} "
            f"reward={ep_reward:.2f} "
            f"steps={step+1} "
            f"eps={epsilon:.3f} "
            f"replay={len(self.learner.replay)}"
        )
        # notify the observers that the episode is ending
        self.notify_episode_end(episode_idx, {"steps": step})

    def run(self, num_episodes: int) -> None:
        """Learn over a fixed number of episodes"""
        for episode_idx in range(num_episodes):
            self._run_single_episode(episode_idx)

    def notify_episode_start(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """Let all observers know that the episode has started"""
        for obs in self.observers:
            obs.on_episode_start(episode_idx, info)

    def notify_step(self, step_idx: int, transition: Dict[str, Any]) -> None:
        """Let all observers know that a simulation timestep has been taken"""
        for obs in self.observers:
            obs.on_step(step_idx, transition)

    def notify_episode_end(self, episode_idx: int, info: Dict[str, Any]) -> None:
        """Let all observers know that this episode has ended"""
        for obs in self.observers:
            obs.on_episode_end(episode_idx, info)
