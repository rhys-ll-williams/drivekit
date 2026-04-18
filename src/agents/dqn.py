"""DQN - an implementation of Deep Q Networks"""
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from agents.base import AgentStrategy, ReplayBuffer

@dataclass
class DQNHyperparameterConfig:
    """A data class containing the hyperparameter tunings for DQN"""
    learning_rate: float = 1e-7
    gamma: float = 0.99
    min_replay: int = 500
    batch_size: int = 64
    buffer_size: int = 100000


# Our DQN - Deep Q Network has 2 hidden layers with ReLU activation
class DQN(nn.Module):
    """A pytorch network with 2 hidden layers and  ReLU activation"""
    def __init__(self, input_dim, n_actions, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        """forward pass the data, x, through the nework"""
        return self.net(x)

# DQNAgent has a lot of attributes due to supporting
# 2 networks and the optimizer and all the hyperparameters
# pylint: disable=too-many-instance-attributes
class DQNAgent(AgentStrategy):
    """DQNAgent - implments AgentStrategy and wrappers the DQN to make it a learning agent"""
    def __init__(self,obs_dim, nactions,
                 config : DQNHyperparameterConfig = DQNHyperparameterConfig()):
        """initialize DQNAgent -interpret the hyperparameter configuration dictionary"""
        learning_rate=config.learning_rate
        gamma=config.gamma
        min_replay = config.min_replay
        batch_size = config.batch_size
        buffer_size = config.buffer_size

        self.n_actions = nactions
        self.policy_net  = DQN(obs_dim, nactions)
        self.target_net  = DQN(obs_dim, nactions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.replay = ReplayBuffer(buffer_size)
        self.min_replay = min_replay
        self.batch_size = batch_size
        self.loss = None

    def explore(self,state):
        action = random.randrange(self.n_actions)
        return action

    def exploit(self,state):
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            qvals = self.policy_net(s_t)
            action = int(torch.argmax(qvals, dim=1).item())
        return action

    def learn(self,transition):
        # allow for offline RL updating based on the ReplayBuffer
        if not transition is None:
            #the * separates the transition into state, action, next_state, reward, done)
            self.replay.push(*transition['transition'])

        # learning, once the replay buffer is full enough
        if len(self.replay) >= self.min_replay:
            batch = self.replay.sample(self.batch_size)
            states = torch.tensor(np.vstack(batch.state), dtype=torch.float32)
            actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32)
            dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

            q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                # --- DDQN CHANGE #1: action selection using policy_net ---
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                # --- DDQN CHANGE #2: action evaluation using target_net ---
                q_next = self.target_net(next_states).gather(1, next_actions)

                #q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
                q_target = rewards + self.gamma * q_next * (1.0 - dones)

            loss = nn.functional.mse_loss(q_values, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.loss = loss
            # Gradient clipping - to avoid exploding Q-values
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),
                                            clip_value=0.05) # local variability
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                           max_norm=1) # global variability
            self.optimizer.step()

    def update(self, tau=0.8):
        """a soft update that blends the target and policy"""
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def report(self):
        """simplest reporting, just returns the current loss value"""
        return self.loss

    def save(self,ep=None):
        """Save either an interim result for an episode, or the final policy"""
        if not ep is None:
            torch.save(self.policy_net.state_dict(), f"policy_ep{ep+1}.pth")
        else:
            torch.save(self.policy_net.state_dict(), "policy_final.pth")

    def load(self, path):
        """load either the shards into the replay buffer (e.g. for hot start
           OR
           a previously trained network"""
        if path is None:
            self.replay.load_all_shards()
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
