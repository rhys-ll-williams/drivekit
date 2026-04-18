"""
AgentStratey - the abstract class for all learning agents
ReplayBuffer - a sharded storage of all transitions to allow
               flexibility in the design of learning systems like DQN
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from collections import deque, namedtuple
import os
import pickle
import random


# ---------- Agent strategy (Strategy pattern) ----------

class AgentStrategy(ABC):
    """the abstract class for all learning agents"""
    @abstractmethod
    def save(self) -> None:
        """Save the currently learned policy."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a policy from file"""

    @abstractmethod
    def explore(self, state : Any) -> Any:
        """Choose an exploratory action given the current state."""

    @abstractmethod
    def exploit(self,state):
        """Choose a policy-based action given the current state."""

    @abstractmethod
    def learn(self,transition: Dict[str,Any]) -> None:
        """Learn from a transition, a typical transition contains
          states,actions,rewards,next_states,dones"""

    @abstractmethod
    def report(self) -> Any:
        """Return a report on the current status, such as the loss"""

# ------ ReplayBuffer - all agents use the same ReplayBuffer si
# that, for example, the rules-based system can build a dataset
# for hot-start training the DQN
# The ReplayBuffer stores up state transitions in a buffer
# then the learning works by batch training from the ReplayBuffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
# pylint: disable=consider-using-with
class ReplayBuffer:
    """a sharded storage of all transitions to allow
       flexibility in the design of learning systems like DQN"""
    def __init__(self, capacity,
                 shard_dir="replay_shards",
                 shard_size_limit=10000000): # default 10 MB shard size
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        self.shard_dir = shard_dir
        os.makedirs(shard_dir, exist_ok=True)

        self.shard_size_limit = shard_size_limit
        self.current_shard = None
        self.current_shard_path = None

        self._open_new_shard()

    def _open_new_shard(self):
        """PRIVATE: open a new shard"""
        shard_id = len(os.listdir(self.shard_dir)) + 1
        self.current_shard_path = os.path.join(self.shard_dir, f"shard_{shard_id:05d}.pkl")
        # pylint suggests using "with", but the file should remain open for appending.
        self.current_shard = open(self.current_shard_path, "ab")  # append-binary

    def push(self, *args):
        """Write the provided Transition to memory and the shard file"""
        transition = Transition(*args)
        self.buffer.append(transition)

        # Write a single pickled record
        pickle.dump(transition, self.current_shard)

        # Rotate shard if too large
        if self.current_shard.tell() >= self.shard_size_limit:
            self.current_shard.close()
            self._open_new_shard()

    def sample(self, batch_size):
        """Recover a random batch of Transitions"""
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

    def load_all_shards(self):
        """read shards into the memory buffer"""
        merged = []

        for fname in sorted(os.listdir(self.shard_dir)):
            if not fname.endswith(".pkl"):
                continue

            path = os.path.join(self.shard_dir, fname)
            with open(path, "rb") as f:
                while True:
                    try:
                        t = pickle.load(f)
                        merged.append(t)
                    except EOFError:
                        break

        # Keep only the newest transitions
        if len(merged) > self.capacity:
            merged = merged[-self.capacity:]

        self.buffer = deque(merged, maxlen=self.capacity)
        print(f"Loaded {len(merged)} transitions from {len(os.listdir(self.shard_dir))} shards.")
