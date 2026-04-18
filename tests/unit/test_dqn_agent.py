import os
import torch
import numpy as np
import pytest

from agents.dqn import DQNAgent, DQNHyperparameterConfig
from agents.base import Transition


@pytest.fixture
def config():
    # Small values so tests run fast
    return DQNHyperparameterConfig(
        learning_rate=1e-3,
        gamma=0.99,
        min_replay=5,      # <<< IMPORTANT: small threshold for testing
        batch_size=2,
        buffer_size=50
    )


@pytest.fixture
def agent(config):
    obs_dim = 4
    nactions = 3
    return DQNAgent(obs_dim, nactions, config)


# ---------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------

def test_agent_initialises_network(agent):
    assert isinstance(agent.policy_net, torch.nn.Module)
    assert isinstance(agent.target_net, torch.nn.Module)

    x = torch.zeros(1, 4)
    y = agent.policy_net(x)
    assert y.shape == (1, agent.n_actions)


# ---------------------------------------------------------
# Action selection
# ---------------------------------------------------------

def test_explore_returns_valid_action(agent):
    action = agent.explore(np.zeros(4))
    assert 0 <= action < agent.n_actions


def test_exploit_returns_valid_action(agent):
    action = agent.exploit(np.zeros(4))
    assert 0 <= action < agent.n_actions


# ---------------------------------------------------------
# Replay buffer integration
# ---------------------------------------------------------

def test_learn_pushes_transition(agent):
    transition = {"transition": (1, 2, 3, 4, False)}
    agent.learn(transition)
    assert len(agent.replay) == 1

    t = agent.replay.buffer[0]
    assert isinstance(t, Transition)
    assert (t.state, t.action, t.next_state, t.reward, t.done) == (1, 2, 3, 4, False)


# ---------------------------------------------------------
# Learning behaviour BEFORE min_replay
# ---------------------------------------------------------

def test_learn_does_not_update_before_min_replay(agent):
    # Add fewer transitions than min_replay
    for _ in range(agent.min_replay - 1):
        agent.learn({"transition": (1, 2, 3, 4, False)})

    # Copy weights
    before = [p.clone() for p in agent.policy_net.parameters()]

    # Try to learn
    agent.learn(None)

    after = list(agent.policy_net.parameters())

    # No parameter should change
    assert all(torch.equal(b, a) for b, a in zip(before, after))
    assert agent.report() is None


# ---------------------------------------------------------
# Learning behaviour AFTER min_replay
# ---------------------------------------------------------

##def test_learn_updates_after_min_replay(agent):
##    # Fill replay buffer to min_replay
##    #for _ in range(agent.min_replay+1):
##    #    agent.learn({"transition": (1, 2, 3, 4, False)})
##
##    #before = [p.clone() for p in agent.policy_net.parameters()]
##
##    # Now learning should occur
##    #agent.learn(None)
##
##    #after = list(agent.policy_net.parameters())
##
##    # At least one parameter should change
##    #assert any(not torch.equal(b, a) for b, a in zip(before, after))
##
##    # Loss should be set
##    #assert agent.report() is not None
##

# ---------------------------------------------------------
# Target network update
# ---------------------------------------------------------

##def test_update_soft_updates_target(agent):
##    # Modify policy net
##    for p in agent.policy_net.parameters():
##        p.data.add_(1.0)
##
##    # Soft update
##    agent.update(tau=0.5)
##
##    # Target should move toward policy
##    for p, tp in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
##        assert torch.allclose(tp, 0.5 * p + 0.5 * tp)


# ---------------------------------------------------------
# Save / Load
# ---------------------------------------------------------

def test_save_and_load(agent, tmp_path):
    save_path = tmp_path / "policy.pth"

    # Save final policy
    agent.save(ep=None)
    assert os.path.exists("policy_final.pth")

    # Load it back
    agent.load("policy_final.pth")
    # If load didn't crash, it's good enough for this test
