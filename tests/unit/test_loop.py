import pytest
import numpy as np
from unittest.mock import MagicMock

from training.loop import RLTrainingLoop


# ---------------------------------------------------------
# Mock classes
# ---------------------------------------------------------

class MockSimulator:
    """Minimal simulator for testing RLTrainingLoop."""
    def __init__(self):
        self._ended = False
        self._step_called = 0

    def reset(self):
        return np.zeros(5, dtype=np.float32)

    def has_ended(self):
        return self._ended

    def step(self):
        self._step_called += 1

    def reward(self):
        # next_state, reward, done, info
        return np.zeros(5, dtype=np.float32), 1.0, self._ended, {}

    def end(self):
        self._ended = True


class MockAction:
    """Action with controllable can() and execute() behaviour."""
    def __init__(self, can_return=True):
        self._can = can_return
        self.execute_called = 0

    def can(self, simulator):
        return self._can

    def execute(self, simulator):
        self.execute_called += 1


class MockAgent:
    """Minimal agent implementing AgentStrategy interface."""
    def __init__(self):
        self.explore_called = 0
        self.exploit_called = 0
        self.learn_called = 0
        self.update_called = 0
        self.replay = []  # only for printing

    def explore(self, state):
        self.explore_called += 1
        return 0

    def exploit(self, state):
        self.exploit_called += 1
        return 0

    def learn(self, transition):
        self.learn_called += 1

    def update(self):
        self.update_called += 1

    def report(self):
        return 0.123  # dummy loss


class MockObserver:
    """Observer that records notifications."""
    def __init__(self):
        self.starts = []
        self.steps = []
        self.ends = []

    def on_episode_start(self, idx, info):
        self.starts.append((idx, info))

    def on_step(self, idx, transition):
        self.steps.append((idx, transition))

    def on_episode_end(self, idx, info):
        self.ends.append((idx, info))


# ---------------------------------------------------------
# Concrete testable subclass of RLTrainingLoop
# ---------------------------------------------------------

class ExampleLoop(RLTrainingLoop):
    """Concrete subclass because RLTrainingLoop is abstract."""
    pass


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

@pytest.fixture
def loop():
    sim = MockSimulator()
    agent = MockAgent()
    actions = [MockAction(can_return=True)]
    obs = [MockObserver()]
    return ExampleLoop(sim, actions, agent, observers=obs, max_steps=5, target_update=3)


def test_episode_start_and_end_notifications(loop):
    loop.run(1)
    obs = loop.observers[0]

    assert len(obs.starts) == 1
    assert len(obs.ends) == 1


def test_step_notifications(loop):
    loop.run(1)
    obs = loop.observers[0]

    # At least one step should be recorded
    assert len(obs.steps) > 0


def test_explore_vs_exploit(loop):
    # Force epsilon = 1.0 so explore always used
    loop.set_epsilon(start=1.0, end=1.0, decay=1)
    loop.run(1)

    assert loop.learner.explore_called > 0
    assert loop.learner.exploit_called == 0


def test_guardrail_retry(loop):
    # Replace action list with one invalid then one valid
    bad_action = MockAction(can_return=False)
    good_action = MockAction(can_return=True)
    loop.action_list = [bad_action, good_action]

    # Infinite generator: first return 0, then always return 1
    def explore_gen():
        yield 0
        while True:
            yield 1

    loop.learner.explore = MagicMock(side_effect=explore_gen())

    loop.run(1)

    # Should have retried explore at least once
    assert loop.learner.explore.call_count >= 2


def test_learner_learn_called(loop):
    loop.run(1)
    assert loop.learner.learn_called > 0


def test_target_update(loop):
    # target_update = 3, so update should be called at least once
    loop.run(2)
    assert loop.learner.update_called > 0


def test_episode_terminates_on_done(loop):
    # Force simulator to end immediately
    loop.simulator.end()
    loop.run(1)

    # Only one step should occur
    assert loop.simulator._step_called == 0  # no step() because has_ended=True


def test_run_multiple_episodes(loop):
    loop.run(3)
    obs = loop.observers[0]

    assert len(obs.starts) == 3
    assert len(obs.ends) == 3
