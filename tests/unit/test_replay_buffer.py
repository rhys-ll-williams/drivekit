import os
import pickle
import random
from collections import deque

import pytest

from agents.base import ReplayBuffer, Transition


# -----------------------------
# FIXTURE: fresh buffer per test
# -----------------------------
@pytest.fixture
def rb(tmp_path):
    return ReplayBuffer(
        capacity=5,
        shard_dir=tmp_path,
        shard_size_limit=10_000_000,  # large so no rotation unless forced
    )


# -----------------------------
# TEST: initialization
# -----------------------------
def test_init_creates_directory_and_first_shard(tmp_path):
    rb = ReplayBuffer(capacity=5, shard_dir=tmp_path)

    # directory exists
    assert os.path.isdir(tmp_path)

    # shard file created
    files = os.listdir(tmp_path)
    assert len(files) == 1
    assert files[0].startswith("shard_00001")

    # buffer initialized
    assert isinstance(rb.buffer, deque)
    assert rb.buffer.maxlen == 5

    # shard handle open
    assert rb.current_shard is not None
    assert rb.current_shard_path.endswith("shard_00001.pkl")


# -----------------------------
# TEST: push() stores in memory
# -----------------------------
def test_push_adds_transition_to_buffer(rb):
    rb.push(1, 2, 3, 4, False)
    assert len(rb) == 1
    t = rb.buffer[0]
    assert isinstance(t, Transition)
    assert (t.state, t.action, t.reward) == (1, 2, 4)


# -----------------------------
# TEST: push() writes to shard
# -----------------------------
def test_push_writes_to_shard(rb):
    rb.push(10, 20, 30, 40, False)

    # read back from shard file
    shard_path = rb.current_shard_path
    assert os.path.exists(shard_path)


# -----------------------------
# TEST: shard rotation
# -----------------------------
def test_shard_rotation(tmp_path):
    # force rotation after every write
    rb = ReplayBuffer(capacity=5, shard_dir=tmp_path, shard_size_limit=1)

    rb.push(1, 2, 3, 4, False)
    rb.push(4, 5, 6, 7, False)

    files = sorted(os.listdir(tmp_path))
    assert len(files) == 3
    assert files[0].startswith("shard_00001")
    assert files[1].startswith("shard_00002")


# -----------------------------
# TEST: capacity enforcement
# -----------------------------
def test_capacity_limit(rb):
    for i in range(10):
        rb.push(i, i, i, i, False)

    assert len(rb) == rb.capacity
    # newest 5 should remain
    assert [t.state for t in rb.buffer] == [5, 6, 7, 8, 9]


# -----------------------------
# TEST: sample()
# -----------------------------
def test_sample_returns_batched_transition(rb):
    for i in range(5):
        rb.push(i, i + 1, i + 2, i+3, False)

    random.seed(0)
    batch = rb.sample(3)

    assert isinstance(batch, Transition)
    assert len(batch.state) == 3
    assert len(batch.action) == 3
    assert len(batch.reward) == 3


# -----------------------------
# TEST: load_all_shards()
# -----------------------------
def test_load_all_shards(tmp_path):
    # create buffer and write two shards manually
    rb = ReplayBuffer(capacity=10, shard_dir=tmp_path, shard_size_limit=1)

    # two pushes → two shards
    rb.push(1, 2, 3, 4, False)
    rb.push(4, 5, 6, 7, False)

    # new buffer loads from disk
    rb2 = ReplayBuffer(capacity=10, shard_dir=tmp_path)
    rb2.load_all_shards()

    assert len(rb2) == 2
    states = [t.state for t in rb2.buffer]
    assert states == [1, 4]


# -----------------------------
# TEST: load_all_shards respects capacity
# -----------------------------
def test_load_all_shards_capacity(tmp_path):
    rb = ReplayBuffer(capacity=3, shard_dir=tmp_path, shard_size_limit=1)

    # 5 pushes → 5 shards
    for i in range(5):
        rb.push(i, i, i, i , False)

    rb2 = ReplayBuffer(capacity=3, shard_dir=tmp_path)
    rb2.load_all_shards()

    assert len(rb2) == 3
    assert [t.state for t in rb2.buffer] == [2, 3, 4]


# -----------------------------
# TEST: ignores non-pkl files
# -----------------------------
def test_load_all_shards_ignores_non_pkl(tmp_path):
    # create junk file
    with open(tmp_path / "junk.txt", "w") as f:
        f.write("ignore me")

    rb = ReplayBuffer(capacity=5, shard_dir=tmp_path)
    rb.push(1, 2, 3, 4, False)

    rb2 = ReplayBuffer(capacity=5, shard_dir=tmp_path)
    rb2.load_all_shards()

    assert len(rb2) == 0
