"""Tests for fifo_rl/agents/replay_buffer.py (Contract E)."""

import numpy as np
import pytest

from fifo_rl.agents.replay_buffer import ReplayBuffer

DIM = 87


def _transition(i):
    """A transition whose every field encodes i, so a sampled row says where it came from."""
    s = np.full(DIM, i, dtype=np.float32)
    return s, i % 22, float(i) / 10, s + 0.5, i % 2 == 0


def _filled(n, capacity=100, seed=0):
    buf = ReplayBuffer(capacity, DIM, seed=seed)
    for i in range(n):
        buf.push(*_transition(i))
    return buf


def test_shapes_and_dtypes():
    batch = _filled(10).sample(64)
    assert batch.states.shape == (64, DIM) and batch.states.dtype == np.float32
    assert batch.next_states.shape == (64, DIM) and batch.next_states.dtype == np.float32
    assert batch.actions.shape == (64,) and batch.actions.dtype == np.int64
    assert batch.rewards.shape == (64,) and batch.rewards.dtype == np.float32
    assert batch.dones.shape == (64,) and batch.dones.dtype == np.float32


def test_rows_stay_together():
    """Every sampled row is one pushed transition, fields not mixed between transitions."""
    batch = _filled(50).sample(200)
    i = batch.states[:, 0].astype(int)
    assert (batch.states == i[:, None]).all()
    assert (batch.next_states == i[:, None] + 0.5).all()
    assert (batch.actions == i % 22).all()
    np.testing.assert_allclose(batch.rewards, i / 10, rtol=1e-6)
    assert (batch.dones == (i % 2 == 0)).all()


def test_len_grows_then_stops_at_capacity():
    buf = ReplayBuffer(5, DIM)
    assert len(buf) == 0
    for i in range(12):
        buf.push(*_transition(i))
        assert len(buf) == min(i + 1, 5)


def test_wrap_around_keeps_the_newest():
    """After 12 pushes into 5 slots only transitions 7..11 remain."""
    buf = _filled(12, capacity=5)
    seen = set(buf.sample(500).states[:, 0].astype(int))
    assert seen == {7, 8, 9, 10, 11}


def test_same_seed_same_sample():
    a, b = _filled(30, seed=4).sample(16), _filled(30, seed=4).sample(16)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    c = _filled(30, seed=5).sample(16)
    assert not np.array_equal(a.states, c.states)


def test_sample_is_a_copy():
    buf = _filled(3)
    batch = buf.sample(8)
    batch.states[:] = -1
    assert (buf.sample(50).states >= 0).all()


def test_bad_arguments():
    with pytest.raises(ValueError):
        ReplayBuffer(0, DIM)
    with pytest.raises(ValueError):
        ReplayBuffer(10, 0)
    with pytest.raises(ValueError):
        ReplayBuffer(10, DIM).sample(4)  # empty
    with pytest.raises(ValueError):
        _filled(3).sample(0)
