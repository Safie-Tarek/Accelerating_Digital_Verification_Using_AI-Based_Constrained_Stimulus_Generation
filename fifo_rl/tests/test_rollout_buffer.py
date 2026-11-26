"""Tests for fifo_rl/agents/rollout_buffer.py: the A2C memory and the advantage maths (GAE)."""

import numpy as np
import pytest

from fifo_rl.agents.rollout_buffer import RolloutBuffer, compute_gae
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID

WRITE, IDLE = ACTION_INDEX["WRITE"], ACTION_INDEX["IDLE"]


def state(x, size=93):
    return np.full(size, x, dtype=np.float32)


# --- the buffer ---------------------------------------------------------------------------------

def test_counts_tests_by_end_test_only():
    buf = RolloutBuffer()
    buf.add(state(0), WRITE, 0.1, state(1), False)
    buf.add(state(1), IDLE, 0.0, state(2), False)
    assert len(buf) == 2 and buf.n_tests == 0          # half a test: not finished yet
    buf.add(state(2), END_TEST_ID, -0.01, state(3), False)
    assert buf.n_tests == 1 and not buf.ended
    buf.add(state(3), END_TEST_ID, 0.2, state(4), True)  # an empty test that ends the campaign
    assert buf.n_tests == 2 and buf.ended


def test_arrays_keep_order_shapes_and_types():
    buf = RolloutBuffer()
    buf.add(state(0), WRITE, 0.5, state(1), False)
    buf.add(state(1), END_TEST_ID, -1.0, state(2), True)
    r = buf.arrays()
    assert r.states.shape == r.next_states.shape == (2, 93)
    assert r.states.dtype == r.next_states.dtype == r.rewards.dtype == r.dones.dtype == np.float32
    assert r.actions.dtype == np.int64
    assert r.actions.tolist() == [WRITE, END_TEST_ID]
    assert r.rewards.tolist() == [0.5, -1.0] and r.dones.tolist() == [0.0, 1.0]
    assert r.states[1, 0] == 1.0 and r.next_states[1, 0] == 2.0


def test_add_keeps_a_copy_of_the_state():
    buf = RolloutBuffer()
    s = state(0)
    buf.add(s, WRITE, 0.0, s, False)
    s[:] = 7.0                                          # the caller reuses its array
    assert buf.arrays().states[0, 0] == 0.0


def test_clear_empties_everything():
    buf = RolloutBuffer()
    buf.add(state(0), END_TEST_ID, 0.0, state(1), True)
    buf.clear()
    assert len(buf) == 0 and buf.n_tests == 0 and not buf.ended
    with pytest.raises(ValueError):
        buf.arrays()


# --- compute_gae --------------------------------------------------------------------------------

def test_gae_worked_out_by_hand():
    # delta = [0.1 + 0.9*0.3 - 0.2, 0 + 0.9*0.4 - 0.3, 0.5 + 0.9*0.6 - 0.4] = [0.17, 0.06, 0.64]
    # A (lam=1, backwards) = [0.17 + 0.9*0.636, 0.06 + 0.9*0.64, 0.64] = [0.7424, 0.636, 0.64]
    # R = A + V = [0.9424, 0.936, 1.04]
    adv, ret = compute_gae([0.1, 0.0, 0.5], [0.2, 0.3, 0.4], [0.3, 0.4, 0.6], [0, 0, 0], gamma=0.9, lam=1.0)
    np.testing.assert_allclose(adv, [0.7424, 0.636, 0.64], atol=1e-6)
    np.testing.assert_allclose(ret, [0.9424, 0.936, 1.04], atol=1e-6)
    assert adv.dtype == ret.dtype == np.float32


def test_done_stops_the_return_between_campaigns():
    # step 0 ends a campaign: its next value (5) and step 1's advantage must not reach it
    adv, ret = compute_gae([1.0, 1.0], [0.0, 0.0], [5.0, 5.0], [1, 0], gamma=0.5, lam=1.0)
    np.testing.assert_allclose(adv, [1.0, 3.5])
    np.testing.assert_allclose(ret, [1.0, 3.5])


def test_lam_0_is_the_one_step_td_error():
    rewards, values, next_values = [0.1, 0.0, 0.5], [0.2, 0.3, 0.4], [0.3, 0.4, 0.6]
    adv, _ = compute_gae(rewards, values, next_values, [0, 0, 0], gamma=0.9, lam=0.0)
    np.testing.assert_allclose(adv, [0.17, 0.06, 0.64], atol=1e-6)


def test_lam_1_returns_are_the_discounted_rewards_until_done():
    # campaign ends on the last step, so the return is 1 + 0.9*2 + 0.81*3 = 5.23 whatever V guesses
    _, ret = compute_gae([1.0, 2.0, 3.0], [9.0, -4.0, 0.5], [-4.0, 0.5, 100.0], [0, 0, 1], gamma=0.9, lam=1.0)
    np.testing.assert_allclose(ret, [5.23, 4.7, 3.0], atol=1e-5)


def test_gae_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.0], [0.0, 0.0], [0, 0], 0.9, 1.0)   # lengths differ
    with pytest.raises(ValueError):
        compute_gae([1.0], [0.0], [0.0], [0], 1.5, 1.0)                # gamma out of range
    with pytest.raises(ValueError):
        compute_gae([1.0], [0.0], [0.0], [0], 0.9, -0.1)               # lam out of range


def test_buffer_and_gae_together():
    """A whole test of 2 blocks + END_TEST through the buffer, then GAE on its arrays."""
    buf = RolloutBuffer()
    buf.add(state(0), WRITE, 0.1, state(1), False)
    buf.add(state(1), IDLE, 0.0, state(2), False)
    buf.add(state(2), END_TEST_ID, 0.5, state(3), False)
    r = buf.arrays()
    adv, ret = compute_gae(r.rewards, [0.2, 0.3, 0.4], [0.3, 0.4, 0.6], r.dones, gamma=0.9, lam=1.0)
    np.testing.assert_allclose(ret, [0.9424, 0.936, 1.04], atol=1e-6)
    assert buf.n_tests == 1
