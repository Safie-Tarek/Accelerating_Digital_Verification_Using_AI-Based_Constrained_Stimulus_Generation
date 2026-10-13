import numpy as np
import pytest

from fifo_rl.state import StateEncoder


def hits(*indices):
    run_hits = np.zeros(64, dtype=bool)
    run_hits[list(indices)] = True
    return run_hits


def test_size():
    assert StateEncoder(budget=20).size == 87
    assert StateEncoder(budget=20, include_last_action=False).size == 65
    assert StateEncoder(budget=20, n_actions=10).size == 75


def test_fresh_state():
    state = StateEncoder(budget=20).reset()
    assert state.shape == (87,)
    assert state.dtype == np.float32
    assert state[64] == 1.0
    assert state[:64].sum() == 0
    assert state[65:].sum() == 0


def test_update_ticks_bins_and_counts_new_ones():
    enc = StateEncoder(budget=20)
    state, n_new = enc.update(hits(0, 5, 40), action_id=3)
    assert n_new == 3
    assert state[[0, 5, 40]].tolist() == [1.0, 1.0, 1.0]
    assert state[:64].sum() == 3


def test_bins_stay_ticked_and_old_bins_are_not_new():
    enc = StateEncoder(budget=20)
    enc.update(hits(0, 5, 40), action_id=3)
    state, n_new = enc.update(hits(5, 41), action_id=4)
    assert n_new == 1
    assert state[[0, 5, 40, 41]].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert state[:64].sum() == 4


def test_budget_goes_down_one_run_at_a_time():
    enc = StateEncoder(budget=20)
    state, _ = enc.update(hits(), action_id=0)
    assert state[64] == pytest.approx(0.95)
    state, _ = enc.update(hits(), action_id=0)
    assert state[64] == pytest.approx(0.90)
    assert enc.runs_left == 18


def test_last_action_is_one_hot():
    enc = StateEncoder(budget=20)
    state, _ = enc.update(hits(), action_id=7)
    assert state[72] == 1.0
    assert state[65:].sum() == 1
    state, _ = enc.update(hits(), action_id=12)
    assert state[77] == 1.0
    assert state[65:].sum() == 1


def test_without_last_action():
    enc = StateEncoder(budget=20, include_last_action=False)
    state, n_new = enc.update(hits(1, 2), action_id=7)
    assert state.shape == (65,)
    assert n_new == 2
    assert state[64] == pytest.approx(0.95)


def test_values_stay_between_0_and_1():
    enc = StateEncoder(budget=3)
    for action_id in (0, 21, 5):
        state, _ = enc.update(np.ones(64, dtype=bool), action_id)
        assert state.min() >= 0.0
        assert state.max() <= 1.0
    assert state[64] == 0.0


def test_all_covered():
    enc = StateEncoder(budget=20)
    assert not enc.all_covered
    enc.update(np.ones(64, dtype=bool), action_id=0)
    assert enc.all_covered


def test_reset_starts_a_new_episode():
    enc = StateEncoder(budget=20)
    enc.update(hits(1, 2, 3), action_id=9)
    state = enc.reset()
    assert state[:64].sum() == 0
    assert state[64] == 1.0
    assert state[65:].sum() == 0
    assert enc.runs_left == 20


def test_checklist_is_a_copy():
    enc = StateEncoder(budget=20)
    enc.checklist[:] = True
    assert not enc.checklist.any()


def test_run_hits_accepts_0_1_lists():
    enc = StateEncoder(budget=20)
    _, n_new = enc.update([1] * 10 + [0] * 54, action_id=0)
    assert n_new == 10


def test_errors():
    with pytest.raises(ValueError):
        StateEncoder(budget=0)
    with pytest.raises(ValueError):
        StateEncoder(budget=2.5)

    enc = StateEncoder(budget=1)
    with pytest.raises(ValueError):
        enc.update(np.zeros(63, dtype=bool), action_id=0)
    with pytest.raises(ValueError):
        enc.update(hits(), action_id=22)
    with pytest.raises(ValueError):
        enc.update(hits(), action_id=-1)

    enc.update(hits(), action_id=0)
    with pytest.raises(ValueError):
        enc.update(hits(), action_id=0)
