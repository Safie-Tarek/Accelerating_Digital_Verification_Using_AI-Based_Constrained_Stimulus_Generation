"""Tests for fifo_rl/fake_fifo.py: the Python FIFO must behave like FIFO.sv + the monitor."""

from pathlib import Path

import numpy as np
import pytest

from fifo_rl import fake_fifo
from fifo_rl.coverage_model import BIN_INDEX, N_BINS
from fifo_rl.fake_fifo import DEPTH, simulate, step
from fifo_rl.questa.report_parser import hits_vector

FIXTURES = Path(__file__).parent / "fixtures"


def count_after(cycles):
    """Helper. Input: a list of (rst_n, wr_en, rd_en). Output: the FIFO count after the last cycle."""
    count = 0
    for rst_n, wr_en, rd_en in cycles:
        count, _ = step(count, rst_n, wr_en, rd_en)
    return count


def test_count_rules():
    """count changes exactly like FIFO.sv for every input combination and corner (empty / full)."""
    assert step(5, 0, 1, 1)[0] == 0         # reset wins
    assert step(3, 1, 1, 0)[0] == 4         # write
    assert step(DEPTH, 1, 1, 0)[0] == DEPTH  # write when full is dropped
    assert step(3, 1, 0, 1)[0] == 2         # read
    assert step(0, 1, 0, 1)[0] == 0         # read when empty is dropped
    assert step(3, 1, 1, 1)[0] == 3         # write + read in the middle
    assert step(0, 1, 1, 1)[0] == 1         # write + read when empty: only the write
    assert step(DEPTH, 1, 1, 1)[0] == DEPTH - 1  # write + read when full: only the read
    assert step(3, 1, 0, 0)[0] == 3         # idle


def test_flags_use_the_new_count():
    """Flags are computed from the count AFTER the cycle, as the monitor samples them."""
    _, s = step(DEPTH - 1, 1, 1, 0)  # the write that fills the FIFO
    assert (s.full, s.overflow, s.wr_ack, s.almostfull) == (1, 1, 0, 0)
    _, s = step(DEPTH, 1, 0, 1)      # read from full -> 7
    assert (s.full, s.almostfull) == (0, 1)
    _, s = step(0, 1, 1, 0)          # first write -> 1
    assert (s.empty, s.almostempty, s.wr_ack) == (0, 1, 1)
    _, s = step(0, 1, 0, 1)          # read when empty
    assert (s.empty, s.underflow) == (1, 1)


def test_during_reset_underflow_follows_rd_en_and_wr_ack_follows_wr_en():
    """In a reset cycle count = 0, so underflow = rd_en and wr_ack = wr_en (the reset-only bins)."""
    _, s = step(5, 0, 1, 1)
    assert (s.empty, s.underflow, s.wr_ack, s.overflow) == (1, 1, 1, 0)


def test_run_must_start_with_reset():
    """simulate() refuses a run that does not begin with a reset cycle, or an empty run."""
    with pytest.raises(ValueError):
        simulate([(1, 1, 0)])
    with pytest.raises(ValueError):
        simulate([])


def test_fill_to_full_then_drain():
    """Extra writes stop at 8; extra reads stop at 0."""
    assert count_after([(0, 0, 0)] + [(1, 1, 0)] * 12) == DEPTH
    assert count_after([(0, 0, 0)] + [(1, 1, 0)] * 8 + [(1, 0, 1)] * 12) == 0


def test_every_sample_hits_16_bins():
    """Each cycle ticks exactly 16 bins (9 coverpoints + 7 crosses), like cg_FIFO.sample()."""
    rng = np.random.default_rng(0)
    cycles = [(0, 0, 0)] + [tuple(int(x) for x in rng.integers(0, 2, 3)) for _ in range(500)]
    counts = fake_fifo.bin_hit_counts(simulate(cycles))
    assert counts.shape == (N_BINS,)
    assert counts.sum() == 16 * len(cycles)


def test_reset_cycle_alone_hits_the_16_free_bins():
    """The opening reset cycle alone ticks 16 bins - every run gets these for free."""
    hit = fake_fifo.hits([(0, 0, 0)])
    assert hit.sum() == 16
    assert hit[BIN_INDEX["cx_empty/<auto[0],auto[0],auto[1]>"]]
    assert hit[BIN_INDEX["cp_empty/auto[1]"]]
    assert not hit[BIN_INDEX["cp_empty/auto[0]"]]


def test_long_random_run_closes_coverage_and_never_hits_an_ignore_bin():
    """3000 cycles of plain 70/30/10 random reach all 64 bins, and no ignore bin is ever hit."""
    rng = np.random.default_rng(1)
    cycles = [(0, 0, 0)] + [
        (int(rng.random() >= 0.1), int(rng.random() < 0.7), int(rng.random() < 0.3)) for _ in range(3000)
    ]
    assert fake_fifo.hits(cycles).all()  # sample_bins raises on an ignore bin


@pytest.mark.parametrize("seed", range(20))
def test_matches_real_questa_100_cycle_report(seed):
    """Check against real Questa.

    Questa seed 1, first 100 cycles = reset + 99 write-only cycles with 10% random reset.
    Whatever the reset pattern, the set of bins must be exactly the one Questa reported.
    """
    real = hits_vector((FIXTURES / "cov_seed1_100cyc.txt").read_text())
    rng = np.random.default_rng(seed)
    cycles = [(0, 0, 0)] + [(int(rng.random() >= 0.1), 1, 0) for _ in range(99)]
    assert (fake_fifo.hits(cycles) == real).all()
