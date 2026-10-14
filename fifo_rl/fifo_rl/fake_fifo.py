"""Python copy of the FIFO and of how the UVM monitor samples it ("fake FIFO").

Mirrors "src codes/FIFO.sv" (corrected version) and FIFO_monitor.sv. Each clock cycle:
    - the driver applies (rst_n, wr_en, rd_en) at negedge
    - the DUT updates count at posedge (reset is asynchronous)
    - the monitor samples at the next negedge: this cycle's wr_en/rd_en together with the
      flags of the NEW count, and the covergroup ticks one bin per coverpoint and per cross

So the write that makes the FIFO full is already seen as full=1, overflow=1, wr_ack=0.
"""

from typing import NamedTuple

import numpy as np

from fifo_rl.coverage_model import BIN_INDEX, COVERPOINTS, CROSSES, N_BINS, bin_name, coverpoint_label, cross_label

DEPTH = 8  # the only place the FIFO depth is written in Python


class Sample(NamedTuple):
    """What the monitor sees in one cycle (field names as in FIFO_seq_item)."""
    wr_en: int
    rd_en: int
    wr_ack: int
    overflow: int
    full: int
    empty: int
    almostfull: int
    almostempty: int
    underflow: int


def step(count, rst_n, wr_en, rd_en):
    """Simulate one clock cycle of the FIFO.

    Input:  count, the number of words in the FIFO before this cycle (0..8);
            rst_n, wr_en, rd_en, this cycle's inputs (0 or 1).
    Output: (new count, Sample) - the Sample is what the monitor records for this cycle.
    Logic:  1. update count like FIFO.sv:
               reset -> 0; otherwise a write is accepted if not full, a read if not empty,
               count += accepted write - accepted read
            2. compute the 7 flags from the NEW count (as the monitor sees them at negedge):
               full = 8, empty = 0, almostfull = 7, almostempty = 1,
               wr_ack = wr_en and not full, overflow = full and wr_en, underflow = empty and rd_en
    """
    if not rst_n:
        count = 0
    else:
        write = wr_en and count < DEPTH
        read = rd_en and count > 0
        count += int(bool(write)) - int(bool(read))
    full = int(count == DEPTH)
    empty = int(count == 0)
    sample = Sample(
        wr_en=int(bool(wr_en)),
        rd_en=int(bool(rd_en)),
        wr_ack=int(bool(wr_en) and count != DEPTH),
        overflow=int(bool(full and wr_en)),
        full=full,
        empty=empty,
        almostfull=int(count == DEPTH - 1),
        almostempty=int(count == 1),
        underflow=int(bool(empty and rd_en)),
    )
    return count, sample


def simulate(cycles):
    """Simulate a whole run.

    Input:  cycles, a list of (rst_n, wr_en, rd_en), one per clock cycle (e.g. from to_cycles).
    Output: a list of Sample, one per cycle.
    Logic:  refuse a run whose first cycle is not a reset (in the real DUT count starts as X),
            then start from count = 0 and call step() for every cycle, keeping each Sample.
    """
    cycles = list(cycles)
    if not cycles or cycles[0][0] != 0:
        raise ValueError("a run must start with a reset cycle (rst_n=0)")
    count = 0
    samples = []
    for rst_n, wr_en, rd_en in cycles:
        count, sample = step(count, rst_n, wr_en, rd_en)
        samples.append(sample)
    return samples


def _flag(item):
    """Name of the Sample field behind a coverpoint or cross.

    Input:  item, a coverpoint/cross name such as "cp_full" or "cx_full".
    Output: the field name, e.g. "full".
    Logic:  both prefixes "cp_" and "cx_" are 3 characters, so drop the first 3.
    """
    return item[3:]


# Bin index of every (cross, (wr_en, rd_en, flag)) combo; None for the ignore_bins.
# Built once so that sample_bins() only does dictionary look-ups.
_CROSS_INDEX = {
    (cx, combo): BIN_INDEX.get(bin_name(cx, cross_label(*combo)))
    for cx in CROSSES
    for combo in np.ndindex(2, 2, 2)
}


def sample_bins(sample):
    """Which bins one monitor sample ticks.

    Input:  sample, a Sample.
    Output: a list of 16 bin indices (positions in BIN_NAMES).
    Logic:  like cg_FIFO.sample():
            - each of the 9 coverpoints ticks bin auto[value of its signal]
            - each of the 7 crosses ticks bin <wr_en, rd_en, its flag>
            A cross combo listed in ignore_bins should never happen; if it does the model is
            wrong, so raise instead of silently dropping it like Questa would.
    """
    indices = [BIN_INDEX[bin_name(cp, coverpoint_label(getattr(sample, _flag(cp))))] for cp in COVERPOINTS]
    for cx in CROSSES:
        combo = (sample.wr_en, sample.rd_en, getattr(sample, _flag(cx)))
        index = _CROSS_INDEX[(cx, combo)]
        if index is None:
            raise RuntimeError(f"ignore bin hit: {cx} (wr_en, rd_en, flag) = {combo}")
        indices.append(index)
    return indices


def bin_hit_counts(samples):
    """How many times each bin was hit in a run (same numbers as Questa's report).

    Input:  samples, a list of Sample (from simulate).
    Output: an int numpy array of N_BINS (64), in BIN_NAMES order.
    Logic:  start from zeros and add 1 to the 16 bins of every sample.
    """
    counts = np.zeros(N_BINS, dtype=np.int64)
    for sample in samples:
        counts[sample_bins(sample)] += 1
    return counts


def hits(cycles):
    """Which bins a run hits at least once - the main entry point of this file.

    Input:  cycles, a list of (rst_n, wr_en, rd_en) starting with a reset.
    Output: a bool numpy array of N_BINS (64): True = hit at least once.
    Logic:  simulate the run, count the hits per bin, then keep only "count > 0".
    """
    return bin_hit_counts(simulate(cycles)) > 0
