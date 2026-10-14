"""Backend on the Python fake FIFO: under a millisecond per run, used for training."""

import numpy as np

from fifo_rl import fake_fifo
from fifo_rl.actions import to_cycles
from fifo_rl.backends.base import Backend


class FakeBackend(Backend):
    def run(self, action_id, seed):
        """Run one action on the Python FIFO.

        Input:  action_id, an int 0-21; seed, an int.
        Output: a bool array of 64: True where the run hit the bin.
        Logic:  1. make a random generator from the seed (same seed -> same run, like -sv_seed)
                2. turn the action into its per-cycle stimulus (actions.to_cycles)
                3. simulate it and return the bins hit (fake_fifo.hits)
        """
        cycles = to_cycles(action_id, np.random.default_rng(seed))
        return fake_fifo.hits(cycles)
