"""Contract C: a backend runs one action and says which bins it hit.

The env only talks to this interface, so the fake FIFO and real Questa are interchangeable.
"""


class Backend:
    def run(self, action_id, seed):
        """Simulate one run of an action.

        Input:  action_id, an int 0-21; seed, an int (same seed -> same run).
        Output: a bool array of N_BINS (64): True where the run hit the bin.
        Logic:  none here - every backend (fake FIFO, Questa) must override this method.
        """
        raise NotImplementedError
