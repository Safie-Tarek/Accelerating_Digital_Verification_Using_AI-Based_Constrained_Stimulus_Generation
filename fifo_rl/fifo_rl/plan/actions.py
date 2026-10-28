"""The 9 action blocks of a test plan (plan-based design).

One action = one block of a test plan = one "ACT ..." line of the plan file. A test runs its
blocks back to back in ONE simulation, with no reset between them, so the FIFO keeps its fill
level: WRITE then IDLE holds a full FIFO (cx_full <0,0,1>), which no block can do on its own.

    id  name           wr%  rd%  rst%  cycles
    0   RESET            0    0   100      2   back to count 0 inside the same test
    1   IDLE             0    0     0      4   hold the current level
    2   WRITE          100    0     0     10   fills an empty FIFO into overflow
    3   READ             0  100     0     10   drains a full FIFO into underflow
    4   WRITE_READ     100  100     0      4   empty -> count 1, full -> count 7, else holds
    5   RAND50          50   50     0     10   random traffic around the current level
    6   RESET_TRAFFIC   50   50   100     16   random wr/rd while reset is held (3 reset-only bins)
    7   BASELINE        70   30    10     18   the verification plan's FIFO_4 recipe
    8   END_TEST         -    -     -      0   closes the plan; chosen by the agent like the others

The UVM test only needs the four numbers of a block (wr %, rd %, rst %, cycles); the name is for
people. The old 22-action design (one whole run per action) stays in fifo_rl/actions.py.
"""

from dataclasses import dataclass

RESET_CYCLE = (0, 0, 0)  # (rst_n, wr_en, rd_en): the reset the test applies before the first block


@dataclass(frozen=True)
class ActionSpec:
    id: int
    name: str
    wr_pct: int    # chance (0-100) of wr_en=1 in each cycle of the block
    rd_pct: int    # chance (0-100) of rd_en=1 in each cycle of the block
    rst_pct: int   # chance (0-100) of rst_n=0 in each cycle of the block
    n_cycles: int  # clock cycles of the block (0 for END_TEST)

    @property
    def is_end_test(self):
        """Input: nothing. Output: True for the END_TEST action (it closes the plan, runs no cycles)."""
        return self.name == "END_TEST"

    @property
    def deterministic(self):
        """Does every run of this block give exactly the same cycles?

        Input:  the three percentages.
        Output: True or False.
        Logic:  0 % and 100 % never need a random draw; any value in between does.
        """
        return all(pct in (0, 100) for pct in (self.wr_pct, self.rd_pct, self.rst_pct))


ACTIONS = (
    ActionSpec(0, "RESET", 0, 0, 100, 2),
    ActionSpec(1, "IDLE", 0, 0, 0, 4),
    ActionSpec(2, "WRITE", 100, 0, 0, 10),
    ActionSpec(3, "READ", 0, 100, 0, 10),
    ActionSpec(4, "WRITE_READ", 100, 100, 0, 4),
    ActionSpec(5, "RAND50", 50, 50, 0, 10),
    ActionSpec(6, "RESET_TRAFFIC", 50, 50, 100, 16),
    ActionSpec(7, "BASELINE", 70, 30, 10, 18),
    ActionSpec(8, "END_TEST", 0, 0, 0, 0),
)
N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {spec.name: spec.id for spec in ACTIONS}
END_TEST_ID = ACTION_INDEX["END_TEST"]


def get_action(action_id):
    """Look up one action by its id.

    Input:  action_id, an int 0-8.
    Output: the ActionSpec with that id.
    Logic:  check the id is in range (ValueError if not), then index into ACTIONS.
    """
    if not 0 <= action_id < N_ACTIONS:
        raise ValueError(f"action_id must be in 0..{N_ACTIONS - 1}, got {action_id}")
    return ACTIONS[action_id]


def _draw(pct, rng):
    """Draw one bit that is 1 with probability pct %.

    Input:  pct, an int 0-100; rng, a numpy Generator (may be None when pct is 0 or 100).
    Output: 0 or 1.
    Logic:  0 % -> 0 and 100 % -> 1 without touching rng; otherwise a uniform number in [0, 100)
            below pct gives 1.
    """
    if pct <= 0:
        return 0
    if pct >= 100:
        return 1
    return int(rng.random() * 100 < pct)


def block_cycles(wr_pct, rd_pct, rst_pct, n_cycles, rng=None):
    """The cycles of one block, from its four numbers (what the UVM test receives).

    Input:  the block's wr %, rd %, rst % and number of cycles; rng, a numpy Generator
            (needed only when a percentage is strictly between 0 and 100).
    Output: a list of n_cycles (rst_n, wr_en, rd_en) tuples.
    Logic:  for every cycle draw rst_n = 0 with rst %, wr_en = 1 with wr %, rd_en = 1 with rd %,
            always in that order, so the same seed gives the same cycles.
    """
    if rng is None and not all(pct in (0, 100) for pct in (wr_pct, rd_pct, rst_pct)):
        raise ValueError("a random block needs rng=np.random.default_rng(seed)")
    cycles = []
    for _ in range(n_cycles):
        rst_n = 1 - _draw(rst_pct, rng)
        wr_en = _draw(wr_pct, rng)
        rd_en = _draw(rd_pct, rng)
        cycles.append((rst_n, wr_en, rd_en))
    return cycles


def to_cycles(action_id, rng=None):
    """Expand one action into its clock cycles (no reset in front: blocks follow each other).

    Input:  action_id, an int 0-8; rng, a numpy Generator for the random blocks.
    Output: a list of (rst_n, wr_en, rd_en), length spec.n_cycles (empty for END_TEST).
    """
    spec = get_action(action_id)
    return block_cycles(spec.wr_pct, spec.rd_pct, spec.rst_pct, spec.n_cycles, rng)


def plan_cycles(action_ids, rng=None):
    """All cycles of one test: the reset the test applies first, then every block in order.

    Input:  action_ids, the plan's action ids (END_TEST adds nothing); rng as in to_cycles.
    Output: a list of (rst_n, wr_en, rd_en) starting with RESET_CYCLE.
    """
    cycles = [RESET_CYCLE]
    for action_id in action_ids:
        cycles += to_cycles(action_id, rng)
    return cycles
