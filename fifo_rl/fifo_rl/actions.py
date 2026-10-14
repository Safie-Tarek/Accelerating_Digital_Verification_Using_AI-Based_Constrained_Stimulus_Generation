"""The 22 actions the agent can choose from (Contract B).

One action = the recipe for one whole simulation run:
    1 reset cycle  ->  `level` write cycles (fill)  ->  `k_cycles` cycles of a traffic pattern

    ids  0-19  level L in (0, 1, 7, 8) x pattern P in (idle, wr, rd, wr_rd, rand50)
               id = 5 * (index of L) + (index of P), e.g. L8_idle = 15
    id   20    reset_with_traffic: reset held on while writing/reading at random
    id   21    baseline_random: the verification plan's FIFO_4 recipe (wr 70 / rd 30 / rst 10)

Levels 0, 1, 7, 8 are the counts where a flag is set (empty, almostempty, almostfull, full).
The four fixed patterns are the four (wr_en, rd_en) combinations the crosses ask about.

A recipe is turned into stimulus in two ways:
    to_cycles(id, rng)  -> list of (rst_n, wr_en, rd_en) per cycle, for the Python FIFO
    to_plusargs(id)     -> command-line arguments for FIFO_rl_test in Questa
"""

from dataclasses import dataclass

LEVELS = (0, 1, 7, 8)

# (name, write %, read %)
PATTERNS = (
    ("idle", 0, 0),
    ("wr", 100, 0),
    ("rd", 0, 100),
    ("wr_rd", 100, 100),
    ("rand50", 50, 50),
)

K_CYCLES = 10         # pattern length: drains a full FIFO into underflow, fills an empty one into overflow
K_RESET_TRAFFIC = 16  # all four wr/rd combos appear during reset in ~96% of runs
K_BASELINE = 18       # as long as the longest scenario (fill 8 + 10)

RESET_CYCLE = (0, 0, 0)  # (rst_n, wr_en, rd_en)
FILL_CYCLE = (1, 1, 0)


@dataclass(frozen=True)
class ActionSpec:
    id: int
    name: str
    level: int     # write cycles right after the reset
    wr_pct: int    # chance (0-100) of wr_en=1 in each pattern cycle
    rd_pct: int    # chance (0-100) of rd_en=1 in each pattern cycle
    rst_pct: int   # chance (0-100) of rst_n=0 in each pattern cycle
    k_cycles: int  # number of pattern cycles

    @property
    def deterministic(self):
        """Does every run of this action give exactly the same cycles?

        Input:  the action's three percentages (wr_pct, rd_pct, rst_pct).
        Output: True or False.
        Logic:  a percentage of 0 or 100 never needs a random draw; any value in between does.
                So the action is deterministic only if all three are 0 or 100.
        """
        return all(pct in (0, 100) for pct in (self.wr_pct, self.rd_pct, self.rst_pct))

    @property
    def n_cycles(self):
        """Length of one run in clock cycles.

        Input:  the action's level and k_cycles.
        Output: an int = 1 (reset) + level (fill writes) + k_cycles (pattern).
        """
        return 1 + self.level + self.k_cycles


def _build_actions():
    """Create the 22 ActionSpecs in their fixed id order.

    Input:  nothing (uses LEVELS, PATTERNS and the K_* constants above).
    Output: a tuple of 22 ActionSpec.
    Logic:  for each level, for each pattern -> one action (ids 0-19, rst 0%, K_CYCLES cycles);
            then append reset_with_traffic (id 20) and baseline_random (id 21).
            The id is simply the position in the list, so the order here defines the ids.
    """
    actions = []
    for level in LEVELS:
        for pattern, wr_pct, rd_pct in PATTERNS:
            actions.append(ActionSpec(len(actions), f"L{level}_{pattern}", level, wr_pct, rd_pct, 0, K_CYCLES))
    actions.append(ActionSpec(len(actions), "reset_with_traffic", 0, 50, 50, 100, K_RESET_TRAFFIC))
    actions.append(ActionSpec(len(actions), "baseline_random", 0, 70, 30, 10, K_BASELINE))
    return tuple(actions)


ACTIONS = _build_actions()
N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {spec.name: spec.id for spec in ACTIONS}


def get_action(action_id):
    """Look up one action by its id.

    Input:  action_id, an int.
    Output: the ActionSpec with that id.
    Logic:  check the id is in 0..21 (raise ValueError if not), then index into ACTIONS.
    """
    if not 0 <= action_id < N_ACTIONS:
        raise ValueError(f"action_id must be in 0..{N_ACTIONS - 1}, got {action_id}")
    return ACTIONS[action_id]


def _draw(pct, rng):
    """Draw one bit that is 1 with probability pct %.

    Input:  pct, an int 0-100; rng, a numpy Generator (may be None when pct is 0 or 100).
    Output: 0 or 1.
    Logic:  0% -> always 0 and 100% -> always 1, without touching rng (this is what makes the
            fixed actions seed-independent). Otherwise draw a uniform number in [0, 100) and
            return 1 if it is below pct.
    """
    if pct <= 0:
        return 0
    if pct >= 100:
        return 1
    return int(rng.random() * 100 < pct)


def to_cycles(action_id, rng=None):
    """Expand an action into the stimulus of every clock cycle of one run.

    Input:  action_id, an int 0-21;
            rng, a numpy Generator (np.random.default_rng(seed)); needed only for random actions.
    Output: a list of (rst_n, wr_en, rd_en) tuples, one per cycle, length = spec.n_cycles.
    Logic:  1. one reset cycle (0, 0, 0)  - the DUT's count starts as X, so every run resets first
            2. `level` fill cycles (1, 1, 0) - write only, no reset, no read, to reach the level
            3. `k_cycles` pattern cycles - each bit drawn with its percentage:
               rst_n = 0 with rst_pct %, wr_en = 1 with wr_pct %, rd_en = 1 with rd_pct %
    """
    spec = get_action(action_id)
    if rng is None and not spec.deterministic:
        raise ValueError(f"{spec.name} is random: pass rng=np.random.default_rng(seed)")
    cycles = [RESET_CYCLE] + [FILL_CYCLE] * spec.level
    for _ in range(spec.k_cycles):
        rst_n = 1 - _draw(spec.rst_pct, rng)
        wr_en = _draw(spec.wr_pct, rng)
        rd_en = _draw(spec.rd_pct, rng)
        cycles.append((rst_n, wr_en, rd_en))
    return cycles


def to_plusargs(action_id):
    """Turn an action into the command-line arguments of the real Questa test.

    Input:  action_id, an int 0-21.
    Output: a list of strings, e.g. ["+UVM_TESTNAME=FIFO_rl_test", "+FILL_LEVEL=8", ...].
    Logic:  copy the 5 recipe numbers (level, wr %, rd %, rst %, cycles) into plusargs that
            FIFO_rl_test reads with $value$plusargs. The names are a proposal still to be
            agreed with the verification team. The seed is not included: the Questa runner
            adds -sv_seed itself.
    """
    spec = get_action(action_id)
    return [
        "+UVM_TESTNAME=FIFO_rl_test",
        f"+FILL_LEVEL={spec.level}",
        f"+WR_PCT={spec.wr_pct}",
        f"+RD_PCT={spec.rd_pct}",
        f"+RST_PCT={spec.rst_pct}",
        f"+NUM_CYCLES={spec.k_cycles}",
    ]
