"""Reward of one simulation run (one env step).

    r = scale * (new_bins - step_cost + closure_bonus * [this run completed coverage])

All three numbers are in "bins": step_cost = 1 means one run costs as much as one new bin.
Over an episode that closes coverage in T runs, the unscaled rewards add up to
    N_BINS - T + closure_bonus
(every closing episode collects the same 64 bins), so the only way to earn more is to close in
fewer runs - exactly the project's metric.

Why these defaults (measured on the fake FIFO, random agent over the 22 actions):
    - a random agent already reaches ~63/64 bins; good and bad agents differ in the number of
      runs and in closing the last 1-2 bins.
    - step_cost = 1.0: with 0.1 a good agent (5 runs) scores only ~4% more than random (20 runs);
      with 1.0 the gap is ~30%, a clear learning signal. It does not change which action is
      best inside one step (every action pays it), and the agent has no "stop" action to abuse.
    - closure_bonus = 10: sign-off needs 64/64, not 63/64; the bonus makes closing within the
      budget clearly worth it (~10 runs' cost) without drowning the 64 bins.
    - scale = 1/64: rewards ~[-0.02, 1.2] and returns ~1, the range DQN defaults are tuned for.
      A constant factor: it changes the numbers, not which policy is best.
"""

from dataclasses import dataclass

from fifo_rl.coverage_model import N_BINS


@dataclass(frozen=True)
class RewardConfig:
    step_cost: float = 1.0        # price of one simulation run, in bins
    closure_bonus: float = 10.0   # paid once, on the run that reaches all bins
    scale: float = 1.0 / N_BINS   # 1/64: "fraction of coverage" units

    def __post_init__(self):
        """Reject settings that would break the reward.

        Input:  the three fields.
        Output: nothing; raises ValueError for a negative cost or bonus, or a scale <= 0.
        Logic:  a negative cost would pay the agent for wasting runs; a negative bonus would
                punish closing; scale <= 0 would flip or erase every reward.
        """
        if self.step_cost < 0:
            raise ValueError(f"step_cost must be >= 0, got {self.step_cost}")
        if self.closure_bonus < 0:
            raise ValueError(f"closure_bonus must be >= 0, got {self.closure_bonus}")
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")


# The reward the team agreed first (new bins - 0.1), kept for comparison experiments.
LEGACY_REWARD = RewardConfig(step_cost=0.1, closure_bonus=0.0, scale=1.0)


def compute_reward(new_bins, closed_now, config=RewardConfig()):
    """Reward of one run.

    Input:  new_bins, bins hit for the first time in this episode by this run (int >= 0);
            closed_now, True if this run brought coverage to all bins;
            config, a RewardConfig.
    Output: the reward (float).
    Logic:  new bins minus the run's cost, plus the bonus on the closing run, times the scale.
            Examples with the defaults: 3 new bins -> (3 - 1) / 64; nothing new -> -1 / 64;
            closing run with 4 new bins -> (4 - 1 + 10) / 64.
    """
    if new_bins < 0:
        raise ValueError(f"new_bins must be >= 0, got {new_bins}")
    raw = new_bins - config.step_cost + (config.closure_bonus if closed_now else 0.0)
    return raw * config.scale
