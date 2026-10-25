"""Contract E: exploration schedules, epsilon as a function of the agent's step count.

    schedule = LinearEpsilon(start=1.0, end=0.05, decay_steps=30_000)
    schedule.value(step) -> float

    LinearEpsilon   straight line from start to end over decay_steps, then flat at end
    ExpEpsilon      end + (start - end) * exp(-step / decay_steps): fast drop, long tail
"""

import math


class LinearEpsilon:
    def __init__(self, start, end, decay_steps):
        """Input: start and end epsilon (0..1), decay_steps (> 0). Output: nothing."""
        _check(start, end, decay_steps)
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def value(self, step):
        """Input: step (>= 0). Output: epsilon; start at step 0, end from decay_steps on."""
        frac = min(max(step, 0) / self.decay_steps, 1.0)
        return self.start + frac * (self.end - self.start)


class ExpEpsilon:
    def __init__(self, start, end, decay_steps):
        """Input: start and end epsilon (0..1), decay_steps = time constant (> 0). Output: nothing."""
        _check(start, end, decay_steps)
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def value(self, step):
        """Input: step (>= 0). Output: epsilon; start at step 0, ~63% of the way to end at decay_steps."""
        return self.end + (self.start - self.end) * math.exp(-max(step, 0) / self.decay_steps)


SCHEDULES = {"linear": LinearEpsilon, "exp": ExpEpsilon}


def make_schedule(cfg):
    """Input: a config dict (eps_schedule, eps_start, eps_end, eps_decay_steps). Output: the schedule."""
    kind = cfg.get("eps_schedule", "linear")
    if kind not in SCHEDULES:
        raise ValueError(f"unknown eps_schedule {kind!r}, expected one of {sorted(SCHEDULES)}")
    return SCHEDULES[kind](cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])


def _check(start, end, decay_steps):
    if not (0.0 <= end <= start <= 1.0):
        raise ValueError(f"need 0 <= end <= start <= 1, got start={start}, end={end}")
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be positive, got {decay_steps}")
