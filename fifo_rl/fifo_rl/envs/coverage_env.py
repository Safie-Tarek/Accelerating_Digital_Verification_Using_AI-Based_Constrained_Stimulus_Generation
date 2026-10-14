"""Contract D: the coverage-closure game, Gym style.

    state = env.reset()
    state, reward, done, info = env.step(action_id)

One step = one simulation run on the backend (fake FIFO or Questa).
    state  = StateEncoder vector: 64 bin flags + budget left + one-hot last action (87 numbers)
    reward = reward.compute_reward: (new bins - step_cost + closure_bonus on the closing run) * scale
    done   = all bins covered, or the run budget is used up

Every run gets its own seed, made from (base seed, episode, step): runs differ from each
other, and the same base seed replays the same episodes.
"""

import numpy as np

from fifo_rl.actions import ACTIONS, N_ACTIONS, get_action
from fifo_rl.reward import RewardConfig, compute_reward
from fifo_rl.state import StateEncoder


class CoverageEnv:
    def __init__(self, backend, budget=20, reward=RewardConfig(), seed=0, include_last_action=True):
        """Set up the game.

        Input:  backend, any Backend (FakeBackend for training, Questa later);
                budget, max simulation runs per episode;
                reward, a RewardConfig (step cost, closure bonus, scale - see reward.py);
                seed, base seed from which every run's seed is derived;
                include_last_action, passed to StateEncoder (87 numbers if True, 65 if False).
        Output: nothing (the env must be reset() before the first step).
        Logic:  store the settings and build a StateEncoder that keeps the checklist + budget.
        """
        self.backend = backend
        self.reward_config = reward
        self.base_seed = seed
        self.encoder = StateEncoder(budget, n_actions=N_ACTIONS, include_last_action=include_last_action)
        self.n_actions = N_ACTIONS
        self.state_size = self.encoder.size
        self.episode = -1
        self._step = 0
        self._done = True

    def reset(self):
        """Start a new episode.

        Input:  nothing.
        Output: the first state (float32 array): no bins covered, budget 1.0, no last action.
        Logic:  move to the next episode number, set the step counter to 0, clear the encoder.
        """
        self.episode += 1
        self._step = 0
        self._done = False
        return self.encoder.reset()

    def run_seed(self, episode, step):
        """The seed of one simulation run.

        Input:  episode and step numbers (ints).
        Output: an int in 0 .. 2**32-1, so it can also be passed to Questa's -sv_seed.
        Logic:  numpy's SeedSequence mixes (base seed, episode, step) into one well-spread
                number: every run gets a different seed, and the same base seed gives the
                same seeds again (reproducible experiments).
        """
        return int(np.random.SeedSequence([self.base_seed, episode, step]).generate_state(1)[0])

    def step(self, action_id):
        """Play one turn: run one simulation with the chosen action.

        Input:  action_id, an int 0-21.
        Output: (state, reward, done, info)
                state  - new state vector after this run
                reward - compute_reward(new bins, closed_now) (see reward.py)
                done   - True when all 64 bins are covered or the budget is used up
                info   - dict for logging: action name, seed, new_bins, total_bins, run_bins,
                         runs_left, all_covered, closed_now
        Logic:  1. refuse to play after the episode ended, check the action id
                2. pick this run's seed and ask the backend which bins the run hit
                3. let the StateEncoder OR them into the checklist and count the new ones
                4. compute the reward and whether the episode is over
        """
        if self._done:
            raise RuntimeError("episode is over: call reset() first")
        spec = get_action(action_id)
        seed = self.run_seed(self.episode, self._step)

        run_hits = np.asarray(self.backend.run(action_id, seed), dtype=bool)
        state, n_new = self.encoder.update(run_hits, action_id)
        all_covered = self.encoder.all_covered

        closed_now = all_covered and n_new > 0  # this run (not an earlier one) completed coverage
        reward = compute_reward(n_new, closed_now, self.reward_config)
        self._step += 1
        self._done = all_covered or self.encoder.runs_left == 0

        info = {
            "action": spec.name,
            "seed": seed,
            "new_bins": n_new,
            "total_bins": int(self.encoder.checklist.sum()),
            "run_bins": int(run_hits.sum()),
            "runs_left": self.encoder.runs_left,
            "all_covered": all_covered,
            "closed_now": closed_now,
        }
        return state, reward, self._done, info

    @staticmethod
    def action_names():
        """Names of all actions, for logs and plots.

        Input:  nothing.
        Output: a list of 22 strings, index = action id (e.g. [ "L0_idle", ..., "baseline_random" ]).
        """
        return [spec.name for spec in ACTIONS]
