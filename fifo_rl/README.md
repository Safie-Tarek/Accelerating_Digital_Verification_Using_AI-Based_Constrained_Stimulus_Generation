# fifo_rl

RL agent for the FIFO testbench in `../src codes/`. Two designs live side by side: the original one (22 actions, one simulation per action, tables below) and the plan-based campaigns (several blocks per simulation, see "Plan-based campaigns"). The DQN agent is shared by both; the Questa launcher waits for the verification team's plan test.

| File | What it is |
|---|---|
| `fifo_rl/coverage_model.py` | The 64 coverage bins of `cg_FIFO`, in one fixed order (`BIN_NAMES`, `BIN_INDEX`, `N_BINS`) |
| `fifo_rl/state.py` | `StateEncoder`: 64 bin flags + budget left + one-hot last action = 87 numbers |
| `fifo_rl/actions.py` | The 22 actions (reset → fill to level L → pattern for K cycles): `ACTIONS`, `to_cycles`, `to_plusargs` |
| `fifo_rl/fake_fifo.py` | Python copy of the FIFO + monitor sampling: cycle list → 64-bin hits |
| `fifo_rl/backends/` | `run(action_id, seed) -> bool[64]`: `FakeBackend` (Python FIFO); Questa backend to come |
| `fifo_rl/reward.py` | `compute_reward`: (new bins − 1 + 10 on the run that closes 64/64) / 64; settings in `RewardConfig` |
| `fifo_rl/envs/coverage_env.py` | `CoverageEnv`: `reset()` / `step(action)` → state, reward, done, info |
| `fifo_rl/agents/random_agent.py` | Baselines: `RandomAgent` (random action every run), `FixedActionAgent` (always action 21 = 70/30/10) |
| `fifo_rl/evaluation.py` | `run_episodes`, `summarize` (closure rate, runs to close, coverage curve) + the one-long-random-run baseline |
| `fifo_rl/utils/logger.py` | `CsvLogger`: one CSV row per run, same columns for every agent |
| `fifo_rl/agents/replay_buffer.py` | `ReplayBuffer`: numpy ring buffer, `push` / `sample(B) -> Batch` (Contract E) |
| `fifo_rl/agents/schedules.py` | ε schedules: `LinearEpsilon`, `ExpEpsilon`, `make_schedule(cfg)` |
| `fifo_rl/agents/networks.py` | `QNetwork`: MLP state → one Q-value per action (87 → 128 → 128 → 22 by default) |
| `fifo_rl/agents/dqn.py` | `DQNAgent`: ε-greedy `act`, `observe`, `learn` (DQN or DDQN target, Huber loss, target net), `save` / `load` |
| `fifo_rl/agents/stub_agent.py` | `StubAgent`: Contract E interface without learning, for tests and plumbing checks |
| `fifo_rl/agents/a2c.py` | `A2CAgent` (Contract F): actor + critic (two `QNetwork`s), legality mask, `act`, `observe`, `learn` (GAE, policy + value + entropy loss), `save` / `load`; see "A2C agent" below |
| `fifo_rl/agents/rollout_buffer.py` | `RolloutBuffer` (on-policy memory of the last whole tests) and `compute_gae` (advantages and returns) |
| `configs/default.yaml`, `fifo_rl/config.py` | DQN / DDQN settings; `load_config`, CLI overrides (`--double false`) |
| `scripts/train.py` | Train DQN / DDQN on the fake FIFO, greedy evaluation every `eval_every` episodes, checkpoints |
| `scripts/evaluate.py`, `scripts/plot.py` | Run the baselines / a trained agent and plot them (see below) |
| `fifo_rl/questa/report_parser.py` | `hits_vector(report_text)`: Questa coverage report → 64 true/false flags, one per bin |
| `tests/` | pytest tests for all of the above |
| `tests/fixtures/` | 4 real Questa reports used by the tests |

The report must be created with `vcover report -cvg -details -nocompactcrossbins run.ucdb > report.txt`.

## Setup (WSL Ubuntu)

```bash
python3 -m venv ~/venvs/fifo_rl
source ~/venvs/fifo_rl/bin/activate
cd fifo_rl
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, before the line below
pip install -r requirements.txt
pytest -q
```

Install the CPU build of PyTorch first: a plain `pip install torch` on Linux downloads the much larger GPU build.

## Random baselines

```bash
python -m scripts.evaluate      # writes runs/baselines/*.csv, *_summary.json, long_random.json
python -m scripts.plot          # writes runs/baselines/coverage_vs_runs.png, runs_to_close.png
```

## DQN / DDQN training

```bash
python -m scripts.train                          # DDQN -> runs/train/ddqn_seed0/
python -m scripts.train --double false --seed 1  # vanilla DQN -> runs/train/dqn_seed1/
python -m scripts.train --agent stub             # plumbing check without the DQN (no learning)
python -m scripts.evaluate --agents random fixed dqn --checkpoint runs/train/ddqn_seed0/best.pt
python -m scripts.plot --train runs/train/dqn_seed* runs/train/ddqn_seed*   # learning curves
```

Every training folder holds `config.yaml`, `steps.csv`, `eval.csv`, `last.pt`, `best.pt` and `<agent>_summary.json`.
Training picks `best.pt` on validation episodes (seed + 1000); `scripts.evaluate` then measures it on the same test
episodes as the baselines.

## Plan-based campaigns (new design, step 1)

One test = **one simulation of a plan** of several blocks (no reset between blocks, the FIFO keeps its level);
one campaign = up to `campaign_budget` tests = one DQN episode. The agent picks one block per step (9 actions,
93-number state); the DQN agent itself is unchanged. Step 1 = one coverage report per test, so the test's coverage
reward goes to its `END_TEST` step. Design and reasons: `../docs/fifo_agent_context.md` (STATUS 2026-10-27).

| File | What it is |
|---|---|
| `fifo_rl/plan/actions.py` | The 9 blocks (`RESET`, `IDLE`, `WRITE`, `READ`, `WRITE_READ`, `RAND50`, `RESET_TRAFFIC`, `BASELINE`, `END_TEST`) |
| `fifo_rl/plan/stimulus_plan.py` | `PlannedAction` (idx, action, state when chosen), `StimulusPlan` (one test's blocks) |
| `fifo_rl/plan/state_encoder.py` | `StateEncoder`: 64 covered bins + last 3 blocks one-hot + plan position + tests left = 93 |
| `fifo_rl/plan/plan_builder.py` | `PlanBuilder`: asks the agent for blocks until `END_TEST` (or the plan is full) |
| `fifo_rl/plan/plan_file.py` | `TextPlanWriter` / `read_plan`: the plan file the UVM test reads with `+PLAN_FILE=` |
| `fifo_rl/launchers/` | `FakeLauncher` (Python FIFO) and `QuestaLauncher` (vsim + vcover); both write `result.txt` / `coverage.txt` |
| `fifo_rl/questa/result_parser.py` | `ResultParser`: pass / errors from `result.txt`, else from the UVM summary in the log |
| `fifo_rl/campaign/` | `RewardAttributor`, `CampaignStore`, `TrajectoryBuilder`, `CampaignRunner` (the loop) |
| `scripts/run_campaign.py` | Train / evaluate on campaigns; one CSV row per test (cycles, wall time, new bins, ...) |

```bash
python -m scripts.run_campaign train --gamma 0.99 --eps-decay-steps 80000           # DDQN on the fake FIFO
python -m scripts.run_campaign eval --agent random                                   # baseline
python -m scripts.run_campaign eval --agent dqn --checkpoint runs/campaign/train_ddqn_seed0/best.pt
python -m scripts.run_campaign eval --agent dqn --checkpoint <best.pt> --launcher questa --eval-campaigns 3
```

**Go on training a saved agent** instead of starting from a new one:

```bash
python -m scripts.run_campaign train --resume runs/campaign/long_same_seed1/best.pt --seed 10 \
    --eps-start 0.1 --eps-end 0.1 --n-campaigns 1000 --eval-campaigns 100 --out runs/campaign/resume_seed10
```

- The networks, optimizer and step counters come from the checkpoint. The learning settings (`lr`, `gamma`, `target_sync`, `buffer_capacity`, `eps_*`, …) also stay as saved, unless you give them on the command line. `hidden` and `double` cannot change.
- The replay buffer starts empty, because checkpoints do not store it. The agent does not learn until it holds `learning_starts` transitions.
- ε follows the agent's own step count, so a finished schedule stays at `eps_end`. Give the same `--eps-start` and `--eps-end` for a fixed amount of exploration.
- The loaded agent is validated first (campaign 0 in `eval.csv`), so `best.pt` is only replaced by an agent that scores higher. `summary.json` records `resumed_from` and `resumed_at_step`.
- Use a new `--seed`, because campaign numbers start at 0 again. Validation scores from different seeds are not comparable, so compare checkpoints with `eval` on the same `--seed`.

`--launcher questa` compiles `tb_rl/rl_files.list` once into `questa_work_dir`: the unchanged `../src codes/` files plus
the AI team's stand-in plan test (`tb_rl/FIFO_rl_pkg.sv`: `FIFO_plan_sequence`, `FIFO_rl_test`; `tb_rl/FIFO_rl_top.sv`),
until the verification team delivers its own. Tool folder, suffix and paths are the `questa_*` keys in
`configs/default.yaml`. Every test keeps `plan.txt`, `sim.log`, `result.txt`, `coverage.txt` under
`<questa_work_dir>/tests/<out name>/`. Check that the fake FIFO still equals Questa bin for bin (~30 s):

```bash
FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py
```

## A2C agent (actor-critic, plan-based campaigns)

A second learning agent for the same game as the DQN: the same 9 blocks, the same 93-number state and the same campaigns.
It plugs into `CampaignRunner` like `DQNAgent` does, with no change to the runner, launchers or reward.

- **Actor:** `QNetwork(93, a2c_hidden, 9)`. The softmax of its scores is the policy π, the probability of each block.
- **Critic:** `QNetwork(93, a2c_hidden, 1)`. It gives V(s), its guess of the reward still to come.
- **Advantage:** A = how much better a block did than the critic expected. One update raises π for blocks with A > 0 and lowers it for blocks with A < 0, while V moves toward what really came back.
- **Exploration:** the agent samples from π, and an entropy bonus keeps π spread out. There is no ε, so `agent.epsilon` is always 0.0.
- **On-policy:** `learn()` returns `None` until the rollout holds `a2c_rollout_tests` whole tests (each ends with `END_TEST`) or the campaign ended. So an update never falls in the middle of a test, and the data always comes from the current policy.
- **Legality mask:** in a full plan (`state[-2] == 1.0`), `PlanBuilder` adds `END_TEST` itself. The mask allows only `END_TEST` there, so π(END_TEST) = 1 and the actor learns nothing from a choice it did not make. No other block is ever masked.

**Contract F** (agreed by Mohamed and Jana):

```python
agent = A2CAgent(state_dim=93, n_actions=9, cfg=cfg, seed=0)
agent.name -> "a2c";  agent.steps;  agent.updates;  agent.epsilon -> 0.0
agent.act(state, greedy=False) -> int      # greedy=True: most likely block (GreedyWrapper uses it)
agent.observe(s, a, r, s2, done)           # what CampaignRunner passes for every transition
agent.learn() -> float | None              # None until whole tests are stored, then one update
agent.last_stats -> {"policy_loss", "value_loss", "entropy", "n_transitions"}
agent.save(path);  A2CAgent.load(path)     # one file; load refuses a DQN checkpoint
```

**Config keys it reads.** Every key must also exist in `configs/default.yaml`, because `config.py` rejects unknown keys.

| Key | Value | Meaning |
|---|---|---|
| `gamma` | 0.99 (give `--gamma 0.99`; the file's 0.9 is for the old 22-action DQN) | discount, shared with the DQN |
| `a2c_hidden` | `[128, 128]` | hidden layers of the actor and of the critic |
| `a2c_lr` | 7e-4 | Adam learning rate (one optimizer for both networks) |
| `a2c_grad_clip` | 0.5 | gradient norm limit per update |
| `vf_coef` | 0.5 | weight of the critic's loss |
| `ent_coef` | 0.01 | weight of the entropy bonus |
| `gae_lambda` | 0.95 | GAE λ (1.0 = plain discounted return) |
| `a2c_rollout_tests` | 4 | whole tests per update |
| `a2c_normalize_adv` | true | normalize advantages (only in rollouts of 8 or more transitions) |

**Wiring in `scripts/run_campaign.py`** (Jana's part):
- `--agent a2c`
- a factory: `A2CAgent(runner.state_size, runner.n_actions, cfg, seed=cfg["seed"])`. It needs no buffer or schedule, so `make_agent` cannot be reused.
- `eval_agent`: `GreedyWrapper(A2CAgent.load(checkpoint))`
- refuse `--resume` for a2c (`resume_agent` is DQN-only)
- optionally, the `last_stats` numbers as log columns

`train()` already calls `agent.save(...)` for `best.pt` / `last.pt`, so that part needs no change.

**First results** (fake FIFO, 2026-11-26):
- Set-up: scratch script with the `run_campaign` protocol, step-2 reward, γ 0.99, 3 seeds × 1,000 campaigns.
- `best.pt` was evaluated greedily on the 200 unseen campaigns of seed + 2000, the same set the step-2 DDQN results use.

| Agent | Closed | Tests per campaign | Cycles per campaign |
|---|---|---|---|
| Random block each step | 98.5% | 8.86 | 459 |
| **A2C** (seeds 0 / 1 / 2) | **100%** | **1.81** (1.77 / 1.74 / 1.92) | **327** (320 / 315 / 347) |
| Constrained random in plans (always `BASELINE`) | 100% | 1.77 | 320 |
| Step-2 DDQN (`docs/random_vs_step1_vs_step2.md`) | 100% | 2.14 ± 0.44 | 260 ± 60 |

What the results show:
- A2C learns in 200–400 campaigns, and 1,000 campaigns take 1–2 minutes on CPU.
- With `ent_coef 0.01` no seed collapsed later on: its final weights were as good as its best ones.
- **But every seed settled on `BASELINE` ×10**, which is constrained random inside a plan. So it matches CRV in tests and cycles, and it does not find the ordered 9-block plan that closes in 1 test.
- `ent_coef` 0.05 and 0.1 gave the same plan and less stable training (one seed's last weights fell to 48% closure).
- The reward charges nothing for cycles (`cycle_cost: 0`), so nothing pushes toward short, ordered plans. Changing that is a team decision about the shared reward.

**Tests:** `tests/test_rollout_buffer.py` and `tests/test_a2c.py` cover:
- GAE worked out by hand, and the losses of one update rebuilt from the formulas
- the mask, and the forced `END_TEST` never moving the actor
- no update in the middle of a test
- save / load giving an identical agent that keeps learning
- one run through the real campaign loop

A mutation check (18 bugs planted on purpose, one at a time, in a copy) was caught 18 of 18 times.
