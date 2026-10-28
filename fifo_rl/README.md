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

`--launcher questa` compiles `tb_rl/rl_files.list` once into `questa_work_dir`: the unchanged `../src codes/` files plus
the AI team's stand-in plan test (`tb_rl/FIFO_rl_pkg.sv`: `FIFO_plan_sequence`, `FIFO_rl_test`; `tb_rl/FIFO_rl_top.sv`),
until the verification team delivers its own. Tool folder, suffix and paths are the `questa_*` keys in
`configs/default.yaml`. Every test keeps `plan.txt`, `sim.log`, `result.txt`, `coverage.txt` under
`<questa_work_dir>/tests/<out name>/`. Check that the fake FIFO still equals Questa bin for bin (~30 s):

```bash
FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py
```
