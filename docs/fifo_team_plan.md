# FIFO RL Agent — Team Work Split (4 members)

## 0. Status (2026-10-26)

| Piece | State |
|---|---|
| Contract A — bin list (`coverage_model.py`) | ✅ done |
| Contract B — 22 actions (`actions.py`) | ✅ done (plusarg names still to agree with the verification team) · 🔨 being redesigned as a plan file (Kenzy + Yasmeen, §7) |
| Contract C — backend (`backends/base.py`, `backends/fake_backend.py`) | ✅ fake backend done (~0.4 ms/run) · ⏳ Questa backend not started |
| Contract D — env (`envs/coverage_env.py`, `state.py`, `reward.py`) | ✅ done (new reward needs team approval) |
| Fake FIFO (`fake_fifo.py`) | ✅ done; checked cycle by cycle against 10,500 logged Questa cycles (seed 1 full run + seed 2, 300 cycles): same 7 flags on every cycle, same 64 bin counts (done with a script outside the repo; a repo test is still to add) |
| Questa report parser | ✅ done |
| Random baselines, logger, plots | ✅ done (`python -m scripts.evaluate`, `python -m scripts.plot`) |
| DQN / DDQN — training side (Jana) | ✅ done and merged into `FIFO_RL_dev` (PR #1, `840da5a`): `ReplayBuffer`, ε schedules, `configs/default.yaml` + `config.py`, `scripts/train.py`, `evaluate.py --agents dqn --checkpoint`, `plot.py --train`, `StubAgent` |
| DQN / DDQN — agent core (Mohamed, `feat/ddqn-core`) | ✅ done (`7a48fea`): `agents/networks.py` (`QNetwork`), `agents/dqn.py` (`DQNAgent`, DQN/DDQN via `double`), `tests/test_networks.py` + `tests/test_dqn.py` (23 tests), `torch>=2.2` in `requirements.txt`, new config key `grad_clip: 10.0`. Merged into `FIFO_RL_dev` (PR #2, `63f51a7`) |
| Questa runner + Questa backend | ⏳ not started; the real loop also waits for the testbench to accept the agent's choice (§7) |

183 tests pass on `FIFO_RL_dev`.

**First training results (2026-10-26, fake FIFO, default config, 5 seeds each, ~9 min for all 10 trainings on CPU).** Test = 100 games of up to 40 runs (seed 0), re-checked on 200 games with an unseen seed (777) — same picture:

| Agent | Games closed (64/64) | Runs to close (mean, median) | Clock cycles to close (median) | Coverage AUC |
|---|---|---|---|---|
| Random action each run | 84% | 21.6, 20.5 | — | 0.946 |
| Always `baseline_random` (70/30/10, short runs) | 63% | 21.4, 21 | — | 0.963 |
| One long 70/30/10 run like `FIFO_test` (CRV) | 100% | 1 run | 343 | — |
| **DDQN** | **99.8%** | **5.3, 5** | **92** | **0.99** |
| **DQN** | **100%** | **5.2, 5** | **92** | **0.99** |

- The agent learns this in ~300–900 training games (3000 is more than enough) and reaches the ~5-run greedy figure. DQN ≈ DDQN: the task is too easy to separate them.
- Favourite actions: `L8_rand50`, `reset_with_traffic`, `baseline_random`, `L8_rd`, `L8_idle`, `L0/L1_rand50`.
- Small tuning notes: ε ends at ~0.13 (only ~27.6k of the 30k decay steps happen in 3000 games); one DDQN test game took 39 runs (rare tail).
- **Honest caveat:** in real Questa a run costs ~3 s of start-up whatever its length. Today the agent's 5 short runs (~15 s) lose to one long CRV run (~3 s). With a plan file (many blocks in ONE run, §7) it becomes 1 run vs 1 run: the agent wins on cycles (92 vs 343) but both cost ~3 s. Winning on time needs richer coverage or a bigger design (§5).

**Next:** new plan-based actions (Kenzy + Yasmeen, §7); Questa runner/backend; decision on the "too easy" risk (§5).

## 1. The idea in one picture

The work is cut into 4 pieces that talk to each other through **4 small agreed interfaces ("contracts")**, so everyone can work in parallel without waiting.

```
                    ┌──────────────────────────────┐
                    │  Member 3: DQN / DDQN agent  │
                    └──────────────┬───────────────┘
                                   │  Contract D: env.reset() / env.step(a)
                    ┌──────────────▼───────────────┐
                    │ Member 2: Env + actions +    │
                    │ baselines + plots            │
                    └───────┬──────────────┬───────┘
          Contract C:       │              │      Contract C:
          backend.run(a, s) │              │      backend.run(a, s)
                ┌───────────▼───┐      ┌───▼──────────────────────┐
                │ Member 1:     │      │ Member 4: Questa bridge  │
                │ Fake FIFO     │      │ (WSL → Windows Questa)   │
                │ (pure Python) │      │ + verification-team link │
                └───────────────┘      └──────────────────────────┘
        Contract A (64 bin names) and Contract B (22 actions) are shared by everyone.
```

- **Fake FIFO** = fast Python copy of the chip → used for training (thousands of runs per second).
- **Questa bridge** = the real simulator (~3 s per run) → used for validation and final results.
- Both backends have **the same interface**, so the env and the agent don't care which one is plugged in.

---

## 2. Day 1 — everyone together (one meeting, ~2 hours)

1. Everyone reads the two docs above.
2. Everyone sets up WSL (Ubuntu): `sudo apt update && sudo apt install -y python3-pip python3-venv`.
3. Mohamed pushes the project skeleton (`fifo_rl/` with `requirements.txt`, `README.md` with the venv steps, the bin list and the state encoder — written 2026-10-13); everyone pulls `FIFO_RL_dev` and creates their venv.
4. **Agree the 4 contracts** (write them into `fifo_rl/contracts.md` and don't change them without telling the team):

| Contract | What | Owner | Shape |
|---|---|---|---|
| **A. Bin list** | The 64 coverage bins in a fixed order, named like the Questa report | Member 1 | `BIN_NAMES` (64 names) + `BIN_INDEX` (name → number), e.g. `"cp_full/auto[1]"`, `"cx_full/<auto[0],auto[0],auto[1]>"` — **done**, in `fifo_rl/fifo_rl/coverage_model.py` |
| **B. Action list** | The 22 scenarios | Member 2 | `ACTIONS` (tuple of 22 `ActionSpec(id, name, level, wr_pct, rd_pct, rst_pct, k_cycles)`): ids 0–19 = level∈{0,1,7,8} × pattern∈{idle,wr,rd,wr_rd,rand50}, id = 5·level_idx + pattern_idx; 20 `reset_with_traffic`; 21 `baseline_random` — **done**, in `fifo_rl/fifo_rl/actions.py` |
| **C. Backend** | Run one scenario, return which bins were hit | Members 1 & 4 | `run(action_id: int, seed: int) -> np.ndarray[bool]` of shape (64,) — **fake one done** (`backends/fake_backend.py`); Questa one to do |
| **D. Env** | Gym-style game | Member 2 | `reset() -> obs (87,)`; `step(a) -> (obs, reward, done, info)`; obs = 64 bins + 1 budget left + 22 one-hot last action; reward = (new bins − 1 + 10·[this run closed 64/64]) / 64 (`fifo_rl/reward.py`, proposed 2026-10-14, replaces new bins − 0.1 — needs team approval) — **done**, `CoverageEnv(backend, budget, reward, seed)` in `fifo_rl/fifo_rl/envs/coverage_env.py` |
| **E. Agent / buffer / schedule** | What the DDQN agent and the training script exchange (added 2026-10-25) | Mohamed (agent) & Jana (buffer, schedule, config) | `DQNAgent(state_dim, n_actions, cfg, buffer, epsilon_schedule, seed)` with `act(state, greedy=False)`, `observe(s,a,r,s2,done)`, `learn() -> loss \| None`, `epsilon`, `save/load`; `ReplayBuffer.push/sample -> Batch`; `schedule.value(step)` — full shapes in §6 |

5. Git: everyone branches from `FIFO_RL_dev` (`feat/fake-fifo`, `feat/env-baselines`, `feat/dqn`, `feat/questa-bridge`) and opens a PR into `FIFO_RL_dev`.

---

## 3. Who does what

### Member 1 — Fake FIFO + coverage model ("the fast simulator")
**Goal:** a pure-Python FIFO that ticks the *same* 64 bins as Questa, cycle for cycle.

Tasks:
1. `fifo_rl/coverage_model.py` — **already written (2026-10-13) — don't rewrite it; pull `FIFO_RL_dev` to get it.** Contract A (64 bin names + index map). Get the exact names/order from a Questa report (the first block of `vcover report -cvg -details -nocompactcrossbins`; the report prints the covergroup twice).
2. `fifo_rl/fake_fifo.py` — **done 2026-10-14.** The per-cycle model from `fifo_agent_context.md` §1 (count rules + flags + how the monitor samples). Input: list of `(rst_n, wr_en, rd_en)` per cycle; output: 64-bin hit vector (`hits`) + hit counts (`bin_hit_counts`).
3. `fifo_rl/backends/fake_backend.py` — **done 2026-10-14.** Contract C (uses the actions to build the cycle list, then the model; ~0.4 ms/run).
4. **Exact validation** (**done as a check outside the repo** — 10,500 logged Questa cycles replayed through `fake_fifo.py`: same flags every cycle, same 64 bin counts; turning it into a repo test is what remains) (`tests/test_fake_fifo.py` + a replay script): get one Questa log with `+UVM_VERBOSITY=UVM_HIGH` (Member 4 produces it — the monitor prints every cycle), replay those inputs through the fake FIFO, and require **identical bin hit counts** (seed 1 full run: `cp_wr_en` 3171/7029, `cx_full <0,0,1>` = 163).

Needs: one Questa log from Member 4 (no Questa licence needed on your own PC).
Done when: replay test passes exactly; `run()` takes < 1 ms per scenario.

### Member 2 — Actions, environment, baselines, plots ("the game")
**Goal:** the game the agent plays, and the "random" score it must beat.

Tasks:
1. `fifo_rl/actions.py` — **done 2026-10-14.** Contract B: each of the 22 scenarios → list of `(rst_n, wr_en, rd_en)` per cycle (`to_cycles`) and → Questa plusargs (`to_plusargs`). **Every scenario starts with a reset**, then fills to level L, then applies pattern P for K = 10 cycles.
2. `fifo_rl/envs/coverage_env.py` + `fifo_rl/reward.py` — **done 2026-10-14.** Contract D, wrapping *any* backend (fake or Questa). Episode ends at 64/64 or when the run budget is used.
3. ✅ **Done 2026-10-14:** `fifo_rl/agents/random_agent.py` + `scripts/evaluate.py` — two baselines on `CoverageEnv(FakeBackend(), budget=40)`: (i) random scenario each run, (ii) always `baseline_random` (id 21, plain constrained-random 70/30/10). Average over ≥ 20 seeds; report runs-to-64/64, closure rate and total reward. (First numbers for (i): budget 20 closes 39%, budget 40 closes 84%, mean ≈ 25 runs.)
4. ✅ **Done 2026-10-14:** `scripts/plot.py` + `fifo_rl/utils/logger.py` — CSV logging (episode, step, action, new_bins, total_bins, epsilon, loss) and plots: coverage vs runs (mean ± std), learning curve, action histogram.

Needs: Contract A names (Member 1) — use a stub list until then.
Done when: `evaluate.py --agent random --backend fake` produces the baseline plot.

### Member 3 — Mohamed: DQN → DDQN agent + integration lead ("the brain")
**Goal:** an agent that closes coverage in fewer runs than random.

> **Update 2026-10-25:** tasks 2–4 below are now split between Mohamed (networks + update rule) and Jana (replay buffer, ε schedule, config, `train.py`, eval/plots) — see §6.

Tasks:
1. Day 1: project skeleton, `requirements.txt`, `configs/default.yaml`, contracts doc.
2. `fifo_rl/agents/dqn.py` — MLP 87→128→128→22, replay buffer (10k), batch 64, Adam 1e-3, γ ≈ 0.9, ε 1.0 → 0.05, target-net sync every ~200 steps, Huber loss. No dummy env needed any more: develop directly against `CoverageEnv(FakeBackend(), budget=20)` (state 87, 22 actions; thousands of steps per second).
3. DDQN flag (online net chooses the next action, target net scores it).
4. `scripts/train.py` — training loop, checkpoints, uses Member 2's logger.
5. Integration: merge the PRs, run DQN vs DDQN vs random on the fake backend, write up results.
6. Team-level decisions: answer the engineer's question (Python in WSL + Windows Questa OK?) and decide with the supervisors: FIFO as **pilot** vs **richer coverage model**.

Done when: DQN/DDQN beat the random-scenario baseline on the fake backend (runs-to-100%, area under the coverage curve).

### Member 4 — Questa bridge + verification-team liaison ("the real world")
**Goal:** the same Contract C, but backed by real Questa runs.

Tasks:
1. `fifo_rl/questa/runner.py` — from WSL Python call the Windows tools (`vlib.exe`, `vlog.exe -F`, `vsim.exe -c -sv_seed …`, `vcover.exe report -cvg -details -nocompactcrossbins`); one run folder per run under `runs/` on `/mnt/d`; convert path arguments with `wslpath -w`. Keep tool names and paths in config (so a Linux Questa later = `vsim` instead of `vsim.exe`).
2. `fifo_rl/questa/report_parser.py` + `tests/test_report_parser.py` — **already written (2026-10-13), with 4 real reports in `tests/fixtures/` — don't rewrite it; pull `FIFO_RL_dev` to get it.** Use `hits_vector(report_text)` → Contract A vector (64 true/false). It raises `ReportError` if the report was made without `-nocompactcrossbins` or has no `cg_FIFO` bins.
3. Produce the `+UVM_VERBOSITY=UVM_HIGH` log Member 1 needs (no testbench change required).
4. **Verification-team link:** send them the 6 requests (`fifo_agent_context.md` §7) — most importantly: command-line knobs (`+WR_PCT`, `+RD_PCT`, `+RST_PCT`, length / scenario), an RL test that always starts with a reset and runs only the chosen scenario, coverage save **without** `-du FIFO`. Agree the exact plusarg names with them.
5. `fifo_rl/backends/questa_backend.py` — Contract C, once their RL test exists. Build the command line from `actions.to_plusargs(action_id)` and pass the env's per-run seed as `-sv_seed` (it is already a 32-bit int). Until then, a temporary version using `run <N>us` + seed is enough to test the plumbing.
6. Validate: for each of the 22 actions, fake backend and Questa backend hit the same bins.

Needs: a PC with Questa (Windows install + licence) and WSL.
Done when: `run(action_id, seed)` returns the right 64-bin vector from a real simulation in ~3 s.

---

## 4. Order of work

| Phase | Member 1 | Member 2 | Member 3 (Mohamed) | Member 4 |
|---|---|---|---|---|
| **Day 1** | Contracts meeting | Contracts meeting | Skeleton + contracts | Contracts meeting |
| **Week 1** | Bin list, fake FIFO | Actions, env, random agent | DQN on dummy env | Runner, parser, UVM_HIGH log, send requests to verification team |
| **Week 2** | Exact replay validation | Baseline plots, logger | DDQN, train on fake backend | Temporary Questa backend (`run <N>us`) |
| **Week 3** | Help Member 4 compare fake vs real | Agent-vs-random plots | Experiments + write-up | Real Questa backend (after verification team delivers) |
| **Week 4** | — | Final plots | Final comparison on real Questa | Final runs |

**Blocking dependencies (only these):**
- Member 1's validation needs Member 4's UVM_HIGH log (Week 1).
- The real Questa backend needs the verification team's RL test (Week 3, external).
- Everything else can run in parallel thanks to the contracts and stubs.

---

## 5. Open items for the whole team
- Engineer: is "Python in WSL + Windows Questa" OK, or must Questa run on Linux? (Code is ready for either — one config change.)
- Supervisors: the current covergroup is closed by plain random in one ~3 s run, so RL can't fairly beat random on it. Decide: FIFO as **pipeline pilot** and/or ask the verification team for a **richer coverage model**. (Confirmed by the 2026-10-26 results in §0: the agent needs ~4× fewer cycles than CRV, but on this covergroup Questa start-up time, not cycles, is the real cost.)
- Approve the 22 actions, the 87-number state and the reward: (new bins − 1 + 10 on the closing run) / 64, see `fifo_rl/reward.py` (was new bins − 0.1; `LEGACY_REWARD` keeps the old one).

---

## 6. DDQN build — Mohamed + Jana (added 2026-10-25)

Split: **Mohamed = "the brain"** (networks + update rule), **Jana = "everything around the brain"** (replay memory, exploration schedule, config, training and evaluation scripts). Both code against Contract E below, each with a stub of the other half, and merge at the end.

### 6.1 Contract E (agree first, don't change without telling the other)

**Agent** — Mohamed implements, Jana calls:
```python
agent = DQNAgent(state_dim=87, n_actions=22, cfg: dict, buffer, epsilon_schedule, seed)
agent.name                              # "dqn" or "ddqn" (from cfg["double"])
agent.act(state, greedy=False) -> int   # ε-greedy; greedy=True for evaluation
agent.observe(s, a, r, s2, done)        # pushes into the buffer, counts steps
agent.learn() -> float | None           # one gradient step; None until learning_starts
agent.epsilon -> float                  # current ε (for the logger)
agent.save(path) / DQNAgent.load(path, buffer=None)
```
`act(state)` without the extra argument keeps working, so `evaluation.run_episode` drives it unchanged; evaluation wraps it in a small `GreedyWrapper` that forces `greedy=True`.

**Replay buffer** — Jana implements, the agent uses:
```python
buf = ReplayBuffer(capacity, state_dim, seed)
buf.push(s, a, r, s2, done); len(buf)
buf.sample(batch_size) -> Batch(states f32 (B,87), actions i64 (B,), rewards f32 (B,),
                                next_states f32 (B,87), dones f32 (B,))   # numpy arrays
```

**Epsilon schedule** — Jana implements: `schedule.value(step) -> float`.

**Config keys** (`configs/default.yaml`, shared): `hidden: [128, 128]`, `lr: 1e-3`, `gamma: 0.9`, `batch_size: 64`, `buffer_capacity: 10000`, `learning_starts: 500`, `train_freq: 1`, `target_sync: 200`, `double: true`, `eps_start: 1.0`, `eps_end: 0.05`, `eps_decay_steps` (to agree, e.g. 20k–50k — the fake env is cheap), `budget_train: 20`, `budget_eval: 40`, `n_episodes`, `eval_every`, `eval_episodes: 100`, `seed: 0`.

**Episode end at the budget** is treated as terminal (`done = 1`); valid because the budget left is part of the state.

### 6.2 Mohamed — branch `feat/ddqn-core`

| File | Contents |
|---|---|
| `fifo_rl/agents/networks.py` | `QNetwork`: MLP 87→128→128→22 with ReLU, built from `cfg["hidden"]` |
| `fifo_rl/agents/dqn.py` | `DQNAgent`: online + target nets, Adam, Huber loss, gradient clipping, ε-greedy `act`, hard target sync every `target_sync` steps, `save`/`load` (state dicts + cfg + step count) |
| DDQN switch (inside `learn()`) | vanilla: `y = r + γ(1−d)·max_a Q_target(s′,a)`; double: `a* = argmax_a Q_online(s′,a)`, `y = r + γ(1−d)·Q_target(s′,a*)` |
| `tests/test_dqn.py` | output shapes; DDQN target ≠ DQN target on hand-set nets (checked by hand); target sync exactly every N steps; loss falls when overfitting one fixed batch; save → load gives the same Q-values; `greedy=True` never explores |

Until Jana's buffer is merged: a ~10-line list-based stub buffer returning the same `Batch`.

**✅ Done 2026-10-26 (`7a48fea`)** — built directly on Jana's merged buffer (no stub needed). As built:
- `QNetwork(state_dim, hidden, n_actions)`: Linear + ReLU per size in `cfg["hidden"]`, then a plain Linear to `n_actions`.
- `DQNAgent(state_dim, n_actions, cfg, buffer=None, epsilon_schedule=None, seed=0)`; `steps` = `observe()` calls, `updates` = gradient steps. `learn()` returns `None` until `len(buffer) >= max(learning_starts, batch_size)`, then: Huber loss → clip gradient norm to `grad_clip` → Adam step → copy online → target every `target_sync` **gradient steps**. `epsilon` is `0.0` without a schedule.
- `save(path)` writes one torch file (cfg, sizes, seed, counters, both nets, optimizer); `DQNAgent.load(path)` needs nothing else (CPU, `weights_only=True`).
- New config key `grad_clip: 10.0` in `default.yaml` (other config files get it from the defaults automatically; override with `--grad-clip 5`).
- Tests (15 in `test_dqn.py`, 8 in `test_networks.py`) include the DQN vs DDQN target worked out by hand (online scores [1, 2], target scores [5, 3], r = 1, γ = 0.9 → DQN 5.5, DDQN 3.7, done → 1.0) and a real `train.train` + `evaluate.main --agents dqn` run. 7 deliberately broken versions of `dqn.py` were each caught by at least one test.

### 6.3 Jana — branch `feat/ddqn-training`

| File | Contents |
|---|---|
| `fifo_rl/agents/replay_buffer.py` | `ReplayBuffer`: numpy ring buffer, preallocated arrays, overwrites oldest when full, seeded sampling |
| `fifo_rl/agents/schedules.py` | `LinearEpsilon(start, end, decay_steps)` (optionally `ExpEpsilon`) |
| `configs/default.yaml` + `fifo_rl/config.py` | YAML loader with the keys above + CLI overrides (`--double false`, `--seed 3`) |
| `scripts/train.py` | loop on `CoverageEnv(FakeBackend(), budget=20)`: act → step → observe → learn; every `eval_every` episodes a greedy evaluation on a **separate** `budget=40` env via `evaluation.run_episodes` + `summarize`; CSV via `CsvLogger` (episode, step, action, new_bins, total_bins, epsilon, loss); checkpoints `last.pt` and `best.pt` (best = highest closure rate, then fewest runs); `summary.json` in the format `plot.py` already reads |
| `scripts/evaluate.py` (extend) | `--agent dqn --checkpoint runs/.../best.pt` |
| `scripts/plot.py` (extend) | learning curve: eval runs-to-close and closure rate vs episodes; loss and ε vs steps |
| tests | `test_replay_buffer.py` (wrap-around, shapes/dtypes, same seed → same sample), `test_schedules.py`, `test_train.py` smoke test with a **stub agent** (`RandomAgent` + no-op `observe`/`learn`) so it doesn't wait on Mohamed |

### 6.4 Merge and done-criteria
1. Both open PRs into `FIFO_RL_dev`; Mohamed merges buffer/schedules first, then the core, and removes his stub buffer.
2. Joint integration check: `python -m scripts.train --double false` and `--double true` over ≥ 5 seeds, then `scripts.evaluate` for dqn / ddqn / random / fixed with budget 40, and `scripts.plot`.
3. Done when DQN and DDQN beat the random-scenario baseline (84% closure, mean 21.6 runs) on runs-to-close and area under the coverage curve, aiming at the ~5-run greedy figure. **✅ Met 2026-10-26** (results in §0). All three steps are done; `feat/ddqn-core` merged into `FIFO_RL_dev` (PR #2, `63f51a7`).

Reproduce (Ubuntu, in `fifo_rl/`; results go to the git-ignored `runs/`):
```bash
python -m scripts.train --seed 0                    # DDQN -> runs/train/ddqn_seed0/   (~3-5 min)
python -m scripts.train --double false --seed 0     # DQN  -> runs/train/dqn_seed0/
python -m scripts.evaluate --agents random fixed dqn --checkpoint runs/train/ddqn_seed0/best.pt
python -m scripts.plot --train runs/train/ddqn_seed* runs/train/dqn_seed*   # learning_curve.png, loss_epsilon.png
```

---

## 7. Next: plan-based actions — Kenzy + Yasmeen (added 2026-10-26)

Agreed 2026-10-25: Kenzy + Yasmeen redesign the actions as a **plan file**: several stimulus blocks in **one** Questa run, passed with `+PLAN_FILE=<file>` (DPI / cocotb maybe later). Mohamed + Jana built the agent on the old 22 actions in the meantime (§6).

**What does NOT change** — the agent reads its sizes from the env (`env.state_size`, `env.n_actions`), nothing is hard-coded: `networks.py`, `dqn.py`, replay buffer, ε schedules, `train.py`, `evaluate.py`, `plot.py`.

**What the agent side needs from the new design:**
1. The same env interface (Contract D): `reset() -> state`, `step(action) -> (state, reward, done, info)`, plus `state_size` and `n_actions`.
2. **One block per step**: the agent picks block 1, then block 2, … (each `step` adds one block). If a whole plan must be chosen in one go, DQN does not fit — talk to Mohamed first.
3. Say **when the reward arrives**. If only at the end of a plan (one simulation per full plan), we raise `gamma` (e.g. 0.99) and lengthen `eps_decay_steps` — config only.
4. Keep Contract C for the bins: a backend returning the 64 bins as `bool[64]`; `questa/report_parser.py` already turns a real report into that vector.

**Then, on the agent side:** retrain + re-evaluate (§6.4 commands), check fake vs real bins per block, final comparison on real Questa (runs **and** wall-clock time).
