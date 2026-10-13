# FIFO RL Agent — Team Work Split (4 members)

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
| **B. Action list** | The 22 scenarios | Member 2 | `ACTIONS: list[ActionSpec]`; `ActionSpec(level∈{0,1,7,8}, pattern∈{idle,wr,rd,wr_rd,rand50}, k_cycles)` + `reset_with_traffic` + `baseline_random` |
| **C. Backend** | Run one scenario, return which bins were hit | Members 1 & 4 | `run(action_id: int, seed: int) -> np.ndarray[bool]` of shape (64,) |
| **D. Env** | Gym-style game | Member 2 | `reset() -> obs (87,)`; `step(a) -> (obs, reward, done, info)`; obs = 64 bins + 22 one-hot last action + 1 budget left; reward = new bins − 0.1 |

5. Git: everyone branches from `FIFO_RL_dev` (`feat/fake-fifo`, `feat/env-baselines`, `feat/dqn`, `feat/questa-bridge`) and opens a PR into `FIFO_RL_dev`.

---

## 3. Who does what

### Member 1 — Fake FIFO + coverage model ("the fast simulator")
**Goal:** a pure-Python FIFO that ticks the *same* 64 bins as Questa, cycle for cycle.

Tasks:
1. `fifo_rl/coverage_model.py` — **already written (2026-10-13) — don't rewrite it; pull `FIFO_RL_dev` to get it.** Contract A (64 bin names + index map). Get the exact names/order from a Questa report (the first block of `vcover report -cvg -details -nocompactcrossbins`; the report prints the covergroup twice).
2. `fifo_rl/fake_fifo.py` — the per-cycle model from `fifo_agent_context.md` §1 (count rules + flags + how the monitor samples). Input: list of `(rst_n, wr_en, rd_en)` per cycle; output: 64-bin hit vector (+ hit counts).
3. `fifo_rl/backends/fake_backend.py` — Contract C (uses Member 2's actions to build the cycle list, then the model).
4. **Exact validation** (`tests/test_fake_fifo.py` + a replay script): get one Questa log with `+UVM_VERBOSITY=UVM_HIGH` (Member 4 produces it — the monitor prints every cycle), replay those inputs through the fake FIFO, and require **identical bin hit counts** (seed 1 full run: `cp_wr_en` 3171/7029, `cx_full <0,0,1>` = 163).

Needs: one Questa log from Member 4 (no Questa licence needed on your own PC).
Done when: replay test passes exactly; `run()` takes < 1 ms per scenario.

### Member 2 — Actions, environment, baselines, plots ("the game")
**Goal:** the game the agent plays, and the "random" score it must beat.

Tasks:
1. `fifo_rl/actions.py` — Contract B: each of the 22 scenarios → list of `(rst_n, wr_en, rd_en)` per cycle. **Every scenario starts with a reset**, then fills to level L, then applies pattern P for K cycles.
2. `fifo_rl/envs/coverage_env.py` — Contract D, wrapping *any* backend (fake or Questa). Episode ends at 64/64 or when the run budget is used.
3. `fifo_rl/agents/random_agent.py` + `scripts/evaluate.py` — two baselines: (i) random scenario each run, (ii) plain constrained-random 70/30/10 runs. Average over ≥ 20 seeds.
4. `scripts/plot.py` + `fifo_rl/utils/logger.py` — CSV logging (episode, step, action, new_bins, total_bins, epsilon, loss) and plots: coverage vs runs (mean ± std), learning curve, action histogram.

Needs: Contract A names (Member 1) — use a stub list until then.
Done when: `evaluate.py --agent random --backend fake` produces the baseline plot.

### Member 3 — Mohamed: DQN → DDQN agent + integration lead ("the brain")
**Goal:** an agent that closes coverage in fewer runs than random.

Tasks:
1. Day 1: project skeleton, `requirements.txt`, `configs/default.yaml`, contracts doc.
2. `fifo_rl/agents/dqn.py` — MLP 87→128→128→22, replay buffer (10k), batch 64, Adam 1e-3, γ ≈ 0.9, ε 1.0 → 0.05, target-net sync every ~200 steps, Huber loss. Develop first against a tiny **dummy env** with the Contract D interface (e.g. a 10-bin toy) so you don't wait for Members 1–2.
3. DDQN flag (online net chooses the next action, target net scores it).
4. `scripts/train.py` — training loop, checkpoints, uses Member 2's logger.
5. Integration: merge the PRs, run DQN vs DDQN vs random on the fake backend, write up results.
6. Team-level decisions: answer the engineer's question (Python in WSL + Windows Questa OK?) and decide with the supervisors: FIFO as **pilot** vs **richer coverage model**.

Done when: DQN/DDQN beat the random-scenario baseline on the fake backend (runs-to-100%, area under the coverage curve).

### Member 4 — Questa bridge + verification-team liaison ("the real world")
**Goal:** the same Contract C, but backed by real Questa runs.

Tasks:
1. `fifo_rl/questa/runner.py` — from WSL Python call the Windows tools (`vlib.exe`, `vlog.exe -F`, `vsim.exe -c -sv_seed …`, `vcover.exe report -cvg -details -nocompactcrossbins`); one run folder per run under `runs/` on `/mnt/d`; convert path arguments with `wslpath -w`. Keep tool names and paths in config (so a Linux Questa later = `vsim` instead of `vsim.exe`).
2. `fifo_rl/questa/report_parser.py` + `tests/test_parser.py` — report text → Contract A vector. Save 2–3 real reports as test fixtures. Remember: the report lists the covergroup **twice** — parse only the first block.
3. Produce the `+UVM_VERBOSITY=UVM_HIGH` log Member 1 needs (no testbench change required).
4. **Verification-team link:** send them the 6 requests (`fifo_agent_context.md` §7) — most importantly: command-line knobs (`+WR_PCT`, `+RD_PCT`, `+RST_PCT`, length / scenario), an RL test that always starts with a reset and runs only the chosen scenario, coverage save **without** `-du FIFO`. Agree the exact plusarg names with them.
5. `fifo_rl/backends/questa_backend.py` — Contract C, once their RL test exists. Until then, a temporary version using `run <N>us` + seed is enough to test the plumbing.
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
- Supervisors: the current covergroup is closed by plain random in one ~3 s run, so RL can't fairly beat random on it. Decide: FIFO as **pipeline pilot** and/or ask the verification team for a **richer coverage model**.
- Approve the 22 actions, the 87-number state and the reward (new bins − 0.1).
