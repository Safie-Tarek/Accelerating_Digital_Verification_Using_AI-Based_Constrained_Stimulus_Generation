# FIFO RL Agent — Technical Context (team copy)

> The AI team's shared working notes. It holds everything the DQN agent needs from the FIFO testbench in `src codes/` and `FIFO Plan.xlsx`. For a gentler, step-by-step explanation read [`fifo_walkthrough.md`](fifo_walkthrough.md); for who builds what, read [`fifo_team_plan.md`](fifo_team_plan.md). Last updated: 2026-10-27.

## STATUS 2026-10-27 — plan-based campaign (newest — read this first)

**Why a new design.** With the 22 actions, every action is one Questa run (~3 s start-up), so 5 agent runs (~15 s) lose to one constrained-random run (~3 s). New design: **one Questa run = one test plan made of several action blocks**, run back to back **without a reset between blocks** (the FIFO keeps its fill level). The old 22-action env (`actions.py`, `state.py`, `envs/`, `backends/`, `scripts/train.py`) is kept unchanged for comparison; the new code lives in new packages. **The DQN agent is not changed** (`DQNAgent`, `QNetwork`, replay buffer, schedules): it only sees new sizes (93 → 9).

**Two steps:**
1. **Step 1 (this commit):** Questa gives **one coverage report per test** (`cg_FIFO` only — no per-action coverage in the testbench). The coverage reward of the test goes to the plan's last step (`END_TEST`); each block only pays its own cycle cost. Goal: measure simulation time, cycles per test and how the agent behaves.
2. **Step 2 (later):** the testbench stamps every transaction with `action_idx` and keeps one covergroup instance per block (`act_0`, `act_1`, … next to `cg_FIFO`) → a reward per block. Then compare step 1 vs step 2 (tests to close, cycles, wall-clock, learning speed).

**Actions = 9 blocks** (`fifo_rl/plan/actions.py`; one block = one line of the plan; no `level`, no reset of its own):

| id | name | wr % | rd % | rst % | cycles | why |
|---|---|---|---|---|---|---|
| 0 | RESET | 0 | 0 | 100 | 2 | back to count 0 without starting a new test |
| 1 | IDLE | 0 | 0 | 0 | 4 | hold the current level (idle while full / almostfull / almostempty) |
| 2 | WRITE | 100 | 0 | 0 | 10 | fills an empty FIFO into overflow |
| 3 | READ | 0 | 100 | 0 | 10 | drains a full FIFO into underflow |
| 4 | WRITE_READ | 100 | 100 | 0 | 4 | from empty → count 1, from full → count 7, else holds the level |
| 5 | RAND50 | 50 | 50 | 0 | 10 | random wr/rd around the current level |
| 6 | RESET_TRAFFIC | 50 | 50 | 100 | 16 | random wr/rd while reset is held: the only way to the 3 reset-only bins |
| 7 | BASELINE | 70 | 30 | 10 | 18 | the plan's FIFO_4 recipe (constrained random) |
| 8 | END_TEST | – | – | – | 0 | closes the plan; no cycles |

**Plan file** (written by `TextPlanWriter`, read by the UVM test through `+PLAN_FILE=<path>`; `name` is for people, the UVM can ignore it):
```
PLAN id=12 seed=3141592653 version=1
ACT idx=0 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=1 name=IDLE wr=0 rd=0 rst=0 cycles=4
END_TEST
```
The test always applies **one reset cycle before the first block** (the DUT's count starts as X). A plan that is full (`plan_max_len` blocks) gets `END_TEST` added automatically; an `END_TEST` chosen first gives an empty test (only the reset cycle) that still costs a whole run.

**State = 93 numbers** (`fifo_rl/plan/state_encoder.py`): [0:64] bins covered in the campaign so far · [64:91] last 3 blocks of this plan, one-hot 3 × 9 (newest first, zeros where there is none) · [91] blocks in this plan ÷ `plan_max_len` · [92] tests left ÷ campaign budget. No legality mask.

**Episode / steps / reward (step 1):**
- Episode = one **campaign** of up to `campaign_budget` tests; it ends at 64/64 or when the tests are used up (`done`).
- One DQN step = **one block** of a plan; one transition per block: `(state when chosen, block, reward, state of the next choice, done)`. The next state of `END_TEST` is the first state of the next test (coverage already updated).
- Reward of block k = `−cycle_cost · cycles_k · scale`; reward of `END_TEST` = `reward.compute_reward(new bins of the whole test, closed_now)` (= `(new − step_cost + closure_bonus·[closed]) / 64`, so one *test* costs one bin, exactly like one run before) + its own cycle cost (0). `cycle_cost` defaults to 0 → the agent is judged on **number of tests** (Questa start-up dominates); set it > 0 to also charge cycles.
- The reward arrives only at the end of a plan → train with `gamma 0.99` and a longer `eps_decay_steps` (config overrides, no code change).

**Classes (new code, see `fifo_rl/README.md`):**

| File | Class | Role |
|---|---|---|
| `fifo_rl/plan/actions.py` | `ActionSpec`, `ACTIONS`, `END_TEST_ID` | the 9 blocks; `to_cycles(id, rng)` for the fake launcher |
| `fifo_rl/plan/stimulus_plan.py` | `PlannedAction`, `StimulusPlan` | one chosen block (`idx`, `action_id`, the state seen when chosen) and the ordered plan |
| `fifo_rl/plan/state_encoder.py` | `StateEncoder` | (campaign store, plan so far) → 93 numbers |
| `fifo_rl/plan/plan_builder.py` | `PlanBuilder` | asks the agent for blocks until `END_TEST` or the plan is full |
| `fifo_rl/plan/plan_file.py` | `TextPlanWriter`, `read_plan` | plan ↔ text file |
| `fifo_rl/launchers/` | `SimulatorLauncher`, `FakeLauncher`, `QuestaLauncher`, `RunOutput` | run one plan → result file + coverage report (+ wall-clock times) |
| `fifo_rl/questa/result_parser.py` | `ResultParser`, `RunResult` | pass / errors from `result.txt`, or from the UVM summary in the log when there is no `result.txt` |
| `fifo_rl/questa/report_parser.py` | (existing) `hits_vector` | the report → 64 bins of the test |
| `fifo_rl/campaign/reward_attributor.py` | `RewardAttributor`, `AttributionResult` | rewards of the plan's blocks (step 1: test-level) |
| `fifo_rl/campaign/campaign_store.py` | `CampaignStore`, `TestSummary` | the campaign's true coverage, tests left, history |
| `fifo_rl/campaign/trajectory_builder.py` | `TrajectoryBuilder`, `Transition` | plan + rewards → one transition per block |
| `fifo_rl/campaign/campaign_runner.py` | `CampaignRunner`, `RunConfig` | the loop: plan → write → launch → parse → reward → store → transitions → `agent.observe` / `learn` |
| `scripts/run_campaign.py` | – | train / evaluate with the fake or the Questa launcher; one CSV row per test (cycles, wall time, new bins, …) |

**What the testbench must provide for step 1** (verification team; `src codes/` is read-only for us): a test (default name `FIFO_rl_test`, chosen with `+UVM_TESTNAME`) that reads `+PLAN_FILE`, applies one reset cycle, then for each `ACT` line drives `cycles` transactions with those wr/rd/rst percentages, and stops at `END_TEST`; batch run with `coverage save` **without** `-du FIFO`. Optional `result.txt` (`passed= errors= cycles= actions=`); without it the UVM error summary in the log is used.

**As built 2026-10-27 (not committed yet; 244 tests pass = 183 old + 61 new):**
- `plan_max_len` default is **10**, not 8: the shortest plan reaching all 64 bins needs **9 blocks** — `WRITE, IDLE, READ, WRITE_READ, IDLE, WRITE, WRITE_READ, IDLE, RESET_TRAFFIC` (67 cycles; closes 64/64 in 97% of seeds, the rest is RESET_TRAFFIC luck). `cx_almostfull <0,1,1>` needs a READ that *starts from full* (8 → 7), which is why IDLE at 7 then READ is not enough.
- **Fake-FIFO results** (eval on 200 unseen campaigns, budget 30 tests, seed + 2000):

  | agent | closes | tests to close (median / mean) | cycles to close (median) |
  |---|---|---|---|
  | random block each step | 98.5% | 7 / 8.5 | 395 |
  | DDQN `best.pt` (2000 campaigns, `--gamma 0.99 --lr 5e-4 --eps-decay-steps 80000 --buffer-capacity 50000`, ~8 min CPU) | 99.5% | 3 / 3.6 | 287 |
  | best possible (the 9-block plan) | ~97% in 1 test | 1 | 67 |
- **Step-1 learning is unstable:** greedy validation went 87% → 77% → **100% (campaign 1000, kept as best.pt)** → 80% → 47% → 0% → 0% as ε fell to 0.05; a first try (600 campaigns, lr 1e-3) peaked at 40%. Typical collapse: END_TEST at once every test, or full 10-block plans that never close. Cause: the reward arrives only on END_TEST, so the agent cannot tell which block helped (the credit problem step 2 removes). Always keep `best.pt` (picked on validation), never `last.pt`. Worth trying before step 2: more seeds, a lower lr, a slower ε decay.
- The DDQN favours WRITE, READ, RAND50, RESET_TRAFFIC and BASELINE and never uses RESET, IDLE or WRITE_READ — so it does not find the 1-test plan; it closes in ~3 tests by mixing random blocks.
- **Step 1 runs on real Questa (2026-10-27, Questa Sim-64 2024.1 on Windows, `C:\questasim64_2024.1\win64`, license `MGLS_LICENSE_FILE`).** Until the verification team delivers its plan test, `fifo_rl/tb_rl/` holds the AI team's **stand-in**: `FIFO_rl_pkg.sv` (`FIFO_plan_sequence` reads `+PLAN_FILE`, sets `WR_EN_ON_DIST` / `RD_EN_ON_DIST` per block and replaces the fixed 10% `reset_con` by the block's rst %; `FIFO_rl_test extends FIFO_test`: reset sequence, plan, `#1ns`, then writes `+RESULT_FILE` = `passed= errors= fatals= cycles= actions=`), `FIFO_rl_top.sv` (copy of `FIFO_top` that imports the new package), `rl_files.list` (the 14 unchanged `src codes/` files + those two). Nothing in `src codes/` is modified. Config: `questa_src_list: tb_rl/rl_files.list`, `questa_top: work.FIFO_rl_top`.
- **Fake FIFO = Questa for the plan test, bin for bin:** `FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py` (3 tests, ~30 s) runs 3 deterministic plans (WRITE+IDLE; 9 fixed blocks; empty test) and requires the same hit count on all 64 bins, the cycle count and pass. All pass → training on the fake FIFO is valid for the plan test.
- **Real-Questa numbers (this laptop):** one test ≈ **6.2 s** wall (vsim ≈ 5.6–6.3 s, almost all start-up — 45 vs 93 cycles makes no difference; vcover report ≈ 0.3 s); first vsim after compiling ≈ 18–20 s (optimisation). 60 tests, all passed.

  | on Questa (3 campaigns each, seed + 2000) | closes | tests to close | cycles per campaign | wall time per campaign |
  |---|---|---|---|---|
  | random block each step | 2 of 3 | 10 | 747 | ~117 s |
  | DDQN `best.pt` (trained on the fake FIFO) | 3 of 3 | 3.3 | 309 | ~21 s |
  | original `FIFO_test` (one CRV run, 10,200 cycles) | always | 1 | 10,200 | **6–7 s** |
  The DDQN plans on Questa look like on the fake FIFO (e.g. test 1 = `WRITE RAND50 WRITE WRITE RAND50 …` → 55 bins, test 2 mixes `RESET_TRAFFIC` → 62, test 3 `BASELINE ×3` → 64). **RL beats random block choice ~5×, but one long CRV run is still ~3× cheaper** — start-up per test dominates and `cg_FIFO` closes in ~400 random cycles. The §9 risk stands; the honest wins are a richer coverage model or keeping one vsim alive across tests.
- Run on Windows from `fifo_rl/`: Questa on PATH (or `--questa-bin C:/questasim64_2024.1/win64`), then `python -m scripts.run_campaign eval --agent dqn --checkpoint <best.pt> --launcher questa --eval-campaigns 3`; every test keeps `plan.txt`, `sim.log`, `result.txt`, `run.ucdb`, `coverage.txt` under `runs/questa/tests/<out name>/c<campaign>_t<test>/`.

**Step 2 requests** (later): `action_idx` in `FIFO_seq_item`, an `action_idx` wire in `FIFO_if` (not in the DUT modport), driver + monitor copy it, and in `FIFO_coverage.sv` a package-level `covergroup cg_action(string name) with function sample(FIFO_seq_item t)` (same bins as `cg_FIFO`, `option.per_instance = 1`), one new instance `act_<idx>` per block, skipped for the reset cycle (`action_idx = -1`).

## STATUS 2026-10-26 (older — 22-action design; still valid for the old env)
- **The whole loop works on the fake FIFO:** fake FIFO → 64 bins → 87-number state → `DQNAgent` → one of 22 actions → reward → replay memory → learning. 183 tests pass on `FIFO_RL_dev` (everything below is merged there as of `63f51a7`).
- **Who built what:**
  - Kenzy: `actions.py` (22 actions), `fake_fifo.py`, `backends/`, `reward.py`, `envs/coverage_env.py`.
  - Jana: random baselines, `evaluation.py`, `utils/logger.py`, `scripts/evaluate.py` / `plot.py`; then (PR #1, merged into `FIFO_RL_dev` at `840da5a`) `agents/replay_buffer.py`, `agents/schedules.py`, `agents/stub_agent.py`, `configs/default.yaml` + `config.py`, `scripts/train.py`.
  - Mohamed (PR #2 `feat/ddqn-core`, `7a48fea`, merged into `FIFO_RL_dev` at `63f51a7`): `agents/networks.py` (`QNetwork`), `agents/dqn.py` (`DQNAgent`), `tests/test_networks.py` + `tests/test_dqn.py`, `torch>=2.2`, config key `grad_clip: 10.0`.
- **`DQNAgent` API:** `DQNAgent(state_dim, n_actions, cfg, buffer=None, epsilon_schedule=None, seed=0)`; `name` = `"ddqn"` if `cfg["double"]` else `"dqn"`; `act(state, greedy=False) -> int`; `observe(s, a, r, s2, done)`; `learn() -> float | None` (`None` until `len(buffer) >= max(learning_starts, batch_size)`; Huber loss, gradient-norm clip `grad_clip`, Adam, target copy every `target_sync` gradient steps); `epsilon` (0.0 without a schedule); `q_values(states)`; `td_target(rewards, next_states, dones)`; `save(path)` / `DQNAgent.load(path, buffer=None, epsilon_schedule=None)` (one torch file with cfg, sizes, seed, counters, both nets, optimizer; CPU).
  - DQN target: `y = r + γ(1−done)·max_a Q_target(s′, a)`. DDQN: `a* = argmax_a Q_online(s′, a)`, `y = r + γ(1−done)·Q_target(s′, a*)`.
- **Fake FIFO = Questa, cycle by cycle:** replaying 10,500 logged Questa cycles (`+UVM_VERBOSITY=UVM_HIGH`; seed 1 full run + seed 2, 300 cycles) through `fake_fifo.py` gives the same 7 flags on every cycle and the same 64 bin counts (checked with a script outside the repo; a repo test is still to add). This also closes the negedge sampling question.
- **Results (fake FIFO, default config, 5 seeds × DQN and DDQN, test = 100 games of ≤ 40 runs, re-checked on 200 games of unseen seed 777):** DDQN closes 99.8%, DQN 100%, both in ~5.2–5.4 runs (median 5) and ~92 clock cycles (median) — vs random action 84% in 21.6 runs, always-70/30/10 63% in 21.4 runs, one long CRV run like `FIFO_test` 1 run of 343 cycles (median). AUC 0.99 vs 0.946. Learns in ~300–900 games; DQN ≈ DDQN (task too easy to separate them). Favourite actions: `L8_rand50`, `reset_with_traffic`, `baseline_random`, `L8_rd`, `L8_idle`.
- **The agent has never run on real Questa.** The testbench cannot receive the agent's choice yet. Real Questa cost ≈ 3 s start-up per run → today 5 agent runs (~15 s) lose to 1 CRV run (~3 s). See the biggest risk in §9.
- **Split since 2026-10-25:** Kenzy + Yasmeen redesign the actions as a plan file (`+PLAN_FILE=`, several blocks in one Questa run; DPI / cocotb later). The agent needs: the same env interface (`reset`, `step`, `state_size`, `n_actions`), **one block per step**, and to know when the reward arrives (only at plan end → raise `gamma`, lengthen `eps_decay_steps`). Agent code does not change. Details: `fifo_team_plan.md` §7.
- Setup: `python3 -m venv ~/venvs/fifo_rl`, activate, `pip install torch --index-url https://download.pytorch.org/whl/cpu`, `pip install -r requirements.txt`, `pytest -q`. Run git from Windows only (WSL git shows line-ending noise on every file).

## Project Context
- Graduation project with Siemens DISW: use a Reinforcement Learning agent to steer stimulus generation toward uncovered functional-coverage bins, so coverage closes in fewer simulation runs than plain constrained-random verification.
- The verification team builds UVM testbenches; the AI team (including Mohamed) builds the RL agent.

## Current Focus: FIFO
- DUT: a synchronous FIFO. The code in `src codes/` comes from https://github.com/SeiffAhmed22/Synchronous-FIFO-UVM-Verification
- Goal: build a DQN agent for this FIFO testbench.
- Agent <-> simulator loop is file-based: after each simulation run, the agent reads the coverage report, the last stimulus it applied and the remaining run budget, then chooses the next stimulus. Reward = new coverage gained.
- The agent controls the stimulus directly (e.g. which sequence/traffic pattern and its parameters), not by editing SystemVerilog constraints.
- Environment: WSL2 + Ubuntu, QuestaSim installed locally, Python + PyTorch, CPU only (no GPU).
- Only `src codes/` (note the space; quote it in commands) and `FIFO Plan.xlsx` are relevant; other folders are out of scope.

## Working rules
- `src codes/` is read-only for the AI team. Testbench changes are requested from the verification team (see section 7).
- Never run simulations inside `src codes/` (they create `work/`, `*.ucdb`, `transcript`, …); use a separate run folder.

---

## 1. DUT facts (`FIFO.sv`, `FIFO_if.sv`)
- Synchronous FIFO, DEPTH = 8, WIDTH = 16, one clock (period 10 ns), async active-low reset `rst_n`.
- Inputs (the only things stimulus can control): `rst_n`, `wr_en`, `rd_en`, `data_in`.
- Outputs: `data_out` + 7 flags: `full, empty, almostfull, almostempty, wr_ack, overflow, underflow`.
- `data_in` value is NOT in any coverage → the agent only decides *when* to write / read / reset.
- The DUT is the corrected version (lines marked `// BUG HERE` are commented out, fixes at the bottom).
- DEPTH/WIDTH are duplicated in `FIFO_seq_item.sv` and `FIFO_scoreboard.sv` (not shared) — keep DEPTH=8 in one place in our Python code.

### Python-ready model of one clock cycle (DUT + how the monitor samples it)
**Every run must start with a reset**: `count`, `wr_ptr`, `rd_ptr` are `reg` and start as X; without a reset nothing updates, and the monitor turns X into 0 in its `bit` fields → false bins get ticked. The model starts at `count = 0` after that reset.
```python
# state: count in 0..8
if not rst_n:
    count = 0                      # async reset
else:
    w = wr_en and count < 8        # write accepted?
    r = rd_en and count > 0        # read accepted?
    count += int(w) - int(r)
# flags as SAMPLED by the monitor (combinational, using the NEW count and THIS cycle's wr_en/rd_en)
full        = count == 8
empty       = count == 0
almostfull  = count == 7
almostempty = count == 1
wr_ack      = wr_en and count != 8
overflow    = full and wr_en
underflow   = empty and rd_en
# coverage sample = (wr_en, rd_en, full, empty, almostfull, almostempty, wr_ack, overflow, underflow)
```
Sampling assumption: driver applies inputs at negedge, DUT updates at posedge, monitor samples at the next negedge (inputs of cycle k + count after cycle k). Driver and monitor both wake at negedge (possible race); UVM's sequencer handshake most likely makes the monitor sample first. **Verify on a waveform** — the fake env must match.

## 2. Testbench map (`src codes/`)
| File | Role | What matters for the agent |
|---|---|---|
| `FIFO_if.sv` | Bundle of wires (interface) | 4 inputs, 7 flags, DEPTH=8 |
| `FIFO.sv` | The DUT | Count/flag rules above (basis of fake env) |
| `FIFO_seq_item.sv` | One cycle of stimulus + random rules | **Knobs**: `WR_EN_ON_DIST` (default 70), `RD_EN_ON_DIST` (default 30); reset is **hardcoded** 10% (`rst_n dist {0:/10, 1:/90}`); `data_in` unconstrained |
| `FIFO_sequence.sv` | Traffic patterns | `FIFO_reset_sequence` (1 cycle reset), `FIFO_write_only_sequence` (100 cycles, wr 100% rd 0%), `FIFO_read_only_sequence` (100 cycles, rd 100% wr 0%), `FIFO_write_read_sequence` (9999 cycles, defaults 70/30). Lengths hardcoded. All randomized sequences still reset ~10% of cycles |
| `FIFO_sequencer.sv` | Hand-off sequence→driver | Boilerplate, nothing |
| `FIFO_config_obj.sv` | Config object | Only holds the vif today; natural place to add agent knobs |
| `FIFO_driver.sv` | Items → wires | 1 item = 1 clock cycle; drives at negedge |
| `FIFO_monitor.sv` | Wires → items | Samples all signals every negedge → feeds scoreboard + coverage |
| `FIFO_agent.sv` | sequencer+driver+monitor | Wiring only |
| `FIFO_coverage.sv` | Covergroup `cg_FIFO` | **State + reward** (section 4) |
| `FIFO_scoreboard.sv` | Golden model `ref_model()` + data check | Checks only `data_out` (flags checked by SVA). Not needed for reward; useful reference for fake env |
| `FIFO_env.sv` | agent+scoreboard+coverage | Wiring only |
| `FIFO_test.sv` | The only test, `FIFO_test` | **Fixed order**: reset → write-only(100) → read-only(100) → write/read(9999) ≈ 10,200 cycles. No plusargs, no config knobs. This is where agent choices must plug in |
| `FIFO_top.sv` | Clock, interface, DUT, `bind` SVA, `run_test("FIFO_test")` | `+UVM_TESTNAME=<test>` on the command line overrides the test name → new tests can be selected without editing top |
| `FIFO_SVA.sv` | Assertions + 17 cover directives | 7 `cover property` + 10 `cover final` (reset/full/empty/almost*/overflow/underflow/wr_ack). Extra coverage source, not in `cg_FIFO` |
| `src_files.list` | Compile order | — |
| `run.do` | Questa script | See section 5 |

## 3. Stimulus knobs (candidate actions)
Existing (inside SV, not reachable from the command line yet):
- write probability `WR_EN_ON_DIST` (0–100), read probability `RD_EN_ON_DIST` (0–100), choice of sequence, number of cycles (hardcoded).
Missing (must be added by verification team):
- reset probability knob (today fixed 10%), sequence length knob, command-line access to all knobs (plusargs), seed control.

**Key constraint: FIFO state does NOT carry over between runs.** Each run = fresh simulation = reset → count 0. So an action must be a *self-contained scenario* that first reaches the target fill level and then applies the pattern, inside the same run. (A plain "wr+rd 100%" run from empty just sits at count 1 and never sees full.)

Action space (implemented 2026-10-14 in `fifo_rl/actions.py`; K = 10, `reset_with_traffic` 16 cycles, `baseline_random` 18 cycles; still pending Mohamed's final OK) — action = (target level L, pattern P), run = `reset → fill to L → apply P for K cycles`:
- L ∈ {0 (empty), 1 (almostempty), 7 (almostfull), 8 (full)}
- P ∈ {idle (0/0), write-only (100/0), read-only (0/100), write+read (100/100), random mix (50/50)}
- → 4 × 5 = 20 scenario actions, plus:
  - `reset_with_traffic`: cycles with `rst_n=0` and random wr/rd (only way to hit cx_empty (1,0,1),(1,1,1) and cx_underflow (1,1,1))
  - `baseline_random`: wr 70 / rd 30 / rst 10 for N cycles (the plan's FIFO_4 recipe)
- K (cycles of pattern) fixed small, e.g. 4–16.
Examples: (8, idle) → cx_full (0,0,1); (8, write+read) → wr&rd at full, then cx_almostfull (1,1,1); (8, write-only) → overflow, cx_wr_ack (1,0,0); (0, write+read) → cx_almostempty (1,1,1).

## 4. Coverage model = state vector + reward (`FIFO_coverage.sv`, covergroup `cg_FIFO`)
Sampled once per clock from the monitor. 9 coverpoints (auto bins 0/1 → 18 bins) + 7 crosses of `wr_en × rd_en × flag` (46 bins after ignore_bins) = **64 bins, 16 coverage items**. Questa covergroup % = average of the 16 items' percentages.

| Item | Bins | Ignored (wr,rd,flag) | Notes (derived from section 1 model — verify in sim) |
|---|---|---|---|
| cp_wr_en, cp_rd_en, cp_wr_ack, cp_overflow, cp_full, cp_empty, cp_almostfull, cp_almostempty, cp_underflow | 2 each | – | |
| cx_wr_ack | 5 | (0,1,1),(0,0,1),(1,1,0) | (1,0,0) needs write-only ending at count 8 |
| cx_overflow | 5 | (0,1,1),(0,0,1),(1,1,1) | (1,0,1) needs write while full |
| cx_full | 6 | (0,1,1),(1,1,1) | (0,0,1) needs idle while full |
| cx_empty | 8 | – | (1,0,1) and (1,1,1) reachable **only during a reset cycle** with wr_en=1 |
| cx_almostfull | 8 | – | needs count 7 with every wr/rd combo |
| cx_almostempty | 8 | – | needs count 1 with every wr/rd combo |
| cx_underflow | 6 | (1,0,1),(0,0,1) | (1,1,1) reachable **only during a reset cycle** |

RL interface (implemented 2026-10-14; reward still pending team approval):
- **State** (`state.py`) = 64 bin-hit flags (0/1, cumulative over the episode) + remaining budget (normalized) + one-hot last action = 87 floats. Possible later extras: the 16 item percentages and the 17 SVA cover directives.
- **Reward** (`reward.py`) = (new bins − 1 + 10·[this run closed 64/64]) / 64. `LEGACY_REWARD` keeps the first idea (new bins − 0.1). See §9 for why.
- **Episode** (`envs/coverage_env.py`) = a budget of N simulation runs (train 20, evaluate 40); cumulative coverage is tracked in Python by OR-ing per-run bin vectors (no `vcover merge` needed).

## 5. Running the simulation (original `run.do`, for reference)
```
vlib work
vlog -f src_files.list +cover -covercells
vsim -voptargs=+acc work.FIFO_top -classdebug -uvmcontrol=all -cover
coverage save FIFO.ucdb -onexit -du FIFO
add wave /FIFO_top/FIFOif/*
run -all
```
### Confirmed in Step 6 (2026-10-12, by running it)
- **Questa = Windows install** `C:\questasim64_2024.1\win64` (Questa Sim-64 2024.1, UVM-1.1d, license via `MGLS_LICENSE_FILE`). There is **no Linux Questa in WSL**; WSL has `/mnt/c/questasim64_2024.1/win64` on PATH, so `vsim.exe` / `vcover.exe` work from WSL (Python 3.10 there) **as long as the working dir is under `/mnt/c` or `/mnt/d`**. Windows Python 3.12 + openpyxl also available.
- Original `run.do` works in batch: `vsim -c -do "do run.do; quit -f"` → 10,200 transactions, 0 scoreboard errors, 0 assertion failures, ~10 s.
- **`coverage save ... -du FIFO` DROPS `cg_FIFO`** — the UCDB only has code/assertion/directive coverage (all already 100%). Never use `-du FIFO` for the agent.
- **No seed → every run identical** (verified). Same `-sv_seed` → identical; different seed → different.
- **Default test (200 directed write-only/read-only cycles + random 70/30/10) reaches 64/64 `cg_FIFO` by 400–500 cycles across seeds 1–5** (seed 1: 100 cyc 36/64, 200 cyc 48/64, 300 cyc 63/64, 500 cyc 64/64; seeds 2–5: 61–63/64 at 300, 64/64 by 400–500). Last bin typically `cx_almostfull <0,0,1>` (idle at count 7). The full 10,200-cycle test is massive overkill.
- **Time per run ≈ 3 s**, almost all start-up (100 cycles 2.9 s, 10,200 cycles 3.3 s); compile 0.7 s once; first sim after compile ~7 s (vopt). `+acc`/`-classdebug` only matter for debugging.
- Ignore bins never hit (confirms the sampling model in section 1).

### Commands for the agent loop (tested; run in a separate run dir, NEVER inside `src codes/` — sims create `work/`, `transcript`, `*.ucdb`, `*.wlf`)
```
# once
vlib work
vlog -F "<repo>/src codes/src_files.list" +cover -covercells        # -F = paths relative to the list file
# per run k
vsim -c -sv_seed <seed_k> work.FIFO_top -coverage -l run_k.log -do "coverage save -onexit run_k.ucdb; run -all; quit -f"
vcover report -cvg -details -nocompactcrossbins run_k.ucdb > run_k_cov.txt
# optional cumulative: vcover merge cum_new.ucdb cum_old.ucdb run_k.ucdb   (or OR the bin vectors in Python — simpler)
```
- Inside vsim, `coverage report -cvg -details -file x.txt` also works (skips the vcover call).
- `run <N>us` truncates the test after N×100 cycles (useful for experiments until the RL test exists).
- **Always use `-nocompactcrossbins`**: without it, missing cross bins are merged into wildcard rows like `bin <auto[1],auto[1],*>  0  1  2  ZERO`.

### Report format to parse (per bin line)
```
    Coverpoint cp_full                                100.00%        100    Covered
        bin auto[0]                                      9240          1    Covered
    Cross cx_almostfull                                87.50%        100    Uncovered
            bin <auto[0],auto[0],auto[1]>                   0          1    ZERO
            ignore_bin auto_full_1                          0               ZERO     <- skip ignore_bin lines
    ...
            Covergroup Bins         64        63         1    98.43%                 <- summary line
```
Parser: track current `Coverpoint`/`Cross` name, read each `bin <...>` / `bin auto[x]` line → (item, bin, hits). Map to a fixed 64-index vector (hit = hits > 0).

## 6. Verification plan (`FIFO Plan.xlsx`, 1 sheet, 4 rows)
| Label | Feature | Stimulus in plan | Implemented in | Coverage in plan | Implemented? |
|---|---|---|---|---|---|
| FIFO_1 | Reset clears pointers, counter, overflow, underflow | Directed reset at start, then random reset mostly off | `FIFO_reset_sequence` + `reset_con` (10%) | "cover the reset signal" | **Not in `cg_FIFO`** (no `rst_n` coverpoint); only SVA `rst_*_cover` |
| FIFO_2 | Write-only | Only writes, no reads | `FIFO_write_only_sequence` (100 cycles; still 10% reset) | Cross wr_en × rd_en × each output flag | Yes (`cx_*`) |
| FIFO_3 | Read-only | Only reads, no writes | `FIFO_read_only_sequence` (100 cycles) | Same crosses | Yes |
| FIFO_4 | Constrained random | repeat 9999, reset off 90%, wr 70%, rd 30% | `FIFO_write_read_sequence` + defaults | Same crosses | Yes |
Check column says "scoreboard checks internal signals and counter" — in reality the scoreboard only compares `data_out`; counter/pointers are checked by SVA.
For the agent: FIFO_4 is the **constrained-random baseline** to beat; the plan's coverage target is exactly `cg_FIFO`. If a `rst_n` coverpoint is added later, the state vector grows by 2 bins.

## 7. Requests for the verification team (since `src codes/` is read-only for us)
1. Read knobs from the command line (`$value$plusargs` / `uvm_cmdline_processor`): `+WR_PCT`, `+RD_PCT`, `+RST_PCT`, `+NUM_CYCLES` (or `+SEQ=<name>`), passed via `FIFO_config_obj`.
2. Make reset probability a variable (like `WR_EN_ON_DIST`).
3. A configurable test (e.g. `FIFO_rl_test`, selected with `+UVM_TESTNAME`) that **always starts with a reset**, then runs only the scenario the agent chose (fill to level L, then pattern P for K cycles) instead of the fixed 4-step order.
4. Batch-friendly do-file: seed argument, per-run UCDB name, `quit -f`, no `add wave`, coverage save **without `-du FIFO`** (it drops `cg_FIFO` — confirmed).
5. Optional: add a `rst_n` coverpoint (plan item FIFO_1).
6. Discuss a richer coverage model (current `cg_FIFO` closes in ~500 random cycles).

## 8. Build plan for the agent (steps 1–9 and the parser half of step 10 are done — see STATUS at the top; left: Questa runner/backend, steps 10–12)

**Step 6 done (2026-10-14, 135 tests pass):** random action each run closes 84% within 40 runs (mean 21.6 runs when it closes); always-`baseline_random` closes 63%; a greedy agent that knows each action's hit odds closes 100% in ~5 runs (the headroom for the DQN). One long 70/30/10 run like FIFO_test closes in a median of 343 cycles (95% by 654), in line with Questa's 400–500. Hardest bins for random: the 3 reset-only bins (cx_empty <1,0,1>, <1,1,1>, cx_underflow <1,1,1>), hit almost only by `reset_with_traffic` (p ≈ 0.98–1.0 per run) which a random agent picks 1 run in 22. **Next to implement: step 7, the DQN**, evaluated with `evaluation.run_episodes`/`summarize`; target: get close to the ~5-run greedy figure. Original step-6 plan, for reference: step 6 — `fifo_rl/agents/random_agent.py`, `fifo_rl/utils/logger.py`, `scripts/evaluate.py`, `scripts/plot.py`: run (i) a uniform-random-action agent and (ii) an always-`baseline_random` (id 21) agent on `CoverageEnv(FakeBackend(), budget=40)` over ≥ 20 seeds; log runs-to-64/64, closure rate and coverage-vs-runs (mean ± std). Fake-env numbers so far for (i): budget 20 closes 39%, budget 40 closes 84%, mean ≈ 25 runs. In parallel: step 7 (DQN, now directly on the real env, no dummy env needed) and step 10's `questa/runner.py` (testable with `run <N>us`).
Order (fake env first; real Questa only when the verification team delivers the RL test):
0. Decisions: ~~Windows vs WSL Python~~ → WSL (decided); FIFO = pilot (b) or richer coverage (a); send requests (section 7) to verification team; install pip/venv in WSL.
1. Project skeleton + venv + requirements.
2. `coverage_model.py`: the 64 bins in a fixed order, names matching the Questa report.
3. Fake FIFO env (section 1 model). **Exact validation (no testbench change needed):** run Questa with `+UVM_VERBOSITY=UVM_HIGH` (the monitor then prints every sampled cycle: inputs + flags), replay those exact inputs through the fake FIFO, and require identical per-cycle flags and identical bin hit counts (seed 1 full run: `cp_wr_en` 3171/7029, `cx_full <0,0,1>` = 163). This also settles the negedge sampling question.
4. `actions.py`: 22 scenarios → list of (rst_n, wr_en, rd_en) per cycle (always start with reset).
5. Env interface: `reset()` → obs; `step(a)` → (obs, reward, done, info). obs = 64 bins + one-hot last action (22) + budget left (1) = 87 floats. reward = (new bins − 1 + 10·[this run closed 64/64]) / 64 (`fifo_rl/reward.py`, proposed 2026-10-14, replaces new bins − 0.1 — needs team approval). done = budget used or 64/64.
6. Random baselines (uniform random scenario; plain constrained-random run of same length) + coverage-vs-runs plot over ≥20 seeds.
7. DQN (MLP 87→128→128→22, replay 10k, batch 64, Adam 1e-3, γ≈0.9, ε 1.0→0.05, target sync every ~200 steps, Huber loss).
8. DDQN (online net picks argmax, target net evaluates) — one-line change.
9. Compare on fake env: runs-to-100%, area under coverage curve.
10. Questa runner + report parser (testable now with `run <N>us`; unit-test the parser on a saved report).
11. `backends/questa_backend.py` (Contract C, same `CoverageEnv`) once the RL test + plusargs exist; check fake vs real bins per action.
12. Final comparison on real Questa (runs and wall-clock); logs + plots.

Layout (top level `fifo_rl/`, next to `src codes/`; ✔ = exists 2026-10-26). One env + swappable backends replaced the planned `fake_fifo_env.py` / `questa_env.py` pair:
```
fifo_rl/
  requirements.txt ✔  README.md ✔  pyproject.toml ✔  configs/default.yaml ✔
  fifo_rl/{coverage_model.py ✔, state.py ✔, actions.py ✔, fake_fifo.py ✔, reward.py ✔}
  fifo_rl/backends/{base.py ✔, fake_backend.py ✔, questa_backend.py}
  fifo_rl/envs/coverage_env.py ✔
  fifo_rl/questa/{report_parser.py ✔, runner.py}
  fifo_rl/{evaluation.py ✔, config.py ✔}
  fifo_rl/agents/{random_agent.py ✔, replay_buffer.py ✔, schedules.py ✔, stub_agent.py ✔, networks.py ✔, dqn.py ✔}
  fifo_rl/utils/logger.py ✔
  scripts/{evaluate.py ✔, plot.py ✔, train.py ✔}
  tests/ ✔ (16 test files, 183 tests) + tests/fixtures/ ✔
  runs/   (git-ignored: sim dirs, logs, checkpoints)
```
requirements.txt (as of 2026-10-26): `numpy>=1.26`, `matplotlib>=3.8`, `pyyaml>=6.0`, `torch>=2.2` (install the CPU wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`), `pytest>=8.0`.
Logging: CSV per step (episode, step, action, new_bins, total_bins, epsilon, loss); plots: coverage vs runs (agent vs random, mean ± std), learning curve, action histogram.

## 9. Open questions / risks
- **BIGGEST RISK (confirmed): `cg_FIFO` is too easy for RL to beat random fairly.** The default test closes 64/64 in one ~3 s run (400–500 cycles). A run costs ~3 s regardless of length (start-up dominates), so forcing the agent's runs to be short is not an honest win — random could simply run longer for the same cost. On this covergroup RL cannot beat random by any fair cost metric. Honest options (**decision for the team + supervisors**):
  (a) richer coverage model from the verification team (e.g. `count` value 0..8 × wr × rd, transitions such as full→empty, back-to-back overflow, reset at each fill level);
  (b) treat the FIFO as a **pipeline pilot**: prove the loop, parser, fake env and DQN work end-to-end here, and show coverage-closure gains on a harder DUT later.
- At ~3 s per real run, 1,000 training runs ≈ 50 min → train mostly in the Python fake env; use Questa for validation/final comparison. (Possible speed-up later: keep one vsim alive and loop runs via Tcl.)
- Monitor/driver sampling race at negedge: **closed 2026-10-14** — the section 1 model gives Questa's logged flags on every one of 10,500 cycles.
- Reward granularity: bins (64) vs Questa % (16 items averaged) — bins preferred.
- Since no FIFO state carries across runs, the MDP state is only cumulative coverage + last action + budget → the problem is close to a contextual bandit. DQN still works (use a small discount γ), but don't expect long-horizon planning to matter much.
- `underflow = empty && rd_en` is combinational, so during reset it equals `rd_en`; cx_underflow (1,1,1) is only reachable this way. Plan FIFO_1 says reset clears underflow → ask the verification team whether that bin is intended.
- **DECIDED (2026-10-12, engineer's instruction): agent runs in Linux/WSL** (Ubuntu 22.04.1, Python 3.10.12). One-time setup: `sudo apt install -y python3-pip python3-venv`.
- **Tested 2026-10-12 from a Python script in WSL** calling the Windows Questa tools (`vlib.exe`, `vlog.exe`, `vsim.exe`, `vcover.exe`) via `subprocess`: works from `/mnt/d/...` folders (compile 0 errors; sim 7.0 s first run, then ~2.8 s; report parsed in Python) and also from Linux-only folders (`/tmp`). `/mnt/d` preferred (repo lives there). Path arguments to `.exe` tools must be converted with `wslpath -w`. PyTorch `2.14.1+cpu` installs fine; a DQN-sized MLP trains at ~810 steps/s on CPU.
- **Parser gotcha:** `vcover report -cvg -details` prints the covergroup twice (TYPE + instance) → 128 bin lines for 64 bins; parse only the first block or de-duplicate.
- **How the test reports in `fifo_rl/tests/fixtures/` were made (to rebuild them):** default test, `-sv_seed 1`, three runs that stop at different points — `vsim -c -sv_seed 1 work.FIFO_top -coverage -do "coverage save -onexit x.ucdb; run 1us; quit -f"` (100 cycles), the same with `run 3us` (300 cycles) and with `run -all` (full run) — then `vcover report -cvg -details -nocompactcrossbins x.ucdb > file.txt`. The fourth file is the 100-cycle run reported *without* `-nocompactcrossbins`.
- **Parser gotcha 2:** inside a cross, Questa lists covered bins first and ZERO bins last, so a bin's line position changes from run to run. Map bins **by name** (item + bin label) into our own fixed 64-index order; never by line position.
- **Code that exists (2026-10-13), in `fifo_rl/` (pushed to `FIFO_RL_dev`):** `fifo_rl/coverage_model.py` (the 64 bins in a fixed order: `BIN_NAMES`, `BIN_INDEX`, `N_BINS`; names are `"<item>/<Questa label>"`, e.g. `cx_full/<auto[0],auto[0],auto[1]>`; checked against real Questa reports), `fifo_rl/state.py` (`StateEncoder`: `.reset()`, `.update(run_hits, action_id) -> (state, n_new)`, `.encode()`; no reward inside) and `fifo_rl/questa/report_parser.py` (`hits_vector(report_text) -> bool[64]`, `parse_bin_hits(report_text) -> {bin name: hit count}`; raises `ReportError` for a report made without `-nocompactcrossbins`, for unknown bins and for missing bins). `tests/fixtures/` holds 4 real reports (seed 1: 100 cycles = 36 bins, 300 cycles = 63, full run = 64, and one compacted report that must be rejected). 35 pytest tests pass, including the whole chain report → 64 flags → 87-number state. Setup: `python3 -m venv ~/venvs/fifo_rl`, activate, `pip install -r requirements.txt`, `pytest -q`.
- **Added 2026-10-14 (102 tests pass):** `fifo_rl/actions.py` (Contract B: 22 `ActionSpec`s, id = 5·level_idx + pattern_idx, then 20 `reset_with_traffic` (K=16), 21 `baseline_random` (K=18); pattern K=10; `to_cycles(id, rng)`, `to_plusargs(id)` with proposed names `+FILL_LEVEL +WR_PCT +RD_PCT +RST_PCT +NUM_CYCLES` — still to agree with the verification team), `fifo_rl/fake_fifo.py` (§1 model, `DEPTH` lives here; `hits(cycles)`, `bin_hit_counts(samples)`; raises if an ignore bin is ever hit), `fifo_rl/backends/{base,fake_backend}.py` (Contract C, ~0.4 ms/run), `fifo_rl/envs/coverage_env.py` (Contract D, per-run seed from (base seed, episode, step) as a 32-bit int usable as `-sv_seed`). `state.py` now takes `N_ACTIONS` from `actions.py`. **Fake FIFO checked against real Questa:** the seed-1 100-cycle report's 36 bins equal the fake FIFO's bins for any 10%-reset pattern (test `test_matches_real_questa_100_cycle_report`). Exact per-cycle replay still needs the UVM_HIGH log. Not there yet: Questa runner/backend, random baseline + plots, DQN.
- **Reward (added 2026-10-14, 117 tests pass, needs team approval):** `fifo_rl/reward.py` — `r = scale·(new_bins − step_cost + closure_bonus·[this run closed 64/64])`, defaults `step_cost = 1` (one run costs one bin), `closure_bonus = 10`, `scale = 1/64`; `LEGACY_REWARD` = old `new bins − 0.1`. Property (tested): an episode closing in T runs sums (unscaled) to exactly `64 − T + 10`, so the only way to score more is fewer runs. Why: on the fake FIFO a random agent already reaches ~63/64 (budget 20: closes 39%, budget 40: 84%, mean 25 runs; 1st run ≈ 38 bins); with −0.1 a 5-run agent scores only ~4% above random, with −1 + bonus ~30%. Train with budget 20, evaluate with 40; γ ≈ 0.9. `CoverageEnv(backend, budget, reward=RewardConfig(...))`; `info['closed_now']` added.
- **State encoding (as built):** float32 vector of 87 = [0:64] cumulative bin flags (0–17 coverpoints in `FIFO_coverage.sv` order × {0,1}; 18–22 cx_wr_ack, 23–27 cx_overflow, 28–33 cx_full, 34–41 cx_empty, 42–49 cx_almostfull, 50–57 cx_almostempty, 58–63 cx_underflow), [64] runs_left ÷ budget, [65:87] one-hot last action (all zeros before the first run). The last-action part is switchable (65 numbers without it).
- **Branch layout (2026-10-13):** `FIFO_RL_dev` holds everything: the iDMA folders (`AI/ doc/ results/ rtl/ scripts/ sim/ sw/ tb/`), the FIFO pilot (`src codes/`, `FIFO Plan.xlsx`, `README_FIFO_UVM.md`), these `docs/`, and the agent code in `fifo_rl/`. Only the 2 FIFO PDFs are not tracked (they stay in the branch's history). For one commit (`a9fc70c`) the iDMA folders were removed from this branch; if you pulled at that point, pull again to get them back. **Merging into `main` is done by Mohamed only.**
- Portability rule: keep all OS-specific bits (tool names `vsim.exe` vs `vsim`, `wslpath` conversion, run-dir root) in one runner module + config, so switching Windows⇄Linux later is a config change. Venvs are not portable → recreate from `requirements.txt`.
- Open question for the engineer: is "WSL Python + Windows Questa" OK, or must Questa itself run on Linux? The latter needs the Linux Questa installer + a license usable from WSL (a node-locked Windows license may not work in WSL2; a license server would).
- WSL interop translates the **working directory** but NOT path arguments: never pass `/mnt/d/...` paths to `vsim.exe`/`vcover.exe`; use relative paths or convert with `wslpath -w`.
- Today, without testbench changes, the only per-run controls are **seed** (`-sv_seed`) and **run length** (`run <N>us`) — neither steers the stimulus. `+UVM_TESTNAME` works but only one test exists; `+uvm_set_config_int` can't reach the knobs (nothing reads them from config_db). → The real-sim loop is **blocked on verification-team requests 1–3**; the fake env is not.
