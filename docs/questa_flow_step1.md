
# From agent action to Questa coverage: the step-1 flow

How one test goes from the agent's choices to a plan file, through Questa, and back as a coverage report and a reward. All examples are real files from `fifo_rl/runs/questa/tests/eval_ddqn_questa/c00000_t000/` (DDQN `best.pt`, campaign 0, test 0, run on 2026-10-27).

---

## 1. The big picture

```
Agent ──action ids──► PlanBuilder ──StimulusPlan──► TextPlanWriter ──plan.txt──► QuestaLauncher
                                                                                     │
                     vsim (FIFO_rl_test reads plan.txt, drives the FIFO, samples cg_FIFO)
                                                                                     │
                     sim.log + result.txt + run.ucdb ──vcover report──► coverage.txt
                                                                                     │
ResultParser (pass? cycles?) ◄── result.txt / sim.log                                │
report_parser (64 bins)      ◄── coverage.txt ◄──────────────────────────────────────┘
        │
RewardAttributor ──► CampaignStore ──► StateEncoder ──► TrajectoryBuilder ──► DQN replay buffer
```

- **One test** = one Questa run = one plan of up to 10 blocks + `END_TEST`.
- **One campaign** = up to 30 tests = one DQN episode.
- The FIFO is reset **once**, at the start of the test. It is **not** reset between blocks, so a block starts from wherever the previous block left the fill level.

The code that runs one test is `CampaignRunner.run_one_test` ([campaign_runner.py](../fifo_rl/fifo_rl/campaign/campaign_runner.py)):

| # | Line of code | What happens |
|---|---|---|
| 1 | `seed = config.seed_for(campaign, test_no)` | 32-bit seed for this test (passed to Questa as `-sv_seed`) |
| 2 | `plan = planner.build(agent, store, test_no, seed)` | agent picks blocks one by one until `END_TEST` |
| 3 | `writer.write(plan, run_dir / "plan.txt")` | plan → text file |
| 4 | `launcher.launch(plan_file, seed, run_dir)` | vsim + vcover → `sim.log`, `result.txt`, `run.ucdb`, `coverage.txt` |
| 5 | `result_parser.parse(result_file, log_file)` | did the test pass? how many cycles? |
| 6 | cycle check | Questa's cycles must equal `plan.sim_cycles()`, otherwise the plan was not applied as written |
| 7 | `hits_vector(coverage.txt)` | 64 true/false flags, one per bin |
| 8 | `attributor.attribute(...)` | rewards for each step of the plan |
| 9 | `store.commit(hits)` → `encoder.encode(store)` | campaign coverage OR-ed, next state |
| 10 | `traj_builder.build(...)` | one transition per block for the replay buffer |

---

## 2. How the agent's actions become a plan

### 2.1 The 9 actions ([plan/actions.py](../fifo_rl/fifo_rl/plan/actions.py))

An action is **not** one transaction. It is a **block**: `n_cycles` transactions, each randomized with the same percentages.

| id | name | wr % | rd % | rst % | cycles | Purpose |
|---|---|---|---|---|---|---|
| 0 | RESET | 0 | 0 | 100 | 2 | empty the FIFO |
| 1 | IDLE | 0 | 0 | 0 | 4 | hold the level (flags with wr=rd=0) |
| 2 | WRITE | 100 | 0 | 0 | 10 | fill up, reach full, overflow |
| 3 | READ | 0 | 100 | 0 | 10 | drain, reach empty, underflow |
| 4 | WRITE_READ | 100 | 100 | 0 | 4 | simultaneous write + read at the current level |
| 5 | RAND50 | 50 | 50 | 0 | 10 | random mix |
| 6 | RESET_TRAFFIC | 50 | 50 | 100 | 16 | reset while writing/reading (the 3 reset-only bins) |
| 7 | BASELINE | 70 | 30 | 10 | 18 | same knobs as the original `FIFO_test` |
| 8 | END_TEST | – | – | – | 0 | close the plan, run the test |

### 2.2 Building the plan ([plan/plan_builder.py](../fifo_rl/fifo_rl/plan/plan_builder.py))

```
plan = StimulusPlan(plan_id, seed, max_len=10)
loop:
    state  = encoder.encode(store, plan)     # 93 numbers: 64 covered bins + last 3 blocks + position + tests left
    action = agent.act(state)
    plan.add(action, state)                  # PlannedAction(idx, action_id, state, forced=False)
    stop when action == END_TEST
    if plan is full (10 blocks): add END_TEST with forced=True
```

Each block gets an index `idx = 0, 1, 2, …` in the order it was chosen. The state it was chosen in is stored with it, because the trainer needs it later.

---

## 3. The plan file sent to Questa

### 3.1 Format ([plan/plan_file.py](../fifo_rl/fifo_rl/plan/plan_file.py))

```
PLAN id=<test number> seed=<sv_seed> version=1
ACT idx=<k> name=<block name> wr=<0-100> rd=<0-100> rst=<0-100> cycles=<n>
...
END_TEST
```

- `PLAN` is the header. The testbench skips it; it is there for people and for `read_plan`.
- There is one `ACT` line per block. `idx` must be 0, 1, 2, … with no gaps.
- `name` is only for logs. The testbench uses only `wr`, `rd`, `rst` and `cycles`.
- `END_TEST` must be the last line.
- `read_plan` / `parse_plan` refuse any broken file (missing field, wrong idx, a percentage outside 0–100, …), so a bad file never reaches the simulator unnoticed.

### 3.2 Real example (`c00000_t000/plan.txt`)

```
PLAN id=0 seed=1733582241 version=1
ACT idx=0 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=1 name=RAND50 wr=50 rd=50 rst=0 cycles=10
ACT idx=2 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=3 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=4 name=RAND50 wr=50 rd=50 rst=0 cycles=10
ACT idx=5 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=6 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=7 name=RAND50 wr=50 rd=50 rst=0 cycles=10
ACT idx=8 name=WRITE wr=100 rd=0 rst=0 cycles=10
ACT idx=9 name=RAND50 wr=50 rd=50 rst=0 cycles=10
END_TEST
```

There are 10 blocks of 10 cycles, so 100 cycles, plus 1 reset cycle = **101 cycles** (`plan.sim_cycles()`). The agent never chose `END_TEST` itself here: the plan reached 10 blocks, so `END_TEST` was forced (`forced_end=1` in `tests.csv`).

---

## 4. How Python starts Questa ([launchers/questa_launcher.py](../fifo_rl/fifo_rl/launchers/questa_launcher.py))

### 4.1 Once per session: compile

Run from `questa_work_dir` (`fifo_rl/runs/questa/`):

```
vlib.exe work
vlog.exe -F <abs path>/tb_rl/rl_files.list +cover -covercells
```

`rl_files.list` = the 14 **unchanged** files of `src codes/` + `tb_rl/FIFO_rl_pkg.sv` + `tb_rl/FIFO_rl_top.sv`.

### 4.2 Per test: simulate

The exact command, copied from line 1 of `sim.log`:

```
vsim -c -sv_seed 1733582241 work.FIFO_rl_top -coverage
     -l tests/eval_ddqn_questa/c00000_t000/sim.log
     +UVM_TESTNAME=FIFO_rl_test
     +PLAN_FILE=tests/eval_ddqn_questa/c00000_t000/plan.txt
     +RESULT_FILE=tests/eval_ddqn_questa/c00000_t000/result.txt
     -do "coverage save -onexit tests/eval_ddqn_questa/c00000_t000/run.ucdb; run -all; quit -f"
```

| Argument | Why |
|---|---|
| `-c` | batch mode, no GUI |
| `-sv_seed <seed>` | the same seed replays the same test; each test gets a different seed from `seed_for(campaign, test)` |
| `work.FIFO_rl_top` | the stand-in top (a copy of `FIFO_top` that imports `FIFO_rl_pkg`) |
| `-coverage` | collect coverage during the run |
| `-l …/sim.log` | the transcript of this test |
| `+UVM_TESTNAME=FIFO_rl_test` | run the plan test instead of `FIFO_test` |
| `+PLAN_FILE=…` | where the test reads the plan |
| `+RESULT_FILE=…` | where the test writes pass / errors / cycles |
| `coverage save -onexit …run.ucdb` | write the coverage database when the run ends. **No `-du FIFO`**: that option keeps only the DUT and drops `cg_FIFO` |

All paths are **relative to the work dir**, because the Windows tools called from WSL translate the working directory but not path arguments.

### 4.3 Per test: report

```
vcover.exe report -cvg -details -nocompactcrossbins tests/.../run.ucdb   > coverage.txt
```

| Option | Why |
|---|---|
| `-cvg` | covergroup coverage only |
| `-details` | print every bin with its hit count (without it you get only percentages) |
| `-nocompactcrossbins` | print every cross bin separately. Without it Questa merges bins as `<auto[0],*,...>` and the parser cannot map them, so it refuses the report |

The launcher times both steps: `sim_time` (vsim, including start-up) and `report_time` (vcover).

---

## 5. What happens inside Questa ([tb_rl/FIFO_rl_pkg.sv](../fifo_rl/tb_rl/FIFO_rl_pkg.sv))

This is the AI team's **stand-in** until the verification team delivers its own plan test. It reuses the original env, agent, driver, monitor, scoreboard, coverage and SVA unchanged.

### 5.1 `FIFO_rl_test extends FIFO_test`

- **build_phase:** `super.build_phase` builds the original env and reset sequence. It then reads `+PLAN_FILE` (fatal error if missing) and `+RESULT_FILE` (optional).
- **run_phase:** the original 4-sequence run is **not** called. It does this instead:
  1. `reset_seq.start(...)`: **1 reset cycle**
  2. `plan_seq.start(...)`: all the blocks of the plan
  3. `#1ns`: lets the last sample reach `cg_FIFO` before the simulation stops
- **report_phase:** writes `result.txt`:
  ```
  passed=<errors==0 && fatals==0> errors=<UVM_ERROR> fatals=<UVM_FATAL> cycles=<1 + transactions> actions=<blocks run>
  ```

### 5.2 `FIFO_plan_sequence`

It reads the file line by line:

- `PLAN` and empty lines: skipped
- `ACT …`: parsed with `$sscanf`. If the line is wrong or `idx` is out of order, it raises `uvm_fatal`.
- `END_TEST`: stop

For each ACT line, `run_block(wr, rd, rst, cycles)` repeats `cycles` times:

```systemverilog
item.WR_EN_ON_DIST = wr;              // existing knobs of FIFO_seq_item
item.RD_EN_ON_DIST = rd;
item.reset_con.constraint_mode(0);    // switch off the fixed 10% reset constraint
item.randomize() with { rst_n dist {0 :/ rst, 1 :/ (100 - rst)}; };   // block's reset %
```

One sequence item = **one clock cycle**: driver → FIFO → monitor (samples at negedge) → scoreboard + `cg_FIFO.sample()`.

### 5.3 Timing seen in the log

The clock period is 10 ns, so 10 cycles = 100 ns:

```
@ 10:  [PLAN] block 0 WRITE wr=100 rd=0 rst=0 cycles=10      ← after the 1 reset cycle
@ 110: [PLAN] block 1 RAND50 wr=50 rd=50 rst=0 cycles=10
...
@ 910: [PLAN] block 9 RAND50 wr=50 rd=50 rst=0 cycles=10
@ 1011: [TEST_DONE]                                          ← 101 cycles + #1ns
```

---

## 6. What Questa gives back

Each test folder `runs/questa/tests/<out name>/c<campaign>_t<test>/` holds:

| File | Written by | Content | Read by |
|---|---|---|---|
| `plan.txt` | Python (`TextPlanWriter`) | the plan | the UVM test |
| `sim.log` | vsim (`-l`) | full transcript: `[PLAN]` lines, scoreboard totals, UVM summary, `** Error` lines from SVA | `ResultParser` (fallback + SVA errors) |
| `result.txt` | `FIFO_rl_test.report_phase` | one line: pass / errors / cycles / actions | `ResultParser` |
| `run.ucdb` | vsim (`coverage save -onexit`) | binary coverage database | `vcover` |
| `coverage.txt` | `vcover report` | text coverage report | `report_parser` |

### 6.1 `result.txt` (real)

```
passed=1 errors=0 fatals=0 cycles=101 actions=10
```

`ResultParser` ([questa/result_parser.py](../fifo_rl/fifo_rl/questa/result_parser.py)):

- **Main source** is `result.txt`.
- It also adds the `** Error` lines of `sim.log`. SVA `$error` messages never reach the UVM counters, so a passed `result.txt` with an assertion failure in the log counts as **failed**.
- If there is no `result.txt`, it reads the UVM Report Summary in the log. If there is no summary either, the result is "did not finish".
- The runner then checks `cycles == plan.sim_cycles()` (101 = 101 here).

### 6.2 `sim.log` (end of the real file)

```
# [report_phase] Total successful transactions: 101
# [report_phase] Total failed transactions: 0
# --- UVM Report Summary ---
# UVM_INFO :   16
# UVM_WARNING :    0
# UVM_ERROR :    0
# UVM_FATAL :    0
# [PLAN]    10                     ← one info line per block run
# Saving coverage database on exit...
# Errors: 0, Warnings: 0
```

---

## 7. The coverage report (`coverage.txt`)

### 7.1 What is measured: `cg_FIFO` = 64 bins

| Item | Kind | Bins |
|---|---|---|
| `cp_wr_en`, `cp_rd_en`, `cp_wr_ack`, `cp_overflow`, `cp_full`, `cp_empty`, `cp_almostfull`, `cp_almostempty`, `cp_underflow` | coverpoints | 2 each (`auto[0]`, `auto[1]`) = 18 |
| `cx_full`, `cx_empty`, `cx_almostfull`, `cx_almostempty`, `cx_wr_ack`, `cx_overflow`, `cx_underflow` | crosses `wr_en × rd_en × flag` | 46 after the `ignore_bins` |

That gives 64 bins in 16 items. Questa's covergroup % is the **average of the 16 items' percentages**. The agent uses the **number of bins** instead (55/64 here, not 90.10%).

### 7.2 Layout of the real report (339 lines)

```
QuestaSim-64 vcover 2024.1 ...                          ← header (tool, time, command, license notice)
=== Instance: /FIFO_coverage_pkg
Covergroup Coverage:
    Covergroups                      1   na   na   90.10%
        Coverpoints/Crosses         16   na   na   na
            Covergroup Bins         64   55    9   85.93%   ← total bins / covered / missing

 TYPE /FIFO_coverage_pkg/FIFO_coverage/cg_FIFO   90.10%   100   Uncovered
    covered/total bins:   55   64
    Coverpoint cp_wr_en   100.00%   100   Covered
        bin auto[0]        27   1   Covered               ← hit count 27, goal 1
        bin auto[1]        74   1   Covered
    ...
    Cross cx_full         100.00%   100   Covered
        Auto, Default and User Defined Bins:
            bin <auto[1],auto[1],auto[0]>    8   1   Covered    ← <wr_en, rd_en, full>
            ...
        Illegal and Ignore Bins:
            ignore_bin auto_full_1           0       ZERO       ← not counted
    Cross cx_empty         62.50%   100   Uncovered
            bin <auto[0],auto[0],auto[1]>    1   1   Covered
            ...
            bin <auto[1],auto[1],auto[1]>    0   1   ZERO      ← missing bin
            bin <auto[0],auto[1],auto[1]>    0   1   ZERO
            bin <auto[1],auto[0],auto[1]>    0   1   ZERO

COVERGROUP COVERAGE:                                    ← the same covergroup printed a 2nd time
 TYPE /FIFO_coverage_pkg/FIFO_coverage/cg_FIFO   90.10% ...
    ...
TOTAL COVERGROUP COVERAGE: 90.10%  COVERGROUP TYPES: 1
Errors: 0, Warnings: 0
```

How to read a bin line: `bin <label>  <hits>  <goal>  <status>`.

- A status of `Covered` means hits ≥ goal (1).
- `ZERO` means the bin was never hit.
- Cross labels are `<wr_en, rd_en, flag>`, for example `cx_empty <auto[1],auto[0],auto[1]>` = write, no read, while empty.

### 7.3 What this test shows

The plan had **no reset after the start** (all blocks have rst=0). The 3 missing `cx_empty` bins (empty while writing or reading) are reachable only when the FIFO is held empty by a reset during traffic. So the test closed 55/64.

The **next** test of the same campaign (`c00000_t001`) used `RESET_TRAFFIC ×5` and found 7 new bins (62/64). Test 3 (`BASELINE ×3`) closed 64/64.

### 7.4 How Python reads it ([questa/report_parser.py](../fifo_rl/fifo_rl/questa/report_parser.py))

- A `Coverpoint …` / `Cross …` line sets the current item. A `bin <label> <count>` line gives one bin.
- The bin name is `"<item>/<label>"`, for example `cx_full/<auto[0],auto[0],auto[1]>`. It is mapped **by name** into the fixed 64-bin order of `coverage_model.py`. Never use line position: Questa lists covered bins first, so the order changes from run to run.
- `ignore_bin` lines are skipped.
- The second copy of the covergroup must give the same counts. A different count is an error.
- The parser raises `ReportError` for a compacted report (`*` in a label), an unknown bin or a missing bin.
- `hits_vector(text)` returns `bool[64]` (count > 0), and `parse_bin_hits(text)` returns `{name: count}`.

---

## 8. From the report to the reward and the next state (step 1)

Step 1 has **one report per test**, so Python cannot tell which block hit which bin.

| Plan step | Reward |
|---|---|
| each block k | `−cycle_cost · cycles_k · scale` (0 by default) |
| `END_TEST` | `(new bins of the whole test − 1 + 10 if 64/64 is reached now) / 64` |

The "new bins" are this test's hits that were not already covered earlier in the campaign. Real rows from `runs/campaign/eval_ddqn_questa/tests.csv`:

| test | blocks | cycles | new bins | total | reward | sim_time | report_time | wall |
|---|---|---|---|---|---|---|---|---|
| 0 | WRITE RAND50 … (10, forced end) | 101 | 55 | 55 | (55−1)/64 = 0.844 | 5.42 s | 0.29 s | 5.71 s |
| 1 | WRITE READ WRITE RESET_TRAFFIC … | 131 | 7 | 62 | (7−1)/64 = 0.094 | 5.62 s | 0.26 s | 5.89 s |

Then:

1. **`CampaignStore.commit(hits)`:** covered bins = OR of all tests so far.
2. **`StateEncoder.encode`:** next state, with the new covered bins and tests left − 1.
3. **`TrajectoryBuilder`:** one transition `(state_k, action_k, reward_k, state_{k+1}, done)` per block. The last block's `next_state` is the start of the next test. With γ = 0.99 the `END_TEST` reward flows back to the earlier blocks through the Q-values.
4. Training: `agent.observe(...)` then `agent.learn()`. In eval mode nothing is learned.

---

## 9. Time per test (this laptop)

| Part | Time |
|---|---|
| compile (once) + first vsim (optimisation) | ≈ 18–20 s |
| vsim per test | ≈ 5.4–6.3 s (almost all start-up; 45 vs 131 cycles makes no difference) |
| vcover report | ≈ 0.3 s |
| Python (plan, parse, reward) | milliseconds |

Since start-up dominates, the agent is judged on the **number of tests**, not the cycles.

---

## 10. What changes in step 2 (for comparison)

| | Step 1 (now) | Step 2 |
|---|---|---|
| Testbench | `cg_FIFO` only | + `action_idx` in seq_item → interface → driver → monitor, + `cg_action` with one instance `act_<idx>` per block (`option.per_instance = 1`) |
| Report | one `TYPE …/cg_FIFO` section | + one `INSTANCE …/act_0`, `act_1`, … section per block |
| Parser | `hits_vector` | + `hits_per_action` → `bool[n_blocks][64]` |
| Reward | whole test → `END_TEST` | each block gets its own new bins, in idx order; check that the OR of all `act_k` = `cg_FIFO` |
| Plan file / launcher | – | unchanged (the `idx` field is already there) |

---

## 11. Re-run this yourself

From `fifo_rl/` on Windows with Questa on PATH and `MGLS_LICENSE_FILE` set:

```bash
python -m scripts.run_campaign eval --agent dqn --checkpoint runs/campaign/train_ddqn_seed0/best.pt \
       --launcher questa --eval-campaigns 1
FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py      # fake FIFO == Questa, bin for bin
```

To run one plan by hand from `runs/questa/` (after one compile):

```bash
vsim -c -sv_seed 1 work.FIFO_rl_top -coverage -l my.log +UVM_TESTNAME=FIFO_rl_test \
     +PLAN_FILE=my_plan.txt +RESULT_FILE=my_result.txt \
     -do "coverage save -onexit my.ucdb; run -all; quit -f"
vcover report -cvg -details -nocompactcrossbins my.ucdb > my_coverage.txt
```

---

## 12. Comparison on Questa: CRV vs DDQN vs random agent (2026-10-29)

### 12.1 The four methods

| Method | What it is | Questa runs |
|---|---|---|
| **`FIFO_test`** | the original test, unchanged: reset, 100 write-only, 100 read-only, 9,999 × 70/30/10 = 10,200 cycles | always 1 |
| **Constrained random** | `--agent baseline`: every block is BASELINE (70/30/10); full plans of 10 blocks = 181 cycles per test; repeat until 64/64 | as many as needed |
| **DDQN (step 1)** | `best.pt`, greedy (no exploration) | as many as needed |
| **Random agent** | picks one of the 9 actions uniformly at random every step, no learning | as many as needed |

All four use the **same seeds**: the seed of test *t* of campaign *c* is the same for every method. `FIFO_test` uses the seed of test 0 of each campaign.

The cycle where `FIFO_test` reaches 64/64 comes from Questa itself. It runs with `+UVM_VERBOSITY=UVM_HIGH`, so the monitor prints every sampled cycle. Those printed signals are turned into the 64 bins cycle by cycle, and the result is checked to equal Questa's coverage report. A second run without the verbose log gives the wall time.

### 12.2 Results on Questa (10 campaigns each)

| | `FIFO_test` | Constrained random | DDQN | Random agent |
|---|---|---|---|---|
| Reached 64/64 | 10/10 | 10/10 | 10/10 | 9/10 (one stopped at 62 after 30 runs) |
| Questa runs (mean, range) | **1** | 2 (1–4) | 3.3 (3–4) | 10.9 (4–30) |
| Cycles needed for 64/64 (mean, range) | 379 (275–538) | 362 (181–724) | **312** (263–410) | 520 (132–1062) |
| Cycles actually simulated | 10,200 | 362 | 312 | 520 |
| Wall time per campaign | **5.1 s** | 6.9 s | 12.1 s | 54.7 s |
| Bins after the first run | 64 | 62.6 | 54.7 | 40.1 |

Constrained random counts whole tests (181 cycles each), so its real closing cycle is somewhere inside its last test. Questa start-up was ≈ 3.5 s per run on this day; a `FIFO_test` run is ≈ 1.5 s longer because of its 10,200 cycles.

### 12.3 Same comparison on the fake FIFO (200 campaigns)

| | `FIFO_test`-like run | Constrained random | DDQN | Random agent |
|---|---|---|---|---|
| Reached 64/64 | 100% | 100% | 99.5% | 98.5% |
| Runs (mean / median) | 1 | 1.77 / 2 | 3.75 / 3 | 8.86 / 7 |
| Cycles to close (mean / median) | 346 / 323 | 320 / 362 | 354 / 287 | 447 / 395 |

### 12.4 What it means

- **In time, `FIFO_test` wins.** Questa start-up dominates, and it needs one run. The DDQN needs about 3.
- **In cycles, the DDQN wins on Questa.**
  - 312 vs 379 for `FIFO_test` and 362 for constrained random, about 15–20% fewer.
  - It is also the most consistent: always 3–4 runs.
  - On the fake FIFO, constrained random has a slightly lower mean, so the gain over CRV is small and not yet solid.
- **DDQN vs random agent:** learning clearly works. It needs 3× fewer runs, 40% fewer cycles, and is 4.5× faster.
- **Why the agent can't win on this FIFO:**
  - random 70/30/10 stimulus already covers about 62 of 64 bins in roughly 300 cycles
  - only the reset-during-traffic bins are hard
  - each extra Questa run costs more than the cycles the agent saves
- **Headroom:** one hand-written plan of 9 blocks (`WRITE IDLE READ WRITE_READ IDLE WRITE WRITE_READ IDLE RESET_TRAFFIC`, 67 cycles) closes 64/64 in **one run** for 97% of seeds. That would beat `FIFO_test` in both runs and cycles. The DDQN does not find it yet.

Raw data:
- `fifo_rl/runs/campaign/cmp10_*_questa/` and `cmp200_*_fake/` (`tests.csv`, `summary.json`)
- `fifo_rl/runs/questa/tests/fifo_test_orig/summary.json`
