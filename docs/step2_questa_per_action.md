# Step 2 on real Questa: per-block coverage (`act_<idx>`)

2026-11-17. Questa now reports which bins **each block** of a plan hit, not only the whole test. Everything is in the AI team's stand-in testbench `fifo_rl/tb_rl/`; **`src codes/` is unchanged**. Nothing committed.

---

## 1. What was changed, where, and why

| # | File | Change | Why |
|---|---|---|---|
| 1 | `fifo_rl/tb_rl/FIFO_rl_pkg.sv` | new covergroup **`cg_action`**: same 9 coverpoints, 7 crosses and `ignore_bins` as `cg_FIFO`; `option.per_instance = 1`; instance name from the constructor (`act_0`, `act_1`, …) | every block gets its own copy of the 64 bins, with names the parser already knows |
| 2 | same file | new component **`FIFO_action_coverage`** (`uvm_subscriber`): receives every monitor sample and samples the `act_<k>` of the block that sample belongs to | assigns each cycle to its block **without changing** `seq_item`, driver, monitor or `FIFO_coverage` |
| 3 | same file | `FIFO_plan_sequence`: calls `act_cov.add_block(cycles)` for each ACT line, **before** sending that block's items | the coverage component knows where each block starts and ends in time |
| 4 | same file | `FIFO_rl_test`: reads **`+ACTION_COV=1`**; only then creates `act_cov`, gives it to the plan sequence, and connects it to `env.agt.agt_ap` in `connect_phase` | step 1 stays exactly as before when the plusarg is absent |
| 5 | same file | `report_phase`: a `UVM_ERROR` if any sample falls after the last block (`unmapped != 0`) | if the counting ever got out of step, the test fails instead of silently giving wrong per-block coverage |
| 6 | `fifo_rl/fifo_rl/launchers/questa_launcher.py` | `QuestaLauncher(per_action=False)`; with `True`, vsim gets `+ACTION_COV=1` | Python decides per run whether Questa produces `act_k` |
| 7 | `fifo_rl/scripts/run_campaign.py` | `make_launcher` passes `per_action = (reward_mode == "action")` to the Questa launcher too | `--reward-mode action --launcher questa` now gets real per-block coverage |
| 8 | `fifo_rl/tests/test_launchers.py` | `test_questa_per_action_asks_for_act_instances` | checks the plusarg is added only in step 2 |
| 9 | `fifo_rl/tests/test_questa_real.py` | `test_questa_act_instances_match_the_fake_fifo_block_by_block` | on Questa, every block's 64 hit counts equal the fake FIFO's |

The parser (`report_parser.py`) needed **no change**. It already accepted `Covergroup instance act_<k>` lines, the format Questa actually prints.

### 1.1 How a sample is assigned to its block

One transaction = one clock cycle = one monitor sample:

```
sample 0               -> the test's reset cycle      (no block)
samples 1 .. c0        -> act_0   (c0 = cycles of block 0)
samples c0+1 .. c0+c1  -> act_1
...
```

Two Questa limits found during the build:
- `push_back(new(...))` is a syntax error for covergroups
- a queue of covergroup handles is refused (`'const ref' can't be passed`)

So the instances live in a fixed array of 64, `MAX_BLOCKS`, far above `plan_max_len = 10`. A larger plan stops with `uvm_fatal`.

---

## 2. Checks

| Check | Result |
|---|---|
| All tests (`FIFO_RL_QUESTA=1`, 4 tests on real Questa) | **265 passed** |
| Fixed 9-block plan (no randomness): Questa `act_k` vs fake FIFO, every block, every bin | **identical** |
| Trained step-2 agent, 10 campaigns (19 tests), report without vs with `+ACTION_COV=1` | **same plans, same seeds, `cg_FIFO` identical in all 19 tests**: per-block coverage does not change the stimulus |
| Extra simulation time of per-block coverage (same plan, 4 runs each, alternating) | 2.50 s → 2.64 s per run (+0.14 s, within noise) |
| Unmapped samples | 0 in every run (no `UVM_ERROR`) |

---

## 3. The two reports side by side

Same agent (`step2_seed0/best.pt`), same plan, same seed (1733582241), campaign 0, test 0:

```
BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC   (105 cycles)
```

| | Before (`questa10_step2_seed0/c00000_t000/coverage.txt`) | After (`questa10_step2_seed0_act/c00000_t000/coverage.txt`) |
|---|---|---|
| Lines / size | 339 lines, 30 KB | 3,549 lines, 332 KB |
| Covergroup types | 1 (`cg_FIFO`) | 2 (`cg_FIFO`, `cg_action`) |
| `cg_FIFO` | 61/64 bins, 97.39% | **the same**: 61/64, 97.39%, identical counts |
| Per-block sections | none | `act_0` … `act_9`, 64 bins each |
| What Python can learn | "this test found 61 new bins" | "block 0 found 47, block 2 found 7, block 3 found 2, …" |
| `vcover report` time | 0.09 s | 0.10 s |

### 3.1 Layout of the new report

```
=== Instance: /FIFO_coverage_pkg
 TYPE /FIFO_coverage_pkg/FIFO_coverage/cg_FIFO     97.39%   ← unchanged, the whole test
    ... 64 bins ...
=== Instance: /FIFO_rl_pkg
 TYPE /FIFO_rl_pkg/cg_action                       53.49%   ← NEW: merged over all blocks (ignored by Python)
 Covergroup instance act_0                         77.96%   ← NEW: only the cycles of block 0 (BASELINE)
    covered/total bins:   47   64
    Coverpoint cp_wr_en
        bin auto[0]        6   1   Covered
        bin auto[1]       12   1   Covered                  ← 18 cycles = 6 + 12
    ...
 Covergroup instance act_1                         35.05%   ← block 1 (IDLE)
 ...
 Covergroup instance act_9                         68.33%   ← block 9 (RESET_TRAFFIC)
COVERGROUP COVERAGE:                                         ← Questa prints everything a 2nd time
 ...
TOTAL COVERGROUP COVERAGE: 75.44%  COVERGROUP TYPES: 2
```

`TOTAL COVERGROUP COVERAGE` changes (97.39% → 75.44%) because Questa now averages the two covergroup types. Python never uses that line; it counts the 64 `cg_FIFO` bins, and those are unchanged.

### 3.2 What a block "found": new bins credited to each block

Each block is credited only with bins nobody hit earlier in the campaign. Columns = blocks idx 0–9:

| Campaign, test | Plan | Questa (real `act_k`) | Fake FIFO, same seed |
|---|---|---|---|
| 0, 0 | BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC | 47 0 7 2 1 0 1 3 0 0 (61) | 46 0 7 4 0 0 1 3 0 0 (61) |
| 0, 1 | WRITE RAND50 WRITE RAND50 … IDLE | 0 2 0 0 0 0 0 0 1 0 (3 → 64/64) | 0 2 0 0 0 0 0 0 1 0 |
| 1, 0 | (same plan as 0, 0) | 48 0 7 4 0 0 0 2 0 0 | 40 0 7 12 0 0 0 2 0 0 |
| 2, 0 | (same plan as 0, 0) | 46 0 7 4 1 0 0 3 0 0 | 47 0 7 4 1 0 1 1 1 0 |

- The **deterministic blocks** (IDLE, WRITE, READ, WRITE_READ) always behave the same.
- The **random blocks** (BASELINE, RAND50, RESET_TRAFFIC) differ in detail between Questa and the fake FIFO, because each simulator draws its own random numbers. With no randomness they are identical (section 2).
- The picture is the same in both: BASELINE does the bulk (~46 bins), WRITE then READ add the full and almost-full corners (~7 + 2–4), and RESET_TRAFFIC adds the reset-during-traffic bins (2–3).

---

## 4. Agent results with real per-block coverage (10 campaigns)

| | Before (`cg_FIFO` only) | After (`+ACTION_COV=1`) |
|---|---|---|
| Reached 64/64 | 10/10 | 10/10 |
| Questa runs per campaign | 1.9 | 1.9 |
| Cycles per campaign | 190 | 190 |
| Reward per block in `tests.csv` | no (step-1 logging) | **yes** (`block_new_bins`, from Questa) |

The agent's choices don't change, as expected: it plays greedily and doesn't learn during evaluation. What changes is that Questa now gives the per-block information step 2 learns from. **Training step 2 directly on Questa is now technically possible.** It would still take hours (≈ 11,000 tests at ~2.5 s), so training stays on the fake FIFO, which now matches Questa block by block.

---

## 5. Re-run

From `fifo_rl/`, with Questa and `MGLS_LICENSE_FILE` set:

```bash
FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py                 # 4 tests, includes act_k == fake
python -m scripts.run_campaign eval --agent dqn --checkpoint runs/campaign/s1_vs_s2/step2_seed0/best.pt \
       --launcher questa --questa-bin C:/questasim64_2024.1/win64 --eval-campaigns 10 \
       --reward-mode action --out <new folder>
```

By hand: add `+ACTION_COV=1` to the `vsim` line of `docs/questa_flow_step1.md` §11.

Results: `fifo_rl/runs/campaign/s1_vs_s2/questa10_step2_seed0_act/` and `fifo_rl/runs/questa/tests/questa10_step2_seed0_act/`.
