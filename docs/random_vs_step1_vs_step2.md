# Random agent vs step 1 vs step 2: full comparison (FIFO)

All Questa numbers below were measured **on the same day (2026-11-18), on the same 10 seeds, back to back**, so every column can be compared directly. Fake-FIFO numbers use 200 campaigns on the same seeds. Nothing committed; this experiment wrote only new folders (section 9).

---

## 1. The three agents (plus two references)

| | What chooses the blocks | Reward it learned from | Learns? |
|---|---|---|---|
| **Random agent** | uniform random choice among the 9 actions at every step | – | no |
| **Step 1 (DDQN)** | trained network, greedy | per **test**: all new bins of the test go to `END_TEST` | yes, on the fake FIFO |
| **Step 2 (DDQN)** | trained network, greedy | per **block**: each block gets the new bins it was first to hit (`act_<idx>`) | yes, on the fake FIFO |
| *Constrained random (reference)* | always `BASELINE` (70/30/10, the `FIFO_test` settings), full plans of 181 cycles | – | no |
| *Original `FIFO_test` (reference)* | fixed: reset, 100 write, 100 read, 9,999 × 70/30/10 = 10,200 cycles in one run | – | no |

Same for all three agents: the 9 actions, the plan file, the 93-number state, the campaign (up to 30 Questa runs, stop at 64/64), and the stand-in testbench `FIFO_rl_test`. Step 1 and step 2 also have the same network, the same training settings (2,000 campaigns, γ 0.99, lr 5e-4, ε decay 80,000) and the same 3 training seeds (0, 1, 2).

---

## 2. Results on real Questa (same day, same 10 seeds)

| | Random agent | Step 1 | **Step 2** | *Constrained random* | *`FIFO_test`* |
|---|---|---|---|---|---|
| Reached 64/64 | 9/10 | 29/30 | **30/30** | 10/10 | 10/10 |
| Questa runs per campaign | 10.9 (4–30) | 3.37 ± 2.44 | **2.17 ± 0.38** | 2.0 (1–4) | 1 |
| Cycles needed | 520 (132–1062) | 391 ± 216 | **273 ± 86** | 362 (181–724) | 379 (275–538) |
| Cycles simulated | 520 | 391 | **273** | 362 | 10,200 |
| Wall time per campaign | 28.4 s (10–75) | 8.7 ± 6.0 s | **5.6 ± 0.9 s** | 5.1 s | **3.6 s** |
| Bins after the 1st run | 40.1 | 58.0 | 58.1 | 62.6 | 64 |
| Per-block coverage in the Questa report | no | no | **yes** (`act_0`…`act_k`) | no | no |

- Step 1 and step 2 are the mean ± std over the 3 training seeds; each seed played the same 10 campaigns.
- One step-1 test hung inside Questa for 395 s, a tool problem rather than the agent. Step 1 seed 0 was re-run with identical runs and cycles; the table uses the re-run.
- Questa start-up was ≈ 2.5 s per run on this day.

### 2.1 Per training seed

| Seed | Step 1: runs / cycles / time | Step 2: runs / cycles / time |
|---|---|---|
| 0 | 2.6 / 301 / 7.2 s | **1.9 / 190 / 4.9 s** |
| 1 | 6.1 / 637 / 15.3 s (one campaign failed after 30 runs) | 2.0 / 362 / 5.2 s |
| 2 | 1.4 / 235 / 3.5 s | 2.6 / 266 / 6.7 s |

---

## 3. Results on the fake FIFO (200 campaigns, same seeds)

| | Random agent | Step 1 | **Step 2** | *Constrained random* | *`FIFO_test`-like run* |
|---|---|---|---|---|---|
| Reached 64/64 | 98.5% | 99.2% | **100%** | 100% | 100% |
| Runs per campaign | 8.86 | 3.01 ± 1.57 | **2.14 ± 0.44** | 1.77 | 1 |
| Cycles per campaign | 459 | 340 ± 80 | **260 ± 60** | 320 | 346 (needed) |

The fake FIFO and Questa give the same ranking.

---

## 4. Improvement from one to the next (Questa)

| From → to | Questa runs | Cycles | Wall time | Reached 64/64 |
|---|---|---|---|---|
| Random → step 1 | 10.9 → 3.37 (**−69%**) | 520 → 391 (**−25%**) | 28.4 → 8.7 s (**−69%**) | 9/10 → 29/30 |
| Step 1 → step 2 | 3.37 → 2.17 (**−36%**) | 391 → 273 (**−30%**) | 8.7 → 5.6 s (**−36%**) | 29/30 → 30/30 |
| **Random → step 2 (total)** | 10.9 → 2.17 (**−80%**) | 520 → 273 (**−48%**) | 28.4 → 5.6 s (**−80%, 5×**) | 9/10 → 30/30 |
| Step 2 vs constrained random | 2.17 vs 2.0 (+9%) | 273 vs 362 (**−25%**) | 5.6 vs 5.1 s (+10%) | equal |
| Step 2 vs `FIFO_test` | 2.17 vs 1 | 273 vs 379 needed (**−28%**); 273 vs 10,200 simulated (**37× fewer**) | 5.6 vs 3.6 s (+54%) | equal |

Training speed (first time the validation closure rate hit 100%):
- step 1: campaign ~1,200
- step 2: campaign ~330, **3.6× sooner**

Both still collapse late in training. That is not fixed yet; `best.pt` is used.

---

## 5. What each agent does (most common plans, fake FIFO)

| Agent | Typical behaviour | Effect |
|---|---|---|
| Random agent | random blocks; ends a plan at random (1 step in 9), so many tests are 0–3 blocks | ~40 bins in the 1st run, then many runs that add **nothing** (26 of 30 in its worst campaign) |
| Step 1 | `WRITE BASELINE …`, then `BASELINE ×10` | learned "use constrained random", plus a little `WRITE` at the start |
| Step 2 (seed 0) | `BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC` | an **ordered** plan: fill → hold → read from full → boundary → reset during traffic; 61 bins in 105 cycles |

Per-block report on Questa for that step-2 plan (new bins per block, `act_0`…`act_9`):

```
BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC
   47      0    7     2      1          0        1        3           0          0      = 61
```

---

## 6. Pros and cons

### 6.1 Random agent
**Pros**
- No training, no model, nothing to tune.
- Uses the same plan framework, so it is the fair "no intelligence" baseline.
- Explores every action, so it sometimes hits corners by chance (best campaign: 132 cycles).

**Cons**
- Worst on every measure: 10.9 Questa runs, 520 cycles, 28.4 s per campaign; the only one that failed a campaign on Questa (62/64 after 30 runs).
- Very unpredictable: 4 to 30 runs, 10 to 75 s.
- Ignores coverage: keeps running tests that add nothing.
- Ends plans at random, wasting a whole Questa start-up on 1–20 cycle tests.

### 6.2 Step 1: DDQN, reward per test
**Pros**
- Big gain over random: −69% runs, −25% cycles, −69% time.
- Needs **nothing from the testbench**: works with the standard `cg_FIFO` report.
- Simplest reward, and could be trained on Questa without any testbench change.
- Its best seed is very good (1.4 runs, 3.5 s).

**Cons**
- Weak credit assignment: one reward per test, so the agent can't tell which block helped.
- Very inconsistent between training seeds (1.4–6.1 runs; one seed failed a campaign).
- Learns slowly (~1,200 campaigns to the first 100%).
- Mostly converges to "BASELINE everywhere" (constrained random). On average it does **not** beat plain constrained random (3.37 vs 2.0 runs, 391 vs 362 cycles).
- Training collapses late; it depends on keeping `best.pt`.

### 6.3 Step 2: DDQN, reward per block
**Pros**
- **Best of the three agents on every measure:**
  - 30/30 closed
  - 2.17 runs, 273 cycles, 5.6 s
  - −36% runs and −30% cycles vs step 1
  - −80% runs and time vs random
- **Most consistent:** std of runs 0.38 vs 2.44 for step 1, and every seed closes all campaigns.
- Learns **3.6× sooner** than step 1.
- Fewer cycles than constrained random (−25%) and than `FIFO_test` (−28% needed, 37× fewer simulated).
- Can learn **ordered, coverage-directed plans** (seed 0), which constrained random cannot produce on purpose.
- Its total reward per test equals step 1's, so it optimises the same goal.
- **Per-block coverage now comes from real Questa** (`+ACTION_COV=1`), checked equal to the fake FIFO block by block, without changing `src codes/`.

**Cons**
- Needs per-block coverage: the fake FIFO, or the `cg_action` addition in the testbench (done in our stand-in `tb_rl/`; the verification team's own test would need the same).
- The report is ~10× larger (3,549 vs 339 lines). Simulation cost is small (+0.14 s per run, within noise).
- "First hit wins" doesn't reward set-up blocks (`IDLE` before `READ` gets 0); they still rely on γ, as in step 1.
- Not always a structured plan: one seed learned plain constrained random.
- Still collapses late in training.
- **Still slower than `FIFO_test` in wall time** (5.6 vs 3.6 s), because it usually needs 2 Questa runs and each start-up costs ~2.5 s. It is about equal to constrained random in runs and time, and better only in cycles.

---

## 7. Ranking

| Measure | Best → worst |
|---|---|
| Wall time on Questa | `FIFO_test` (3.6 s) → constrained random (5.1) → **step 2 (5.6)** → step 1 (8.7) → random (28.4) |
| Questa runs | `FIFO_test` (1) → constrained random (2.0) → **step 2 (2.17)** → step 1 (3.37) → random (10.9) |
| Cycles needed | **step 2 (273)** → constrained random (362) → `FIFO_test` (379) → step 1 (391) → random (520) |
| Reliability (closed, spread) | **step 2** = `FIFO_test` = constrained random → step 1 → random |
| Among the RL agents | **step 2** → step 1 → random |

---

## 8. Conclusion

- **Learning works:** step 1 removes ~70% of the random agent's runs and time.
- **Per-block reward works better:** step 2 removes another ~35% and makes training reliable. It is the best agent on every measure.
- On cycles, step 2 is now better than both non-RL references.
- On wall time it is still behind `FIFO_test`, because this FIFO's coverage is easy and each Questa start-up costs more than the cycles saved.

The next gains should come from:
1. fixing the late-training collapse
2. pushing step 2 to close in **one** run (reward or plan length)
3. adding the FIFO fill level to the state
4. keeping one Questa session open between tests

The first two would put step 2 ahead of `FIFO_test` in time as well.

---

## 9. Where the data is (under `fifo_rl/`)

| What | Folder |
|---|---|
| Questa, same day, all agents | `runs/campaign/final_cmp/{random, baseline, step1_seed0_rerun, step1_seed1, step1_seed2, step2_seed0, step2_seed1, step2_seed2}/` |
| Step 1 seed 0 with the 395 s Questa hang (kept, not used) | `runs/campaign/final_cmp/step1_seed0/` |
| `FIFO_test`, same day | `runs/questa/tests/fifo_test_final/summary.json` (`scripts/run_fifo_test.py`) |
| Per-test Questa files (plan, log, result, coverage; step 2 with `act_k`) | `runs/questa/tests/{random, baseline, step1_seed*, step2_seed*}/c*_t*/` |
| Fake FIFO, 200 campaigns | `runs/campaign/cmp200_random_fake/`, `cmp200_baseline_fake/`, `s1_vs_s2/eval200_step{1,2}_seed*/` |
| Trained models | `runs/campaign/s1_vs_s2/step{1,2}_seed{0,1,2}/best.pt` |

Related docs:
- `docs/step1_vs_step2_report.md`: step 1 vs step 2 in detail
- `docs/step2_questa_per_action.md`: the per-block Questa coverage
- `docs/questa_flow_step1.md`: how a plan goes through Questa
