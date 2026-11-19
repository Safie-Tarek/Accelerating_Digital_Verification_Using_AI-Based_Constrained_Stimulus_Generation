# Step 1 vs step 2: per-test reward vs per-action reward (DDQN, FIFO)

Experiment run 2026-10-31. Training on the fake FIFO, evaluation on the fake FIFO (200 campaigns) and on real Questa (10 campaigns). Nothing committed. The step-1 results from earlier days (`runs/campaign/train_ddqn_seed0`, `cmp10_*`, `cmp200_*`, `docs/questa_flow_step1.md`) are unchanged; this experiment wrote only into new folders.

---

## 1. Summary

| | Step 1 (reward per test) | Step 2 (reward per action) |
|---|---|---|
| Reached 64/64 on Questa | 29 / 30 campaigns | **30 / 30** |
| Questa runs per campaign | 3.37 ± 2.44 | **2.17 ± 0.38** |
| Cycles per campaign on Questa | 391 ± 216 | **273 ± 86** |
| Wall time per campaign on Questa | 8.5 ± 6.1 s | **5.4 ± 0.9 s** |
| First time training reached 100% closure | campaign ~1,200 | **campaign ~330** |
| Collapse late in training | yes | yes (not fixed) |

(Mean ± std over 3 training seeds; each seed evaluated on the same 10 Questa seeds.)

**Step 2 is better on every average and much more consistent across training seeds:**
- fewer Questa runs (−36%), fewer cycles (−30%), less wall time (−36%)
- it learns about 3.5× sooner

**But it does not yet beat the original `FIFO_test` in wall time** (3.05 s on the same day), and it doesn't fix the late-training collapse.

---

## 2. What changed between step 1 and step 2

Only **where the reward lands** changed. The agent, network, state (93 numbers), actions (9), plan file, training settings and seeds are identical.

| | Step 1 | Step 2 |
|---|---|---|
| Coverage report | one `cg_FIFO` section per test | `cg_FIFO` plus one `act_<idx>` section per block (bins hit during that block) |
| Reward of block k | 0 (minus an optional cycle cost) | **new bins that block k was the first to hit** / 64 |
| Reward of `END_TEST` | (all new bins of the test − 1 + 10 if closed) / 64 | (new bins only the reset cycle hit − 1 + 10 if closed) / 64 |
| Total reward of a test | X | **the same X** (checked by a unit test) |

Example: one real step-2 test (fake FIFO, `eval200_step2_seed0`, campaign 0, test 0):

```
blocks   : BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC END_TEST
new bins :    46     0     7    4      0          0       1        3            0          0         (61 in total)
```

In step 1 the same test gives all 61 bins to `END_TEST`, so the agent can't tell that `READ` after `WRITE` found 4 bins and the second `WRITE_READ` found none. In step 2 it sees this directly.

### 2.1 Code changes (fake FIFO only, nothing in `src codes/` or `tb_rl/`)

| File | Change |
|---|---|
| `fifo_rl/launchers/fake_launcher.py` | `FakeLauncher(per_action=True)` adds `INSTANCE …/cg_action/act_<idx>` sections; the reset cycle at the start of the test belongs to no block |
| `fifo_rl/questa/report_parser.py` | `hits_vector` reads only `cg_FIFO`; new `hits_per_action(report, n_blocks)`, which raises an error if an `act_k` section is missing or extra |
| `fifo_rl/campaign/reward_attributor.py` | new `ActionRewardAttributor` (first hit wins, blocks in idx order); new field `block_new_bins` |
| `fifo_rl/campaign/campaign_runner.py` | `RunConfig.reward_mode` = `"test"` (step 1, default) or `"action"` (step 2) |
| `scripts/run_campaign.py` | `--reward-mode`; columns `reward_mode`, `block_new_bins` in `tests.csv`; `_action` in default folder names; **refuses to write into a folder that already has results** |
| `configs/default.yaml` | `reward_mode: test` |
| `tests/test_step2.py` | 15 new tests; all **260** pass, plus 3 opt-in Questa tests skipped |

---

## 3. Setup

| | Value |
|---|---|
| Agent | DDQN, `DQNAgent` unchanged; MLP 93 → 128 → 128 → 9 |
| Training | 2,000 campaigns on the fake FIFO, γ 0.99, lr 5e-4, ε 1.0 → 0.05 over 80,000 steps, buffer 50,000 |
| Training seeds | 0, 1, 2 for **both** steps (6 runs) |
| Model kept | `best.pt` = best validation (50 campaigns, seed offset 1000, every 100 campaigns) |
| Fake evaluation | 200 unseen campaigns (seed offset 2000), the same seeds as the earlier CRV comparison |
| Questa evaluation | the 10 seeds of the earlier comparison; Questa Sim-64 2024.1, stand-in `FIFO_rl_test` |
| Campaign | up to 30 tests; one test = one plan of up to 10 blocks + `END_TEST` |

**Evaluation on Questa needs no testbench change.** Rewards are only used for learning, so a step-2 agent runs on today's testbench like a step-1 agent. The verification team only has to add `action_idx` / `cg_action` if we ever want to *train* on Questa.

---

## 4. Results on the fake FIFO (200 campaigns per agent)

| Agent | Closes | Tests (mean / median) | Cycles per campaign (mean / median) | Bins after 1st test |
|---|---|---|---|---|
| Step 1, seed 0 | 100% | 2.54 / 2 | 295 / 230 | 57.5 |
| Step 1, seed 1 | 97.5% | 4.77 / 3 | 432 / 338 | 54.4 |
| Step 1, seed 2 | 100% | 1.73 / 2 | 293 / 346 | 63.1 |
| **Step 1, mean ± std** | 99.2% | **3.01 ± 1.57** | **340 ± 80** | 58.3 |
| Step 2, seed 0 | 100% | 2.03 / 2 | 200 / 200 | 61.5 |
| Step 2, seed 1 | 100% | 1.77 / 2 | 320 / 362 | 63.1 |
| Step 2, seed 2 | 100% | 2.62 / 3 | 259 / 230 | 51.2 |
| **Step 2, mean ± std** | **100%** | **2.14 ± 0.44** | **260 ± 60** | 58.6 |
| *Reference: CRV in plans (always BASELINE)* | 100% | 1.77 / 2 | 320 / 362 | 62.6 |
| *Reference: random agent* | 98.5% | 8.86 / 7 | 447 / 395 | 40.1 |

---

## 5. Results on real Questa (10 campaigns per agent, same seeds, same day)

| Agent | Reached 64/64 | Questa runs (mean, range) | Cycles (mean, range) | Wall time per campaign |
|---|---|---|---|---|
| Step 1, seed 0 | 10/10 | 2.6 (2–4) | 301 (230–544) | 6.5 s |
| Step 1, seed 1 | 9/10 | 6.1 (2–30) | 637 (170–2958) | 15.3 s |
| Step 1, seed 2 | 10/10 | 1.4 (1–2) | 235 (165–346) | 3.6 s |
| **Step 1, mean ± std** | 29/30 | **3.37 ± 2.44** | **391 ± 216** | **8.5 ± 6.1 s** |
| Step 2, seed 0 | 10/10 | 1.9 (1–2) | **190** (105–202) | 4.7 s |
| Step 2, seed 1 | 10/10 | 2.0 (1–4) | 362 (181–724) | 5.2 s |
| Step 2, seed 2 | 10/10 | 2.6 (2–3) | 266 (136–333) | 6.4 s |
| **Step 2, mean ± std** | **30/30** | **2.17 ± 0.38** | **273 ± 86** | **5.4 ± 0.9 s** |
| *Original `FIFO_test`* | 10/10 | **1** | 379 needed (10,200 simulated) | **3.05 s** |
| *CRV in plans (always BASELINE)* | 10/10 | 2.0 (1–4) | 362 (181–724) | 5.0 s |

Questa start-up was ≈ 2.5 s per run on this day, faster than the ≈ 3.5 s of the earlier comparison. So only rows from this table should be compared with each other, not with `docs/questa_flow_step1.md` §12.

---

## 6. Training behaviour

Validation closure rate every 100 campaigns (20 points per run):

| Run | First 100% | Evaluations at 100% | Last evaluation (campaign 2,000) |
|---|---|---|---|
| Step 1, seed 0 | 1,200 | 2 / 20 | 70% |
| Step 1, seed 1 | 1,400 | 3 / 20 | 6% |
| Step 1, seed 2 | 1,000 | 5 / 20 | 72% |
| Step 2, seed 0 | **100** | 4 / 20 | 68% |
| Step 2, seed 1 | 800 | 3 / 20 | 0% |
| Step 2, seed 2 | **100** | 5 / 20 | 74% |

- **Step 2 learns much sooner:** first 100% at campaign ~330 on average vs ~1,200. The per-block reward is a direct signal, so it doesn't have to travel back through 10 steps of γ.
- **Both collapse:** the closure rate swings between 0% and 100% from one evaluation to the next, and all 6 runs end lower than their best. Step 2 doesn't fix this. It is a training problem (learning rate, target updates, ε floor, replay), not a reward problem. Keeping `best.pt` hides it in the results, but it must be fixed before the results can be called robust.

---

## 7. What the agents learned

Most common plans in the 200 fake campaigns:

| Agent | Test 1 | Test 2 |
|---|---|---|
| Step 1, seed 0 | `WRITE RAND50 BASELINE WRITE` | `BASELINE ×10` |
| Step 1, seed 1 | `WRITE BASELINE` | `BASELINE ×10` or `RESET_TRAFFIC ×10` |
| Step 1, seed 2 | `WRITE BASELINE ×7 WRITE BASELINE` | `BASELINE ×10` |
| **Step 2, seed 0** | **`BASELINE IDLE WRITE READ WRITE_READ WRITE_READ IDLE RESET_TRAFFIC BASELINE RESET_TRAFFIC`** | `WRITE RAND50 …` |
| Step 2, seed 1 | `BASELINE ×10` (= plain CRV) | `BASELINE ×10` |
| Step 2, seed 2 | `BASELINE IDLE RESET RAND50` | `WRITE BASELINE ×9` |

- **Step 1 mostly learns "use BASELINE".** That is plain constrained random with a small change. This is what a reward only at the end of a test can teach.
- **Step 2, seed 0 learned a structured plan:** fill → hold → read from full → write+read at the boundary → reset during traffic. This is close to the hand-written 9-block plan that closes in one run. It gets 61.5 bins in test 1 and needs the fewest cycles of all agents (190 on Questa). It closes in 1 run on some Questa seeds, but usually in 2.
- **Step 2, seed 1 converged to pure BASELINE.** On Questa it gives exactly the CRV baseline's numbers (2.0 runs, 362 cycles). Per-action reward makes the structured plan easier to find, but doesn't guarantee it.

---

## 8. Pros and cons

### Step 1: reward per test
**Pros**
- Needs nothing from the testbench: works with today's Questa report (`cg_FIFO` only).
- Could be trained directly on Questa without any testbench change.
- Simple, and already validated (fake FIFO = Questa, bin for bin).

**Cons**
- Credit assignment is weak: the agent can't see which block was useful.
- Learns slowly (~1,200 campaigns to the first 100%).
- Results vary a lot between training seeds (runs 1.4–6.1 on Questa; one seed failed a campaign).
- Mostly converges to "BASELINE everywhere" and never finds the ordered plan.

### Step 2: reward per action
**Pros**
- Better on every average: −36% Questa runs, −30% cycles, −36% wall time vs step 1, and 30/30 closed.
- Much more consistent across seeds (std of runs 0.38 vs 2.44).
- Learns about 3.5× sooner.
- Can find structured, ordered plans (seed 0), the kind of stimulus constrained random can't produce on purpose.
- The total reward is the same as step 1, so the agents are judged on the same objective.
- **No testbench change is needed to use it**: training on the fake FIFO, evaluation on today's Questa.

**Cons**
- Needs per-block coverage: the fake FIFO today, or `action_idx` + `cg_action` from the verification team to train on Questa.
- The real Questa `act_k` report format is assumed, not yet checked (the parser accepts `INSTANCE …` and `Covergroup instance …` lines).
- "First hit wins" is greedy: a block that sets up a later hit (`IDLE` before `READ`) gets 0 itself and relies on γ, like in step 1.
- Does not fix the late-training collapse.
- Still slower than the original `FIFO_test` in wall time (5.4 s vs 3.05 s), and not reliably 1 run.

---

## 9. Comparison with the original test (same day, same seeds, Questa)

| | Runs | Cycles needed | Cycles simulated | Wall time |
|---|---|---|---|---|
| Original `FIFO_test` | 1 | 379 | 10,200 | **3.05 s** |
| Step 2, best seed (0) | 1.9 | **190** | **190** | 4.7 s |
| Step 2, mean of 3 seeds | 2.17 | 273 | 273 | 5.4 s |
| CRV in plans | 2.0 | 362 | 362 | 5.0 s |
| Step 1, mean of 3 seeds | 3.37 | 391 | 391 | 8.5 s |

- **Cycles:** step 2 needs 28% fewer cycles than `FIFO_test` on average (273 vs 379), and the best seed needs **50% fewer** (190). It simulates **54× fewer** cycles (190 vs 10,200).
- **Wall time:** `FIFO_test` still wins, because each extra Questa run costs ~2.5 s of start-up. Step 2 would win if it closed in 1 run (≈ 2.5 s), or if one Questa session stayed open between tests.

---

## 10. Conclusion and next steps

**Step 2 is a clear improvement over step 1**: more reliable, faster to learn, fewer runs and cycles. It is the first version that sometimes learns an ordered, coverage-directed plan instead of plain constrained random. **It is not yet better than the original test in wall time.**

Next steps, in order of expected gain:
1. **Fix the training collapse** (lower lr, slower target updates or soft updates, more training seeds), then repeat this comparison with 5 seeds.
2. **Push towards 1 run per campaign:**
   - raise the step cost per test so a second run costs more
   - or raise `plan_max_len` so a plan has room for the full ordered sequence plus a margin
3. **Add the FIFO fill level to the state** as a separate experiment against step 2. The ordered plan depends on the level, which the agent can only guess today.
4. **Keep one vsim session open across tests,** so start-up is paid once and the cycle savings show in wall time.
5. **Ask the verification team for `action_idx` + `cg_action`** only if we want to train on Questa; check the real `act_k` report format against the parser.

---

## 11. Where the results are

| What | Folder (under `fifo_rl/`) |
|---|---|
| Training runs (`best.pt`, `last.pt`, `eval.csv`, `tests.csv`, logs) | `runs/campaign/s1_vs_s2/step{1,2}_seed{0,1,2}/`, `runs/campaign/s1_vs_s2_step*_seed*.log` |
| Fake evaluation, 200 campaigns | `runs/campaign/s1_vs_s2/eval200_step{1,2}_seed{0,1,2}/` |
| Questa evaluation, 10 campaigns | `runs/campaign/s1_vs_s2/questa10_step{1,2}_seed{0,1,2}/`, `questa10_baseline/` |
| `FIFO_test` on the same day | `runs/questa/tests/fifo_test_s1_vs_s2/summary.json` |
| Per-test Questa files | `runs/questa/tests/questa10_*/c*_t*/` (plan, log, result, coverage) |

Re-run (from `fifo_rl/`):

```bash
H="--gamma 0.99 --lr 5e-4 --eps-decay-steps 80000 --buffer-capacity 50000 --n-campaigns 2000"
python -m scripts.run_campaign train $H --seed 0 --reward-mode test   --out <new folder>   # step 1
python -m scripts.run_campaign train $H --seed 0 --reward-mode action --out <new folder>   # step 2
python -m scripts.run_campaign eval --agent dqn --checkpoint <folder>/best.pt --eval-campaigns 200 --reward-mode action --out <new folder>
python -m scripts.run_campaign eval --agent dqn --checkpoint <folder>/best.pt --launcher questa \
       --questa-bin C:/questasim64_2024.1/win64 --eval-campaigns 10 --out <new folder>
```

The script refuses an `--out` folder that already has results, so no run can overwrite another.
