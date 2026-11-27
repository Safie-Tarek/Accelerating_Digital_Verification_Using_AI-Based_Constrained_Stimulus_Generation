# Trials: A2C without the "random" actions

Run 2026-11-27 (Jana). **Trials only:** no repo code changed, and this file is not committed.
Compare with `docs/a2c_vs_ddqn_results.md` (A2C with all 9 actions).

- **Trial 1** (§1–§5): `BASELINE` forbidden.
- **Trial 2** (§6): `BASELINE` and `RAND50` forbidden. **Best result of the project so far:** a directed plan that closes 64/64 in **one Questa run, ~79 cycles** (ideal: 67).

## 0. How to read this doc (campaigns, tests, blocks)

The agent plays a **coverage game**:

| Word | Meaning | Analogy |
|---|---|---|
| **Campaign** | One attempt to reach 64/64 coverage, starting from 0 bins. It ends when all 64 bins are covered or after 30 tests (`campaign_budget`), whichever comes first. | one game |
| **Test** | One simulation (one Questa run). The agent writes a plan of up to 10 blocks plus `END_TEST`, the simulation runs it, and the new bins are added to the campaign's total. | one turn |
| **Block** | One piece of stimulus inside a plan, e.g. `WRITE` = write for 10 cycles, `IDLE` = do nothing for 4 cycles. 9 blocks exist (`fifo_rl/plan/actions.py`). | one move |

**How many campaigns the agents play:**

| Phase | Campaigns | What happens |
|---|---|---|
| Training | 3,000 per seed (A2C), 2,000 (DDQN) | The agent plays and learns from every campaign. |
| Validation (inside training) | 50, every 100 training campaigns | It plays without learning; the best validation becomes `best.pt`. |
| Final evaluation | 300 on the fake FIFO, 10 on real Questa | Unseen campaigns, no learning: the reported numbers. |

Every campaign has its own random seed, so simulations differ a little between campaigns. Validation always uses the same 50 campaigns (seed + 1000), and evaluation uses unseen ones (seed + 2000).

**Reading the numbers:**
- "**Closes 100% of campaigns**": every campaign reached 64/64 before using its 30 tests.
- "**Closes in 1.8 tests**" (expected tests): on average 1.8 simulations per campaign. A campaign that does not close counts as 30.
- "**Closes in 1 test**": a single simulation reached 64/64.
- "**First 100% at campaign 600**": after 600 training campaigns, the agent closed all 50 validation campaigns for the first time.
- **Greedy** = the agent always takes its most likely block (the normal way to evaluate). **Sampled** = it rolls its dice with its learned probabilities.
- **Seed** = one independent training run. We train 3 seeds to see whether the result is reliable or luck.

## 1. Question

With all 9 actions, A2C settled on `BASELINE` ×10 on 2 of 3 seeds. That is the constrained-random recipe (70% write, 30% read, 10% reset, from the verification plan's FIFO_4), so the agent just copied constrained random. Is `BASELINE` a shortcut that stops the agent from learning a real strategy?

**Test:** forbid `BASELINE` and train again under exactly the same conditions.

## 2. Set-up

- Same as the main A2C run: step-2 reward (`--reward-mode action`), γ 0.99, `cycle_cost` 0, all A2C defaults from `configs/default.yaml` (`ent_coef` 0.01, `a2c_lr` 7e-4, K = 4 tests per update, …).
- 3 seeds (0, 1, 2) × 3,000 campaigns on the fake FIFO. Validation on 50 campaigns every 100 campaigns, with the new checkpoint rule (fewest expected tests, then cycles).
- Evaluation: greedy, 300 unseen fake campaigns (seed + 2000), for `best.pt` and `last.pt`. The same campaigns as the main A2C run.
- **How `BASELINE` was removed:** with A2C's existing legality mask, `BASELINE` is marked illegal in every state, so its probability is exactly 0 in both acting and learning. The state (93 numbers) and the network outputs (9) stay the same. This is done by patching `legal_mask` at runtime in a separate script (§11), so the repo is unchanged. Not done for the DQN.
- Not run on Questa. The fake FIFO matches Questa bin for bin (`tests/test_questa_real.py`), and the main A2C run showed the same ranking on both.

## 3. Results (300 unseen campaigns, mean of 3 seeds)

| Agent | Closed | Expected tests | Cycles |
|---|---|---|---|
| Random block each step | 99.1% | 8.90 | 462 |
| Constrained random (`BASELINE` ×10) | 100% | 1.76 | 319 |
| A2C **with** `BASELINE`, `best.pt` | 100% | **1.70** | 294 |
| A2C **with** `BASELINE`, `last.pt` | 100% | 1.77 | 315 |
| **A2C without `BASELINE`, `best.pt`** | **100%** | 1.80 | **194** |
| A2C without `BASELINE`, `last.pt` | 99.9% | 2.09 | 225 |
| *Step-2 DDQN (200 campaigns, `docs/random_vs_step1_vs_step2.md`)* | 100% | 2.14 | 260 |
| *Best possible (the directed 9-block plan)* | 100% | 1 | 67 |

Per seed, without `BASELINE`:

| Seed | `best.pt`: tests / cycles | `last.pt`: tests / cycles (closed) |
|---|---|---|
| 0 | 1.93 / 207 | 2.17 / 237 (100%) |
| 1 | 1.73 / 186 | 2.27 / 242 (100%) |
| 2 | 1.75 / 189 | 1.83 / 197 (99.7%) |

## 4. What the agent learned

Without `BASELINE`, all three seeds (`best.pt`) play the same 4-block pattern in their first test, repeated to fill the plan:

```
seed 0:  RAND50 WRITE RAND50 RESET_TRAFFIC RAND50 WRITE RAND50 RESET_TRAFFIC RAND50 WRITE
seed 1:  WRITE RAND50 RESET_TRAFFIC RAND50 WRITE RAND50 RESET_TRAFFIC RAND50 WRITE RAND50
seed 2:  RAND50 WRITE RAND50 RESET_TRAFFIC RAND50 WRITE RAND50 RESET_TRAFFIC RAND50 WRITE
```

In words: random traffic → fill toward full (`WRITE`) → random traffic → reset in the middle of traffic (`RESET_TRAFFIC`, the only block that reaches the 3 reset-only bins).

- Three independent seeds converged on the same recipe, so it is a stable solution, not luck.
- It is built from basic blocks and is more structured than `BASELINE` ×10.
- Plans are still always the full 10 blocks (`END_TEST` forced). The agent never ends a test early, because with `cycle_cost` 0 a longer plan costs nothing.

**How it got there** (validation, closure / expected tests):
- **Campaigns 0–500 (all seeds) and up to 1,000 (seed 0):** the greedy policy was `RAND50` ×10. That is the next "random-like" block after `BASELINE`, and it was **stuck at 61/64**, because `RAND50` never resets, so the 3 reset-only bins are never hit.
- **First 100%:** campaign 1,200 (seed 0; 98% at 1,100), 800 (seed 1) and 600 (seed 2), once `RESET_TRAFFIC` entered the plan.

## 5. Stability

Validation closure fell back after first reaching 100%. The main A2C run (with `BASELINE`) never did this after campaign 500.

| Seed | Drops after first reaching 100% |
|---|---|
| 0 | 94% at 1,700; 98% at 2,300 (small) |
| 1 | 0% at 1,000–1,200; 0% at 1,600; 82% at 1,800 |
| 2 | 0% at 800; 0% at 1,200; 20% at 1,400; 84% at 2,300 |

- `last.pt` is worse than `best.pt` by 0.08–0.55 tests. Seed 1's final weights partly slid back to `RAND50` ×10 (it played it in 362 of its 682 eval tests). So, like DDQN, this variant **needs `best.pt`** when played greedily.

### 5.1 Why: greedy evaluation is unstable, not the agent

The drops happen **only in greedy validation**. In the same campaigns, the training games (sampled play, rolling the policy's dice) stayed steady.

| Seed 1, campaign | Greedy validation closure | Training (sampled): tests per campaign | Entropy | Greedy plan |
|---|---|---|---|---|
| 900 | 100% | 2.80 | 0.82 | test 1: `RAND50 RESET_TRAFFIC RAND50 WRITE …`, later tests `RAND50` ×10 |
| 1,000–1,200 | **0%** | 2.36 / 3.10 / 2.15 | 0.91 / 0.80 / 0.74 | `RAND50` ×10 |
| 1,300–1,400 | 100% | 2.26 / 2.10 | 0.67 / 0.72 | `RAND50 WRITE RAND50 RESET_TRAFFIC …` |
| 1,600 | **0%** | 2.37 | 0.47 | `RAND50` ×10 |
| 1,700 | 100% | 2.08 | 0.65 | `RAND50 WRITE RAND50 RESET_TRAFFIC …` |

Seed 2 looks the same: greedy `RAND50 WRITE` ×5 at 800 (0%), `RAND50` ×10 at 1,200 (0%), `WRITE` ×10 at 1,400 (20%), while sampled training stays at 2.0–3.8 tests per campaign.

**The mechanism:**
1. Closing 64/64 needs at least one `RESET_TRAFFIC` (the only block that hits the 3 reset-only bins).
2. At each plan position the policy gives `RESET_TRAFFIC` roughly 20–30% probability, a little less than `RAND50`. Entropy stays around 0.5–1.1, so the policy is still spread out.
3. **Sampled play:** over 10 blocks, at least one `RESET_TRAFFIC` almost always appears, so the campaign closes.
4. **Greedy play** takes the most likely block at every position, and that is never `RESET_TRAFFIC`. The plan becomes `RAND50` ×10 and stops at 61/64. Greedy play is deterministic, so the next test is the same plan again: 30 tests, 0% closure.
5. A tiny update that moves `RESET_TRAFFIC` from just above to just below `RAND50` at one position flips the greedy plan from "closes" to "never closes". The agent barely changed, but its greedy score jumps between 100% and 0%.
6. **A second effect (distribution shift):** during training the dice nearly always hit the reset bins in the first test, so the state "everything covered except the reset bins" is rarely seen. The agent never learns to respond to it, and greedy play walks right into it. At campaign 900 the greedy first test still contained `RESET_TRAFFIC`, so the reset bins were covered and `RAND50` ×10 in the later tests was harmless. At campaign 1,000 the first test flipped to `RAND50` ×10 as well, and every later test repeated it.

**Why `BASELINE` hid this:** `BASELINE` includes 10% reset, so even greedy `BASELINE` ×10 hits the reset bins and closes. With `BASELINE` available, the greedy plan is always safe, so training looked stable.

### 5.2 Test of the cheapest fix: sampled evaluation

Same checkpoints, same 300 unseen campaigns, played by sampling instead of greedily (with `BASELINE` still masked, see the note below):

| | Closed | Expected tests | Cycles | Per seed (tests / cycles) |
|---|---|---|---|---|
| `best.pt`, greedy | 100% | **1.80** | **194** | 1.93/207, 1.73/186, 1.75/189 |
| `last.pt`, greedy | 99.9% | 2.09 | 225 | 2.17/237, 2.27/242, 1.83/197 |
| `best.pt`, sampled | 100% | 2.18 | 227 | 2.09/220, 2.05/221, 2.41/240 |
| **`last.pt`, sampled** | **100%** | 1.87 | 202 | 2.12/224, 1.78/197, 1.71/184 |

**Note (corrected 2026-11-27):** a first version of this table was made with `run_campaign eval --sample`, which does not know a block was forbidden. Those runs played `BASELINE` in 1–10% of blocks, because its never-trained score still gives it a small probability. They were redone with the mask kept (`a2c_forbid_eval.py`, §11). The numbers changed by at most 0.3 tests and the conclusions did not change. The leaked runs are kept as `LEAKED_evalfake300_*`. **Rule: a forbidden block must stay masked at evaluation too.** Greedy evaluations were not affected (0 `BASELINE`).

- With sampled play, the **final weights never fail a campaign** and are almost as good as the best greedy checkpoint. So the end of training was fine; greedy testing made it look broken.
- `best.pt` gets worse when sampled. It was chosen *by greedy validation*, so it is the checkpoint that happened to be good greedily, not the best policy overall. Selection and evaluation should use the same way of playing.

### 5.3 Possible fixes (easiest first)

1. **Validate and evaluate with sampled play** for this variant. `train()` validates greedily today; letting validation sample (with fixed seeds, so it stays repeatable) is a small change. `best.pt` would then no longer depend on which block happens to be on top. Tested in §5.2: it works.
2. **Make the policy commit before greedy play.** Decay `ent_coef` from 0.01 to 0 over training, or train longer, so the policy becomes confident and `RESET_TRAFFIC` becomes the top choice where it is needed. Then greedy play works too.
3. **Smoother updates.** A lower `a2c_lr` (3e-4) or more tests per update (`a2c_rollout_tests` 8) moves the probabilities less per update, so there are fewer flips. Already part of the planned sweep.
4. **Reward / curriculum** (bigger change, team decision). For example, a bonus for closing within the current test would make the agent place `RESET_TRAFFIC` on purpose instead of by chance.

## 6. Trial 2: without `BASELINE` and `RAND50`

**Idea:** in Trial 1 the agent first fell back on `RAND50` ×10, the next "random traffic" block. With `RAND50` also removed, no random block remains that can be repeated. `RESET_TRAFFIC` is random, but reset is held all the time, so the FIFO stays empty. Every other block is directed. The ideal 1-test plan (`WRITE IDLE READ WRITE_READ IDLE WRITE WRITE_READ IDLE RESET_TRAFFIC`) only uses blocks that are still allowed.

**Set-up:** the same as Trial 1 (§2), with both blocks masked during training **and during evaluation** (§5.2 note). Evaluated greedily and sampled, `best.pt` and `last.pt`, on 300 fake campaigns, then `best.pt` on real Questa (10 campaigns, `--seed 0`, the same campaigns as `docs/a2c_vs_ddqn_results.md`).

### 6.1 Results, fake FIFO (300 unseen campaigns)

| Seed | `best.pt` greedy: closed / tests / cycles | `last.pt` greedy | `best.pt` sampled | `last.pt` sampled |
|---|---|---|---|---|
| 0 | 100% / 1.01 / **72** | 100% / 1.01 / 78 | 100% / 1.31 / 94 | 100% / 1.08 / 84 |
| 1 | 99.7% / 1.10 / 90 | 99.7% / 1.10 / 90 | 100% / 2.49 / 204 | 98.3% / 1.64 / 130 |
| 2 | 100% / **1.00** / 83 | 100% / 1.00 / 83 | 100% / 1.91 / 145 | 100% / 1.10 / 91 |
| **Mean** | **99.9% / 1.04 / 82** | 99.9% / 1.04 / 84 | 100% / 1.90 / 148 | 99.4% / 1.27 / 102 |

- Seed 1's 1.10: in 299 of 300 campaigns it closed in one test. In campaign 241 its first test reached 62/64 instead of 64, and in that state the greedy plan has no `RESET_TRAFFIC`, so it repeated a plan without reset 29 times (charged the full 30 tests). It is the same weakness as §5.1, point 6: a state that training rarely visits.
- Seed 0 needed a second test in 3 of its 300 campaigns.
- Here **greedy beats sampled**, the opposite of Trial 1. The policies became confident (they committed to one plan), so sampling only adds mistakes.

### 6.2 Results, real Questa (10 campaigns, same campaigns as all earlier Questa numbers)

| | Reached 64/64 | Questa runs per campaign | Cycles | Wall time per campaign |
|---|---|---|---|---|
| Constrained random (`BASELINE` ×10) | 10/10 | 2.0 | 362 | 4.9 s |
| A2C, all 9 actions | 30/30 | 1.83 | 318 | 4.4 s |
| Step-2 DDQN (2026-11-18) | 30/30 | 2.17 | 273 | 5.6 s |
| **A2C, no `BASELINE` / `RAND50`** (seeds 0 / 1 / 2) | **30/30** | **1.0** | **79** (71 / 83 / 83) | **2.3 s** (2.6 / 1.8 / 2.6) |
| `FIFO_test` (one long CRV run) | 10/10 | 1 | 379 needed, 10,200 simulated | 3.1 s |

- All 30 Questa tests passed (no UVM errors) and each reached 64/64 in its single run. The fake FIFO predicted exactly this (71 / 83 / 83 cycles).
- **This is the first agent that needs fewer cycles than `FIFO_test` needs to reach 64/64 (79 vs 379, about 5×) and uses one run like it.** It simulates 79 cycles instead of 10,200.
- **Wall time:** 2.3 s vs 3.1 s for `FIFO_test`, both measured on 2026-11-27. But single runs varied from 1.7 s to 4.3 s on this PC (Questa start-up dominates), so this gap is within the noise. Claim "as fast as `FIFO_test`, with 130× fewer simulated cycles", not "faster", unless a repeated, interleaved timing confirms it.

### 6.3 The plans it learned

`best.pt`, greedy, identical in all 300 fake campaigns and all 10 Questa campaigns:

```text
seed 0:     RESET_TRAFFIC WRITE IDLE WRITE_READ IDLE WRITE READ WRITE_READ IDLE WRITE_READ      (71 cycles)
seeds 1, 2: RESET_TRAFFIC WRITE READ WRITE_READ IDLE RESET_TRAFFIC WRITE IDLE WRITE_READ IDLE   (83 cycles)
ideal:      WRITE IDLE READ WRITE_READ IDLE WRITE WRITE_READ IDLE RESET_TRAFFIC                 (67 cycles)
```

- Seeds 1 and 2 found **exactly the same plan** independently.
- Both plans are what a verification engineer would write: reset during traffic (reset-only bins), then fill, hold, read, boundary write/read at empty and full, and wait.
- Seed 0's plan uses the same blocks as the ideal plan in a different order, plus one block the ideal plan does not need. The extra cycles come from that, not from random traffic.
- Plans are still 10 blocks long; the agent never ends a plan early (`cycle_cost` 0, §7 point 5).

### 6.4 How training went

| Seed | First 1-test validation | `best.pt` from | Greedy drops after first 100% |
|---|---|---|---|
| 0 | 2,500 | 2,600 | 2 (1,300, 1,900) |
| 1 | 1,600 | 1,600 | 1 (700) |
| 2 | 2,300 | 2,300 | 4 (1,300, 1,500, 2,000, 2,600) |

- **Slow start:** for the first 600–1,200 campaigns, greedy validation was 0% on every seed, stuck at 45–61 bins with plans like `WRITE READ` ×5 or `WRITE WRITE_READ IDLE` ×3. Sampled training still closed 100% of its campaigns, in 5–9 tests each.
- **Then it found the plan and kept it:** by campaign 3,000, all seeds played their 1-test plan in every validation, and `last.pt` equals `best.pt` (within 0.01 tests). Unlike Trial 1, the end of training is stable, because the policy became confident in a plan that does not depend on luck.
- Training wall time: ~2 hours per seed (3 seeds in parallel on one CPU), much longer than Trial 1 (~10–15 min), because stuck greedy validations play all 30 tests in every validation campaign.

### 6.5 Why removing `RAND50` helped so much

- With a random block available, the cheapest thing to learn is "repeat the random block, then patch the missing bins". That closes in ~1.8 tests, so the push towards a precise plan is weak.
- With only directed blocks, each block gives the **same bins every time**. A plan either closes or it does not, and the step-2 reward credits each block exactly for the bins it adds. That is a clean, repeatable signal, and A2C can climb it to a full directed plan.
- It also makes greedy play safe at the end: a directed plan that closes once closes every time (§6.2: 30/30 on Questa).

## 7. What we learned (for the report)

1. **`BASELINE` acts as a crutch.** With it, A2C is very stable but lazy: it rediscovers constrained random and stops. Without it, A2C builds its own stimulus recipe from basic blocks.
2. **That recipe is cheaper.** It needs ~34% fewer cycles than A2C with `BASELINE` (194 vs 294) and fewer than step-2 DDQN (260), with almost the same number of tests (1.80 vs 1.70; constrained random 1.76).
3. **Trade-off: stability vs strategy.** Removing the easy option gives better plans and unsteady training (collapses mid-training, `last.pt` < `best.pt`). That matches what we saw with DDQN, which also learned more structured plans and was also unstable.
4. **Agents first look for "one block repeated".** With `BASELINE` it was `BASELINE` ×10, without it `RAND50` ×10. Only when that fails to close (61/64) do they add the missing block. Reward that a single repeated block can collect is the main thing that slows learning down.
5. **The reward still does not ask for short plans.** Every plan is 10 blocks. The ideal plan (1 test, 67 cycles) needs a reason to stop early: a non-zero `cycle_cost` or a per-block penalty. That is a team decision about the shared reward.
6. **Removing both random blocks gives the best agent of the project (Trial 2).** All 3 seeds learned a directed plan that closes 64/64 in **one Questa run and ~79 cycles** (30/30 campaigns on Questa), against 1.8–2.2 runs and 270–320 cycles for every earlier agent and 67 cycles for the hand-made ideal plan. Two seeds found the identical plan.
7. **The action set matters as much as the algorithm.** The same A2C, settings and reward went from "copies constrained random" (all 9 actions) to "cheaper random mix" (no `BASELINE`) to "near-ideal directed test" (no `BASELINE`, no `RAND50`). Random blocks are a shortcut that gives a weak learning signal; directed blocks give a clean one.
8. **Cost of the harder action set: slower learning.** 1,600–2,500 campaigns to the first 1-test plan (vs ~500 to a stable `BASELINE` plan), with 0% greedy closure for the first 600–1,200 campaigns. Training took ~2 hours per seed on CPU.
9. **Keep the mask at evaluation.** A forbidden block's score is never trained, so without the mask at evaluation the agent can still sample it (it did, in 1–10% of blocks, §5.2 note).

## 8. Is this really a better result? (honest assessment)

**On the numbers, yes.** Trial 2 beats every earlier agent on every metric: 1 Questa run instead of ~2, 79 cycles instead of 270–320, 30/30 campaigns closed. And the agent found the plan by itself: nobody gave it the order of the blocks, and two seeds found the identical plan independently. That is the strongest evidence in the project that RL can do coverage-directed stimulus generation.

**But it is not a clean win yet, because we changed the problem to get there:**

1. **We used knowledge of the answer.** `BASELINE` and `RAND50` were removed because we already knew the ideal plan does not need them. For a new design, nobody knows in advance which blocks are shortcuts. Choosing the action set by hand is part of the solution, and it is our part, not the agent's.
2. **The comparison with DDQN and the 9-action A2C is not fair yet.** They had all 9 actions. To say "A2C is better", DDQN must get the same 7 actions. The gain may come from the action set rather than from A2C (§9, experiment A).
3. **The agent learned one fixed test, not a strategy.** It plays the same plan in every campaign. An engineer could write that test by hand, and someone already did (the 67-cycle plan). So on this FIFO the agent rediscovered something known. It has not shown that it can react to what is still uncovered: its one failed campaign (§6.1, seed 1, campaign 241) is exactly a case where it had to adapt and could not.
4. **The FIFO covergroup is easy.** RL should help most where coverage is hard to hit and nobody knows the best test. This FIFO does not test that.
5. **Wall time is a tie, not a win.** Questa start-up dominates, so 79 vs 10,200 simulated cycles barely shows in seconds (§6.2).

**Suggested wording for the report:**

> With an action set made only of directed blocks, A2C discovers on its own a directed test that closes coverage in one simulation and ~79 cycles, close to the hand-written optimum (67). With random blocks available, both A2C and DDQN settle for constrained-random-like plans. So the action set strongly shapes what the agent learns.

That is an honest result and a real lesson. It is not proof that RL beats constrained random in general.

## 9. Next experiments (agreed: we do all three)

| | Experiment | Question it answers | Status |
|---|---|---|---|
| A | **DDQN with the same 7 actions** (no `BASELINE`, no `RAND50`), step-2 DDQN settings (2,000 campaigns, γ 0.99, lr 5e-4, ε decay 80,000, seeds 0–2) | Does the gain come from the action set or from A2C? | running (2026-11-27) |
| B | **All 9 actions, but the random blocks cost something** (a per-block cost for `BASELINE` / `RAND50`, or a non-zero `cycle_cost`) | Does the agent avoid the shortcuts *by itself*? Then nobody has to choose the action set by hand, which is the more convincing version. | implemented on branch `feat/random-block-cost` (`random_block_cost`, `random_blocks`); A2C sweep running (2026-11-27) |
| C | **A harder coverage model**, where the best test is not obvious (e.g. cross coverage, sequences, rare corner cases) | Does RL beat constrained random where it is supposed to? | longer term |

Notes for B: `cycle_cost` changes the shared reward, so both agents must be retrained, and it is a team decision. A cost only on the random blocks is closer to Trial 2 (it makes them unattractive instead of forbidden). Start with a small value, so closing coverage still pays more than the cost.

### 9.1 Which blocks are "random", and what a cost on them means

The **random blocks** choose write / read at random in each cycle, instead of a fixed pattern:

| Block | Per cycle | Cycles | Role |
|---|---|---|---|
| `BASELINE` | 70% write, 30% read, 10% reset | 18 | the verification plan's constrained-random recipe (FIFO_4) |
| `RAND50` | 50% write, 50% read | 10 | random traffic around the current fill level |
| `RESET_TRAFFIC` | 50% write, 50% read, reset held | 16 | random, but reset is held, so the FIFO stays empty; the only block that hits the 3 reset-only bins. **Not** counted as a shortcut: it stays free. |

**Cost:** today a block's reward (step 2) is the number of new bins it covers (scaled by 1/64). A cost would subtract a fixed amount each time the agent picks `BASELINE` or `RAND50`. Example with a cost of 0.5 bins: a `BASELINE` block that covers 5 new bins scores 5 − 0.5 = 4.5 instead of 5. Random blocks stay allowed but become less attractive. If the agent then prefers directed blocks **by itself**, that is more convincing than us forbidding them.

The other option, **`cycle_cost`**, already exists: it charges every simulated cycle of every block, so long plans cost more (`BASELINE` ×10 = 180 cycles; the directed plan ~70–80). It pushes towards short plans in general, not only away from random blocks, and it can be tried with no code change (`--cycle-cost 0.01`).

### 9.2 Why have random blocks at all?

On this FIFO, directed blocks alone are enough (Trial 2 proves it). The random blocks are there for reasons beyond this FIFO:

1. **It is in the project's title.** "AI-Based **Constrained Stimulus Generation**": constrained random is the industry-standard method, and a common RL set-up is an agent that tunes or mixes constrained-random settings. `BASELINE` and `RAND50` are those settings. Without them the project becomes "AI picks directed tests", a different (also valid) story.
2. **This FIFO is a special case.** All 64 bins are control states (full, empty, overflow, underflow, reset, …), and directed blocks control those exactly. Real covergroups often also have bins on data values, timing between events, or interleaved operations, which a few fixed directed blocks cannot reach. Random traffic explores combinations nobody thought to write.
3. **In a real project nobody knows the ideal sequence.** We know the 67-cycle plan only because it was worked out by hand for this small FIFO. For a bigger design nobody knows whether a directed-only plan exists, so random blocks are a safety net that always makes progress.
4. **Random blocks make early learning easy.** With them, A2C closed coverage within ~500 campaigns. With directed blocks only, it was stuck at 0% for the first 600–1,200 campaigns, because a directed plan works only in the right order.

So: removing the random blocks gives the best result on this FIFO (a valid finding), and keeping them keeps the set-up general. Experiment B asks whether the agent can learn **when** random traffic is worth it, without us deciding for it.

### 9.3 How to implement experiment B (cost on random blocks)

Where the reward is made today: `fifo_rl/campaign/reward_attributor.py`. Every block already gets `-cycle_cost * n_cycles * scale` (`_cycle_costs`), then the new bins it covers (step 2) or the test's new bins on `END_TEST` (step 1). `RewardConfig` (`fifo_rl/reward.py`) holds `step_cost` = 1 bin per test, `closure_bonus` = 10 and `scale` = 1/64.

Steps:

1. **Config** (`configs/default.yaml`): two new keys, defaults that change nothing:
   ```yaml
   random_block_cost: 0.0               # bins charged each time a block below is chosen (0 = off)
   random_blocks: [BASELINE, RAND50]    # the blocks that pay it
   ```
2. **`RunConfig`** (`fifo_rl/campaign/campaign_runner.py`): add a field `block_costs` (a dict block id → cost), built in `RunConfig.from_cfg` from the two keys with `ACTION_INDEX`.
3. **`RewardAttributor`** (`reward_attributor.py`): take `block_costs` in `__init__` (next to `cycle_cost`, with the same "must be ≥ 0" check), and in `_cycle_costs` subtract `block_costs.get(pa.spec.id, 0) * scale` per block. Both the step-1 and the step-2 attributor use `_cycle_costs`, so both get it.
4. **`CampaignRunner`** (line ~114): pass `config.block_costs` when building the attributor.
5. **Nothing else changes:** agents, launchers, state and Questa stay the same. The cost only changes the rewards the agent learns from; `new_bins`, closure and cycles are measured as before, so results stay comparable.
6. **Tests** (`tests/test_campaign.py` / `test_step2.py`):
   - with cost 0, rewards are exactly the same as today (no regression);
   - a plan with `BASELINE` gets `-cost * scale` on that block and nothing on the others;
   - a negative cost is refused.
7. **Logging:** add a column with the number of random blocks per test to `tests.csv`, so the share of random blocks over training can be plotted.
8. **Choosing the value:** the cost is in bins, like `step_cost` (1 bin per test).
   - Too small: nothing changes (`BASELINE` ×10 still wins).
   - Too big: the agent never uses random blocks even where they help.
   - `BASELINE` ×10 then costs 10 × cost bins per test. A first sweep: cost ∈ {0.25, 0.5, 1.0}.
9. **Experiment:** A2C (and later DDQN), all 9 actions, step-2 reward, γ 0.99, 3 seeds × 3,000 campaigns per cost value, plus `cycle_cost` ∈ {0.005, 0.01} as the no-code alternative. Report tests, cycles, closure, the share of random blocks in the learned plans, and Questa for the best setting.

**Success looks like:** with all 9 actions available, the agent's plans use few or no `BASELINE` / `RAND50` blocks and reach Trial 2's level (~1 test, ~80 cycles), with nobody choosing the action set.

## 10. Limits and follow-ups

- Trial 1 was evaluated on the fake FIFO only. Trial 2 was also checked on Questa (§6.2).
- Is removing the random blocks fair? See §8. Options: keep all blocks but give random blocks a cost (§9 B), or train in stages (directed blocks first, then add random ones).
- Trial 2 wall time vs `FIFO_test`: needs a repeated, interleaved timing before saying "faster" (§6.2).
- DDQN with the same 7 actions: §9 A.
- One setting only (default A2C hyperparameters). A larger `ent_coef` or a smaller `a2c_lr` might reduce the collapses.
- Not tried for DDQN. A fair "with vs without `BASELINE`" comparison would remove it for both agents.
- Possible next experiments: no `BASELINE` together with a small `cycle_cost`, and a richer coverage model where repeated random traffic cannot close.

## 11. Reproduce

**Data** (git-ignored, only on Jana's PC):

| What | Folder under `fifo_rl/runs/campaign/` |
|---|---|
| Trial 1 training | `a2c_nobaseline/train_seed{0,1,2}/` (`eval.csv`, `tests.csv`, `best.pt`, `last.pt`) |
| Trial 1 evaluations | `a2c_nobaseline/evalfake300_{best,last}[_sampled]_seed{0,1,2}/`; leaked sampled runs: `LEAKED_evalfake300_*` |
| Trial 2 training | `a2c_no_baseline_rand50/train_seed{0,1,2}/` |
| Trial 2 evaluations | `a2c_no_baseline_rand50/evalfake300_{best,last}_{greedy,sampled}_seed{0,1,2}/` |
| Trial 2 on Questa | `a2c_no_baseline_rand50/questa_best_seed{0,1,2}/` (per-test files under `runs/questa/tests/`) |

**Scripts** (kept outside the repo; copy each into a file and run it from `fifo_rl/`). Both patch A2C's `legal_mask` at runtime, so the repo stays unchanged.

`a2c_forbid.py`: training, then fake-FIFO evaluation (greedy and sampled, `best.pt` and `last.pt`):

```bash
python a2c_forbid.py <seed> runs/campaign/a2c_nobaseline BASELINE                 # Trial 1
python a2c_forbid.py <seed> runs/campaign/a2c_no_baseline_rand50 BASELINE RAND50  # Trial 2
```

```python
import sys

sys.path.insert(0, ".")

import fifo_rl.agents.a2c as a2c  # noqa: E402
from fifo_rl.config import load_config  # noqa: E402
from fifo_rl.evaluation import GreedyWrapper  # noqa: E402
from fifo_rl.plan.actions import ACTION_INDEX  # noqa: E402
from scripts import run_campaign  # noqa: E402

seed, root, names = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
FORBIDDEN = [ACTION_INDEX[n] for n in names]
_original = a2c.legal_mask


def legal_mask_forbidden(states, n_actions, end_action=a2c.END_TEST_ID):
    mask = _original(states, n_actions, end_action)
    for idx in FORBIDDEN:
        mask[..., idx] = False
    return mask


a2c.legal_mask = legal_mask_forbidden  # used by act() and learn() through masked_logits

cfg = load_config(overrides={"reward_mode": "action", "gamma": 0.99, "seed": seed, "n_campaigns": 3000})
out = f"{root}/train_seed{seed}"
run_campaign.train(cfg, out, a2c.A2CAgent)
ecfg = {**cfg, "eval_campaigns": 300}
for ck in ("best", "last"):
    for mode in ("greedy", "sampled"):
        agent = a2c.A2CAgent.load(f"{out}/{ck}.pt")
        player = (run_campaign.SampledPlay(agent, cfg["seed"] + run_campaign.EVAL_SEED_OFFSET) if mode == "sampled"
                  else GreedyWrapper(agent))
        s = run_campaign.evaluate(ecfg, f"{root}/evalfake300_{ck}_{mode}_seed{seed}", player)
        print(f"EVAL {ck} {mode} seed {seed}: closes {100 * s['closure_rate']:.1f}%, expected tests "
              f"{s['tests_charged']['mean']:.3f}, cycles {s['cycles_per_campaign']['mean']:.1f}, "
              f"blocks/test {s['blocks_per_test']['mean']:.2f}", flush=True)
```

`a2c_forbid_eval.py`: evaluate one checkpoint **with the mask kept**, on the fake FIFO or Questa:

```bash
python a2c_forbid_eval.py runs/campaign/a2c_no_baseline_rand50/train_seed0/best.pt greedy questa 10 0 \
       runs/campaign/a2c_no_baseline_rand50/questa_best_seed0 BASELINE RAND50
```

```python
import sys

sys.path.insert(0, ".")

import fifo_rl.agents.a2c as a2c  # noqa: E402
from fifo_rl.config import load_config  # noqa: E402
from fifo_rl.evaluation import GreedyWrapper  # noqa: E402
from fifo_rl.plan.actions import ACTION_INDEX  # noqa: E402
from scripts import run_campaign  # noqa: E402

ckpt, mode, launcher, n, seed, out = sys.argv[1:7]
FORBIDDEN = [ACTION_INDEX[b] for b in sys.argv[7:]]
_original = a2c.legal_mask


def legal_mask_forbidden(states, n_actions, end_action=a2c.END_TEST_ID):
    mask = _original(states, n_actions, end_action)
    for idx in FORBIDDEN:
        mask[..., idx] = False
    return mask


a2c.legal_mask = legal_mask_forbidden

overrides = {"reward_mode": "action", "gamma": 0.99, "seed": int(seed), "eval_campaigns": int(n)}
if launcher == "questa":
    overrides["questa_bin"] = "C:/questasim64_2024.1/win64"
cfg = load_config(overrides=overrides)
agent = a2c.A2CAgent.load(ckpt)
player = (run_campaign.SampledPlay(agent, cfg["seed"] + run_campaign.EVAL_SEED_OFFSET) if mode == "sampled"
          else GreedyWrapper(agent))
s = run_campaign.evaluate(cfg, out, player, launcher)
bad = {run_campaign.ACTIONS[i].name: s["action_counts"][run_campaign.ACTIONS[i].name] for i in FORBIDDEN}
print(f"{out}: closes {100 * s['closure_rate']:.1f}%, expected tests {s['tests_charged']['mean']:.3f}, "
      f"cycles {s['cycles_per_campaign']['mean']:.1f}, wall/campaign {s['wall_time_per_campaign_s']['mean']:.2f}s, "
      f"forbidden used {bad}", flush=True)
```

Do **not** use `python -m scripts.run_campaign eval --agent a2c` on these checkpoints: it does not know which blocks were forbidden, so it can play them (§5.2 note).

To keep this as a real feature later, add a config key (for example `a2c_forbid: [BASELINE, RAND50]`) that `legal_mask` reads at training and evaluation, instead of patching at runtime.
