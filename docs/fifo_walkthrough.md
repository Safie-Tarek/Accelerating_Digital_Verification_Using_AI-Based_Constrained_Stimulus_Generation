# FIFO Testbench Walkthrough — for the AI Team

**Who this is for:** AI-team members who will build the DQN agent and have little or no UVM / SystemVerilog background.
**What it covers:** how the FIFO UVM testbench in `src codes/` works, what the verification plan (`FIFO Plan.xlsx`) says, what we measured when we ran it in Questa, and a step-by-step plan for building the agent.
**Companion file:** [`fifo_agent_context.md`](fifo_agent_context.md) — the dense technical reference (all numbers, commands, bin list). Read this walkthrough first, then use that file as a cheat sheet.

Source of the testbench: https://github.com/SeiffAhmed22/Synchronous-FIFO-UVM-Verification (copied into `src codes/`; the AI team does not modify it).

---

## Contents
1. [UVM in 15 minutes (with code)](#1-uvm-in-15-minutes-with-code)
2. [The testbench, file by file](#2-the-testbench-file-by-file)
3. [The verification plan (`FIFO Plan.xlsx`)](#3-the-verification-plan-fifo-planxlsx)
4. [Running it: what we measured](#4-running-it-what-we-measured)
5. [The plan for building the agent](#5-the-plan-for-building-the-agent)

---

## 1. UVM in 15 minutes (with code)

### The big idea
We build a fake world around a chip, push inputs into it, watch its outputs, check they are correct, and **tick boxes on a checklist of situations we have seen**. That checklist ("functional coverage") is what our RL agent cares about: its job is to choose inputs that tick the empty boxes quickly.

- **SystemVerilog (SV)** describes hardware *and* has Python-like classes for writing tests.
- **UVM** is a standard class library in SV, so every testbench has the same parts.

SV classes look a lot like Python:
```systemverilog
class Dog;                 // class Dog:
  int age;                 //     self.age
  function new();          //     def __init__(self):
    age = 0;
  endfunction
  function void bark();    //     def bark(self):
    $display("woof");      //         print("woof")
  endfunction
endclass
Dog d = new();             // d = Dog()
```

### The parts

**1. DUT (Design Under Test)** — the chip being checked. *Analogy: the engine on the test stand.*
```systemverilog
module counter (input logic clk, rst_n, output logic [3:0] q);
  always @(posedge clk)            // every rising clock edge
    if (!rst_n) q <= 0;            // reset → 0
    else        q <= q + 1;        // otherwise count up
endmodule
```

**2. Interface** — a named bundle of wires. *Analogy: one cable harness instead of 12 loose wires.*
```systemverilog
interface my_if (input bit clk);
  logic        wr_en, rd_en;
  logic [15:0] data_in;
endinterface
```

**3. Virtual interface (vif)** — a *pointer* to the wires, because classes can't hold real wires. *Analogy: a remote control for the harness.*
```systemverilog
class my_driver;
  virtual my_if vif;
  task write_one(); vif.wr_en = 1; endtask
endclass
```

**4. Transaction / sequence item** — one clock cycle of input. *Analogy: one instruction card.*
```systemverilog
class my_item extends uvm_sequence_item;
  rand bit        rst_n, wr_en, rd_en;   // "rand" = simulator picks randomly
  rand bit [15:0] data_in;
  bit             full, empty;           // outputs, filled in later by the monitor
endclass
```

**5. Constraint + randomize** — loaded dice. *Python: `random.choices([1, 0], weights=[70, 30])`.*
```systemverilog
constraint c_wr { wr_en dist {1 :/ 70, 0 :/ 30}; }   // wr_en = 1 about 70% of the time
assert(item.randomize());                         // roll the dice
assert(item.randomize() with { wr_en == 1; });    // roll, but force wr_en = 1
```

**6. Sequence** — a script that produces many items. *Python: a generator that `yield`s items.*
```systemverilog
class write_10_seq extends uvm_sequence #(my_item);
  task body();
    repeat (10) begin
      my_item it = my_item::type_id::create("it");
      start_item(it);
      assert(it.randomize() with { wr_en == 1; rd_en == 0; });
      finish_item(it);
    end
  endtask
endclass
```

**7. Sequencer** — hand-off point between sequence and driver. *Python: a `queue.Queue`.*
```systemverilog
class my_sqr extends uvm_sequencer #(my_item); endclass   // usually empty
```

**8. Driver** — puts items onto the wires. *Analogy: the operator pressing the pedals.*
```systemverilog
forever begin
  seq_item_port.get_next_item(it);   // take next card
  vif.wr_en = it.wr_en;              // drive the wires
  @(negedge vif.clk);                // wait one clock
  seq_item_port.item_done();         // ask for the next card
end
```

**9. Monitor** — watches the wires and records what happened (never drives). *Analogy: a camera on the dashboard.*
```systemverilog
forever begin
  @(negedge vif.clk);
  it = my_item::type_id::create("it");
  it.wr_en = vif.wr_en;  it.full = vif.full;   // copy inputs and outputs
  ap.write(it);                                 // broadcast
end
```

**10. Analysis port** — one sender, many listeners. *Python: publish/subscribe.*
```systemverilog
mon.ap.connect(sb.export);    // listener 1: scoreboard
mon.ap.connect(cov.export);   // listener 2: coverage
```

**11. Agent** — box holding sequencer + driver + monitor.

**12. Scoreboard** — compares the chip's answers with a **golden/reference model** (the same behaviour written as simple software). *Analogy: the examiner with the answer key.*
```systemverilog
if (it.rd_en && q.size() > 0) expected = q.pop_front();
if (it.wr_en && q.size() < 8) q.push_back(it.data_in);
if (it.data_out !== expected) `uvm_error("SB", "wrong data!")
```

**13. Coverage (the important one for RL)** — the checklist.
- *covergroup* = the whole checklist, *coverpoint* = watches one signal, *bin* = one box (a 1-bit signal has bins "saw 0" and "saw 1"), *cross* = boxes for combinations, *ignore_bins* = impossible boxes removed. Coverage % = ticked / total.
```systemverilog
covergroup cg;
  cp_rd:  coverpoint it.rd_en;         // 2 boxes
  cp_udf: coverpoint it.underflow;     // 2 boxes
  x_rd_udf: cross cp_rd, cp_udf {      // 4 combination boxes...
    ignore_bins impossible = binsof(cp_rd) intersect {0} && binsof(cp_udf) intersect {1};
  }                                    // ...minus 1 impossible = 3
endgroup
cg.sample();   // tick the boxes matching current values
```
**For RL:** boxes = **state**; newly ticked boxes = **reward**.

**14. Environment (env)** — box holding agent + scoreboard + coverage, and wiring them together.

**15. Test** — decides which sequences run and in which order. **This is where the RL agent will plug in.**
```systemverilog
task run_phase(uvm_phase phase);
  phase.raise_objection(this);       // "keep simulating"
  reset_seq.start(env.agt.sqr);
  write_seq.start(env.agt.sqr);
  phase.drop_objection(this);        // "done, you may stop"
endtask
```

**16. Objection** — while anyone has one raised, the simulation keeps running.

**17. Phases** — fixed stages UVM calls on every part, in order: `build_phase` (create children) → `connect_phase` (plug together) → `run_phase` (simulate; time passes) → `report_phase` (print results).

**18. Factory** — `type_id::create` instead of `new`, so a test can swap classes later without editing code.
```systemverilog
`uvm_component_utils(my_driver)                  // register with the factory
drv = my_driver::type_id::create("drv", this);   // instead of new(...)
```

**19. uvm_config_db** — a global noticeboard. *Python: a global dict.*
```systemverilog
uvm_config_db#(virtual my_if)::set(null, "uvm_test_top", "MY_IF", intf);  // top pins it
uvm_config_db#(virtual my_if)::get(this, "", "MY_IF", vif);               // test takes it
```

**20. Top module** — makes the clock, wires and chip, then starts the test.
```systemverilog
module top;
  bit clk;  always #5 clk = ~clk;   // one cycle = 10 ns
  my_if intf (clk);
  FIFO  dut  (intf);
  initial run_test("my_test");
endmodule
```

**21. Assertions (SVA) and `bind`** — rules checked automatically every cycle; `cover property` counts how often something happened; `bind` attaches the checker without editing the chip.
```systemverilog
assert property (@(posedge clk) (count == 8) |-> full) else $error("full missing!");
bind FIFO FIFO_SVA chk (FIFOif);
```

### The whole testbench in one picture
```
 TEST: reset → write-only → read-only → random
   │
   ▼
 SEQUENCE ──items──► SEQUENCER ──items──► DRIVER ──wires──► INTERFACE ◄──► FIFO (DUT)
                                                               │            ▲ assertions bound on it
                                                     MONITOR (every clock)
                                                               │ broadcast
                                                ┌──────────────┴──────────────┐
                                                ▼                             ▼
                                   SCOREBOARD (golden model)      COVERAGE (tick boxes) → .ucdb file
```

### This testbench in plain words
- The chip is a queue holding **8 numbers of 16 bits** (First-In-First-Out, like a bakery line).
- The only test always does: reset → 100 writes → 100 reads → 9,999 random cycles (≈70% write, 30% read). About 10% of random cycles also reset the chip.
- Checking: scoreboard checks `data_out`; assertions check the counter, pointers and flags; the covergroup ticks combinations of write/read and flags.

---

## 2. The testbench, file by file

Reading order: simplest and most fundamental first. Each file ends with **For the RL agent**.

### 2.1 `FIFO_if.sv` — the wires
A bundle of all wires between the testbench and the chip, with sizes and directions.
```systemverilog
interface FIFO_if (input bit clk);
  parameter FIFO_WIDTH = 16;   parameter FIFO_DEPTH = 8;
  logic [FIFO_WIDTH-1:0] data_in;   logic rst_n, wr_en, rd_en;          // inputs
  logic [FIFO_WIDTH-1:0] data_out;  logic wr_ack, overflow;             // outputs
  logic full, empty, almostfull, almostempty, underflow;
  modport DUT(input clk, data_in, rst_n, wr_en, rd_en,
              output data_out, wr_ack, overflow, full, empty, almostfull, almostempty, underflow);
endinterface
```
| Wire | Meaning |
|---|---|
| `rst_n` | Reset, **active low** (0 = reset) |
| `wr_en` / `rd_en` | Write / read this cycle |
| `data_in` / `data_out` | Number in / number out |
| `full` / `empty` | 8 stored / 0 stored |
| `almostfull` / `almostempty` | 7 stored / 1 stored |
| `wr_ack` | Write accepted |
| `overflow` | Write attempted while full (dropped) |
| `underflow` | Read attempted while empty |

**For the RL agent:** only 4 inputs exist (`rst_n`, `wr_en`, `rd_en`, `data_in`), and `data_in` is in no coverage → the agent only decides *when* to write / read / reset. The 7 flags are what coverage watches. DEPTH = 8 is small. Nothing here needs changing.

### 2.2 `FIFO.sv` — the chip (DUT)
Stores numbers in an 8-slot ring buffer (`mem`, `wr_ptr`, `rd_ptr`) and keeps a `count` of stored items. All flags are computed from `count`.

New Verilog: `always @(posedge clk or negedge rst_n)` = do this every clock tick, or immediately on reset; `<=` = all registers update together at the tick; `assign` = a wire that is always recomputed (like a Python `@property`).

The counter rules (heart of the FIFO):
| wr_en | rd_en | FIFO is… | What happens | count |
|---|---|---|---|---|
| 1 | 0 | not full | write | +1 |
| 1 | 0 | full | nothing (**overflow**) | same |
| 0 | 1 | not empty | read | −1 |
| 0 | 1 | empty | nothing (**underflow**) | same |
| 1 | 1 | middle | write and read | same |
| 1 | 1 | empty | only write works | +1 |
| 1 | 1 | full | only read works | −1 |
| 0 | 0 | any | nothing | same |

Flags (instant, no clock):
```systemverilog
assign full = (count == 8);         assign empty = (count == 0);
assign almostfull = (count == 7);   assign almostempty = (count == 1);
assign wr_ack = (count != 8) && wr_en;
assign overflow = full && wr_en;    assign underflow = empty && rd_en;
```
Lines marked `// BUG HERE` are planted course bugs, commented out; the fixed versions are at the bottom — we use the **corrected** design.

**For the RL agent:** `count` is the hidden variable; every coverage box = "count at some level" × "some wr/rd combo". These ~20 lines are all we need for a **Python fake FIFO**. Resets (10% in random stimulus) drop `count` to 0 and make "full" harder to reach. Quirk: during a reset cycle `count = 0`, so `underflow` = `rd_en` and `wr_ack` = `wr_en` — some boxes can be ticked only during reset.

### 2.3 `FIFO_seq_item.sv` — one cycle of stimulus + dice rules
```systemverilog
rand bit [15:0] data_in;  rand bit rst_n, wr_en, rd_en;   // random inputs
bit full, empty, ...;                                     // outputs, filled by monitor
constraint reset_con { rst_n dist {0 :/ 10, 1 :/ 90}; }                           // reset 10%
constraint wr_en_con { wr_en dist {1 :/ WR_EN_ON_DIST, 0 :/ (100 - WR_EN_ON_DIST)}; } // default 70
constraint rd_en_con { rd_en dist {1 :/ RD_EN_ON_DIST, 0 :/ (100 - RD_EN_ON_DIST)}; } // default 30
```
**For the RL agent:** the knobs are write % and read % (variables), and reset % (**hardcoded** 10 — needs to become a variable). These are natural action parameters.

### 2.4 `FIFO_sequence.sv` — ready-made traffic patterns
| Sequence | Cycles | Write % | Read % | Reset % |
|---|---|---|---|---|
| `FIFO_reset_sequence` | 1 | 0 | 0 | 100 (directed) |
| `FIFO_write_only_sequence` | 100 | 100 | 0 | 10 |
| `FIFO_read_only_sequence` | 100 | 0 | 100 | 10 |
| `FIFO_write_read_sequence` | 9999 | 70 | 30 | 10 |

**For the RL agent:** sequences are natural actions, but lengths are hardcoded and even "write-only" resets 10% of the time.

### 2.5 `FIFO_sequencer.sv` — boilerplate hand-off point. Nothing for us.

### 2.6 `FIFO_config_obj.sv` — config object that carries the vif pointer
Today it only holds `virtual FIFO_if FIFO_config_vif`. **For the RL agent:** natural place to add knob values (write %, read %, reset %, length) read from the command line.

### 2.7 `FIFO_driver.sv` — items → wires
Takes one item, sets `rst_n, data_in, wr_en, rd_en`, waits for the falling clock edge. **1 item = 1 clock cycle.** (It fetches the config object but its vif is actually assigned by the agent.)

### 2.8 `FIFO_monitor.sv` — wires → items
Every falling clock edge it copies all inputs and outputs into an item and broadcasts it to the scoreboard and coverage. **For the RL agent:** coverage sees (this cycle's wr_en/rd_en) together with (flags after this cycle's update). Our fake FIFO must sample the same way (see the model in `fifo_agent_context.md` §1).

### 2.9 `FIFO_agent.sv` — creates sequencer, driver, monitor; connects driver↔sequencer and gives both the vif. Wiring only.

### 2.10 `FIFO_coverage.sv` — the checklist (**state + reward**)
Covergroup `cg_FIFO`, sampled once per clock:
- 9 coverpoints, one per signal (`wr_en, rd_en, wr_ack, overflow, full, empty, almostfull, almostempty, underflow`) → 2 bins each = **18 bins**
- 7 crosses `wr_en × rd_en × flag`, minus impossible combos → **46 bins**
- **Total 64 bins.**

| Cross | Bins | Hard / special boxes |
|---|---|---|
| cx_wr_ack | 5 | (wr=1, rd=0, ack=0) needs a write that ends at full |
| cx_overflow | 5 | (1,0,1) needs write while full |
| cx_full | 6 | (0,0,1) needs idle while full |
| cx_empty | 8 | (1,0,1) and (1,1,1) only reachable **during a reset cycle** |
| cx_almostfull | 8 | needs count 7 with every wr/rd combo; (0,0,1) idle at 7 is usually the last box |
| cx_almostempty | 8 | needs count 1 with every wr/rd combo |
| cx_underflow | 6 | (1,1,1) only reachable **during a reset cycle** |

**For the RL agent:** the 64 bins are the state vector; newly hit bins are the reward.

### 2.11 `FIFO_scoreboard.sv` — golden model
Task `ref_model()` re-implements the FIFO (memory, pointers, count) in software; `check_data()` compares the predicted `data_out` with the chip's. It checks **only `data_out`** — the flags are checked by assertions. **For the RL agent:** not used for reward, but it's a ready reference for the Python fake FIFO.

### 2.12 `FIFO_env.sv` — creates agent + scoreboard + coverage and connects the monitor to both. Wiring only.

### 2.13 `FIFO_test.sv` — the only test
Fixed order: reset → write-only (100) → read-only (100) → write/read (9999) ≈ 10,200 cycles. No command-line options. **For the RL agent:** this is where agent choices must plug in — the verification team needs to add a configurable RL test (see §4).

### 2.14 `FIFO_top.sv` — clock (10 ns period), interface, DUT, `bind` of assertions, `run_test("FIFO_test")`. `+UVM_TESTNAME=<name>` on the command line can select a different test without editing this file.

### 2.15 `FIFO_SVA.sv` — assertions bound to the DUT
Checks counter/pointer rules every cycle and the flags whenever their condition holds; also has 17 cover directives. An extra coverage source, separate from `cg_FIFO`.

### 2.16 `src_files.list` + `run.do` — compile list and Questa script
```
vlib work
vlog -f src_files.list +cover -covercells
vsim -voptargs=+acc work.FIFO_top -classdebug -uvmcontrol=all -cover
coverage save FIFO.ucdb -onexit -du FIFO     ← drops our covergroup (see §4)
add wave /FIFO_top/FIFOif/*
run -all
```

---

## 3. The verification plan (`FIFO Plan.xlsx`)

One sheet, four rows — one per test phase.

| Label | Feature | Stimulus (plan) | Where in code | Coverage (plan) | Implemented? |
|---|---|---|---|---|---|
| FIFO_1 | Reset clears pointers, counter, overflow, underflow | Directed reset at start, then random reset mostly off | `FIFO_reset_sequence` + 10% `reset_con` | "cover the reset signal" | **No** — no `rst_n` coverpoint in `cg_FIFO` (only SVA reset covers) |
| FIFO_2 | Write-only | Writes only | `FIFO_write_only_sequence` | wr_en × rd_en × each flag | Yes |
| FIFO_3 | Read-only | Reads only | `FIFO_read_only_sequence` | same | Yes |
| FIFO_4 | Constrained random | 9999 cycles, reset off 90%, wr 70%, rd 30% | `FIFO_write_read_sequence` | same | Yes |

Note: the plan says "the scoreboard checks internal signals and counter"; in the code the scoreboard checks only `data_out`, and the counter/pointers are checked by assertions.

**For the RL agent:** FIFO_4 is the constrained-random **baseline**; `cg_FIFO` is the coverage target.

---

## 4. Running it: what we measured

All runs were done on 2026-10-12 from copies in a separate folder (never inside `src codes/`).

### Where Questa is
- **Windows install**: `C:\questasim64_2024.1\win64` (Questa Sim-64 2024.1, UVM 1.1d).
- There is **no Linux Questa inside WSL**, but Python in WSL can call the Windows `vsim.exe` / `vcover.exe` (they're on the WSL PATH) — tested from a Python script, from both `/mnt/d/...` folders and Linux-only folders. Keep run folders on `/mnt/d` (where the repo is), and pass file paths as relative paths or convert them with `wslpath -w`.

### Tested commands
```
# compile once (from any run folder; -F makes paths relative to the list file)
vlib work
vlog -F "<repo>/src codes/src_files.list" +cover -covercells

# one simulation run with seed 7
vsim -c -sv_seed 7 work.FIFO_top -coverage -l run7.log -do "coverage save -onexit run7.ucdb; run -all; quit -f"

# coverage as text the agent can parse
vcover report -cvg -details -nocompactcrossbins run7.ucdb > run7_cov.txt
```
Sample of the text report:
```
    Cross cx_almostfull                                87.50%        100    Uncovered
            bin <auto[1],auto[1],auto[1]>                   1          1    Covered
            bin <auto[0],auto[0],auto[1]>                   0          1    ZERO
            Covergroup Bins         64        63         1    98.43%
```
Without `-nocompactcrossbins`, missing cross bins are merged into wildcard rows (`<auto[1],auto[1],*>`), which is hard to parse.

### Findings
| Finding | Consequence |
|---|---|
| Original `run.do` passes: 10,200 transactions, 0 errors | Testbench is healthy |
| `coverage save -du FIFO` **drops `cg_FIFO`** from the UCDB | Must save without `-du FIFO` |
| No `-sv_seed` → **every run identical** | Always pass a seed |
| ≈ **3 s per run**, almost all start-up (100 cycles 2.9 s, 10,200 cycles 3.3 s); compile 0.7 s | Training must happen mostly in a Python fake env |
| Default test reaches **64/64 bins by 400–500 cycles** (seeds 1–5; 200 directed + 200–300 random cycles) | See "biggest risk" below |
| Ignore bins are never hit | Confirms our model of how the monitor samples |

### How the agent can pass stimulus choices in — today
Only **seed** and **run length** (`run <N>us`) can be changed without editing the testbench; neither steers the stimulus. So the real-simulator loop needs these from the verification team:
1. Command-line knobs (`+WR_PCT`, `+RD_PCT`, `+RST_PCT`, `+NUM_CYCLES` or `+SEQ=<name>`) read with `$value$plusargs` and passed via `FIFO_config_obj`.
2. Reset probability as a variable.
3. A configurable RL test (selected with `+UVM_TESTNAME`) that always starts with a reset and then runs only the scenario the agent chose.
4. A batch do-file: seed argument, per-run UCDB name, `quit -f`, no `add wave`, coverage save without `-du FIFO`.
5. Optional: a `rst_n` coverpoint (plan item FIFO_1).
6. Discussion of a richer coverage model (next point).

### ⚠ Biggest risk
`cg_FIFO` is too easy: one ~3 s random run closes all 64 bins. Because a run costs ~3 s whatever its length, making the agent's runs short is not a fair win — random could just run longer for the same cost. **On this covergroup, RL cannot fairly beat random.** Options (to be decided with supervisors):
- **(a)** a richer coverage model from the verification team (e.g. every fill level 0..8 × wr × rd, transitions such as full → empty, back-to-back overflow, reset at each fill level);
- **(b)** treat the FIFO as a **pipeline pilot** — prove the loop, parser, fake env and DQN end-to-end here, and show coverage-closure gains on a harder DUT later.

---

## 5. The plan for building the agent

### The idea in simple words
Think of it as a **game**:
- The **board** is the checklist of 64 boxes, all empty at the start.
- Each **turn**, the agent picks one **scenario** to simulate (e.g. "fill the FIFO to full, then keep writing").
- The simulator runs it and ticks some boxes. The **points** for that turn = how many *new* boxes got ticked (minus a small cost per turn, so the agent hurries).
- The game ends when all boxes are ticked or the **budget** of turns runs out.
- A good agent finishes the board in fewer turns than picking scenarios at random.

| RL word | In our project |
|---|---|
| Environment | The FIFO simulator (fake Python one first, real Questa later) |
| State (what the agent sees) | 64 ticked/empty boxes + which scenario it chose last + turns left → 87 numbers |
| Action (what it chooses) | One of ~22 scenarios |
| Reward | Number of newly ticked boxes − 0.1 |
| Episode | One full game (until 64/64 or budget used) |

**Important:** each simulation starts with an empty FIFO (it begins with a reset), so a scenario must first **reach** the interesting fill level and then **do** something there. That's why actions look like "reset → fill to level L → pattern P for K cycles":
- L ∈ {0 (empty), 1, 7, 8 (full)}
- P ∈ {idle, write-only, read-only, write+read, random 50/50}
- → 20 scenarios, plus "reset with traffic" (only way to tick the reset-only boxes) and "plain random 70/30/10" (the plan's FIFO_4 recipe) = **22 actions**.

Examples: (full, idle) ticks "idle while full"; (full, write-only) ticks overflow; (empty, write+read) ticks "write+read at almost-empty".

Because nothing carries over between runs except which boxes are ticked, this is close to a *contextual bandit*; DQN still works with a smallish discount (γ ≈ 0.9).

### DQN in simple words
- A small neural network looks at the board and guesses, for each action, "how many points will I get from here if I pick this?" (the **Q-value**). The agent picks the action with the highest guess.
- **Epsilon-greedy:** sometimes pick a random action anyway, to discover new things. Start very random (ε = 1.0), slowly become greedy (ε = 0.05).
- **Replay buffer:** a memory of past turns (state, action, reward, next state). Training samples random batches from it, so learning is stable.
- **Target network:** a slowly-updated copy of the network used to compute training targets, so the network isn't chasing its own tail.
- **DDQN (Double DQN):** plain DQN tends to be over-optimistic. DDQN lets the main network *choose* the best next action and the target network *score* it. One-line change.

### Step-by-step plan
| # | Step | What you get | Needs Questa? |
|---|---|---|---|
| 0 | Agent runs in **WSL Ubuntu** (decided; install `python3-pip python3-venv` once). Decide FIFO as pilot (b) or richer coverage (a). Send the 6 requests to the verification team | Clear direction; testbench work starts in parallel | No |
| 1 | Create the project folder, virtual env, `requirements.txt` — **done 2026-10-13** | Empty skeleton that installs | No |
| 2 | `coverage_model.py`: list the 64 bins in a fixed order, names matching the Questa report — **done 2026-10-13**, together with `state.py` (the state encoder: 64 bin flags + budget left + last action = 87 numbers) | One shared definition of the state vector | No |
| 3 | **Fake FIFO env** in Python: the counter rules + flag rules + sampling, ticking the 64 bins | Simulates thousands of runs per second | No |
| 3b | **Exact check vs Questa:** run Questa once with `+UVM_VERBOSITY=UVM_HIGH` (the monitor then prints every cycle's inputs and flags), replay those exact inputs through the fake FIFO, and require the same flags every cycle and the same bin hit counts (seed 1: `cp_wr_en` 3171/7029, `cx_full <0,0,1>` = 163) | Proof the fake matches the real thing | One run |
| 4 | `actions.py`: turn each of the 22 scenarios into a list of (rst_n, wr_en, rd_en) per cycle, always starting with reset | The action space | No |
| 5 | Env interface: `reset()` → state; `step(action)` → (state, reward, done, info) | Gym-style game | No |
| 6 | **Random baselines**: (i) random scenario each turn, (ii) plain 70/30/10 random runs. Plot coverage vs number of runs, averaged over ≥ 20 seeds | The bar the agent must beat | No |
| 7 | **DQN**: MLP 87 → 128 → 128 → 22, replay buffer 10k, batch 64, Adam lr 1e-3, γ ≈ 0.9, ε 1.0 → 0.05, target sync every ~200 steps, Huber loss | First learning agent | No |
| 8 | **DDQN** upgrade | More stable Q-values | No |
| 9 | Compare agent vs random on the fake env: runs-to-100%, area under the coverage curve | First results + plots | No |
| 10 | Questa runner + report parser (`subprocess` → `vsim`/`vcover` → 64-bin vector). Testable now using `run <N>us` and saved reports — **parser done 2026-10-13** (`questa/report_parser.py`, tested on 4 real reports); the runner is still to do | Bridge to the real simulator | Yes |
| 11 | `questa_env.py` once the RL test + plusargs exist; check fake vs real bins for each action | Real file-based loop | Yes (+ verification team) |
| 12 | Final comparison on real Questa (runs and wall-clock time), logs and plots | Project results | Yes |

### Logging and plots
- CSV per step: episode, step, action, new_bins, total_bins, epsilon, loss.
- Plots: coverage vs runs (agent vs random, mean ± std over seeds), learning curve (episode return), action histogram (which scenarios the agent prefers).

### Proposed folder structure (exists so far: `README.md`, `requirements.txt`, `pyproject.toml`, `fifo_rl/coverage_model.py`, `fifo_rl/state.py`, `fifo_rl/questa/report_parser.py`, `tests/test_coverage_model.py`, `tests/test_state.py`, `tests/test_report_parser.py`, `tests/fixtures/`)
```
fifo_rl/
  requirements.txt
  README.md
  configs/default.yaml          # actions, budget, hyper-parameters
  fifo_rl/
    coverage_model.py           # the 64 bins (fixed order)
    actions.py                  # 22 scenarios → per-cycle stimulus
    envs/
      base.py                   # reset() / step() interface
      fake_fifo_env.py          # Python FIFO + coverage
      questa_env.py             # real simulator via files
    questa/
      runner.py                 # vlog / vsim / vcover calls
      report_parser.py          # vcover text → 64-bin vector
    agents/
      random_agent.py           # baseline
      dqn.py                    # DQN + DDQN, replay buffer, target net, ε-greedy
    utils/logger.py
  scripts/
    train.py
    evaluate.py                 # agent vs random
    plot.py
  tests/
    test_fake_fifo.py           # fake FIFO follows the DUT rules
    test_parser.py              # parse a saved Questa report
  runs/                         # git-ignored: sim folders, logs, checkpoints
```

### Proposed `requirements.txt` (today the file has only `numpy` and `pytest`; the rest is added when it is first needed)
```
torch>=2.2        # CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu
numpy>=1.26
matplotlib>=3.8
pandas>=2.1
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
```
Works with Python 3.10 (WSL) and 3.12 (Windows).
