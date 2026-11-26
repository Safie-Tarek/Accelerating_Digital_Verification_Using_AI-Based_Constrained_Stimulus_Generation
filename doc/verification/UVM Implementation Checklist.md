# UVM Implementation Checklist

**Date:** 2026-11-26  

This checklist breaks the verification architecture into assignable work items. Each item has a clear owner boundary, expected files, and a completion check. Agree on the shared transaction and configuration contracts in item 2 before parallel UVM component work begins.

## Design baseline

- DUT: `rtl/idma_wrapper.sv`, using the 1D APB register frontend and AXI backend.
- APB: active master, 32-bit register data path, one register interface and stream.
- AXI: active slave, 64-bit address, 32-bit data, 1-bit ID, 1-bit user.
- Transfer length: use the lower 32 bits; program the upper register word as zero.
- Read `next_id[0]` at `0x044` exactly once to launch a transfer.
- Key APB offsets: status `0x004`, done ID `0x084`, destination `0x0D0/0x0D4`, source `0x0D8/0x0DC`, length `0x0E0/0x0E4`.
- Selected backend rejects zero-length transfers; software-visible AXI error handling is disabled.

## Work items

### 1. Wrapper smoke test - do this first

**Owner:** ____________________

- [ ] Add a non-UVM testbench top under `tb/top/` that instantiates `idma_wrapper`.
- [ ] Provide APB register read/write tasks and a simple AXI memory responder.
- [ ] Initialize source memory, program one aligned transfer, read `next_id` once, wait for completion, and compare destination bytes.
- [ ] Add the testbench and all required elaboration sources to `sim/compile.f`.
- [ ] Run it in Questa and fail the simulation on timeout, bad completion, or data mismatch.
- [ ] **Done when:** Questa elaborates the complete wrapper hierarchy and the transfer passes its data check.

### 2. Shared interfaces and contracts

**Owner:** ____________________

- [ ] Own `tb/interfaces/apb_if.sv`, `tb/interfaces/axi_if.sv`, and the shared verification package/configuration definitions.
- [ ] Define clock/reset conventions, widths, common address constants, transfer configuration fields, and analysis transaction fields.
- [ ] Define the memory model API and the scoreboard/reference-model handoff before those components are implemented.
- [ ] Document reset polarity, APB launch side effect, completion polling behavior, and zero-length policy.
- [ ] **Done when:** Other owners can implement against these declarations without editing shared interface or package files.

### 3. APB master agent

**Owner:** ____________________

- [ ] Own files under `tb/agents/apb_agent/`.
- [ ] Implement APB sequence item, sequencer, driver, monitor, and agent configuration.
- [ ] Support register reads/writes, wait states, reset handling, and observed-transaction analysis output.
- [ ] Ensure the transfer-launch read is issued once per transfer sequence.
- [ ] Add an agent-level test using a small APB responder or the wrapper smoke test.
- [ ] **Done when:** Driver timing follows APB handshakes and monitor output matches observed bus activity.

### 4. AXI slave agent

**Owner:** ____________________

- [ ] Own files under `tb/agents/axi_agent/`, excluding the separately owned memory model.
- [ ] Implement AXI slave request/response items, sequencer, driver, monitor, and agent configuration.
- [ ] Support read data and write responses, configurable latency, and ready backpressure.
- [ ] Publish observed read/write transactions through analysis ports.
- [ ] **Done when:** The agent handles legal single and burst transactions at the selected interface widths and reports the observed transactions.

### 5. AXI memory model

**Owner:** ____________________

- [ ] Own files under `tb/memory_model/` and the memory API defined in item 2.
- [ ] Implement byte-addressed storage, initialization/read APIs, and AXI read/write effects.
- [ ] Apply write strobes correctly and define behavior for unmapped/out-of-range accesses.
- [ ] Keep the expected-memory representation separate from the AXI service memory used by the DUT.
- [ ] **Done when:** Directed memory tests show reads, writes, strobes, and initialization behave as specified.

### 6. Independent reference model

**Owner:** ____________________

- [ ] Own files under `tb/reference_model/`.
- [ ] Model a 1D memory-to-memory copy from transfer configuration and initial source bytes.
- [ ] Handle transfer lengths and byte-level alignment independently of the AXI agent implementation.
- [ ] Publish expected destination updates through the contract from item 2.
- [ ] **Done when:** Directed model checks cover aligned, unaligned, short, and multi-burst transfer examples without using DUT outputs.

### 7. Scoreboard

**Owner:** ____________________

- [ ] Own `tb/env/idma_scoreboard.sv` and its private helper files.
- [ ] Subscribe to APB and AXI monitor outputs and expected results from the reference model.
- [ ] Check programmed addresses/length, AXI read/write behavior, destination data, completion ID, and unexpected or missing activity.
- [ ] Do not use the AXI service memory as the sole golden reference.
- [ ] **Done when:** Injected mismatches cause a clear UVM error/failure and correct transfers pass.

### 8. Functional coverage

**Owner:** ____________________

- [ ] Own `tb/env/idma_coverage.sv` and coverage-only helper files.
- [ ] Subscribe to observed APB/AXI transactions, not sequence intent.
- [ ] Cover transfer length, address alignment, source/destination alignment cross, burst lengths/splitting, response latency, backpressure, completion, and reset scenarios.
- [ ] Keep coverage collection independent of scoreboard decisions.
- [ ] **Done when:** A directed run samples the expected bins and produces a readable coverage report.

### 9. Protocol and reset assertions

**Owner:** ____________________

- [ ] Own files under `tb/assertions/`.
- [ ] Add APB and AXI handshake/protocol properties, reset behavior, and relevant completion/busy checks.
- [ ] Bind or instantiate assertions without changing agent or scoreboard implementation files.
- [ ] **Done when:** A passing smoke run has no assertion failures and a deliberate protocol violation triggers the expected failure.

### 10. Environment and virtual sequencer

**Owner:** ____________________

- [ ] Own `tb/env/idma_env.sv` and `tb/env/idma_virtual_sequencer.sv`.
- [ ] Instantiate the APB/AXI agents, scoreboard, coverage, and memory model using the shared configuration API.
- [ ] Connect monitor analysis ports and virtual sequencer handles; do not modify the internals of owned agents.
- [ ] Support active APB master and active AXI slave operation for the baseline configuration.
- [ ] **Done when:** The environment builds and reports all components and connections through UVM configuration/phase checks.

### 11. Reusable sequences and tests

**Owner:** ____________________

- [ ] Own files under `tb/sequences/` and `tb/tests/`.
- [ ] Add APB register access, memory initialization, transfer programming, and completion-wait sequences.
- [ ] Add a base test and independent tests for register behavior, basic transfer, alignment/length boundaries, burst splitting, backpressure, consecutive requests, zero-length rejection, and reset.
- [ ] Use the public agent sequence APIs; do not drive interface signals directly from tests.
- [ ] **Done when:** Each test has a clear pass condition and can run by name and seed.

### 12. Top-level integration and regression flow

**Owner:** ____________________

- [ ] Own `tb/top/tb_top.sv` and simulation setup/reporting files under `sim/`.
- [ ] Instantiate interfaces and DUT, generate clock/reset, set virtual interfaces/configuration, and call `run_test()`.
- [ ] Add UVM sources and the selected top to `sim/compile.f` in dependency order.
- [ ] Add commands to compile, run one named test/seed, and run the baseline regression; preserve logs and coverage outputs in ignored result directories.
- [ ] **Done when:** A clean checkout can compile, run the smoke-equivalent UVM transfer test, and report pass/fail, seed, assertions, scoreboard results, and coverage.

### 13. RL coverage interface — after baseline UVM

**Owner:** ____________________

- [ ] Define a stable read-only interface for coverage/state feedback and constrained-random parameter selection.
- [ ] Keep the conventional UVM regression runnable without the RL process.
- [ ] **Done when:** The RL loop can consume coverage and submit a new stimulus configuration without changing the DUT, reference model, or scoreboard.

## Parallel assignment guide

Complete item 1 first. Agree on item 2's public API, then items 3–9 can be assigned independently because each owns separate files and connects through the documented interfaces. Items 10–12 integrate those deliverables. Item 13 is intentionally deferred until the baseline environment and coverage model are operational.
