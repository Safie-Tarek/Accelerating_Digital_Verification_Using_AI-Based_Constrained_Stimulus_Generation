# iDMA UVM Verification Environment Architecture

**Project:** UVM Verification Aided by Reinforcement Learning for Functional Coverage Closure  
**DUT:** PULP iDMA — 1D Register Frontend with APB4 and AXI4 Backend  
**Document Version:** 0.2  
**Status:** Architecture Baseline

---

# Revision History

|Version|Date|Description|
|---|---|---|
|0.1|2026-11-18|Initial architecture draft|
|0.2|2026-11-24|Updated architecture based on the selected iDMA RTL configuration and integration wrapper|

# Referenced Documentation

1. PULP Platform — [iDMA repository](https://github.com/pulp-platform/iDMA).

2. SystemVerilog Language Reference Manual.

3. Universal Verification Methodology (UVM) Class Reference.

4. AMBA AXI4 Protocol Specification.

5. AMBA APB Protocol Specification.

# Terminology and Definitions

|Term|Definition|
|---|---|
|DUT|Design Under Test|
|UVM|Universal Verification Methodology|
|APB|Advanced Peripheral Bus|
|AXI4|Advanced eXtensible Interface 4|
|CRV|Constrained-Random Verification|
|TLM|Transaction-Level Modeling|
|Agent|UVM component responsible for driving or monitoring an interface|
|Sequencer|UVM component that provides transactions to a driver|
|Driver|UVM component that converts transactions into interface-level signals|
|Monitor|UVM component that observes interface activity and publishes transactions|
|Scoreboard|Component responsible for checking actual DUT behavior against expected behavior|
|Reference Model|Independent model that predicts expected DUT behavior|
|Functional Coverage|Measurement of whether specified functional scenarios have been exercised|
|DMA|Direct Memory Access|
|FIFO|First-In, First-Out buffer|
|DUT Wrapper|Integration module connecting the selected iDMA modules and exposing the testbench interfaces|

---

# 1. Introduction

## 1.1 Purpose

This document describes the architecture of the UVM-based verification environment developed to verify the PULP iDMA.

It defines:

- The selected DUT configuration and interfaces.

- The DUT integration architecture.

- The UVM verification environment and its components.

- Communication between verification components.

- The reference model and scoreboard architecture.

- Test and sequence organization.

- Functional coverage and assertion strategies.

- Simulation and regression flow.

The architecture also establishes the foundation for a future Reinforcement Learning (RL) component that will improve functional coverage closure by adapting constrained-random stimulus generation.

## 1.2 Scope

The initial verification scope focuses on:

- One-dimensional (1D) memory-to-memory DMA transfers.

- Configuration and control through the APB4 register frontend.

- Memory transactions through the AXI4 backend.

- Verification of source and destination addresses.

- Verification of transfer lengths and data integrity.

- Verification of transfer completion and busy status.

- Verification of supported alignment and burst behavior.

- APB and AXI protocol compliance.

- Functional coverage collection.

- Reset behavior.

The initial environment targets a single register interface and a single DMA stream.

The architecture is designed to support future extensions, including additional iDMA configurations, more extensive transfer scenarios, and RL-based stimulus adaptation.

## 1.3 Verification Objectives

The main verification objectives are:

1. Verify correct configuration of the iDMA through APB.

2. Verify correct generation of AXI read and write transactions.

3. Verify data integrity between source and destination memory locations.

4. Verify correct handling of transfer lengths, address alignment, and burst boundaries.

5. Verify request buffering and transfer completion behavior.

6. Verify the frontend's busy and completion status reporting.

7. Detect APB and AXI protocol violations.

8. Verify reset behavior.

9. Measure functional coverage of the supported scenarios.

---

# 2. DUT Description

## 2.1 DUT Architecture

The selected DUT configuration consists of a register frontend, request buffering, transfer-ID tracking, and an AXI backend.

The DUT is integrated through a project-specific `idma_wrapper` module.

The wrapper connects the following modules:

1. **Register Frontend (**`**idma_reg64_1d**`**):** Receives APB transactions, stores transfer configuration, and generates DMA requests.

2. **Request FIFO (**`**cc_stream_fifo_optimal_wrap**`**):** Buffers DMA requests between the frontend and backend.

3. **Transfer ID Generator (**`**idma_transfer_id_gen**`**):** Generates transfer IDs and tracks issued and completed requests.

4. **AXI Backend (**`**idma_backend_rw_axi**`**):** Performs memory reads and writes and manages AXI transactions.

5. **AXI Read/Write Join (**`**axi_rw_join**`**):** Combines the backend's separate read and write interfaces into the AXI interface exposed by the wrapper.

The midend is not instantiated because the initial verification scope is limited to 1D transfers.

### 2.1.1 DUT Block Diagram

```
                  APB4 Interface
                        |
                        v
              +--------------------+
              |  idma_reg64_1d     |
              |  Register Frontend |
              +--------------------+
                        |
                  DMA Request
                        |
                        v
              +--------------------+
              |  Request FIFO      |
              |  Depth = 3         |
              +--------------------+
                        |
                        v
              +--------------------+
              | idma_backend_rw_axi|
              |    AXI Backend     |
              +--------------------+
                   |          |
              AXI Read     AXI Write
                   |          |
                   +-----+----+
                         |
                         v
                 +---------------+
                 |  axi_rw_join  |
                 +---------------+
                         |
                         v
                   AXI4 Interface
                         |
                         v
                 Testbench Memory
```

The transfer-ID generator operates alongside the request path. It provides the next transfer ID to the frontend and tracks request issue and completion events.

The wrapper also feeds backend busy information and completion IDs back to the frontend.

## 2.2 DUT Interfaces

The selected iDMA configuration uses APB4 for configuration and control and AXI4 for memory access.

### 2.2.1 APB Interface

The APB4 interface is used to configure and control the iDMA through its register frontend.

The testbench acts as the APB master, while the iDMA acts as the APB slave.

The APB interface is responsible for:

- Writing source and destination addresses.

- Writing the transfer length.

- Configuring backend options when required.

- Triggering DMA operations.

- Reading the DMA busy status and completion information.

An APB UVM agent will drive and monitor this interface. It will be configured as an active master.

The selected frontend is `idma_reg64_1d`. It uses a 32-bit APB register data path and provides 64-bit source and destination address registers. Each address is accessed through two 32-bit registers.

The generated register block uses a 9-bit register address path.

#### Register Map

The following registers are relevant to the initial 1D configuration. Offsets are byte addresses.

|Register|Offset|Access|Description|
|---|---|---|---|
|`conf`|`0x000`|Read/Write|Backend configuration|
|`status[0]`|`0x004`|Read-only|DMA busy status|
|`next_id[0]`|`0x044`|Read-only|Reading launches a transfer and returns its ID|
|`done_id[0]`|`0x084`|Read-only|Reports the last completed transfer ID|
|`dst_addr[0]`|`0x0D0`|Read/Write|Destination address, lower 32 bits|
|`dst_addr[1]`|`0x0D4`|Read/Write|Destination address, upper 32 bits|
|`src_addr[0]`|`0x0D8`|Read/Write|Source address, lower 32 bits|
|`src_addr[1]`|`0x0DC`|Read/Write|Source address, upper 32 bits|
|`length[0]`|`0x0E0`|Read/Write|Transfer length, lower 32 bits|
|`length[1]`|`0x0E4`|Read/Write|Transfer length, upper 32 bits|
|`compute_cfg`|`0x0E8`|Read/Write|On-the-fly compute configuration|

The initial configuration uses one register interface and one DMA stream. Therefore, only the registers associated with stream 0 are used.

#### Transfer Programming Sequence

A DMA transfer follows this sequence:

1. Write the source address.

2. Write the destination address.

3. Write the transfer length.

4. Configure backend options if required.

5. Read `next_id[0]` to launch the transfer and obtain its assigned ID.

6. Monitor or poll `done_id[0]` until the assigned transfer ID is reported as completed.

Reading `next_id[0]` is a side-effecting operation: it triggers the transfer. The APB driver must perform this read exactly once per transfer.

The frontend captures the configured transfer parameters and submits a request using a request-valid/request-ready handshake.

The transfer-ID generator provides the next ID to the frontend and updates the completion ID when a backend response is retired.

#### Busy and Completion Reporting

The frontend receives:

- The backend busy status.

- The request FIFO occupancy indication through the FIFO valid signal.

- The completed transfer ID.

The wrapper combines the backend busy status with the FIFO valid signal so that the frontend reports the DMA as busy while a request is waiting in the FIFO.

The backend response is accepted unconditionally because `rsp_ready` is tied high in the wrapper.

### 2.2.2 AXI4 Interface

The AXI4 interface is used by the iDMA to access the memory system.

The iDMA acts as an AXI4 master, while the testbench AXI agent acts as an AXI4 slave.

The AXI4 interface is responsible for:

- Reading data from source memory locations.

- Writing data to destination memory locations.

- Handling read and write responses.

- Generating incremental bursts.

- Respecting AXI burst-length and address-boundary restrictions.

The testbench will use an AXI UVM agent configured as an active slave. The agent will respond to transactions initiated by the iDMA and maintain a memory model representing the accessible memory space.

The backend generates separate read and write AXI requests. These interfaces are combined by `axi_rw_join` into the single AXI request/response interface exposed by the wrapper.

#### AXI Configuration

|Parameter|Configuration|
|---|---|
|Address width|64 bits|
|Data width|32 bits|
|ID width|1 bit|
|User width|1 bit|
|Data bytes per beat|4 bytes|
|Burst type|INCR|
|Maximum AXI4 burst length|256 beats|
|Backend `BurstLen` parameter|8|
|`NumAxInFlight`|3|
|Backend `BufferDepth`|3|

The 64-bit AXI address width is the intended configuration to match the selected 64-bit register frontend. The wrapper must be updated accordingly before implementation is considered complete.

#### Burst Behavior

The backend's `BurstLen` parameter defines an additional burst-size limit:

[  
MaximumBurstSize = 8 \times 2^{BurstLen}\ \text{bytes}  
]

With `BurstLen = 8`, this parameter corresponds to 2048 bytes.

However, AXI4 limits an INCR burst to 256 beats. With a 32-bit data width, this gives a maximum of:

[  
256 \times 4 = 1024\ \text{bytes}  
]

Therefore, the effective AXI4 burst limit is 1024 bytes for the selected data width, subject to the 4 KB address-boundary restriction and backend legalizer behavior.

The hardware legalizer is enabled. It splits transfers into legal AXI bursts and handles the required data realignment.

#### Address Alignment

The backend supports data realignment for transfers whose source and destination addresses are not aligned to the 4-byte data width.

The verification environment will include aligned and unaligned source and destination addresses.

#### Outstanding Transactions and Buffering

The backend is configured with:

- `NumAxInFlight = 3`.

- `BufferDepth = 3`.

- `MemSysDepth = 0`.

The exact outstanding transaction accounting is defined by the backend implementation and will be respected by the AXI agent.

A separate request FIFO with a depth of 3 is instantiated between the frontend and backend. This FIFO buffers submitted DMA requests before they are accepted by the backend.

#### Response and Error Behavior

The backend is configured with:

`ErrorCap = idma_pkg::NO_ERROR_HANDLING`

Consequently, the initial configuration does not provide the normal iDMA error-handling path for propagating AXI errors through the DMA completion response.

The backend is also configured with:

`RejectZeroTransfers = 1`

Zero-length transfers are therefore rejected.

The initial verification scope will focus on successful transfers and zero-length rejection. Software-visible AXI error recovery is not part of the initial verification scope.

### 2.2.3 Clock and Reset

The testbench top-level module (`tb_top`) generates and distributes the clock and reset signals to the DUT and the required testbench components.

The wrapper exposes:

- `clk_i`: Common clock.

- `rst_ni`: Active-low reset.

The reset is connected to the register frontend, request FIFO, transfer-ID generator, backend, and AXI read/write join module.

The selected RTL uses active-low asynchronous reset behavior.

Reset scenarios will verify:

- Initialization of the DUT.

- Reset of the transfer-ID generator.

- Clearing of the request FIFO.

- Return of the frontend to an idle state.

- Ability to accept new transfers after reset.

Reset during an active transfer requires special attention. Resetting the DUT does not roll back memory writes that have already completed. The verification plan will distinguish between internal state reset and memory transactions completed before reset.

The clock period and reset assertion duration are testbench configuration choices.

## 2.3 DUT Configuration

The initial configuration targets 1D memory-to-memory transfers using the APB register frontend and AXI4 backend.

The selected configuration uses:

- Frontend: `idma_reg64_1d`.

- Backend: `idma_backend_rw_axi`.

- Integration wrapper: `idma_wrapper`.

- One APB register interface.

- One DMA stream.

- No midend.

The wrapper exposes APB request/response and AXI request/response ports to the testbench.

### 2.3.1 Frontend Parameters

|Parameter|Value|Description|
|---|---|---|
|`NumRegs`|1|Number of APB register interfaces|
|`NumStreams`|1|Number of DMA streams|
|`IdCounterWidth`|32|Transfer ID counter width|
|Register address width|9 bits|Internal register address path|
|APB data width|32 bits|Register data width|

### 2.3.2 Backend and Wrapper Parameters

|Parameter|Value|Description|
|---|---|---|
|`DataWidth`|32|AXI data width in bits|
|`AddrWidth`|64|Intended DMA and AXI address width|
|`AxiIdWidth`|1|AXI ID width|
|`UserWidth`|1|AXI user width|
|`NumAxInFlight`|3|Configured outstanding AX transaction limit|
|`BufferDepth`|3|Backend buffer depth|
|`TFLenWidth`|32|DMA transfer-length field width|
|`BurstLen`|8|Backend burst-size parameter|
|`RequestFifoDepth`|3|Request FIFO depth|
|`MemSysDepth`|0|Memory-system pipeline depth|

### 2.3.3 Backend Feature Parameters

|Parameter|Value|Description|
|---|---|---|
|`CombinedShifter`|0|Combined shifter disabled|
|`EnableCompute`|0|Compute functionality disabled|
|`RAWCouplingAvail`|1|RAW coupling support enabled|
|`MaskInvalidData`|0|Invalid-data masking disabled|
|`HardwareLegalizer`|1|Hardware legalizer enabled|
|`RejectZeroTransfers`|1|Zero-length transfers rejected|
|`ErrorCap`|`NO_ERROR_HANDLING`|Backend error-handling hardware disabled|
|`PrintFifoInfo`|0|FIFO diagnostic printing disabled|

### 2.3.4 Address and Transfer-Length Configuration

The selected frontend provides 64-bit source and destination address registers.

The DMA request type, backend address width, and AXI address channel types must use the same 64-bit address width to preserve the complete address generated by the frontend.

The current wrapper must therefore be updated from `AddrWidth = 32` to `AddrWidth = 64`, and the externally supplied AXI types must also use 64-bit addresses.

The frontend register map provides a 64-bit transfer-length register, while the wrapper currently defines `TFLenWidth = 32`.

For the initial configuration, transfer lengths will be constrained to the lower 32-bit range, and the upper 32 bits of the programmed length must be zero.

Supporting larger transfer lengths would require increasing `TFLenWidth` and validating the resulting backend configuration.

### 2.3.5 Testbench Configuration Choices

Some parameters are defined by the verification environment rather than the DUT:

- Clock period and frequency.

- Memory model address range.

- Initial memory contents.

- Memory access latency.

- AXI response delays.

- AXI backpressure behavior.

- Reset duration and reset scenarios.

The memory model will initially use a limited address range within the 64-bit AXI address space. This keeps simulation practical while preserving the full address width of the DUT interface.

All shared configuration values will be centralized to prevent inconsistencies between the DUT wrapper, UVM agents, and memory model.

---

# 3. Verification Environment

## 3.1 Introduction

The verification environment is built using UVM and consists of reusable agents, a reference model, a scoreboard, a coverage collector, and a virtual sequencer.

The environment separates:

- Stimulus generation.

- Signal driving.

- Transaction monitoring.

- Expected-result generation.

- Functional checking.

- Coverage collection.

This separation improves maintainability and allows individual components to be modified or replaced independently.

## 3.2 Verification Environment Block Diagram

```
                         TESTBENCH
+-----------------------------------------------------------+
|                                                           |
|  +-----------------------------------------------------+  |
|  |                      tb_top                         |  |
|  |                                                     |  |
|  |  - Clock / Reset                                    |  |
|  |  - idma_wrapper                                     |  |
|  |  - APB Interface                                    |  |
|  |  - AXI Interface                                    |  |
|  |  - run_test()                                       |  |
|  +---------------------------+-------------------------+  |
|                              |                            |
|                       Virtual Interfaces                  |
|                              |                            |
|                              v                            |
|  +-----------------------------------------------------+  |
|  |                    UVM Test                         |  |
|  |                                                     |  |
|  |  +-----------------------------------------------+  |  |
|  |  |                 idma_env                      |  |  |
|  |  |                                               |  |  |
|  |  |  +----------------+  +---------------------+  |  |  |
|  |  |  | APB Agent      |  | AXI Agent            |  |  |  |
|  |  |  | Active Master  |  | Active Slave         |  |  |  |
|  |  |  +-------+--------+  +----------+----------+  |  |  |
|  |  |          |                      |             |  |  |
|  |  |          +----------+-----------+             |  |  |
|  |  |                     |                         |  |  |
|  |  |              +------+-------+                 |  |  |
|  |  |              | Scoreboard   |                 |  |  |
|  |  |              +------+-------+                 |  |  |
|  |  |                     ^                         |  |  |
|  |  |              +------+-------+                 |  |  |
|  |  |              | Reference    |                 |  |  |
|  |  |              | Model        |                 |  |  |
|  |  |              +--------------+                 |  |  |
|  |  |                                               |  |  |
|  |  |              +--------------+                 |  |  |
|  |  |              | Coverage     |                 |  |  |
|  |  |              | Collector    |                 |  |  |
|  |  |              +--------------+                 |  |  |
|  |  |                                               |  |  |
|  |  |              +--------------+                 |  |  |
|  |  |              | Virtual      |                 |  |  |
|  |  |              | Sequencer    |                 |  |  |
|  |  |              +--------------+                 |  |  |
|  |  +-----------------------------------------------+  |  |
|  +-----------------------------------------------------+  |
|                                                           |
+-----------------------------------------------------------+
```

The diagram is conceptual.

`tb_top` is a SystemVerilog module that instantiates the DUT wrapper and physical interfaces, generates the clock and reset, and calls `run_test()`.

The UVM test and environment are created dynamically when `run_test()` is called.

Virtual interfaces provide access to the physical APB and AXI interfaces from UVM components.

## 3.3 Component Architecture and Connections

|Component|Responsibility|
|---|---|
|APB Agent|Drives APB register accesses and monitors APB activity|
|AXI Agent|Responds to AXI transactions and provides memory behavior|
|Reference Model|Predicts expected destination memory contents|
|Scoreboard|Checks observed DUT behavior against expected behavior|
|Coverage Collector|Collects functional coverage from observed transactions|
|Virtual Sequencer|Coordinates sequences across the APB and AXI agents|

The APB and AXI monitors publish observed transactions through TLM analysis ports.

The scoreboard and coverage collector subscribe to the relevant monitor outputs.

The reference model receives the intended transfer configuration and source memory contents, then calculates the expected destination data.

The reference model must remain independent of the AXI agent's memory implementation. Expected results must not be generated by simply comparing the DUT against the same memory-model implementation that services its AXI accesses.

## 3.4 Configuration Mechanism

UVM configuration mechanisms will provide components with their required settings.

The `uvm_config_db` mechanism will distribute virtual interface handles and configuration objects.

Configuration objects may contain:

- APB and AXI interface settings.

- Agent active/passive modes.

- Memory model settings.

- Address and data widths.

- DUT configuration parameters.

- Default response delays and backpressure settings.

This approach avoids hardcoding configuration values inside individual components.

## 3.5 Directory Structure

The proposed directory structure is:

```
idma_verification/
|
+-- rtl/
|   +-- idma/
|   +-- wrapper/
|
+-- tb/
|   +-- top/
|   |   +-- tb_top.sv
|   |
|   +-- interfaces/
|   |   +-- apb_if.sv
|   |   +-- axi_if.sv
|   |
|   +-- agents/
|   |   +-- apb_agent/
|   |   +-- axi_agent/
|   |
|   +-- env/
|   |   +-- idma_env.sv
|   |   +-- idma_virtual_sequencer.sv
|   |   +-- idma_scoreboard.sv
|   |   +-- idma_coverage.sv
|   |
|   +-- reference_model/
|   |
|   +-- memory_model/
|   |
|   +-- tests/
|   |
|   +-- sequences/
|   |
|   +-- assertions/
|
+-- sim/
|   +-- filelist.f
|   +-- scripts/
|
+-- results/
|
+-- docs/
```

The reference model is kept in a separate directory to facilitate its future replacement with a C-based implementation.

---

# 4. UVM Agents and Components

## 4.1 APB Agent

The APB agent acts as an APB master and is responsible for configuring and controlling the iDMA.

It consists of:

- **Sequencer:** Provides APB transactions to the driver.

- **Driver:** Converts APB sequence items into APB signal activity.

- **Monitor:** Observes APB transactions and publishes them through an analysis port.

The APB agent will support the register accesses required by the verification scenarios.

The monitor will publish observed transactions rather than relying on sequence intent. This allows the scoreboard to check the actual APB activity.

## 4.2 AXI Agent

The AXI agent acts as an AXI4 slave because the iDMA is the AXI4 master.

It consists of:

- **Sequencer:** Provides response and memory-behavior control transactions.

- **Driver:** Drives AXI slave-side responses and read data.

- **Monitor:** Observes AXI transactions and publishes them through analysis ports.

- **Memory Model:** Stores memory contents and provides read/write behavior.

The AXI agent will support:

- Normal read and write responses.

- Configurable response delays.

- Configurable backpressure.

- Memory initialization.

- AXI response injection for protocol-level robustness testing, where applicable.

The memory model represents the memory accessed by the iDMA during simulation.

## 4.3 Reference Model

The reference model predicts the expected result of each DMA transfer independently of the DUT.

It receives:

- The observed transfer configuration.

- The initial source memory contents.

- The transfer parameters, including addresses and length.

It calculates the expected destination memory contents and publishes the expected result to the scoreboard.

The reference model will be implemented independently of the AXI agent's memory implementation.

This separation prevents the scoreboard from relying on the same implementation for both actual and expected behavior.

The reference model will remain independent of UVM-specific agent implementation details to facilitate a possible future replacement with a C-based implementation.

## 4.4 Scoreboard

The scoreboard checks whether the actual DUT behavior matches the expected behavior.

It receives observed transactions from the APB and AXI monitors and expected results from the reference model.

Its responsibilities include:

1. Checking observed APB programming transactions.

2. Checking AXI read and write addresses.

3. Checking transferred data and data integrity.

4. Checking transfer lengths and expected memory updates.

5. Checking transfer completion behavior.

6. Detecting unexpected or missing transactions.

7. Reporting mismatches and verification failures.

The scoreboard must not use the AXI agent's memory contents as its only golden reference.

Expected data must originate from the independent reference model or an independently maintained expected-memory representation.

## 4.5 Coverage Collector

The coverage collector measures whether the required functional scenarios have been exercised.

It receives observed transactions from the APB and AXI monitors.

Coverage will be organized around:

- Transfer configuration.

- Source and destination address characteristics.

- Transfer length.

- Alignment.

- AXI burst behavior.

- AXI response behavior.

- Transaction latency.

- Backpressure.

- Request buffering.

- Transfer completion.

- Reset scenarios.

The coverage collector will remain independent of the scoreboard so that coverage collection does not interfere with correctness checking.

## 4.6 Virtual Sequencer

The virtual sequencer coordinates the APB and AXI sequencers.

It provides a common control point for end-to-end scenarios that require coordinated activity across multiple interfaces.

For example, an end-to-end sequence may:

1. Initialize source memory contents.

2. Configure the iDMA through APB.

3. Trigger the transfer.

4. Control AXI response delays and backpressure.

5. Wait for transfer completion.

6. Initiate result checking.

The virtual sequencer will not replace the individual APB and AXI sequencers. It provides access to them for coordinated stimulus generation.

---

# 5. UVM Tests and Sequences

## 5.1 Base Test

The base test creates and configures the verification environment.

It is responsible for:

- Setting environment configuration.

- Selecting agent operating modes.

- Configuring the memory model.

- Creating the required test sequences.

- Reporting the final test result.

Common configuration and initialization procedures should be implemented in reusable base classes where appropriate.

## 5.2 Base Sequences

Base sequences provide reusable transaction-generation and initialization functionality.

They may include:

- APB register access sequences.

- AXI response configuration sequences.

- Memory initialization sequences.

- Common DMA transfer configuration sequences.

- Transfer completion polling sequences.

## 5.3 Test Categories

### 5.3.1 APB-Focused Tests

These tests verify the register frontend.

Scenarios include:

- Register read and write operations.

- Register reset values.

- Register access permissions.

- Configuration programming behavior.

- Transfer launch behavior.

- Busy and completion status reporting.

### 5.3.2 AXI-Focused Tests

These tests focus on the AXI backend.

Scenarios include:

- Read and write transaction behavior.

- Different AXI response latencies.

- Backpressure conditions.

- Burst splitting.

- 4 KB boundary handling.

- Unaligned source and destination addresses.

- Multiple outstanding transactions within the configured limits.

### 5.3.3 End-to-End DMA Tests

These tests verify complete memory-to-memory transfers.

The virtual sequence coordinates the APB and AXI agents.

Scenarios include:

- Basic memory-to-memory transfer.

- Different source and destination addresses.

- Different transfer lengths.

- Aligned and unaligned addresses.

- Transfers requiring multiple AXI bursts.

- Consecutive DMA requests.

- Requests buffered while the backend is busy.

- Data integrity verification.

### 5.3.4 Boundary and Reset Tests

These tests target corner cases and reset behavior.

Scenarios include:

- Minimum supported transfer length.

- Maximum practical transfer length for the testbench.

- Zero-length transfer rejection.

- Address alignment boundaries.

- AXI 4 KB burst boundaries.

- Reset during idle operation.

- Reset during an active transfer.

- New transfer after reset.

AXI error-response injection may be used to assess protocol robustness, but software-visible error recovery is not a guaranteed feature of the selected `NO_ERROR_HANDLING` configuration.

The final test list will remain aligned with the capabilities of the selected iDMA configuration.

---

# 6. Coverage Collection

## 6.1 Coverage Architecture

Functional coverage is collected independently from the scoreboard.

The coverage collector subscribes to APB and AXI monitor analysis ports and samples observed transactions.

This ensures that coverage represents actual DUT behavior rather than merely the stimulus requested by sequences.

## 6.2 Coverage Groups and Coverpoints

The initial coverage model will include:

|Coverage Area|Potential Coverpoints|
|---|---|
|Transfer configuration|Transfer length, configuration fields|
|Source address|Alignment and address range|
|Destination address|Alignment and address range|
|AXI read behavior|Burst length, burst splitting, response behavior|
|AXI write behavior|Burst length, burst splitting, response behavior|
|AXI timing|Response latency and backpressure|
|Request buffering|FIFO occupancy and request stalls|
|Completion|Successful completion and completion latency|
|Reset|Reset state and recovery behavior|

The exact coverpoints and bins will be finalized during implementation.

## 6.3 Cross Coverage

Cross coverage will measure combinations of relevant features.

Examples include:

- Source alignment × destination alignment.

- Transfer length × AXI burst length.

- Source alignment × transfer length.

- Backpressure × burst behavior.

- Request FIFO occupancy × backend busy state.

- Transfer length × completion latency.

Cross coverage helps identify combinations that have not been exercised even when individual coverpoints have reached high coverage.

## 6.4 Coverage Goals

Coverage goals will be defined for the supported feature set.

Coverage reports will be used to identify uncovered scenarios and guide subsequent test generation.

The coverage model will provide a structured interface for future RL integration.

The RL component may later consume coverage information and adapt stimulus-generation parameters without modifying the scoreboard or reference model.

---

# 7. Assertions

Assertions will be used to verify protocol-level and temporal requirements.

The initial assertion set will focus on:

1. APB protocol compliance.

2. AXI4 protocol compliance.

3. Reset behavior.

4. Validity of request and response handshakes.

5. AXI burst termination and response requirements.

6. Request FIFO handshake behavior.

7. Transfer completion behavior.

8. Relevant busy-status behavior.

Assertions will complement the scoreboard:

- Assertions check local signal-level and temporal properties.

- The scoreboard checks end-to-end functional correctness.

Assertions should be based on the relevant protocol specifications and the selected iDMA configuration.

---

# 8. Assumptions and Constraints

The initial architecture is based on the following assumptions:

1. The selected DUT is the PULP iDMA `idma_reg64_1d` frontend with the `idma_backend_rw_axi` backend.

2. The APB interface is used for configuration and control.

3. The AXI4 interface is used for memory access.

4. The iDMA acts as an APB slave and AXI4 master.

5. The verification environment provides an AXI4 slave memory model.

6. The initial verification scope is limited to 1D memory-to-memory transfers.

7. The configuration uses one register interface and one DMA stream.

8. The reference model is independent of the AXI agent memory implementation.

9. The environment uses a common clock and active-low asynchronous reset.

10. The intended address width is 64 bits throughout the frontend request, backend, and AXI path.

11. Transfer lengths are initially constrained to the 32-bit range.

12. Backend error handling is disabled in the selected configuration.

These assumptions define the initial verification baseline.

---

# 9. Out-of-Scope Areas

The following features are excluded from the initial verification scope:

- 2D and 3D DMA transfers.

- Advanced descriptor-based frontend functionality.

- Multiple DMA streams.

- Dedicated interrupt verification.

- On-the-fly compute operations.

- Software-visible AXI error recovery.

- Alternative iDMA frontend configurations.

- RL-based adaptive stimulus generation during initial environment development.

These areas may be considered for future extensions.

---

# 10. Simulation and Regression Flow

## 10.1 Simulation Setup

The simulation environment will compile:

- Selected iDMA RTL and required dependencies.

- The project-specific DUT wrapper.

- APB and AXI interfaces.

- UVM agents and components.

- The reference model and memory model.

- The testbench top-level module.

The simulator will be configured with the required SystemVerilog and UVM support.

## 10.2 Test Execution

Each test will be executed with its selected configuration and random seed.

The regression flow will support running multiple tests and seeds to improve scenario exploration.

## 10.3 Results and Reporting

Simulation results will include:

- Test pass/fail status.

- UVM reports and error messages.

- Assertion failures.

- Scoreboard mismatches.

- Functional coverage reports.

- Random seed and test configuration.

- Simulation runtime.

Regression results will be used to track verification progress and identify remaining coverage gaps.

---

# 11. Implementation Items

The architecture is based on the selected PULP iDMA RTL configuration and the project-specific wrapper.

The following implementation items must be completed or validated before the verification environment is considered ready.

|Item|Required Action|
|---|---|
|Address width consistency|Update the wrapper's `AddrWidth` from 32 to 64 and ensure the AXI typedefs use 64-bit addresses|
|Transfer length width|Keep `TFLenWidth = 32` for the initial scope and ensure the upper 32 bits of the programmed length are zero|
|Wrapper elaboration|Compile and elaborate the complete wrapper with the selected RTL dependencies|
|Register access|Validate the APB register offsets, reset values, and transfer programming sequence against the generated register RTL|
|AXI configuration|Validate the actual AXI channel types and backend parameter interpretation|
|Burst behavior|Confirm the generated burst sizes and 4 KB boundary handling through simulation|
|Reset behavior|Validate reset during idle and active transfers|
|End-to-end smoke test|Complete at least one APB-configured memory-to-memory transfer and check destination data|
|Error behavior|Confirm the observed behavior of AXI error responses with `NO_ERROR_HANDLING` enabled|

The wrapper must pass compilation, elaboration, and an end-to-end smoke test before it is used as the baseline DUT integration for UVM development.

---

# 12. Future RL Integration

The initial verification environment will use conventional constrained-random stimulus generation.

RL integration will be introduced after the baseline UVM environment, reference model, scoreboard, and functional coverage model are operational.

The future RL component will interact with the verification environment through a defined interface.

Its intended responsibilities are:

1. Observe coverage information and selected verification state.

2. Identify scenarios that remain uncovered or difficult to reach.

3. Adapt selected constrained-random stimulus parameters.

4. Generate new stimulus configurations.

5. Evaluate the resulting coverage progress.

The RL component will not modify the DUT, scoreboard, or reference model.

The conventional UVM environment must remain independently executable so that results from RL-assisted stimulus generation can be compared against the baseline constrained-random approach.

---

# 13. Conclusion

This document defines the baseline architecture for verifying the selected PULP iDMA 1D memory-to-memory configuration.

The DUT is accessed through an APB4 register frontend and an AXI4 memory interface. The UVM environment provides active APB master and AXI slave agents, an independent reference model, a scoreboard, a coverage collector, and coordinated end-to-end sequences.

The initial verification effort focuses on functional correctness, data integrity, protocol compliance, transfer completion, buffering, and reset behavior.

The architecture is designed to remain modular and extensible, allowing future configurations and RL-based coverage closure to be integrated without changing the fundamental verification structure.
