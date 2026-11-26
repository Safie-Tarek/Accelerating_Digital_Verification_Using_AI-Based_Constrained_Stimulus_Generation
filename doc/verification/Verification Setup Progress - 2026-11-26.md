# Verification Setup Progress

**Date:** 2026-11-26  
**Author:** Arsany

## Purpose

This note records the repository structure, RTL cleanup, and QuestaSim setup completed while preparing to verify the selected iDMA configuration before implementing the UVM environment.

## Architecture and source layout

The selected design is a 1D APB register frontend with an AXI backend, integrated by `rtl/idma_wrapper.sv`. The wrapper connects the register frontend, request FIFO, transfer ID generator, backend, and AXI read/write join.

The repository's `src/` directory is primarily iDMA hardware source and generation input, not software. `rtl/idma_generated.sv` is a bundled generated file containing multiple iDMA configurations. It is not, by itself, the complete integrated DUT: the wrapper and shared implementation modules are also required.

The proposed UVM directory structure was created under `tb/`:

- `top/`
- `interfaces/`
- `agents/apb_agent/` and `agents/axi_agent/`
- `env/`
- `reference_model/`
- `memory_model/`
- `tests/`
- `sequences/`
- `assertions/`

These directories are currently scaffolding; no UVM components or testbench top have been implemented.

## RTL cleanup

The standalone generated RTL files that duplicated modules or packages in `rtl/idma_generated.sv` were removed. The bundle was kept to retain the other generated configurations. The project wrapper and existing `rtl/tb_idma_*.sv` testbench files were kept. The tiny `rtl/idma_desc64_top.sv` placeholder was also left in place; it contains no module declaration.

During wrapper compilation, Questa reported an error at the `idma_busy_t` array declaration. The type references in `rtl/idma_wrapper.sv` were qualified as `idma_pkg::idma_busy_t`.

## QuestaSim setup

The required PULP dependency sources and headers were copied into `rtl/vendor/pulp/` so the compile setup does not rely on a local `.bender` checkout. Their original license files were retained, and the upstream revisions are recorded in `rtl/vendor/pulp/README.md`.

`sim/compile.f` lists the local dependency sources, bundled iDMA RTL, wrapper, transfer ID generator, and backend support RTL. Its paths are relative to the repository root. The intended command is:

```tcl
vlib work
vlog -sv -work work -f sim/compile.f
```

The root and `sim/.gitignore` files ignore generated QuestaSim libraries and artifacts, including `work/`, transcripts, logs, waveforms, and coverage databases. Questa `.mpf` files are GUI project settings; they are not needed for the command-line flow and should generally remain untracked unless shared GUI settings are desired.

## Compile result and limits

On 2026-11-26, QuestaSim 2021.1 compiled the bundled RTL and wrapper with **0 errors and 302 warnings** after the package qualification fix. That run compiled RTL; it did not elaborate the integrated wrapper or simulate a transfer.

After that successful run, `sim/compile.f` was updated to use root-relative paths and include the backend support files, and the `src/` cleanup below was performed. The updated file list has **not yet been recompiled**.

There is no project UVM test top or wrapper smoke-test top yet, so there is no project test to run. The retained PULP testbenches also depend on PULP test support and job files that are not part of the current Questa file list. A small non-UVM wrapper smoke test is the next functional gate: program a transfer over APB, service AXI requests with a memory model, and check destination data and completion before starting UVM.

## `src/` cleanup

Files outside the current 1D APB/AXI scope were removed:

- The descriptor and instruction frontends (`src/frontend/desc64/` and `src/frontend/inst64/`).
- The multi-dimensional midend sources (`src/midend/`).
- Compute-only RTL and support files, including the floating-point package and DMOPC template/database entry.
- The compute software-header template (`src/sw/`).

The package, transfer ID generator, AXI backend implementation files, legalizer splitters, shared headers, register-generation inputs, and other protocol backend sources were retained. Some are needed by the selected wrapper; others support the bundled variants, retained testbenches, or future scope changes.

## Current readiness

- The proposed `tb/` folder structure exists, but its directories are empty.
- The generated RTL bundle, project wrapper, and retained testbench source files remain in the repository.
- The compile file list is local to the repository and does not refer to `.bender`.
- The updated compile file list still needs a fresh Questa compile.
- Wrapper elaboration and an end-to-end APB-to-AXI memory transfer have not yet been demonstrated.
- The UVM environment should begin after the wrapper smoke test passes.
