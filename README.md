# Accelerating Digital Verification Using AI-Based Constrained Stimulus Generation

This project accelerates functional verification by using a Reinforcement Learning (RL) agent to steer stimulus generation toward uncovered coverage bins, instead of relying on purely constrained-random stimulus. The agent and the UVM simulation communicate through a file-based loop: after each run the agent reads the coverage report, the last applied stimulus and the remaining run budget, then chooses the stimulus for the next run and is rewarded for new coverage gained.

Two agent architectures are implemented and compared: a value-based Double Deep Q-Network (DDQN) and an actor-critic model (A2C). Both are first developed and benchmarked on a UART pilot IP; the best-performing model is then carried to the main DUT, an AXI + DMA design (iDMA with an AXI interface), where RL-aided verification is compared against traditional constrained-random verification in coverage closure speed and simulation cost.

## Folder Structure

```txt
Accelerating_Digital_Verification_Using_AI-Based_Constrained_Stimulus_Generation/
├── README.md
├── .gitignore
│
├── rtl/
│   └── idma/                       # Pinned upstream iDMA RTL
│
├── tb/                             # UVM verification environment
│   ├── top.sv
│   ├── interfaces/
│   │   ├── dma_if.sv
│   │   └── axi_if.sv
│   ├── agents/
│   │   ├── dma_agent/
│   │   └── axi_slave_agent/
│   ├── env/
│   │   ├── idma_env.sv
│   │   ├── memory_model.sv
│   │   ├── reference_model.sv
│   │   ├── scoreboard.sv
│   │   └── coverage.sv
│   ├── sequences/
│   │   └── dma_sequences.sv
│   ├── tests/
│   │   └── idma_base_test.sv
│   └── assertions/
│       └── axi_checks.sv
│
├── rl/                             # Reinforcement-learning component
│   ├── agent.py                    # Chooses stimulus strategies
│   ├── state.py                    # Builds state from coverage feedback
│   ├── reward.py                   # Calculates reward for progress
│   ├── policy.py                   # Selects actions
│   └── config.yaml                 # RL hyperparameters and settings
│
├── integration/                    # Connects RL and UVM
│   ├── rl_interface.py             # Communication API / adapter
│   └── action_mapping.py           # Converts actions into UVM parameters
│
├── sim/
│   ├── files.f
│   └── run.do
│
├── scripts/
│   ├── run_test.py                 # Runs one simulation
│   └── regression.py               # Runs multiple seeds/experiments
│
└── results/
    ├── baseline/                   # Conventional CRV results
    └── rl_guided/                  # RL-guided results
```
