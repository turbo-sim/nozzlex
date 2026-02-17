# nozzlex

JAX-based Python toolkit for quasi-1D nozzle and vaneless-channel flow modeling, with support for single-phase and two-phase formulations.

## What is included

- **Core package**: `nozzlex/`
  - `functions/`: collocation/BVP-style single-phase nozzle solvers.
  - `two_fluid/`: two-fluid nozzle model components.
  - `homogeneous_equilibrium/`, `homogeneous_nonequilibrium/`, `homogeneous_relaxation/`: two-phase model variants.
  - `vaneless_channel/`: vaneless diffuser/channel model with friction and heat-transfer options.
- **Examples**: `examples/` for single-phase, flashing/two-phase, and vaneless diffuser studies.
- **Project setup**: Poetry project (`pyproject.toml`) targeting Python 3.11–3.13.

## Installation

```bash
git clone https://github.com/AndCiof/nozzlex.git
cd nozzlex
poetry install
```

## Quick start

Run a vaneless-channel example:

```bash
cd examples/vaneless_diffuser
python demo_curved_channel.py
```

## Notes

- The package defaults JAX execution to CPU and enables 64-bit mode.
- Several legacy/demo scripts exist in `examples/` and may represent work-in-progress research workflows.
