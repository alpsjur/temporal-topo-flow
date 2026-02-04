# temporal-topo-flow
Time-dependent flow over topography. ⏳⛰️🌊

This repository contains code for simulating and analyzing time-variable flow over corrugated topography using the shallow-water model in Oceananigans.jl.

The analysis supports the study:
> **Nonlinear dynamics of time-variable slope circulation**, *Sjur, A. L. P., Isachsen, P. E., Nilsson, J., Allen, S.*, *2026*  


Processed data needed to reproduce the manuscript figures are included in `output/processed/`.  
Raw model output is not included by default (it is quite large); see *How to run the simulations* below if you want to generate it.

---
## Project structure

```
temporal-topo-flow/
├─ configs/                     # Experiment configs (JSON)
├─ input/                       # Generated inputs (e.g., forcing files)
├─ notebooks/                   # Postprocessing and plotting notebooks
├─ output/
│  ├─ processed/                # Data needed to reproduce manuscript figures
│  └─ raw/                      # (Generated) raw simulation output
├─ reproduce_brink_2010/        # Validation: reproduce Brink (2010)
│  ├─ code/
│  ├─ figures/
│  └─ output/
├─ scripts/                     # Simulation and helper scripts (Julia)
├─ utils/                       # Python helper functions (IO, grids, plotting, etc.)
├─ environment-lock.yml         # Excact python environment for reproducability
├─ environment.yml              # Minimal, readable python environment
├─ Project.toml                 # Julia project environment
└─ README.md
```

---

## Analysis presented in the paper

### Shallow water simulations
Model setup can be found in: 
- `scripts/simulation.jl`.

### Postprocessing
Postprocessing and diagnostics are performed in:
- `notebooks/postprocess_modeloutput.ipynb`

This notebook reads raw simulation output from full production runs and computes the diagnostics used in the paper. It writes compact processed datasets to `output/processed/`.


Re-running the full simulations and postprocessing pipeline is optional and computationally demanding. The processed datasets required to reproduce all manuscript figures are already included in this repository.

### Plotting
Figures for the paper are generated in:
- `notebooks/make_figures.ipynb`

This notebook reads from `output/processed/` and produces all figures used in the manuscript.  
It can be run independently of the simulations and postprocessing, and is suitable for exploring the results and modifying figure appearance.

---

## How to run the simulations

### Requirements 
- Julia, can be installed from [here](https://julialang.org/downloads/)
- Dependencies are managed via `Project.toml`

### 1) Clone the repository
```
git clone https://github.com/alpsjur/temporal-topo-flow.git
cd temporal-topo-flow
```

### 2) Instantiate the Julia environment
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Running simulations

#### Single experiment
To run a single experiment specified by a JSON config in `configs/`:
```
julia --project=. scripts/simulation.jl configs/experiment_name.json
```

By default, outputs are written under `output/raw/`.

#### Batch runs
To run the full set of experiments used in the paper, use the provided batch script:
```
./run_simulations.bash
```

This script loops over a predefined set of configurations and launches the corresponding simulations.  
It is intended for use on HPC systems and may require local adaptations.

### Optional: forcing from file
Some experiments use forcing read from file. Forcing files can be generated with scripts such as:
- `scripts/generate_crossslope_forcing.py`

Generated forcing files are stored in `input/`.

---

## Python environment (for notebooks)

Two environment specifications are provided:

- `environment.yml`  
  A minimal, readable specification intended for creating a working analysis
  environment. This pins the xarray–xgcm versions required for correct behavior.

- `environment-lock.yml`  
  An exact snapshot of the environment used to generate the results in the paper.
  This can be used for full reproducibility.

---

## Topographic wave calculations (bwavesp)

Topographic wave properties are computed using the MATLAB code **bwavesp**:
- reference: https://darchive.mblwhoilibrary.org/entities/publication/5433c043-2cc9-4906-a63e-c80a57f524e3

In this project, bwavesp is run via:
- `scripts/run_bwavesp.m`

with input from:
- `input/bwavesp_input.txt`

If you only want to reproduce manuscript figures, the wave-model outputs used for plotting are already included where needed.

---

## Brink (2010) validation case

From the repository root:
```
# 1) Run the simulation (writes to reproduce_brink_2010/output)
julia --project=. reproduce_brink_2010/code/brink_2010.jl

# 2) Recreate figures (writes to reproduce_brink_2010/figures)
julia --project=. reproduce_brink_2010/code/plot_brink_fig2.jl
julia --project=. reproduce_brink_2010/code/plot_brink_fig3.jl
```

Small example outputs are included so the plotting scripts can be tested without a full rerun.


---
Enjoy!