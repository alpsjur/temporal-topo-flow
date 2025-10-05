# temporal-topo-flow
Time-dependent flow over topography. ⏳⛰️🌊

So far, this repository contains scripts for simulating and reproducing time-variable flow over corrugated topography, using the shallow water model from [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/).  

Stay tuned for more content exploring how time-dependent flow over topography responds differently to prograde[^1] and retrograde[^2] wind-forcing!

[^1]: In the same direction as Topographic Rossby wave propagation.

[^2]: In the opposite direction of Topographic Rossby wave propagation.


## Project structure

```
temporal-topo-flow/
├─ configs/                     # Experiment configs (JSON) for main runs
├─ notebooks/                   # Jupyter notebooks (exploration / plotting)
├─ reproduce_brink_2010/        # Validation: reproduce figures from Brink (2010)
│  ├─ code/
│  │  ├─ brink_2010.jl          # Run SW model & save output
│  │  ├─ plot_brink_fig2.jl     # Recreate Fig. 2
│  │  └─ plot_brink_fig3.jl     # Recreate Fig. 3
│  ├─ figures/                  # Generated figures 
│  └─ output/                   # Example output
├─ scripts/                     # utilities for experiments, including the simulation setup in "simulations.jl"
├─ utils/                       # Shared helper functions
├─ Project.toml                 # Julia project manifest (dependencies)
└─ README.md
```

## Quickstart

### Requirements
- **Julia:** 1.10+ recommended
- **Packages:** managed via `Project.toml` (instantiated automatically)

### 1) Clone the correct branch
```bash
git clone --branch paper-prep https://github.com/alpsjur/temporal-topo-flow.git
cd temporal-topo-flow
```

### 2) Instantiate the Julia environment
Run once to install all packages for this project:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```
If you prefer the REPL:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

That’s it—no global installs needed. The `--project` flag makes Julia use this repo’s environment.

### 3) Smoke test
Run a quick Julia command that imports the core dependency to verify the env:
```bash
julia --project -e 'using Oceananigans; println("Oceananigans OK: ", Oceananigans.versioninfo())'
```


## Configuration

- Experiment configs live under `configs/`.

## Batch simulations

To run multiple experiments conveniently, use the provided bash script:

```bash
./runs_simulations.bash
```

- This script loops through a set of configs in `configs/` and launches simulations.  
- Each run writes output into its own subfolder (by default under `runs/`).  
- You can edit the script to add/remove experiments, or adjust paths for your system.


## Postprocessing notebook

Most analysis and figure generation for the paper is done in  
`notebooks/postprocess.ipynb`.


## Running the Brink (2010) validation case

From the repo root:
```bash
# 1) Run the simulation (saves .jld2 in reproduce_brink_2010/output)
julia --project reproduce_brink_2010/code/brink_2010.jl

# 2) Recreate figures (writes to reproduce_brink_2010/figures)
julia --project reproduce_brink_2010/code/plot_brink_fig2.jl
julia --project reproduce_brink_2010/code/plot_brink_fig3.jl
```
The repo includes small example outputs so you can test the plotting scripts without a full rerun.
