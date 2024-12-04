# temporal-topo-flow
Time-dependent flow over topography. ⏳⛰️🌊

So far, this repository contains scripts for simulating and reproducing results of topographic rectified flow, using the shallow water model from [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/).  

Stay tuned for more content exploring how time-dependent flow over topography responds differently to prograde[^1] and retrograde[^2] wind-forcing!

[^1]: In the same direction as Topographic Rossby wave propagation.

[^2]: In the opposite direction of Topographic Rossby wave propagation.

## Repository structure

### `reproduce_brink_2010`
To validate the shallow water model in Oceananigans.jl, this folder reproduces results from [Brink (2010)](https://www.researchgate.net/publication/50405295_Topographic_rectification_in_a_forced_dissipative_barotropic_ocean). The following files and directories are included:

#### **code**
- **`brink_2010.jl`**: Julia script for running the simulation and generating output files.
- **`plot_brink_fig2.jl`**: Julia script to recreate Figure 2 from Brink (2010).
- **`plot_brink_fig3.jl`**: Julia script to recreate Figure 3 from Brink (2010).

#### **figures**
- **`brink_2010-300-fig2.png`**: Reproduced Figure 2.
- **`brink_2010-300-fig3.png`**: Reproduced Figure 3.

#### **output**
Example output files corresponding to run 300 (see Table 1 in Brink, 2010, for an overview of the runs):
- **`brink_2010-300_bathymetry.jld2`**: Bathymetric data.
- **`brink_2010-300.jld2`**: Model output fields. (Time resolution is 1 day to keep file sizes manageable.)


## Setting up the Julia environment

To run the provided scripts and simulations, you'll need to set up a Julia environment based on the `Project.toml` file in the repository. Follow these steps:

1. **Install Julia**:  
   Download and install Julia from the [official website](https://julialang.org/downloads/).

2. **Clone the repository**:  
   Open a terminal and run:
   ```bash
   git clone https://github.com/alpsjur/temporal-topo-flow.git
   cd temporal-topo-flow
   ```

3. **Activate the Julia environment**:  
   Launch Julia in the repository directory and activate the project:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
   This will install all required dependencies listed in `Project.toml`.

4. **Run scripts**:  
   Use Julia to execute the scripts. For example, to run the simulation `brink_2010.jl`:
   ```bash
   julia reproduce_brink_2010/code/brink_2010.jl
   ```

---
Enjoy!