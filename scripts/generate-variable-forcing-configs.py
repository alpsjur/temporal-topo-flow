"""
This script generates multiple JSON configuration files for simulations with varying wind stress amplitude (tau0).
Each configuration file is named short_001.json, short_002.json, ..., short_020
"""

import json
import os


# Parameters
T = 1382400.0
tmax = 16588800

# Output directory
outdir = "configs/varying_forcing/"
os.makedirs(outdir, exist_ok=True)

# Tau0 values: first increments of 0.000025 up to 0.0001, then increments of 0.00005 up to 0.0008
tau0_values = []

# First range: 0.000025 to 0.0001 (inclusive), step 0.000025
tau0 = 0.000025
while tau0 <= 0.0001 + 1e-12:
    tau0_values.append(round(tau0, 8))
    tau0 += 0.000025

# Second range: 0.00015 to 0.0008 (inclusive), step 0.00005
tau0 = 0.00015
while tau0 <= 0.0008 + 1e-12:
    tau0_values.append(round(tau0, 8))
    tau0 += 0.00005

# Create JSON files
for i, tau0 in enumerate(tau0_values, start=1):
    data = {
        "name": f"short_{i:03d}",
        "T": T,
        "tau0": tau0,
        "tmax": tmax
    }
    filename = os.path.join(outdir, f"short_{i:03d}.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

print(f"Created {len(tau0_values)} JSON files in '{outdir}'")
