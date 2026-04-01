#!/bin/bash

# Script to animate model output for multiple configurations.
# Run from command line:
#     bash scripts/run_animate_fileds.sh


# spesify which configs to animate
CONFIGS=("configs/baseline_forcing/short.json" "configs/baseline_forcing/long.json")

# loop over configs and run animation script
for CONFIG in "${CONFIGS[@]}"
do
    python scripts/animate_fields.py $CONFIG
done