#!/bin/bash

# spesify which configs to animate
CONFIGS=("configs/baseline_forcing/short.json" "configs/baseline_forcing/long.json")

# loop over configs and run animation script
for CONFIG in "${CONFIGS[@]}"
do
    python scripts/animate_fields.py $CONFIG
done