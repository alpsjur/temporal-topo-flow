#!/bin/bash

# Define the filename variable
FILENAME="shallow_water.jl"
CONFIGS=("slope-no_bumps-no_noise-reversed_tau.jl" "slope-no_bumps-noise-reversed_tau.jl" "slope-bumps-no_noise-reversed_tau.jl" "slope-bumps-noise-reversed_tau.jl")

SUCSESS="Skript ferdig! :) \n\nHilsen\n$(hostname)"
FAIL="Oi, nå har det skjedd noe galt. Skript feila :( \nDet går bra, dette fikser du! \n\nHilsen\n$(hostname)"

# multithreading 
export JULIA_NUM_THREADS=4

for CONFIG in "${CONFIGS[@]}"
do
    nice julia --project=. $FILENAME "configs/$CONFIG" && echo -e "$SUCSESS" | mail -s "$FILENAME $CONFIG" alsjur@uio.no || echo -e "$FAIL" | mail -s "$FILENAME $CONFIG" alsjur@uio.no 
done

