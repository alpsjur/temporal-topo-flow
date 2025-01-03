#!/bin/bash

# Define the filename variable
FILENAME="slope/code/simulation.jl"

SUCSESS="Skript ferdig! :) \n\nHilsen\n$(hostname)"
FAIL="Oi, nå har det skjedd noe galt. Skript feila :( \nDet går bra, dette fikser du! \n\nHilsen\n$(hostname)"

# multithreading 
export JULIA_NUM_THREADS=4

for CONFIG in slope/configs/run/*
do
    nice julia --project=. $FILENAME $CONFIG && echo -e "$SUCSESS" | mail -s "$FILENAME $CONFIG" alsjur@uio.no || echo -e "$FAIL" | mail -s "$FILENAME $CONFIG" alsjur@uio.no 
done

