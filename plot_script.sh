#!/bin/bash
FILENAMES=("slope/code/plot_Hcontour_analysis.py" "slope/code/plot_streamline_evolution.py")

conda activate general
for FILENAME in "${FILENAMES[@]}"
do
    for CONFIG in slope/configs/*
    do
        python "$FILENAME" "$CONFIG"
    done
done
