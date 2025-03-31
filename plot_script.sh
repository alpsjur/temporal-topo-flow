#!/bin/bash
#FILENAMES=("slope/code/plot_xmomentumterms.py" "slope/code/plot_residual_flow.py" "slope/code/plot_Hmomentumterms.py" "slope/code/plot_Hcontour_analysis.py" "slope/code/plot_streamline_evolution.py")
FILENAMES=("slope/code/plot_Hcontour_analysis.py")


for FILENAME in "${FILENAMES[@]}"
do
    for CONFIG in slope/configs/*
    do
        python "$FILENAME" "$CONFIG"
    done
done
