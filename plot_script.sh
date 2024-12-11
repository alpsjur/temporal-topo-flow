#!/bin/bash
FILENAMES=("slope/code/plot_circulation_Hcontour.py", "slope/code/plot_streamline_evolution.py")


for FILENAME in FILENAMES
do
    for CONFIG in slope/configs/*
    do
        python $FILENAME $CONFIG 
    done
done
