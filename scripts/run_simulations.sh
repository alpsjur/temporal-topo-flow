#!/bin/bash

# Ask for user email
read -p "Enter your email address for a notification when the simulation is done: " user_email

# Confirm input (optional)
echo "Email entered: $user_email"

# Create output/raw directory if it doesn't exist
mkdir -p output/raw

# Define the filename variable
FILENAME="scripts/simulation.jl"

SUCSESS="Skript done! :) \n\nBest,\n$(hostname)"
FAIL="Something went wrong with the script :( No worries, you'll fix this \n\nBest\n$(hostname)"


for CONFIG in configs/run/*
do
    nice julia --project=. $FILENAME $CONFIG && echo -e "$SUCSESS" | mail -s "$FILENAME $CONFIG" $user_email || echo -e "$FAIL" | mail -s "$FILENAME $CONFIG" $user_email
done