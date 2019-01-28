#!/usr/bin/env bash
# run_generate_noise_validation.sh: bash script to launch generate_noise_validation.py for an individual participant.

#SBATCH --output=logs/generate_noise_validation-%j.out
#SBATCH -t 360
#SBATCH --mem 5000

source ./Code/setup_brainiak_environment.sh

# Inputs are (full paths):
# 1: Input volume of real data
# 2: Output name for initial noise dict based on the real data
# 3: Output Mask name of the real data
# 4: Output name for generated noise dict based on the simulation
# 5: Output volume of the simulated data
# 6: Whether or not to fit the noise
python ./Code/Validation/generate_noise_validation.py $1 $2 $3 $4 $5 $6
