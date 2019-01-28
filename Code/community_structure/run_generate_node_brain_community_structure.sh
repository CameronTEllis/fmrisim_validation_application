#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/generate_community_structure-%j.out
#SBATCH -t 360
#SBATCH --mem 20000

# Setup environment
source ./Code/setup_brainiak_environment.sh

# Inputs are:
#1. Participant number (0 to 20)
#2. Are all the signal coordinates positive
#3. Extra time between onsets
#4. Resamples
#5. Randomize order
#6. Size of the scaling factor
#7. Specify the density of the community structure
#8. Do you want to limit the duration of the run to match the original, even if you extend the ISI?

python ./Code/community_structure/generate_node_brain_community_structure.py $1 $2 $3 $4 $5 $6 $7 $8
