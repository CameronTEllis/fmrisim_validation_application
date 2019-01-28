#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/searchlight_analysis-%j.out
#SBATCH -t 30
#SBATCH --mem 40000
#SBATCH -n 2

# Set up environment
source ./Code/setup_brainiak_environment.sh

# Inputs are:
# 1: Participant (full path to node_brain file)
# 2: Analysis type: clusters, clusters_norm, loops, loop_counter
srun -n $SLURM_NTASKS --mpi=pmi2 python ./Code/community_structure/searchlight_community_structure.py $1 $2
