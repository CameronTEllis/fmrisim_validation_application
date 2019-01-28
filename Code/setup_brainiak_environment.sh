#!/usr/bin/env bash
# Set up the brainiak environment so that brainiak commands will run
# numpy, anaconda, mpi and nibabel are important dependencies; however, read brainiak documentation for the full list

module load python/3.6
source activate brainiak_extras
module load fsl

short_partition=other
long_partition=other
