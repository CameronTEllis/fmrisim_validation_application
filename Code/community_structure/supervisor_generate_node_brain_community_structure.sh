#!/usr/bin/env bash
# Generate data for all participants for this specific parameter configuration

# Setup environment
source ./Code/setup_brainiak_environment.sh

# Inputs are:
#1. Are all the signal coordinates positive
#2. Extra time between onsets
#3. Resamples
#4. Randomize order
#5. Size of the scaling factor
#6. Specify the density of the community structure

all_pos=$1
event_delay=$2
resamples=$3
randomise_onsets=$4
scale_percentage=$5
community_density=$6
limit=$7

# Move into the folder
for participant in `seq 0 19`
do
    # Check to see if the file exist
    ppt=$((ppt+1))
    if [ $limit -eq 1 ]
    then
         limit_name='_limit'
    fi
    
    # Compose the output name
    output_name=community_structure/simulated_data/node_brain/sub-${ppt}_pos-${all_pos}_t-${event_delay}_r-${randomise_onsets}_density_${community_density}_s-${scale_percentage}${limit_name}_resample-${resamples}.nii.gz

    # Check
    if [ ! -e $output_name ]
    then
	echo Creating $output_name
        sbatch -p $short_partition ./Code/community_structure/run_generate_node_brain_community_structure.sh $participant $all_pos $event_delay $resamples $randomise_onsets $scale_percentage $community_density $limit
    else
	echo Skipping $output_name
    fi
done
