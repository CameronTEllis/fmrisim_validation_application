#!/bin/bash
#
# Run through the process of generating and doing super subject analyses on participants
#
#SBATCH --output=logs/signal_fit-%j.out
#SBATCH -t 1000
#SBATCH --mem 2000

# Set up environment
source Code/setup_brainiak_environment.sh

# Input the variables
#1. Are all the signal coordinates positive
#2. Extra time between onsets
#3. Randomize order
#4. Resample number
#5. Size of the scaling factor
#6. Specify the density of the community structure
#7. Do you want to put a limit on the duration (so that there are fewer trials with longer delays)
#8. Output name

if [ $# -eq 0 ]
then
	pos=1.0
	t=0
	r=0
	permutation=0
	s=2.0
	density=1
	limit=0
else
	pos=$1 # should be a float
	t=$2
	r=$3
        permutation=$4
	s=$5 # should be a float
	density=$6
	limit=$7
	output_file=$8
fi

base_path=$(pwd)
code_path=${base_path}/Code/community_structure/
node_path=${base_path}/community_structure/simulated_data/node_brain/
searchlights_path=${base_path}/community_structure/simulated_data/searchlights/
randomise_path=${base_path}/community_structure/simulated_data/randomise/
real_data_path=${base_path}/community_structure/simulator_parameters/real_results/
output_path=${base_path}/community_structure/signal_fit/

# Specify the condition name
condition=pos-${pos}_t-${t}_r-${r}_density_${density}_s-${s}

if [ $limit -eq 1 ]
then
        condition=${condition}_limit
fi


if [ $permutation != 0 ]
then
	condition=${condition}_resample-${permutation}
fi

# Run randomise on this data
new_randomise=${randomise_path}/output_${condition}_tstat1.nii.gz
if [ ! -e $new_randomise ]
then
	# Wait to finish
	participant_num=`ls ${node_path}/*$condition.nii.gz | wc -l`

	# Have you made all the participants already?
	if [ $participant_num -ne 20 ]
	then

		${code_path}/supervisor_generate_node_brain_community_structure.sh ${pos} ${t} $permutation ${r} ${s} ${density} ${limit}

		while [ $participant_num -ne 20 ]
		do
				participant_num=`ls ${node_path}/*${condition}.nii.gz | wc -l`
				sleep 30s
		done
	fi


	# Perform searchlights if you haven't already
	participant_num=`ls ${searchlights_path}/*${condition}.nii.gz | wc -l`

	if [ $participant_num -ne 20 ]
	then

		${code_path}/supervisor_searchlight_community_structure.sh ${node_path} ${condition}

		while [ $participant_num -ne 20 ]
		do
				participant_num=`ls ${searchlights_path}/*${condition}.nii.gz | wc -l`
				sleep 30s
		done
	fi

	# Run randomise
	${code_path}/run_randomise_community_structure.sh ${searchlights_path} ${condition}
	
else
	echo Randomise has already been created, not running
fi

# What is the original randomise output
original_randomise=${real_data_path}/permuted_t_stat.nii.gz
mask=${real_data_path}/significant_mask.nii.gz

fslmaths $new_randomise -sub $original_randomise -mas $mask -nan temp_${condition}.nii.gz
difference=`fslstats temp_${condition}.nii.gz -M`

# Report the difference
echo ${condition}: ${difference} >> $output_file

# Delete all the files generated
#rm -f ${searchlights_path}/*${condition}.nii.gz ${base_path}/community_structure/simulated_data/nifti/*${condition}* ${node_path}/*$condition.nii.gz ${randomise_path}/input_${condition}.nii.gz ${randomise_path}/output_${condition}_vox*.nii.gz ${randomise_path}/output_${condition}_clusterm*.nii.gz
