#!/bin/bash
# bash script to launch generate_noise_validation.py for all participants, all runs, as well as multiple iterations.

# Setup the environment
source ./Code/setup_brainiak_environment.sh

# Cycle through the participants and runs and make the files, one that is the template and many that are resampling.
resamples=10
for subj in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17
do
	for run in 01 02
	do
		input_volume=Validation/real_data/Participant_${subj}_rest_run${run}.nii
		input_noise_dict=Validation/real_noise_dict/Participant_${subj}_rest_run${run}.txt
		mask=Validation/masks/Participant_${subj}_rest_run${run}.nii.gz
		output_noise_dict=Validation/simulated_noise_dict/Participant_${subj}_rest_run${run}.txt
		output_volume=Validation/simulated_data/Participant_${subj}_rest_run${run}.nii.gz

		# Submit as a job
		jid_init=`sbatch -p $short_partition ./Code/Validation/run_generate_noise_validation.sh $input_volume $input_noise_dict $mask ${output_noise_dict} ${output_volume} 1`
		
		for resample in `seq 1 $resamples`
		do
			output_noise_dict_resample=Validation/simulated_noise_dict_resample/Participant_${subj}_rest_run${run}_resample-${resample}.txt
			
			# Wait until the noise dict has been created (otherwise it may overwite)
			sbatch --dependency=afterok:${jid_init:20} -p $short_partition ./Code/Validation/run_generate_noise_validation.sh $input_volume $input_noise_dict $mask ${output_noise_dict_resample} None 1

		done
	done
done
