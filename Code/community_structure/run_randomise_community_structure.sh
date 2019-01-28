#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/merge_randomise-%j.out
#SBATCH -p all
#SBATCH -t 30
#SBATCH --mem 2000

# Specify the inputs
input_folder=$1 # Where is the data stored
condition=$2 # What identifies the volumes to be concatenated. Make sure this has the extension on the end, if not, add it it

if [ ${condition:-1} != z ]
then
condition=${condition}.nii.gz
fi

echo Looking for $input_folder/*$condition


output_folder="${input_folder/searchlights\//randomise/}"

fslmerge -t ${output_folder}/input_${condition} ${input_folder}/*${condition}

participants=`fslval ${output_folder}/input_${condition} dim4`
if [ $participants -ne 20 ]
then
echo $participants participants found. Aborting
exit
fi

# What is the participant condition
condition=`echo $condition | rev | cut -c 8- | rev`

# Make the mean file
fslmaths ${output_folder}/input_${condition}.nii.gz -Tmean ${output_folder}/output_${condition}_mean.nii.gz

# Make the randomise
randomise -i ${output_folder}/input_${condition}.nii.gz -o ${output_folder}/output_${condition} -1 -n 1000 -x -T -C 2.09 -v 6

