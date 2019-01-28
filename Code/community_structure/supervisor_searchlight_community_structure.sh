#!/usr/bin/env bash
# Bash script to launch searchlight_community_structure.py for all participants. It checks whether this participant has been run and only runs it if they haven't been.

# Setup environment
source ./Code/setup_brainiak_environment.sh

input_path=$1 # Path to file
condition=$2 # What is the name that characterizes this participant

files=`ls $input_path*$condition*`

for file in $files
do

	output_name="${file/node_brain\//searchlights/}"

	if [ ! -e ${output_name} ]
	then 

		#Print
		echo "Running $file"

		# Run the command
		sbatch -p $short_partition ./Code/community_structure/run_searchlight_community_structure.sh $file
	fi

done
