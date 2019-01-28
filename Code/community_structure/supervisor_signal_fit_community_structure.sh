#!/bin/bash
#
# Search over a number of different signal parameters relevant for fitting the signal of the data using generate_signal_community_structure.sh.
#
#SBATCH --output=logs/supervisor_signal_fit-%j.out
#SBATCH -t 2000
#SBATCH --mem 2000

source Code/setup_brainiak_environment.sh

# Input the variables
#1. Are all the signal coordinates positive
#2. Extra time between onsets
#3. Randomize order
#4. Resample number
#5. Size of the scaling factor
#6. Specify the density of the community structure
#7. Do you want to put a limit on the duration (so that there are fewer trials with longer d$
#8. Output file

pos=1.0
t=0
randomise=0
resample=0
permutation=0
limit=0
output_file=community_structure/signal_fit/signal_fit.txt

job_limit=200 # Limit how many jobs can be submitted before running this. Otherwise you will have too many scripts that just launch the

for signal in `seq 0.0 0.1 0.55`
do 
for density in `seq 0.0 0.1 1`
do 

job_num=`squeue -u $USER | wc -l`

# Sleep Until there few enough jobs running
while [ $job_num -gt $job_limit ]
do
	job_num=`squeue -u $USER | wc -l`
	sleep 5s
done

# Wait until enough jobs have finished before launching
sbatch -p $short_partition Code/community_structure/generate_signal_community_structure.sh $pos $t $randomise $resample $signal $density $limit $output_file

# Wait before launching
sleep 60s	

done; 
done
