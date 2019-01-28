#!/bin/bash
# Search over a number of different signal parameters relevant for determining the effect of delay (both with limited experiment duration and without) and randomization using generate_signal_community_structure.sh. 
#
#SBATCH --output=logs/supervisor_signal_fit-%j.out
#SBATCH -t 2000
#SBATCH --mem 2000

# Input the variables
#1. Are all the signal coordinates positive
#2. Extra time between onsets
#3. Randomize order
#4. Resample number
#5. Size of the scaling factor
#6. Specify the density of the community structure
#7. Do you want to put a limit on the duration (so that there are fewer trials with longer d$
#8. Output_file

pos=1.0
resamples=10
permutation=0
limit=1
output_file=community_structure/delay/delay.txt

job_limit=500 # Limit how many jobs can be submitted before running this. Otherwise you will have too many scripts that just launch the

# How many times do you want to rerun this whole pipeline
for resample in `seq 1 $resamples`
do 

# What is the density of the community structure
for density in 0.0 0.5 1.0
do 

# Based on the density, determine what the signal ought to be (determined empirically based on signal_fit analyses)
if [ "$density" == "0.0" ]
then
signal=0.5
elif [ "$density" == "0.5" ]
then
signal=0.35
elif [ "$density" == "1.0" ]
then
signal=0.25
fi

# Loop through different ISI
for t in `seq 0 10`
do

# Specify whether you want to randomize the hamiltionian (1) or not (0)
for randomise in 0 #1
do

job_num=`squeue -u $USER | wc -l`

# Sleep Until there few enough jobs running
while [ $job_num -gt $job_limit ]
do
job_num=`squeue -u $USER | wc -l`
sleep 5s
done

# Specify the conditions that you will be running to check whether that file exists
condition="t-${t}_r-${randomise}_density_${density}_s-${signal}"
if [ $limit -eq 1 ]
then
condition=${condition}_limit_resample-${resample}
else
condition=${condition}_resample-${resample}
fi
file_txt=`cat community_structure/delay/delay.txt` # What is the file content

# Check the file txt to see if it exists
if [[ $file_txt != *${condition}* ]]
then

# Wait until enough jobs have finished before launching
sbatch Code/community_structure/generate_signal_community_structure.sh $pos $t $randomise $resample $signal $density $limit $output_file

# Wait before launching
#sleep 60s
else
echo ${condition} already exists
fi

done 
done
done
done
