# Code to validate the noise generation of fmrisim

This folder contains the code to generate and test the noise parameters that fmrisim can estimate and generate. This assumes that you have put the [Corr_MVPA dataset (Bejjanki, Da Silveira, Cohen, & Turk-Browne, 2017)](http://arks.princeton.edu/ark:/88435/dsp01dn39x4181) in the ~/Validation/real_data/ folder.

The functions stored in this folder are described below:

> generate_noise_validation.py: Take in a real fMRI volume from ~/Validation/real_data/ and compute the noise properties of the volume. Then simulate a new volume with those same noise properties and store that. Finally, estimate the noise properties of the simulation and store those noise properties.  
>  
> run_generate_noise_validation.sh: bash script to launch generate_noise_validation.py for an individual participant.  
> 
> supervisor_generate_noise_validation.sh: bash script to launch generate_noise_validation.py for all participants, all runs, as well as multiple iterations.  
>  
> testing_noise_calc_noise_validation.py: Take in the noise parameter estimates from real and simulated participants and compare them.  
>  
> run_testing_noise_calc_noise_validation.sh: bash script to launch testing_noise_calc_noise_validation.py for a single participant.
>  
> plot_noise_validation.ipynb: Explore the data and reproduce the plots reported in the manuscript.  
>  

Hence to replicate the analyses in the associated manuscript you must run Code/Validation/supervisor_generate_noise_validation.sh and then Code/Validation/run_testing_noise_calc_noise_validation.sh.
