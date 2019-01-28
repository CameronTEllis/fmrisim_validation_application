# Code to explore design choices from Schapiro et al., (2013) using fmrisim

This folder contains the code necessary to explore different design choices, such as ISI, using fmrisim to generate realistic noise models of the data.

The functions stored in this folder are described below: 

> generate_node_brain_community_structure.py: this script generates fMRI data based on the noise parameters stored in '~/community_structure/simulator_parameters/' and the signal parameters provided as inputs. The data is then averaged in order to extract node/stimulus specific brain responses.  
>  
>run_generate_node_brain_community_structure.sh: bash script to launch generate_node_brain_community_structure.py for an individual participant.  
>  
> supervisor_generate_node_brain_community_structure.sh: bash script to launch generate_node_brain_community_structure.py for all participants. It checks whether this participant has been run and only runs it if they haven't been.  
>  
>searchlight_community_structure.py: Uses brainiak to perform a voxelwise searchlight analysis on the output of generate_node_brain_community_structure.py that compares the within vs between community distance.  
>  
> run_searchlight_community_structure.sh: bash script to launch searchlight_community_structure.py for an individual participant.  
>  
> supervisor_searchlight_community_structure.sh: bash script to launch searchlight_community_structure.py for all participants. It checks whether this participant has been run and only runs it if they haven't been.  
>  
> run_randomise_community_structure.sh: Perform a permutation test with FSL randomise on the data to extract the test statistic for a cohort of participants.  
>  
> generate_signal_community_structure.sh: Perform the pipeline laid out above for a given set of signal parameters - supervisor_generate_node_brain_community_structure.sh, supervisor_searchlight_community_structure.sh, and run_randomise_community_structure.sh.  
>  
> supervisor_signal_fit_community_structure.sh: Search over a number of different signal parameters relevant for fitting the signal of the data using generate_signal_community_structure.sh.  
>  
> supervisor_delay_community_structure.sh: Search over a number of different signal parameters relevant for determining the effect of delay (both with limited experiment duration and without) and randomization using generate_signal_community_structure.sh.  
>  
> plot_commity_structure.ipynb: Notebook used for exploring the results of the analysis and producing the figures used in the manuscript  
>  
> utils: Various functions used throughout.  

Hence to replicate the analysis described in the manuscript, all you need to do is run *supervisor_signal_fit_community_structure.sh* and *supervisor_delay_community_structure.sh*
