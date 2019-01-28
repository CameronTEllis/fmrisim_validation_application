# Validation and application analyses of fmrisim

fmrisim is a Python toolbox for realistic simulation of fMRI data. This toolbox can estimate the noise parameters from fMRI and create synthetic data with approximately matched noise properties. 

To evaluate the efficacy of this toolbox we performed two sets of validation analyses: to what extent is simulated data like real data that it is based on. An example application of fmrisim is also described here in which we can evaluate how different design decisions affect power. To show this we simulate data from Schapiro, Rogers, Cordova, Turk-Browne, & Botvinick (2013) with changes to their original design, such as a different ISI or stimulus order, and compared these new simulated results with their original results.

For more details on these analyses refer to the following manuscript:  

Cameron T. Ellis, Christopher Baldassano, Anna C. Schapiro, Ming Bo Cai,  & Jonathan D. Cohen, Faclitating open-science with realistic fMRI simulation: validation and application.

Data for validation analyses can be retrieved from online: http://arks.princeton.edu/ark:/88435/dsp01dn39x4181. Necessary data for Application analyses is included.

Scripts should be run from the repo directory (referred to elsewhere as ~/) and they assume that you are using a slurm scheduler.

To run these functions, you must have BrainIAK installed. Depending on your installation method, you will need to edit the Code/setup_brainiak_environment.sh script. Moreover for some analyses you will need FSL. To execute the notebooks, all of the necessary information is provided.
