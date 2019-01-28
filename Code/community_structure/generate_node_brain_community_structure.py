# Simulate data containing a representation of community structure
#
# Read in the data of each participant, calculate the noise properties of
# their brain, then input the signal somewhere in their brain. Then store it
#  as an average representation per node
#
# Take in a participants preprocessed data, read in the timing file and
# create the node by node distance matrix.

import numpy as np
import nibabel
import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import copy
import sys
import utils
from scipy.stats import zscore
from brainiak.utils import fmrisim as sim
import logging

# What are the system inputs?

# What participant are you analyzing
if len(sys.argv) > 1:
    participant_counter = int(sys.argv[1])

if len(sys.argv) > 2:
    all_pos = float(sys.argv[2])
else:
    all_pos = 1

# How many seconds, if any, are you adding in between onsets?
if len(sys.argv) > 3:
    extra_isi = int(sys.argv[3])
else:
    extra_isi = 0

# What resample is it of this participant
if len(sys.argv) > 4:
    resample = int(sys.argv[4])
else:
    resample = 0

# Do you want to randomise the order of events within run
if len(sys.argv) > 5:
    randomise_timing = int(sys.argv[5])
else:
    randomise_timing = 0

# What is the percentage change in signal that these graphs represent?
if len(sys.argv) > 6:
    scale_percentage = float(sys.argv[6])
else:
    scale_percentage = 2

# Community density
if len(sys.argv) > 7:
    community_density = float(sys.argv[7])
else:
    community_density = 1

# Do you want to limit the scan duration
if len(sys.argv) > 8:
    restrict_overall_duration = int(sys.argv[8])
else:
    restrict_overall_duration = 0

logger = logging.getLogger(__name__)

# If it is not a resample, do you want to save some plots
save_signal_func = 1  # Save the plot of the signal function?
save_functional = 0  # Do you want to save the functional (with node_brain)

# Inputs for generate_signal
temporal_res = 100  # How many samples per second are there for timing files
tr_duration = 2  # Force this value
event_durations = [1.0]  # The events are 1s long

# Graph structure
nodes = 15  # How many nodes are to be created
# Do you want to always exclude the first 3 parameters (even when the 
# events may be very temporally separated), or do you want to just ensure 
# that there is at least 12s of downtime at the start of the hamilitonian path 
always_exclude_first_three_events = 1  

# Set analysis parameters
hrf_lag = 2 # How many TRs is the event offset by?
zscore_time = 1  # Do you want to z score each voxel in time
zscore_volume = 0  # Do you want to z score a volume

# What is the participant name with these parameters (change from default)
effect_name = '_pos-' + str(all_pos)
effect_name = effect_name + '_t-' + str(extra_isi)
effect_name = effect_name + '_r-' + str(randomise_timing)
effect_name = effect_name + '_density_' + str(community_density)
effect_name = effect_name + '_s-' + str(scale_percentage)
if restrict_overall_duration == 1:
    effect_name = effect_name + '_limit'

# Only add this at the end
if resample > 0:
    effect_name = effect_name + '_resample-' + str(resample)

# Specify the paths and names
parameters_path = './community_structure/simulator_parameters/'
simulated_data_path = './community_structure/simulated_data/'
timing_path = parameters_path + 'timing/'

participant = 'sub-' + str(participant_counter + 1)
SigMask = parameters_path + '/real_results/significant_mask.nii.gz'
savename = participant + effect_name
node_brain_save = simulated_data_path + '/node_brain/' + savename + \
                  '.nii.gz'

# Load data
print('Loading ' + participant)

# Load significant voxels
nii = nibabel.load(SigMask)
signal_mask = nii.get_data()  # Takes a while

dimsize = nii.header.get_zooms()  # x, y, z and TR size

# Load in the timing information but deal with it differently if you want 
# to exclude different numbers of events from the start of a hamiltonian 
# path depending on the ISI.
if always_exclude_first_three_events == 1:
    # Load the timing information
    onsets_runs = np.load(timing_path + participant + '.npy')
else:
    # Generate the timing for this participant
    runs = 5
    repetitions_per_run = 5  # How many times do you loop per run

    average_threshold = 12  # What is the average time cut off from the
    # loop start (based on Schapiro: tr duration * middle_isi * 3 excluded
    # events)

    excluded_events = average_threshold / (extra_isi + (tr_duration * 2))
    nodes_per_run = int(nodes - np.ceil(excluded_events))  # How many events
    #  are included?

    onsets_runs = [-1] * runs
    for run_counter in list(range(runs)):
        current_time = 0  # Initialize
        onsets = [-1] * nodes  # Ignore first entry
        for hamilton_counter in list(range(0, repetitions_per_run)):

            # Pick on this hamilton the starting node and the direction
            node = np.random.randint(0, nodes - 1)
            direction = np.random.choice([-1, 1], 1)[0]

            # Loop through the events
            for node_counter in list(range(0, nodes_per_run)):

                # Append this time
                if np.all(onsets[node] == -1):
                    onsets[node] = np.array(current_time)
                else:
                    onsets[node] = np.append(onsets[node],
                                             current_time)

                # What is the isi (between 1 and 5) ?
                isi = (np.random.randint(3) * tr_duration) + 1

                # Increment the time
                current_time += event_durations[0] + isi

                # Update the node
                node += direction
                if node >= nodes:
                    node = 0
                elif node < 0:
                    node = (nodes - 1)

        # Store the runs
        onsets_runs[run_counter] = np.asarray(onsets)

# Generate the indexes of all voxels that will contain signal
vector_size = int(signal_mask.sum())

# Find all the indices that contain signal
idx_list = np.where(signal_mask == 1)

idxs = np.zeros([vector_size, 3])
for idx_counter in list(range(0, len(idx_list[0]))):
    idxs[idx_counter, 0] = int(idx_list[0][idx_counter])
    idxs[idx_counter, 1] = int(idx_list[1][idx_counter])
    idxs[idx_counter, 2] = int(idx_list[2][idx_counter])

idxs = idxs.astype('int8')

# What voxels are they
dimensions = signal_mask.shape

# Cycle through the runs and generate the data
node_brain = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                       nodes, 5], dtype='double')  # Preset

# Generate the graph structure (based on the ratio)
signal_coords = utils.community_structure(1 - community_density,
                                          )

# Perform an orthonormal transformation of the data
if vector_size > signal_coords.shape[1]:
    signal_coords = utils.orthonormal_transform(vector_size,
                                                signal_coords,
                                                )

# Do you want these coordinates to be all positive? This means that
# these coordinates are represented as different magnitudes of
# activation
if all_pos == 1:
    mins = np.abs(np.min(signal_coords, 0))
    for voxel_counter in list(range(0, len(mins))):
        signal_coords[:, voxel_counter] += mins[voxel_counter]

# Bound the value to have a max of 1 so that the signal magnitude is more interpretable
signal_coords /= np.max(signal_coords)

for run_counter in list(range(1, 6)):

    # Get run specific names
    template_name = parameters_path + 'template/' + participant + '_r' + \
                    str(run_counter) + '.nii.gz'
    noise_dict_name = parameters_path + 'noise_dict/' + participant + '_r' + str(run_counter) + \
                      '.txt'
    nifti_save = simulated_data_path + 'nifti/' + participant + '_r' + str(run_counter)\
                 + effect_name + '.nii.gz'
    signal_func_save = './community_structure/plots/' + participant + '_r' +\
                       str(run_counter) + effect_name + '.eps'

    # Load the template (not yet scaled
    nii = nibabel.load(template_name)
    template = nii.get_data()  # Takes a while

    # Create the mask and rescale the template
    mask, template = sim.mask_brain(template,
                                    mask_self=True,
                                    )

    # Pull out the onsets for this participant (copy it so you don't alter it)
    onsets = copy.deepcopy(onsets_runs[run_counter - 1])

    # What is the original max duration of the onsets
    max_duration_orig = np.max([np.max(onsets[x]) for x in range(onsets.size)])
    max_duration_orig += 10 # Add some wiggle room

    # Do you want to randomise the onsets (so that the events do not have a
    # fixed order)
    if randomise_timing == 1:
        onsets = utils.randomise_timing(onsets,
                                        )

    # If you want to use different timing then take the order of the data
    # and then create a new timecourse
    onsets = utils.extra_isi(onsets,
                             extra_isi,
                             )

    # If necessary, remove all the values greater than the max
    if restrict_overall_duration == 1:
        onsets = [onsets[x][onsets[x] < max_duration_orig] for x in range(
            onsets.size)]
        onsets = np.asarray(onsets)

    # Determine how long the simulated time course is by finding the max of maxs
    last_event = 0
    for node_counter in range(len(onsets)):
        if len(onsets[node_counter]) > 0 and onsets[node_counter].max() > last_event:
            last_event = onsets[node_counter].max()    

    duration = int(last_event + 10) # Add a decay buffer

    # Specify the dimensions of the volume to be created
    dimensions = np.array([template.shape[0], template.shape[1],
                           template.shape[2], int(duration / tr_duration)])

    # Load the noise parameters in
    with open(noise_dict_name, 'r') as f:
        noise_dict = f.read()

    noise_dict = eval(noise_dict)
    stimfunc_all = []
    for node_counter in list(range(0, nodes)):

        print('Node ' + str(node_counter))

        # Preset value
        volume = np.zeros(dimensions[0:3])

        # Preset the signal
        signal_pattern = signal_coords[node_counter, :]
        onsets_node = onsets[node_counter]

        # Only do it if there are onsets
        if len(onsets_node) > 0:
     
            # Create the time course for the signal to be generated
            stimfunc = sim.generate_stimfunction(onsets=onsets_node,
                                                 event_durations=event_durations,
                                                 total_time=duration,
                                                 temporal_resolution=temporal_res,
                                                 )

            # Aggregate the timecourse
            if len(stimfunc_all) == 0:
                stimfunc_all = np.zeros((len(stimfunc), vector_size))
                for voxel_counter in list(range(0, vector_size)):
                    stimfunc_all[:, voxel_counter] = np.asarray(
                        stimfunc).transpose() * signal_pattern[voxel_counter]
            else:

                # Add these elements together
                temp = np.zeros((len(stimfunc), vector_size))
                for voxel_counter in list(range(0, vector_size)):
                    temp[:, voxel_counter] = np.asarray(stimfunc).transpose() * \
                                             signal_pattern[voxel_counter]

                stimfunc_all += temp

    # After you have gone through all the nodes, convolve the HRF and
    # stimulation for each voxel
    print('Convolving HRF')
    signal_func = sim.convolve_hrf(stimfunction=stimfunc_all,
                                   tr_duration=tr_duration,
                                   temporal_resolution=temporal_res,
                                   )

    if save_signal_func == 1 and resample == 0 and run_counter == 0:
        plt.plot(stimfunc_all[::int(temporal_res * tr_duration), 0])
        plt.plot(signal_func[:,0])
        plt.xlim((0, 200))
        plt.ylim((-1, 5))
        plt.savefig(signal_func_save)

    # Convert the stim func into a binary vector of dim 1
    stimfunc_binary = np.mean(np.abs(stimfunc_all)>0, 1) > 0
    stimfunc_binary = stimfunc_binary[::int(tr_duration * temporal_res)]

    # Bound, can happen if the duration is not rounded to a TR
    stimfunc_binary = stimfunc_binary[0:signal_func.shape[0]]

    # Create the noise volumes (using the default parameters)
    noise = sim.generate_noise(dimensions=dimensions[0:3],
                               stimfunction_tr=stimfunc_binary,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=noise_dict,
                               )

    # Change the type of noise
    noise = noise.astype('double')

    # Create a noise function (same voxels for signal function as for noise)
    noise_function = noise[idxs[:, 0], idxs[:, 1], idxs[:, 2], :].T

    # Compute the signal magnitude for the data
    signal_func_scaled = sim.compute_signal_change(signal_function=signal_func,
                                                   noise_function=noise_function,
                                                   noise_dict=noise_dict,
                                                   magnitude=[
                                                       scale_percentage],
                                                   method='PSC',
                                                   )
    
    # Multiply the voxels with signal by the HRF function
    brain_signal = sim.apply_signal(signal_function=signal_func_scaled,
                                    volume_signal=signal_mask,
                                    )

    # Convert any nans to 0s
    brain_signal[np.isnan(brain_signal)] = 0

    # Combine the signal and the noise
    brain = brain_signal + noise

    # Save the participant data
    if save_functional == 1 and resample == 0:

        print('Saving ' + nifti_save)
        brain_nifti = nibabel.Nifti1Image(brain, nii.affine)

        hdr = brain_nifti.header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], 2.0))
        nibabel.save(brain_nifti, nifti_save)  # Save

    # Z score the data
    if zscore_time == 1:
        brain = zscore(brain, 3)

    # Mask brain
    brain = brain * mask.reshape((mask.shape[0], mask.shape[1], mask.shape[
        2], 1))

    # Average the timepoints corresponding to each node
    for node in list(range(0, nodes)):

        node_trs = onsets[node]

        # Z score all the TRs where a node was created
        temp = np.zeros((dimensions[0], dimensions[1], dimensions[2],
                        len(node_trs)))
        for tr_counter in list(range(0, len(node_trs))):

            # When does it onset (first TR is zero so minus 1)
            onset = int(np.round(node_trs[tr_counter] / tr_duration) +
                        hrf_lag)

            if onset < brain.shape[3]:
                m = np.mean(brain[:, :, :, onset])
                s = np.std(brain[:, :, :, onset])

                # Do you want to z score the volumes
                if zscore_volume == 1:
                    temp[:, :, :, tr_counter-1] = (brain[:, :, :, onset] - m) / s
                else:
                    temp[:, :, :, tr_counter - 1] = brain[:, :, :, onset]

        # Average the TRs
        node_brain[:, :, :, node, run_counter - 1] = np.mean(temp, 3)

print('Wrapping up')

# average the brains across runs
node_brain = np.mean(node_brain, 4)

# Mask again
node_brain = node_brain * mask.reshape(dimensions[0], dimensions[
    1], dimensions[2], 1)

# Save the volume
brain_nifti = nibabel.Nifti1Image(node_brain.astype('double'), nii.affine)

hdr = brain_nifti.header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], tr_duration))
nibabel.save(brain_nifti, node_brain_save)  # Save
print('Saving ' + node_brain_save)
