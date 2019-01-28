
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp_distance
import sklearn.manifold as manifold
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path
sys.path.append('/Users/cellis/Documents/MATLAB/Analysis_BrainIAK/')
from brainiak.utils import fmrisim as sim
import copy
from scipy.stats import zscore

# Perform an orthonormal transformation of the data
def orthonormal_transform(vector_size,
                          signal_coords):

    # Find normal vectors
    ortho_normal_matrix = np.zeros((signal_coords.shape[1], vector_size))
    for vec_counter in list(range(signal_coords.shape[1])):
        vector = np.random.randn(vector_size)

        # Combine vectors
        ortho_normal_matrix[vec_counter, :] = vector

    # Orthogonalize the vectors using the Gram-Schmidt method
    def gramschmidt(input):
        output = input[0:1, :].copy()
        for i in range(1, input.shape[0]):
            proj = np.diag(
                (input[i, :].dot(output.T) / np.linalg.norm(output,
                                                            axis=1) ** 2).flat).dot(
                output)
            output = np.vstack((output, input[i, :] - proj.sum(0)))
        output = np.diag(1 / np.linalg.norm(output, axis=1)).dot(output)
        return output.T

    ortho_normal_matrix = gramschmidt(ortho_normal_matrix)

    # Re store the coordinates in the new dimensional space
    new_coords = np.dot(ortho_normal_matrix, np.transpose(signal_coords))
    return np.transpose(new_coords)


# Specify some definitions
def community_structure(comm_spread=1.0,
                        ):
    # Generate the coordinates of each point specifying the community
    # structure of the data.

    # First specify the points of a pentagon (vertically symmetric,
    # one point on top) with origin 0,0 and radius R.

    pent_coord = np.zeros(shape=[5, 2])
    tri_coord = np.zeros(shape=[3, 2])

    # What is the triangle radius that defines the centre of the clusters
    # Value is set in order to make all nodes equidistant at stride 1
    tri_dist = 1.8270909

    # What angle is the start
    start_angle = 0

    for coord_counter in list(range(0, 5)):
        angle = (((start_angle + (72 * coord_counter)) / 180) * np.pi)

        x = np.sin(angle)
        y = np.cos(angle)

        pent_coord[coord_counter, :] = [x, y]

    # Specify the origins of each pentagon in a triangle
    bottom_y = -1 * tri_dist * np.sin((30 / 180) * np.pi)
    side_x = (tri_dist * np.cos((30 / 180) * np.pi))
    top_y = np.sqrt((side_x) ** 2 + bottom_y ** 2)

    # Set the points (rotating clockwise from the top)
    tri_coord[0, :] = [0, top_y]
    tri_coord[1, :] = [side_x, bottom_y]
    tri_coord[2, :] = [-side_x, bottom_y]

    # Make two copies of the pentagon with the points rotated
    pent_1_coord = pent_coord
    pent_2_coord = np.zeros(shape=[5, 2])
    pent_3_coord = np.zeros(shape=[5, 2])
    pent_2_coord[:, 0] = np.cos(-2 * np.pi / 3) * pent_coord[:, 0] - \
                         np.sin(-2 * np.pi / 3) * pent_coord[:, 1]
    pent_2_coord[:, 1] = np.sin(-2 * np.pi / 3) * pent_coord[:, 0] + \
                         np.cos(-2 * np.pi / 3) * pent_coord[:, 1]
    pent_3_coord[:, 0] = np.cos(2 * np.pi / 3) * pent_coord[:, 0] - \
                         np.sin(2 * np.pi / 3) * pent_coord[:, 1]
    pent_3_coord[:, 1] = np.sin(2 * np.pi / 3) * pent_coord[:, 0] + \
                         np.cos(2 * np.pi / 3) * pent_coord[:, 1]

    # Translate the points
    pent_1_coord[:, 0] = tri_coord[0, 0] + pent_1_coord[:, 0] * comm_spread
    pent_1_coord[:, 1] = tri_coord[0, 1] + pent_1_coord[:, 1] * comm_spread
    pent_2_coord[:, 0] = tri_coord[1, 0] + pent_2_coord[:, 0] * comm_spread
    pent_2_coord[:, 1] = tri_coord[1, 1] + pent_2_coord[:, 1] * comm_spread
    pent_3_coord[:, 0] = tri_coord[2, 0] + pent_3_coord[:, 0] * comm_spread
    pent_3_coord[:, 1] = tri_coord[2, 1] + pent_3_coord[:, 1] * comm_spread

    # Concatenate all the points
    reorder = [3,4,0,1,2]
    signal_coords = np.concatenate([pent_1_coord[reorder,:],
                                    pent_2_coord[reorder,:],
                                    pent_3_coord[reorder,:]],
                                   axis=0)

    # Return the coordinates
    return signal_coords


def randomise_timing(onsets,
                     ):
    # Extract all the timing information
    shuffle_all = 0  # Do you want to shuffle all or shuffle within block?
    nodes = len(onsets)

    # Initialize with the first node
    onsets_all = copy.deepcopy(onsets[0])
    output = [-1] * nodes
    output[0] = np.zeros(len(onsets_all), )

    # Add the rest of the nodes
    for node_counter in list(range(1, nodes)):
        onsets_all = np.append(onsets_all, onsets[node_counter])
        output[node_counter] = np.zeros(len(onsets[node_counter]), )

    if shuffle_all == 1:
        # Shuffle all the onsets, ignoring blocks
        np.random.shuffle(onsets_all)

        for node_counter in list(range(0, nodes)):
            node_num = len(onsets[node_counter])
            output[node_counter] = np.sort(onsets_all[0:node_num])
            onsets_all = onsets_all[node_num:]  # Remove the onsets

    else:
        # Shuffle the onsets within a block
        # Sort the onsets from big to small
        onsets_all = np.sort(onsets_all)

        # What it is the time between blocks
        block_gap = np.max(np.diff(onsets_all)) * 0.5

        # Cycle through blocks
        idx_start = 0
        onset_counter = 0
        while onset_counter < onsets_all.shape[0] - 1:

            # Is this onset less than the block gap
            while onset_counter < onsets_all.shape[0] - 1 and (
                onsets_all[onset_counter + 1] - onsets_all[
                onset_counter]) < block_gap:
                onset_counter += 1

            # Update
            onset_counter += 1
            idx_end = onset_counter

            # Pull out the blocks
            block_times = onsets_all[idx_start:idx_end]

            # Update
            idx_start = idx_end

            # Find which nodes were shown on this block (not always all of them)
            idxs = np.zeros((len(block_times),))
            idx_counter = 0
            for node_counter in list(range(nodes)):
                set_diff = list(set(onsets[node_counter]) - set(block_times))
                if len(set_diff) < len(onsets[node_counter]):
                    idxs[idx_counter] = node_counter
                    idx_counter += 1

            # Randomise block times
            np.random.shuffle(block_times)

            # Assign the times from the block
            for node_counter in list(range(len(block_times))):
                non_zero_idx = np.min(
                    np.where(output[int(idxs[node_counter])] == 0))
                output[int(idxs[node_counter])][non_zero_idx] = block_times[
                    node_counter]

    return output


def extra_isi(onsets,
              extra_isi,
              ):

    # Determine the number of nodes
    nodes = len(onsets)

    # If you want to use different timing then take the order of the data
    # and then create a new timecourse
    if extra_isi > 0:

        # Iterate through all the onset values
        onsets_all = []  # Reset
        for node_counter in list(range(0, nodes)):
            onsets_all = np.append(onsets_all, onsets[node_counter])

            # Take the idx of all of the elements in onset all
            Idxs = np.zeros((len(onsets[node_counter]), 2))

            Idxs[:, 0] = [node_counter] * len(onsets[node_counter])
            Idxs[:, 1] = list(range(0, len(onsets[node_counter])))

            # Append the indexes
            if node_counter == 0:
                onset_idxs = Idxs
            else:
                onset_idxs = np.concatenate((onset_idxs, Idxs))

        # Change the values of the onsets
        sorted_idxs = np.ndarray.argsort(onsets_all)
        cumulative_add = 0
        for onset_counter in list(range(0, len(onsets_all))):
            # What is the onset being considered
            onset = onsets_all[sorted_idxs[onset_counter]]

            # Add time to this onset
            onsets_all[sorted_idxs[onset_counter]] = onset + cumulative_add

            # Insert the onsets at the right time
            node_counter = int(onset_idxs[sorted_idxs[onset_counter], 0])
            idx_counter = int(onset_idxs[sorted_idxs[onset_counter], 1])
            onsets[node_counter][idx_counter] = onset + cumulative_add

            cumulative_add += extra_isi
    return onsets


# Make an mds plot of a distance matrix
def make_mds(dist,
             dim=2,
             mds_file=None,
             ):

    # Make an MDS plot of a given distance matrix
    fig = plt.figure()

    # Create an MDS plot
    mds = manifold.MDS(n_components=dim, dissimilarity='precomputed')  # Fit the
    # mds
    # object
    coords = mds.fit(dist).embedding_  # Find the mds coordinates
    coords = np.vstack(
        [coords, coords[0, :]])  # Duplicate first row for display

    if dim == 2:
        plt.plot(coords[:, 0],
                 coords[:, 1],
                 'k--')
        plt.scatter(coords[:, 0], coords[:, 1], s=100)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])


    # Save if a name is supplied
    if mds_file is not None:
        plt.savefig(mds_file)


# What is the within minus between distance of these data points?
def test_rsa(node_mat,
             distance_type='correlation',
             ):
    def fisher(r):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

    # Calculate the RSA of the data

    if distance_type == 'correlation':
        corr_matrix = np.corrcoef(node_mat)
    elif distance_type == 'distance':
        corr_matrix = sp_distance.squareform(
            sp_distance.pdist(node_mat.astype('double')))

    # Only take the upper matrix
    corr_matrix = np.triu(corr_matrix)

    # Iterate through the distance steps
    within_dist_all = []
    between_dist_all = []
    nodes = 15
    for distance in list(range(1, 5)):

        within_dist = []
        between_dist = []

        for node_counter in list(range(0, nodes)):

            for direction in list([-1, 1]):

                comparison = node_counter + (direction * distance)

                # Deal with the wrapping of numbers
                if comparison < 0:
                    comparison = nodes + comparison
                elif comparison >= nodes:
                    comparison = comparison - nodes

                if corr_matrix[node_counter, comparison] != 0:

                    if distance_type == 'correlation':
                        r = fisher(corr_matrix[node_counter, comparison])
                    else:
                        r = corr_matrix[node_counter, comparison]

                    corr_matrix[node_counter, comparison] = 0

                    # Determine whether these nodes are within or between
                    node_community = np.ceil((node_counter + 1) / 5)
                    comparison_community = np.ceil((comparison + 1) / 5)

                    # Add this distance to either the between or within list
                    if node_community == comparison_community:
                        within_dist.append(r)
                    else:
                        between_dist.append(r)

        # Average at each step
        within_dist_all.append(np.mean(within_dist))
        between_dist_all.append(np.mean(between_dist))

    # What is the within minus between distance
    community_difference = np.mean(within_dist_all) - np.mean(between_dist_all)

    # Do you want to use distance instead
    if distance_type is 'distance':
        community_difference *= -1

    return community_difference


def toy_simulation(community_density=1,
                   added_isi=0,
                   rand=0,
                   signal_magnitude=1,
                   noise_type='coordinates',
                   noise_parameter=0,
                   restrict_overall_duration=0,
                   ):

    # Default these values
    nodes = 15
    runs = 5
    vector_size = 2  # How many voxels did you make
    all_pos = 1
    ppt = '1'  # What participant would you like to simulate
    tr_duration = 2
    event_durations = [1]
    hrf_lag = 2
    temporal_res = 100  # How many samples per second are there for timing files

    # Select what the noise is applied to
    noise_coordinates = 0  # How much noise are you adding to the coordinates
    noise_timecourse = 0  # How many random noise are you adding to the timecourse
    if noise_type == 'coordinates':
        noise_coordinates = noise_parameter
    elif noise_type == 'timecourse':
        noise_timecourse = noise_parameter

    # Load the timing information
    timing_path = '../../community_structure/simulator_parameters/timing/'
    #timing_path = '/Volumes/pniintel/ntb/TDA/Validation/code/community_structure/simulator_parameters/timing/'
    onsets_runs = np.load(timing_path + 'sub-' + ppt + '.npy')
    
    # Generate the graph structure (based on the ratio)
    signal_coords = community_structure(1 - community_density,
                                        )

    # Add noise to these coordinates
    noise = np.random.randn(np.prod(signal_coords.shape)).reshape(
        signal_coords.shape) * noise_coordinates

    signal_coords += noise

    # Perform an orthonormal transformation of the data
    if vector_size > signal_coords.shape[1]:
        signal_coords = orthonormal_transform(vector_size,
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

    # Determine the size of the signal
    signal_coords *= signal_magnitude

    # Cycle through the runs and generate the data

    node_brain = np.zeros([vector_size, nodes, runs], dtype='double')  # Preset
    for run_counter in list(range(1, runs + 1)):

        # Pull out the onsets for this participant (reload it each time to deal with copying issues)
        onsets_runs = np.load(timing_path + 'sub-' + ppt + '.npy')
        onsets_run = onsets_runs[run_counter - 1]

        # What is the original max duration of the onsets
        max_duration_orig = np.max([np.max(onsets_run[x]) for x in range(onsets_run.size)])
        max_duration_orig += 10 # Add some wiggle room

        # Do you want to randomise the onsets (so that the events do not have a
        # fixed order)
        if rand == 1:
            onsets_run = randomise_timing(onsets_runs[run_counter - 1],
                                      )

        # If you want to use different timing then take the order of the data
        # and then create a new timecourse
        onsets_run = extra_isi(onsets_run,
                           added_isi,
                           )
        
        # If necessary, remove all the values greater than the max
        if restrict_overall_duration == 1:
            onsets_run = [onsets_run[x][onsets_run[x] < max_duration_orig] for x in range(
                onsets_run.size)]
            onsets_run = np.asarray(onsets_run)

        # Determine how long the simulated time course is by finding the max of maxs
        last_event = 0
        for node_counter in range(len(onsets_run)):
            if len(onsets_run[node_counter]) > 0 and onsets_run[node_counter].max() > last_event:
                last_event = onsets_run[node_counter].max()    
                
        # How long should you model
        duration = int(last_event + 10)  # Add a decay buffer

        # Preset brain size
        brain_signal = np.zeros([2, int(duration / tr_duration)], dtype='double')
        stimfunc_all = []
        for node_counter in list(range(0, nodes)):

            # Preset the signal
            signal_pattern = np.ones(vector_size)

            # Take the coordinates from the signal template
            for coord_counter in list(range(0, signal_coords.shape[1])):
                signal_pattern[coord_counter] = signal_coords[node_counter,
                                                              coord_counter]

            onsets_node = onsets_run[node_counter]
            
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
                            stimfunc).transpose() * signal_coords[node_counter, voxel_counter]
                else:

                    # Add these elements together
                    temp = np.zeros((len(stimfunc), vector_size))
                    for voxel_counter in list(range(0, vector_size)):
                        temp[:, voxel_counter] = np.asarray(stimfunc).transpose() * \
                                                 signal_coords[node_counter,
                                                               voxel_counter]

                    stimfunc_all += temp

        # After you have gone through all the nodes, convolve the HRF and
        # stimulation for each voxel
        signal_func = sim.convolve_hrf(stimfunction=stimfunc_all,
                                       tr_duration=tr_duration,
                                       temporal_resolution=temporal_res,
                                       )

        # Multiply the convolved responses with their magnitudes (after being
        # scaled)
        for voxel_counter in list(range(0, vector_size)):
            # Reset the range of the function to be appropriate for the
            # stimfunc
            signal_func[:, voxel_counter] *= stimfunc_all[:,
                                             voxel_counter].max()

        # Create the noise
        noise = np.random.randn(np.prod(signal_func.shape)).reshape(
            signal_func.shape) * noise_timecourse

        # Combine the signal and the noise
        brain = signal_func + noise
        # Z score the data
        #brain = zscore(brain.astype('float'), 0)

        # Loop through the nodes
        for node in list(range(0, nodes)):

            node_trs = onsets_run[node]
            if len(node_trs) > 0:
                temp = np.zeros((vector_size, len(node_trs) - 1))
                for tr_counter in list(range(1, len(node_trs))):

                    # When does it onset (first TR is zero so minus 1)
                    onset = int(np.round(node_trs[tr_counter] / tr_duration) +
                                hrf_lag)

                    # Add the TR if it is included when considering the hrf lag
                    if onset < brain.shape[0]:
                        temp[:, tr_counter - 1] = brain[onset, :]

                # Average the TRs
                node_brain[:, node, run_counter - 1] = np.mean(temp, 1)

    # average the brains across runs
    node_brain = np.mean(node_brain, 2)

    return node_brain

