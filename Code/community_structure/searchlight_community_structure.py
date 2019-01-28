import numpy as np
import nibabel
from brainiak.searchlight.searchlight import Searchlight
import sys
from mpi4py import MPI

# What are the system inputs?

# What is the node_brain file you are running
brain_file = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# What is the within minus between distance of these data points?
def test_rsa(node_mat,
             ):
    def fisher(r):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

    # Calculate the RSA of the data
    corr_matrix = np.corrcoef(node_mat)

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
                    r = fisher(corr_matrix[node_counter, comparison])
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
    return np.mean(within_dist_all) - np.mean(between_dist_all)


# Define voxel function
def rsa_sl(data, mask, myrad, bcvar):
    # What are the dimensions of the data?
    dimsize = data[0].shape

    # Pull out the data
    mat = data[0].reshape((dimsize[0] * dimsize[1] * dimsize[2],
                           dimsize[3])).astype('double').transpose()

    # Pass the distance matrix to the specified TDA test
    sl_outputs = test_rsa(mat)

    # Store the results of the analyses
    return sl_outputs

# Load in the data
nii = nibabel.load(brain_file)  # Load the participant
dimsize=nii.header.get_zooms()
node_brain = nii.get_data()

# Specify paths
searchlights_path = './community_structure/simulated_data/searchlights/'

# Set the names
sub_idx = brain_file.find('_brain/') + 7
subjectName = brain_file[sub_idx:]
output_name = searchlights_path + subjectName

# Make the mask
mask = node_brain != 0
mask = mask[:, :, :, 0]

# Create searchlight object
sl = Searchlight(sl_rad=1, max_blk_edge=5)

# Distribute data to processes
sl.distribute([node_brain], mask)
sl.broadcast(None)

# Run clusters
sl_outputs = sl.run_searchlight(rsa_sl, pool_size=1)

if rank == 0:

    # Convert the output into what can be used
    sl_outputs = sl_outputs.astype('double')
    sl_outputs[np.isnan(sl_outputs)] = 0

    # Save the volume
    sl_nii = nibabel.Nifti1Image(sl_outputs, nii.affine)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nibabel.save(sl_nii, output_name)  # Save




