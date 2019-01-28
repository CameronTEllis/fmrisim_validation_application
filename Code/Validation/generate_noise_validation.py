# Take in a real fMRI volume from ~/Validation/real_data/ and compute the noise properties of the volume. Then simulate a new volume with those same noise properties and store that. Finally, estimate the noise properties of the simulation and store those noise properties.
#
# This code can deal with some missing inputs. For instance if no output_name or no output_noise_dict_name is supplied then the steps to produce these will be skipped.
#
# The run time of this code is stored in ./Validation/simulation_timing.txt

import numpy as np
import nibabel
from brainiak.utils import fmrisim as sim
import sys
from os import path, remove
import logging
import time

logging.basicConfig(filename='./logs/fmrisim.log')
logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

start_time = time.time()

# Inputs are (full paths):
# 1: Input volume of real data
# 2: Output name for initial noise dict based on the real data
# 3: Output Mask name of the real data
# 4: Output name for generated noise dict based on the simulation
# 5: Output volume of the simulated data
# 6: Whether or not to fit the noise

input_name = sys.argv[1]
input_noise_dict_name = sys.argv[2]
output_mask_name = sys.argv[3]
output_noise_dict_name = sys.argv[4]
output_name = sys.argv[5]
match_noise = int(sys.argv[6])

# Convert the inputs if appropriate
if input_noise_dict_name == 'None' or input_noise_dict_name == '':
    input_noise_dict_name = None
if output_mask_name == 'None' or output_mask_name == '':
    output_mask_name = None
if output_noise_dict_name == 'None' or output_noise_dict_name == '':
    output_noise_dict_name = None
if output_name == 'None' or output_name == '':
    output_name = None

# How many TRs are there?
nii = nibabel.load(input_name)
dimsize = nii.header.get_zooms()
tr_duration = dimsize[3]
trs=nii.shape[3]
real_brain = nii.get_data()

dimensions = np.array(real_brain.shape[0:3])  # What is the size of the brain

# Generate the continuous mask from the voxels
mask, template = sim.mask_brain(volume=real_brain,
                                mask_self=True,
                                )

# Save the mask brain
nii = nibabel.Nifti1Image(mask.astype('int16'), nii.affine)
hdr = nii.header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))

# Save the mask if the name is given
if output_mask_name is not None:
    print('Saving ' + output_mask_name)
    nibabel.save(nii, output_mask_name)

# Calculate the noise parameters
if input_noise_dict_name is None or path.exists(input_noise_dict_name) == \
        False:
    print('Making ' + input_noise_dict_name)
    noise_dict = {'voxel_size': [dimsize[0], dimsize[1], dimsize[2]], 'matched': match_noise}
    noise_dict = sim.calc_noise(volume=real_brain,
                                mask=mask,
                                template=template,
                                noise_dict=noise_dict,
                                )

    # Save the file
    if input_noise_dict_name is not None:
        with open(input_noise_dict_name, 'w') as f:
            f.write(str(noise_dict))

else:

    # Load the file name instead
    with open(input_noise_dict_name, 'r') as f:
        noise_dict = f.read()

    print('Loading ' + input_noise_dict_name)
    noise_dict = eval(noise_dict)


print('Generating brain for this permutation ')
noise_dict['matched'] = match_noise
brain = sim.generate_noise(dimensions=dimensions,
                           stimfunction_tr=np.zeros((trs,1)),
                           tr_duration=int(tr_duration),
                           template=template,
                           mask=mask,
                           noise_dict=noise_dict,
                           )

# Save the node brain (if you are given an output name)
if output_name is not None:
    print('Saving noise volume')
    nii = nibabel.Nifti1Image(brain.astype('int16'), nii.affine)
    hdr = nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
    nibabel.save(nii, output_name)  # Save

# Calculate and save the output dict if a value is supplied
if output_noise_dict_name is not None:
    print('Testing noise generation')
    out_noise_dict = {'voxel_size': noise_dict['voxel_size'], 'matched': match_noise}
    out_noise_dict = sim.calc_noise(volume=brain,
                                mask=mask,
                                template=template,
                                noise_dict=out_noise_dict,
                                )

    # Remove file if it exists
    if path.exists(output_noise_dict_name):
        remove(output_noise_dict_name)

    # Save the file
    with open(output_noise_dict_name, 'w') as f:
        f.write(str(out_noise_dict))

# Print the timing of the script
duration = time.time() - start_time
with open('./Validation/simulation_timing.txt', 'a') as f:
    f.write('%s matching_%d: %0.3f\n' % (input_name, match_noise, duration))

print('Complete')

print(noise_dict)
print(out_noise_dict)
