# Take in the noise parameter estimates from real and simulated participants and compare them.

# Load in scripts
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# What system inputs are supplied
input_path = './Validation/'

# Summarise the simulated results
def summarise_p_value(variable_name,
                      real_variable,
                      sim_variable,
                      output_root,
                      plot_fig=0,
                      ):

    # What is the p value
    p_value = ((real_variable > sim_variable).sum() + 1) / (
    len(sim_variable) + 1)

    # # Wrap values if necessary
    # if p_value > 0.5:
    #     p_value = 1 - p_value

    print('P value for whether ' + variable_name + ' is significantly less '
                                                   'than the simulations')
    print(str(p_value))

    # Plot the results
    if plot_fig == 1:
        hist = plt.hist(sim_variable)
        plt.plot(np.array([real_variable, real_variable]), np.array([0.0,
                                                                     hist[
                                                                         0].max()]))
        title = variable_name + ' distribution, p=' + str(p_value)[0:5] + ' n=' + str(len(sim_variable))
        plt.title(title)
        plt.savefig(output_root + '_' + variable_name + '.eps')
        plt.close()

    # What are some summary values
    real_value = real_variable
    sim_value = sim_variable

    return real_value, sim_value

# Pull out the noise parameters from the text files that they are stored in for a participant and file path
def run_test(path, ppt):

    # Convert inputs to outputs
    real_noise_dict_name = path + 'real_noise_dict/' + ppt + '.txt'
    simulated_noise_dict_root = path + 'simulated_noise_dict_resample/' + ppt
    output_root = path + 'testing_noise_calc/plots/' + ppt

    # Load the noise dicts
    with open(real_noise_dict_name, 'r') as f:
        noise_dict = f.read()

    print('Loading ' + real_noise_dict_name)
    real_noise_dict = eval(noise_dict)

    # Pull out the relevant values
    real_snr = real_noise_dict['snr']
    real_sfnr = real_noise_dict['sfnr']
    real_fwhm = real_noise_dict['fwhm']
    real_ar_rho = real_noise_dict['auto_reg_rho']
    real_ma_rho = real_noise_dict['ma_rho']

    # What are all the files that have this file root?
    simulated_noise_dict_names = glob.glob(simulated_noise_dict_root + '*')

    # Cycle through the files
    snr = np.zeros((len(simulated_noise_dict_names), 1))
    sfnr = np.zeros((len(simulated_noise_dict_names), 1))
    fwhm = np.zeros((len(simulated_noise_dict_names), 1))
    ar_rho = np.zeros((len(simulated_noise_dict_names), 1))
    ma_rho = np.zeros((len(simulated_noise_dict_names), 1))
    for file_counter in list(range(0, len(simulated_noise_dict_names))):

        # Load in dict
        with open(simulated_noise_dict_names[file_counter], 'r') as f:
            noise_dict = f.read()

        print('Loading ' + simulated_noise_dict_names[file_counter])
        try:
            simulated_noise_dict = eval(noise_dict)

            # Store the dictionary elements
            snr[file_counter] = simulated_noise_dict['snr']
            sfnr[file_counter] = simulated_noise_dict['sfnr']
            fwhm[file_counter] = simulated_noise_dict['fwhm']
            ar_rho[file_counter] = simulated_noise_dict['auto_reg_rho'][0]
            ma_rho[file_counter] = simulated_noise_dict['ma_rho'][0]
        except:
            print('Deleting ' + simulated_noise_dict_names[file_counter])
            #os.remove(simulated_noise_dict_names[file_counter])

    # Summarise the results
    snr_real, snr_sim = summarise_p_value('snr', real_snr, snr,
                                              output_root)
    sfnr_real, sfnr_sim = summarise_p_value('sfnr', real_sfnr, sfnr,
                                           output_root)
    fwhm_real, fwhm_sim = summarise_p_value('fwhm', real_fwhm, fwhm,
                                             output_root)
    ar0_real, ar0_sim = summarise_p_value('ar_rho_0', real_ar_rho[0],
                                          ar_rho[:, 0], output_root);
    ma_real, ma_sim = summarise_p_value('ma_rho', real_ma_rho[0], ma_rho, output_root);

    ar0_sim = ar0_sim.reshape((ar0_sim.size, 1))

    # Summarise the summary
    real_all = np.asarray([snr_real, sfnr_real, fwhm_real, ar0_real, ma_real])
    sim_all = np.hstack([snr_sim, sfnr_sim, fwhm_sim, ar0_sim, ma_sim])

    return real_all, sim_all

# Run the script, first without matching
noise_types = 5
real_all = np.zeros((17, 2, noise_types))
sim_all = np.zeros((17, 2, noise_types, 10))
means_all = np.zeros((17, 2, noise_types))
stds_all = np.zeros((17, 2, noise_types))
for ppt in list(range(1, 18)):
    for run in list(range(1, 3)):
        if ppt < 10:
            ppt_name = 'Participant_0' + str(ppt) + '_rest_run0' + str(run)
        else:
            ppt_name = 'Participant_' + str(ppt) + '_rest_run0' + str(run)
    
        # Don't match noise
        real, sim = run_test(input_path, ppt_name)
        
        print(sim.shape)
        # Save the outputs
        real_all[ppt - 1, run - 1, :] = real
        sim_all[ppt - 1, run - 1, :, 0:sim.shape[0]] = sim.transpose()
        means_all[ppt - 1, run - 1, :] = np.mean(sim, 0)
        stds_all[ppt - 1, run - 1, :] = np.std(sim, 0)

# Save the output of this analysis
output_root = input_path + 'testing_noise_calc/'
np.save(output_root + 'real_noise_dicts.npy', real_all)
np.save(output_root + 'simulated_noise_dicts.npy', sim_all)
np.save(output_root + 'simulated_noise_dicts_means.npy', means_all)
np.save(output_root + 'simulated_noise_dicts_stds.npy', stds_all)

print(np.nanmean(np.nanmean(real_all, 0),0))
print(np.nanmean(np.nanmean(means_all, 0),0))
print(np.nanmean(real_all - means_all,0))
print(np.sqrt(np.nanmean(stds_all ** 2, 0)))


