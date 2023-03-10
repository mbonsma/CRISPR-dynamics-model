#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:46:00 2021

@author: madeleine

This script loads simulation files and processes them into an array,
then updates all_data.csv

Requires all_data.csv to be available in the same folder
Requires all_params.csv to be premade and available in the same folder
"""

import numpy as np
import pandas as pd
import re
from scipy import sparse
import argparse
from scipy.interpolate import interp1d
from sklearn.cluster import AgglomerativeClustering

from sim_analysis_functions import (find_nearest, load_simulation)

from spacer_model_plotting_functions import (nbi_steady_state, nvi_steady_state, 
                                             get_trajectories, interpolate_trajectories,
                                             get_clone_sizes, get_large_trajectories, 
                                             fraction_remaining, calculate_speed, 
                                             bac_large_clone_extinction, get_bac_large_trajectories, 
					     e_effective_shifted)

def phage_m_to_bac_m(nvi, nb, c0, g, f, alpha, pv, B, n_samples = 15):
    """
    Calculate bacteria m from the distribution of phage clone sizes
    """

    s0 = float(alpha*pv*nb*(B-1) - f*g*c0 - alpha*(1-pv)*nb)
    d0 = float(f*g*c0 + alpha*(1-pv)*nb)

    P0_inf = 1- 2*s0/(B*(s0 + d0)) # extinction probability at long time, independent of nbi
    
    if P0_inf > 1: # can happen if s0 comes out small and negative due to fluctuations in nb
        P0_inf = 1 # set P0 == 1

    N_est = (B*(s0 + d0))/(2*s0) # clone size at which P0 ~ (1/e)
    
    # get list of clone sizes by combining several timepoints
    phage_clone_sizes = (nvi[::int(nvi.shape[0]/n_samples)]).toarray().flatten()
    phage_clone_sizes = np.array(phage_clone_sizes[phage_clone_sizes > 0 ], dtype = 'int')
    
    # list of sizes from 0 to largest observed size
    clone_sizes = np.arange(0, np.max(phage_clone_sizes)+1)

    # survival probability for each clone size
    clone_size_survival = 1 - P0_inf**clone_sizes

    clone_size_survival[int(N_est):] = 1 # set Pest for larger sizes to 1

    # number of clones of size k 
    counts = np.bincount(phage_clone_sizes)

    mean_m = np.sum(clone_size_survival*counts)/n_samples
    
    return mean_m

def distance_sequence_1D(R1, R2, norm = "L1"):
    """
    Calculates distance between centre of masses.
    
    If norm == "L2", normalized so that the max distance is 1
    (i.e. the hypercube sides are length 1 / sqrt(L))
    
    Input: two vectors of dimension (L)
    
    Output: float distance between vectors
    """
    
    if norm == "L2":
        # this is the distance from [1,1,1,...1] to [0,0,0,...0] in L-dimensional space
        #normalization_length = np.sqrt(L)
        return np.sqrt(np.sum((R1 - R2)**2))
    elif norm == "L1":
        return np.sum(np.abs(R1 - R2))
    
    # distance between centre of masses

def distance_matrix_cluster(phage_list, cluster_labels):
    """
    Create a distance matrix sorting the sequences by cluster label.
    
    Example usage: distance_matrix_cluster(all_phages_nonzero, clust_phage.labels_)
    
    Inputs:
    phage_list : array of n sequences being clustered, shape n x L
    cluster_labels : array of shape n with labels for each of the sequences
    """
    
    distance_matrix = np.zeros((len(phage_list), len(phage_list)))
    
    for i in range(len(phage_list)):
        distance_matrix[i] = np.sum(np.abs(phage_list[np.argsort(cluster_labels)] 
                                                 - phage_list[np.argsort(cluster_labels)][i]), axis = 1)
        
    return distance_matrix
    

def centre_of_mass(all_phages, n_i):
    """
    Calculates the centre of mass in sequence space for a given vector of abundances
    Inputs:
    all_phages : Sequence vector of length m giving spacer or protospacer sequences
    ni : abundance vector of length m giving phage or bacteria abundances at a particular time point
    Returns:
    R : centre of mass, an array of length L
    """
    
    pwm = np.multiply(all_phages.T, n_i).T / np.sum(n_i)
    R = np.sum(pwm, axis = 0)
    return R

def average_distance(all_phages, n_i, ancestor, return_lists = False):
    """
    Calculate the average distance between a population of clones and an ancestral population.
    
    Inputs:
    all_phages : list of all phage sequences corresponding to columns in pop_array
    n_i : list of clone sizes corresponding to the rows of all_phages (either bac or phage)
    ancestor : sequence to calculate distance to
    """
    existing_clones = n_i[n_i > 0] # nonzero clones
  
    distance = []
    
    for seq in all_phages[n_i > 0]:
        distance.append(distance_sequence_1D(ancestor, seq, norm = "L1"))
    
    distance = np.array(distance)
    
    avg_distance = np.sum(existing_clones*distance) / np.sum(existing_clones) # weighted average of distances

    if return_lists == False:
        return avg_distance
    else:
        return avg_distance, existing_clones, distance

def max_establishments_pred(nu, nv, nb, m, g, c0, alpha, B, mu, L, pv, e, time_interval):
    """
    Calculate the predicted number of phage establishments over the course of a simulation
    time_interval: time in bacterial generations to calculate establishments during
    """
    
    P_est = 2*e*nu/(m*(B-1))
    mu_bar = alpha*B*mu*L*pv*nv*nb*(1-e*nu/m)/(g*c0) # only approximating 1-e^{-mu L}
    
    return P_est*mu_bar*time_interval

def simulation_stats(folder, timestamp):
    
    # regex to match a year beginning with 20
    folder_date = re.findall("20[0-9][0-9]-[0-1][0-9]-[0-3][0-9]", folder) 
    
    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
         max_m, mutation_times, all_phages = load_simulation(folder, timestamp);
    
    t_ss = gen_max / 5 # minimun t_ss = 2000, otherwise gen_max/5
        
    #if m_init > 1:
    #    continue

    # check for extinction:
    last_tp = pop_array[-1].toarray().flatten()
    if not np.any(last_tp[:max_m+1] > 0):
        return
    if not np.any(last_tp[max_m+1:2*max_m+1] > 0):
        return
    
    # subsample time if necessary - makes matrix much smaller in cases where Gillespie was heavily used

    # create mask for times that are not near the 0.5 save mark
    # CAUTION: if the save timestep is changed, this will do weird things
    timestep = 0.5
    cutoff = 0.02 # increase the cutoff to keep more points, decrease it to keep fewer
    mask1 = np.ma.masked_inside(pop_array[:, -1].toarray().flatten()*g*c0 % timestep, 0, cutoff).mask 
    new_times = (pop_array[:, -1]*g*c0)[mask1]
    timediffs =  new_times[1:] - new_times[:-1]
    pop_array = pop_array[mask1]

    # create mask for timesteps that are 0 (multi-saving)
    mask2 = ~np.ma.masked_where(timediffs.toarray().flatten() == 0, timediffs.toarray().flatten()).mask
    if type(mask2) != np.bool_: # if nothing is masked, mask2 will be a single value. Only mask if not.
        pop_array = pop_array[1:][mask2]

    #resave as sparse
    pop_array = sparse.coo_matrix(pop_array)
    pop_array = sparse.csr_matrix(pop_array)

    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

    if any(x in folder for x in exponential_pv_dates): # then this is a new pv sim
        pv_type = 'exponential'
    elif any(x in folder for x in exponential_pv_025_dates):  # then this is a new pv sim with rate 0.25
        pv_type = 'exponential_025'
    elif any(x in folder for x in theta_pv_dates): # then this is theta function pv
        pv_type = 'theta_function'
    else:
        pv_type = 'binary'
    
    # doing .toarray() is slow and memory-intensive, so do it once per simulation
    nbi = pop_array[t_ss_ind:, 1 : max_m + 1].toarray()
    nvi = pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1].toarray()

    # get trajectories        
    # trim at max size in order to measure establishment rate properly
    time_end = 500 # time in bacterial generations to run trajectories to

    (nvi_trajectories, nbi_trajectories, t_trajectories, nvi_fitness, nbi_fitness, 
         nbi_acquisitions, phage_size_at_acquisition, trajectory_lengths, 
         trajectory_extinct, acquisition_rate, phage_identities) = get_trajectories(pop_array, nvi, nbi, f, 
                                    g, c0, R, eta, alpha, e, pv, B, mu, max_m, m_init, t_ss_ind,
                                    trim_at_max_size = True, aggressive_trim_length = time_end)

    # interpolate trajectories
    fitness_times = np.concatenate([np.arange(0.5,6,0.5), np.arange(6,25,2), 
                                    np.arange(25, 100, 5), np.arange(100, time_end, 10)])
    nvi_interp = interpolate_trajectories(nvi_trajectories, t_trajectories, fitness_times, g, c0)
    
    mean_nvi = np.nanmean(nvi_interp, axis = 1) # conditioned on survival - nan if gone extinct
    mean_phage_fitness = np.gradient(mean_nvi, fitness_times) / mean_nvi
    
    # bacterial spacer acquisition
    nbi_acquisitions = np.sort(np.array(nbi_acquisitions)[~np.isnan(nbi_acquisitions)])
    
    try: 
        t = nbi_acquisitions[int(len(nbi_acquisitions)*0.9)] # time at which 90% of acquisitions have happened
        t_ind = find_nearest(fitness_times, t)
        fitness_at_acquisition = mean_phage_fitness[t_ind]
        mean_ind = find_nearest(fitness_times, np.mean(nbi_acquisitions))
        first_ind = find_nearest(fitness_times, nbi_acquisitions[0])
        
        if t > fitness_times[-1]: # print warning that trajectories aren't long enough
            print(str(timestamp) + " Longer mean trajectories needed: " + str(t) + " > " + str(fitness_times[-1]))
        
        first_acquisition_time = nbi_acquisitions[0]
        median_acquisition_time = nbi_acquisitions[int(len(nbi_acquisitions)/2) - 1]
        fitness_at_mean_acquisition = mean_phage_fitness[mean_ind]
        fitness_at_first_acquisition = mean_phage_fitness[first_ind]
        
        mean_bac_acquisition_time = np.mean(nbi_acquisitions)
        
    except IndexError: # no bacterial acquisitions
        fitness_at_acquisition = np.nan
        first_acquisition_time = np.nan
        median_acquisition_time = np.nan
        fitness_at_mean_acquisition = np.nan
        fitness_at_first_acquisition = np.nan
        mean_bac_acquisition_time = np.nan
        
    # get establishment time

    # calculate predicted large clone extinction
    nv = np.sum(pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1], axis = 1)
    nb = np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)
    nb0 = pop_array[t_ss_ind:, 0]
    C = pop_array[t_ss_ind:, -2]

    mean_nb = np.mean(nb[::int(len(nb)/n_snapshots)])
    mean_nv = np.mean(nv[::int(len(nb)/n_snapshots)])
    mean_C = np.mean(C[::int(len(nb)/n_snapshots)])
    mean_nb0 = np.mean(nb0[::int(len(nb)/n_snapshots)])
    
    # get mean field predictions for clone size
    nvi_ss = nvi_steady_state(mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, 
                              pv, R, eta)
    nbi_ss = nbi_steady_state(mean_nb, f, g, c0, e, alpha, B, mu, pv)
    
    # if nvi_ss is negative (happens sometimes)
    while nvi_ss < 0: # recalculate means with different sampling
        shift = np.random.randint(0,100)
        print("negative nvi_ss: %s" %timestamp)
        mean_nb = np.mean(nb[shift::int(len(nb-shift)/n_snapshots)])
        mean_nv = np.mean(nv[shift::int(len(nb-shift)/n_snapshots)])
        mean_C = np.mean(C[shift::int(len(nb-shift)/n_snapshots)])
        mean_nb0 = np.mean(nb0[shift::int(len(nb-shift)/n_snapshots)])
        nvi_ss = nvi_steady_state(mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, 
                              pv, R, eta)
    
    # get phage clone sizes
    (mean_m, mean_phage_m, mean_large_phage_m, mean_large_phage_size,
         mean_nu, e_effective) = get_clone_sizes(pop_array, c0, e, max_m, t_ss_ind, pv_type, theta, all_phages, 1, 
                                                 n_snapshots = n_snapshots)

    # use simulation nbi_ss to get extinction times, same as for nvi    
    bac_extinction_times_large, bac_extinction_times_large_phage_present = bac_large_clone_extinction(pop_array, nbi, nvi,
                                                                        max_m, nbi_ss, t_ss_ind)


    # get large trajectories with size cutoff = nvi_ss
    sim_length_ss = last_tp[-1]*g*c0 - t_ss
    mean_lifetime_large, establishment_rate, establishment_time = get_large_trajectories(nvi_trajectories, 
                    t_trajectories, trajectory_lengths, trajectory_extinct, nvi_ss, g, c0, sim_length_ss)
    
    bac_establishment_rate, establishment_time_bac = get_bac_large_trajectories(nbi_trajectories, 
                                                    t_trajectories, nbi_ss, g, c0, sim_length_ss)

    # get spacer turnover and turnover speed
    turnover_array, interp_t = fraction_remaining(pop_array, t_ss, t_ss_ind, g, c0, gen_max, max_m)
    speed, start_ind = calculate_speed(turnover_array, interp_t)

    F = f*g*c0
    beta = mean_nb*alpha*pv
    delta = F + alpha*mean_nb*(1-pv)
    freq = nvi_ss / mean_nv
    mean_T_backwards_nvi_ss = 2*mean_nv*freq*(1-np.log(freq))*g*c0/((B-1)**2 * beta + delta)
    
    p = beta / (beta + delta)
    predicted_establishment_fraction = (1 - (2-3*B*p + p*B**2)/(B*p*(B-1)))
    
    nvi_sparse = pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1]
    rescaled_phage_m = phage_m_to_bac_m(nvi_sparse, mean_nb, c0, g, f, alpha, pv, B)
    
    all_mutation_times = []
    
    for times in mutation_times:
        all_mutation_times += list(times)
    
    all_mutation_times = np.sort(all_mutation_times)
    all_mutation_times = all_mutation_times[all_mutation_times > 0]
    all_mutation_times_ss = all_mutation_times[all_mutation_times*g*c0 > t_ss]
    
    mutation_rate_actual = len(all_mutation_times_ss)/((all_mutation_times_ss[-1] - all_mutation_times_ss[0])*g*c0)
    
    ### speed and distance calculations
    
    timepoints = pop_array[t_ss_ind:, -1].toarray().flatten()*g*c0 - t_ss

    # interpolate population sizes to have consistent time spacing
    interp_fun_nbi = interp1d(timepoints, nbi, kind='linear', axis = 0)
    interp_fun_nvi = interp1d(timepoints, nvi, kind='linear', axis = 0)
        
    # create distance matrix for all the phages that ever lived
    all_phages = np.array(all_phages)

    distance_matrix = np.zeros((len(all_phages), len(all_phages)))

    for i in range(len(all_phages)):
        distance_matrix[i] = np.sum(np.abs(all_phages - all_phages[i]), axis = 1)

    timestep = 5 # timestep in generations
    
    # subsample the simulation times
    interp_times = np.arange(t_ss + timestep, gen_max, timestep) - t_ss
    
    phage_array = interp_fun_nvi(interp_times)
    bac_array = interp_fun_nbi(interp_times)
    
    try:
        pos_phage_ancestor = centre_of_mass(all_phages, phage_array[0]) # CM sequence at t_ss
        pos_bac_ancestor = centre_of_mass(all_phages, bac_array[0]) # CM sequence at t_ss
    except ValueError:
        print("ValueError: %s" %timestamp)
        raise
        
    count = 1
    while np.any(np.isnan(pos_bac_ancestor)):
        pos_bac_ancestor = centre_of_mass(all_phages, bac_array[count])
        count += 1

    bac_spread = []
    phage_spread = []

    pos_phage = []
    pos_bac = []
    
    for t_ind in range(phage_array.shape[0]):
        R_phage = centre_of_mass(all_phages, phage_array[t_ind])
        pos_phage.append(R_phage)
        R_bac = centre_of_mass(all_phages, bac_array[t_ind])
        pos_bac.append(R_bac)

        # weighted average of distances from current centre of mass
        avg_bac_spread = average_distance(all_phages, bac_array[t_ind], R_bac)
        avg_phage_spread = average_distance(all_phages, phage_array[t_ind], R_phage)

        avg_bac_distance, existing_bac, distance_existing_bac = average_distance(all_phages, bac_array[t_ind], 
                                                                                 pos_bac_ancestor, return_lists = True)
        avg_phage_distance, existing_phage, distance_existing_phage = average_distance(all_phages, phage_array[t_ind], 
                                                                                       pos_phage_ancestor, return_lists = True)

        dist_and_size_phage = np.stack([distance_existing_phage[np.argsort(distance_existing_phage)], 
                      existing_phage[np.argsort(distance_existing_phage)]]).T

        dist_and_size_bac = np.stack([distance_existing_bac[np.argsort(distance_existing_bac)], 
                      existing_bac[np.argsort(distance_existing_bac)]]).T

        # phage population size grouped by distance from ancestor
        pop_sizes_grouped = np.split(dist_and_size_phage[:,1], np.unique(dist_and_size_phage[:, 0], return_index=True)[1][1:])
        pop_sizes_grouped_bac = np.split(dist_and_size_bac[:,1], np.unique(dist_and_size_bac[:, 0], return_index=True)[1][1:])

        pop_sizes_phage = []
        for l in pop_sizes_grouped:
            pop_sizes_phage.append(np.sum(l))

        pop_sizes_bac = []
        for l in pop_sizes_grouped_bac:
            pop_sizes_bac.append(np.sum(l))

        pop_sizes_phage = np.array(pop_sizes_phage)
        pop_sizes_bac = np.array(pop_sizes_bac)

        bac_spread.append(avg_bac_spread)
        phage_spread.append(avg_phage_spread)

    # get distances 
    bac_distance = []  # distance from starting position
    bac_phage_distance = [] # current distance between bacteria and phage
    bac_instantaneous_distance = []
    for j, pos in enumerate(pos_bac):
        bac_distance.append(distance_sequence_1D(pos_bac_ancestor, pos, norm = "L1"))
        bac_phage_distance.append(distance_sequence_1D(pos, pos_phage[j], norm = "L1"))
        if j > 1:
            bac_instantaneous_distance.append(distance_sequence_1D(pos_bac[j-1], pos, norm = "L1"))

    phage_distance = []  # distance from starting position
    phage_instantaneous_distance = []
    for j, pos in enumerate(pos_phage):
        phage_distance.append(distance_sequence_1D(pos_phage_ancestor, pos, norm = "L1"))
        if j > 1:
            phage_instantaneous_distance.append(distance_sequence_1D(pos_phage[j-1], pos, norm = "L1"))
    
    
    # get first passage times for distance
    
    time_to_reach_bac = []
    time_to_reach_phage = []

    for d in distance_checkpoints:
        if d <= np.max(bac_distance):
            time_to_reach_bac.append(interp_times[np.where(np.array(bac_distance) >= d)[0][0]])
        else:
            time_to_reach_bac.append(np.nan)

        if d <= np.max(phage_distance):
            time_to_reach_phage.append(interp_times[np.where(np.array(phage_distance) >= d)[0][0]])
        else:
            time_to_reach_phage.append(np.nan)
    
    bac_speed_mean = np.mean(bac_instantaneous_distance) / timestep
    phage_speed_mean = np.mean(phage_instantaneous_distance) / timestep
    bac_speed_std = np.std(bac_instantaneous_distance) / timestep
    phage_speed_std = np.std(phage_instantaneous_distance) / timestep
    
    # speed and distance in abundance space
    # normalize so that each time point abundance vector is 1

    phage_array_norm = phage_array / np.sum(phage_array, axis = 1)[:,None]
    bac_array_norm = bac_array / np.sum(bac_array, axis = 1)[:,None]
    
    skip = int(timestep*2) # assuming each time point is every 0.5 generations
    
    abundance_distance_phage = np.sum(np.abs(np.subtract(phage_array_norm[: -(skip)] , 
                                                         phage_array_norm[skip:])), axis = 1)
    abundance_distance_phage_ancestor = np.sum(np.abs(np.subtract(phage_array_norm , 
                                                                  phage_array_norm[0][None, :])), axis = 1)
    abundance_distance_bac = np.sum(np.abs(np.subtract(bac_array_norm[: -(skip)] , 
                                                         bac_array_norm[skip:])), axis = 1)
    abundance_distance_bac_ancestor = np.sum(np.abs(np.subtract(bac_array_norm , 
                                                                  bac_array_norm[0][None, :])), axis = 1)
    try:
        time_to_full_turnover_phage = interp_times[np.where(abundance_distance_phage_ancestor == 2)[0]][0]
    except:
        time_to_full_turnover_phage = np.nan
        
    try:
        time_to_full_turnover_bac = interp_times[np.where(abundance_distance_bac_ancestor == 2)[0]][0]
    except:
        time_to_full_turnover_bac = np.nan
        
        
    ## clan number and size
    t_skip = 50 # time interval in generations to sample at

    clan_number_bac = []
    clan_number_phage = []
    clan_size_mean_bac = []
    clan_size_mean_phage = []

    times = np.arange(t_ss, gen_max, t_skip)
    for t in times:
    
        t_ind = find_nearest(pop_array[:, -1].toarray().flatten()*g*c0, t)

        nonzero_inds = pop_array[t_ind, max_m+1:2*max_m +1].toarray().flatten() > 0
        all_phages_nonzero = all_phages[nonzero_inds]

        nonzero_inds_bac = pop_array[t_ind, 1:max_m +1].toarray().flatten() > 0
        bac_nonzero = all_phages[nonzero_inds_bac]

        model_phage = AgglomerativeClustering(n_clusters = None, distance_threshold = 2,
                                        affinity = 'l1', linkage = 'single')
        model_bac = AgglomerativeClustering(n_clusters = None, distance_threshold = 2,
                                        affinity = 'l1', linkage = 'single')

        if bac_nonzero.shape[0] <= 1: # only 0 or 1 spacer, can't do clustering
            clan_number_bac.append(bac_nonzero.shape[0])
            clan_size_mean_bac.append(bac_nonzero.shape[0])
        else:
            clust_bac = model_bac.fit(bac_nonzero)
            clan_sizes_bac = np.unique(clust_bac.labels_, return_counts = True)[1]
            clan_number_bac.append(len(np.unique(clust_bac.labels_)))
            clan_size_mean_bac.append(np.mean(clan_sizes_bac))
            
        if all_phages_nonzero.shape[0] <= 1: # only 0 or 1 protospacer, can't do clustering
            clan_number_phage.append(all_phages_nonzero.shape[0])
            clan_size_mean_phage.append(all_phages_nonzero.shape[0])
        else:
            clust_phage = model_phage.fit(all_phages_nonzero)
            clan_sizes_phage = np.unique(clust_phage.labels_, return_counts = True)[1]
            clan_number_phage.append(len(np.unique(clust_phage.labels_)))
            clan_size_mean_phage.append(np.mean(clan_sizes_phage))

    ### time shift

    nbi_interp = bac_array
    nvj_interp = phage_array
    
    e_eff_mean_past, e_eff_std_past = e_effective_shifted(e, nbi_interp, 
                                                          nvj_interp, max_shift = max_shift, direction = 'past')
    
    e_eff_mean_future, e_eff_std_future = e_effective_shifted(e, nbi_interp, 
                                                          nvj_interp, max_shift = max_shift, direction = 'future')
    
    peak_time = interp_times[np.argmax(e_eff_mean_past[:100])]
    # not sure if :100 is the best cutoff to use, monitor its value
    if np.argmax(e_eff_mean_past[:100]) > 99:
        print(peak_time)
        print(timestamp)

    slope = (e_eff_mean_future[slope_width] - e_eff_mean_past[slope_width]) / interp_times[slope_width*2]
        
    
    # add to data frame
    
    df = pd.DataFrame()
    
    df['C0'] = [c0]
    df['mu'] = [mu]
    df['eta'] = [eta]
    df['e'] = [e]
    df['B'] = [B]
    df['f'] = [f]
    df['pv'] = [pv]
    df['m_init'] = [m_init]
    df['pv_type'] = [pv_type]
    df['gen_max'] = [gen_max]
    df['max_save'] = [max_save]
    df['theta'] = [theta]
    df['t_ss'] = [t_ss]
    df['mean_m'] = [mean_m]
    df['mean_phage_m'] = [mean_phage_m]
    df['mean_large_phage_m'] = [mean_large_phage_m]
    df['mean_large_phage_size'] = [mean_large_phage_size]
    df['rescaled_phage_m'] = [rescaled_phage_m]
    df['timestamp'] = [timestamp]
    df['folder_date'] = folder_date
    df['mean_nu'] = [mean_nu]
    df['mean_nb'] =  [mean_nb]
    df['mean_nv'] = [mean_nv]
    df['mean_C'] = [mean_C]
    df['e_effective'] = [e_effective]
    df['fitness_discrepancy'] = [mean_phage_fitness[0]]
    df['mean_size_at_acquisition'] = [np.nanmean(phage_size_at_acquisition)] # mean phage clone size at the time that a spacer is acquired, ignoring trajectories for which no spacer is acquired
    df['std_size_at_acquisition'] = [np.nanstd(phage_size_at_acquisition)]# std dev phage clone size at the time that a spacer is acquired, ignoring trajectories for which no spacer is acquired
    df['fitness_at_90percent_acquisition'] = [fitness_at_acquisition]
    df['fitness_at_mean_acquisition'] = [fitness_at_mean_acquisition]
    df['fitness_at_first_acquisition'] = [fitness_at_first_acquisition]
    df['num_bac_acquisitions'] = [len(nbi_acquisitions)]
    df['mean_bac_acquisition_time'] = [mean_bac_acquisition_time]
    df['median_bac_acquisition_time'] = [median_acquisition_time]
    df['first_bac_acquisition_time'] = [first_acquisition_time]
    df['mean_large_trajectory_length_nvi_ss'] = [mean_lifetime_large] 
    df['mean_trajectory_length'] = [np.mean(trajectory_lengths)]
    df['mean_T_backwards_nvi_ss'] =  [mean_T_backwards_nvi_ss]
    df['mean_bac_extinction_time'] = [np.mean(bac_extinction_times_large)*g*c0] # simulation average
    df['mean_bac_extinction_time_phage_present'] = [np.mean(bac_extinction_times_large_phage_present)*g*c0]
    df['establishment_rate_nvi_ss'] = [establishment_rate]
    df['turnover_speed'] = [speed]
    df['predicted_establishment_fraction'] = [predicted_establishment_fraction]
    df['measured_mutation_rate'] = [mutation_rate_actual]
    df['mean_establishment_time'] = [establishment_time]
    df['max_m'] = [max_m]
    df['establishment_rate_bac'] = [bac_establishment_rate]
    df['mean_bac_establishment_time'] = [establishment_time_bac]
    
    ### speed and distance stuff
    df["bac_speed_mean"] = [bac_speed_mean]
    df["bac_speed_std"] = [bac_speed_std]
    df["phage_speed_mean"] = [phage_speed_mean]
    df["phage_speed_std"] = [phage_speed_std]
    df["bac_spread_mean"] = [np.nanmean(bac_spread)]
    df["bac_spread_std"] = [np.nanstd(bac_spread)]
    df["phage_spread_mean"] = [np.nanmean(phage_spread)]
    df["phage_spread_std"] = [np.nanstd(phage_spread)]
    df["net_phage_displacement"] = [phage_distance[-1]]
    df["net_bac_displacement"] = [bac_distance[-1]]
    df["max_phage_displacement"] = [np.max(phage_distance)]
    df["max_phage_displacement_10000"] = [np.max(np.array(phage_distance)[interp_times <=10000])] # max until 10000 gens
    df["max_bac_displacement"] = [np.max(bac_distance)]
    df["max_bac_displacement_10000"] = [np.max(np.array(bac_distance)[interp_times <=10000])]
    df["bac_phage_distance_mean"] = [np.nanmean(bac_phage_distance)] # mean distance between bacteria and phage
    df["bac_phage_distance_std"] = [np.nanstd(bac_phage_distance)]
    df["sim_length_ss"] = [sim_length_ss]
    # note: in this version, this time already has the steady-state start time subtracted (usually 2000 gens)
    df["time_to_full_turnover_phage"] = [time_to_full_turnover_phage]
    df["time_to_full_turnover_bac"] = [time_to_full_turnover_bac]
    df["phage_abundance_speed_mean"] = [np.nanmean(abundance_distance_phage) / timestep]
    df["phage_abundance_speed_std"] = [np.nanstd(abundance_distance_phage) / timestep]
    df["bac_abundance_speed_mean"] = [np.nanmean(abundance_distance_bac) / timestep]
    df["bac_abundance_speed_std"] = [np.nanstd(abundance_distance_bac) / timestep]
    df["bac_clan_number"] = [np.nanmean(clan_number_bac)]
    df["phage_clan_number"] = [np.nanmean(clan_number_phage)]
    df["bac_clan_number_std"] = [np.nanstd(clan_number_bac)]
    df["phage_clan_number_std"] = [np.nanstd(clan_number_phage)]
    df["bac_clan_size"] = [np.nanmean(clan_size_mean_bac)]
    df["phage_clan_size"] = [np.nanmean(clan_size_mean_phage)]
    df["bac_clan_size_std"] = [np.nanstd(clan_size_mean_bac)]
    df["phage_clan_size_std"] = [np.nanstd(clan_size_mean_phage)]
    
    for n, d in enumerate(distance_checkpoints):
        df["time_to_reach_bac_%s" %d] = [time_to_reach_bac[n]]
        df["time_to_reach_phage_%s" %d] = [time_to_reach_phage[n]]
    
    df['pred_num_establishments'] = max_establishments_pred(mean_nu, mean_nv, mean_nb, 
                        mean_m, g, c0, alpha, B, mu, L, pv, e, gen_max - t_ss)
    
    df['measured_num_establishments'] = establishment_rate*(gen_max - t_ss)
    # number of establishments in 8000 steady-state generations (for more uniform comparison)
    df['measured_num_establishments_8000'] = establishment_rate*8000
    
    ### time shift stuff
    df['slope'] = [slope]
    df['peak_immunity'] = [peak_time]
    
    # add mean_m to dataframe by joining on parameters that vary
    new_data = all_params.merge(df, on = ['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv', 'm_init', 'theta'])
    
    try:
        all_data.columns == new_data.columns
    except:
        raise
        
    # add new data to original df
    result = pd.concat([all_data, new_data], sort = True).reset_index()
    result = result.drop("index", axis = 1)
    
    result = result.drop_duplicates()
    
    result.to_csv("all_data.csv")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simulation analysis')
    parser.add_argument('timestamp',type=str,help='timestamp to analyze')
    parser.add_argument('folder',type=str,help='folder')
    parser.add_argument('all_params',type=str,default='all_params.csv',
                        nargs='?',help='filename for parameters csv')
    
    args = parser.parse_args()
    
    
    # define parameters
    timestamp = args.timestamp
    folder = args.folder
    all_params_fn = args.all_params
    
    all_params = pd.read_csv(all_params_fn, index_col=0)
    all_data = pd.read_csv("all_data.csv", index_col=0)
    
    n_snapshots = 50 # number of points to sample (evenly) to get population averages
    
    distance_checkpoints = [0.25, 0.5, 1, 2, 5, 10, 15] # first passage time distance checkpoints

    max_shift = 400 # largest time shift to use to calculate memory length
    slope_width = 3

    exponential_pv_dates = ["2019-06-24", "2021-09-09"]
    exponential_pv_025_dates = ["2021-02-01", "2021-09-08"]
    theta_pv_dates = ["2021-06-11", "2021-08-26", "2021-09-13"]

    if np.sum(all_data['timestamp'].isin([timestamp])) == 0: # then this timestamp has not been analyzed
        simulation_stats(folder, timestamp)
