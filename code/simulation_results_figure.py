# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: spacer_phage
#     language: python
#     name: spacer_phage
# ---

# # Figure 1 - model overview and simulation results
#
# This script produces panels B through F of Figure 1; panels were assembled in Inkscape.

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns

# %matplotlib inline

from sim_analysis_functions import load_simulation, find_file, find_nearest

from spacer_model_plotting_functions import (nbi_steady_state, nvi_steady_state, analytic_steady_state, 
                                             get_trajectories)
analytic_steady_state_vec = np.vectorize(analytic_steady_state)

# +
# function to find factors of a number 
# (from https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python)
from functools import reduce

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


# -

# ## Load simulation data used in panels B, C, D

# +
# find folder based on timestamp
all_data = pd.read_csv("../data/all_data.csv", index_col = 0)

# simulation identifying timestamp
timestamp = "2019-02-25T14:24:15.599257" # c0=10000.0, eta=0.0001,e=0.95, mu =0.00001, nu = 0.1908, m =7.125 

# get folder for simulation
top_folder = "../data/" + str(all_data[all_data['timestamp'] 
                                                                         == timestamp]['folder_date'].values[0])

# +
# load data

folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, top_folder)

f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
 max_m, mutation_times, parent_list, all_phages = load_simulation(folder, timestamp, return_parents = True)

# -

# ### Load all simulation summary statistics

# +
grouped_data = pd.read_csv("../data/grouped_data.csv", 
                           index_col = 0, header=[0,1])

# remove unnamed levels
new_columns = []

for label in grouped_data.columns:
    if 'Unnamed' in label[1]:
        new_columns.append((label[0], ''))
    else:
        new_columns.append(label)

grouped_data.columns = pd.MultiIndex.from_tuples(new_columns)
# -

# ## Panel B: simulation visualization at three time points

# +
# create distance matrix for all the phages that ever lived
all_phages = np.array(all_phages)

distance_matrix = np.zeros((len(all_phages), len(all_phages)))

for i in range(len(all_phages)):
    distance_matrix[i] = np.sum(np.abs(all_phages - all_phages[i]), axis = 1)

# +
# create dictionary of mutation and phylogeny information
# the initial phages start at position 0 in the phage list

m_init = int(m_init)
    
phages_dict = {}

# keys: 

for i, phage in enumerate(all_phages):
    if i < m_init:
        continue
    
    parent_ids = parent_list[i]
    
    parent_distances = []
    mutation_positions = []
    angle = []
    full_distance = []
    mutation_ts = mutation_times[i]
    angles = []
    total_distances = []
    
    parent_angles = []
    
    loop_list = parent_ids
    #if len(np.unique(parent_ids)) > 1:
    #    loop_list = parent_ids
        
    #else:
    #    loop_list = np.unique(parent_ids)
    #    mutation_ts = [mutation_ts[0]]
        
    for j, pid in enumerate(loop_list):
        pid = int(pid)
        mutation_t = mutation_ts[j]
        parent_distance = distance_matrix[i, pid]
        parent_distances.append(parent_distance)
        
        mutation_pos = np.where(np.abs(phage - np.array(all_phages[pid])) == 1)[0]
        if mutation_pos[0] < 0:
            mutation_pos += 30
        mutation_positions.append(mutation_pos)
          
    phages_dict[i] = {"sequence": phage, "parents": loop_list, "mutation_times": mutation_ts, "parent_distance": parent_distances,
                    "mutation_position": mutation_positions}
# -

# ### Plot

# +
scale = 65.
length = 1.
box_size = 10
transparency = 0.1
colours1 = cm.gist_rainbow(np.linspace(0,1,30))
colour_spread = 1.8  # increase this if colours look too similar
shift = 1
spread = 60. # increase this to smear blobs out more radially
big_fontsize = 22
small_fontsize = 18

fig, axs = plt.subplots(2,3, figsize = (12, 7.56))

times = [1000, 5000, 9000]
#times = [3600, 3660, 3720]

for j, time in enumerate(times):
    t_ind = find_nearest(pop_array[:,-1].toarray(), time/(g*c0))

    ax0 = axs[0, j]
    ax1 = axs[1, j]

    ax0.set_xlim(-box_size,box_size)
    ax0.set_ylim(-box_size,box_size)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_xlim(-box_size,box_size)
    ax1.set_ylim(-box_size,box_size)

    ax1.set_xticks([])
    ax1.set_yticks([])

    # if starting with more than one initial clone, this plots them spaced out in the image
    facs = np.sort(list(factors(m_init)))
    nrows_ncols = facs[int(len(facs)/2 -1): int(len(facs)/2 + 1)]

    if len(nrows_ncols) == 1:
        center_positions_x = [0]
        center_positions_y = [0]
    else:
        center_positions_x = np.linspace(-5,5,nrows_ncols[1])
        center_positions_y = np.linspace(-3,3,nrows_ncols[0])

    X, Y = np.meshgrid(center_positions_x, center_positions_y)
    center_positions_x = X.flatten()
    center_positions_y = Y.flatten()

    # draw size legend
    if j == 0:
        legend_sizes = np.logspace(0,4,5)
        for l in range(len(legend_sizes)):
            ax0.scatter(-box_size + 0.105*box_size, 0-box_size*0.09 - box_size*0.2*l, c = 'none', 
                        s = scale*np.log(shift+legend_sizes[l]), edgecolors = 'k')
            ax0.annotate(str(int(legend_sizes[l])), (-box_size + 0.205*box_size, 0-box_size*0.15 - box_size*0.2*l), 
                         fontsize = small_fontsize)

    for m in range(m_init):
        ax0.scatter(center_positions_x[m], center_positions_y[m], s = scale*np.log(pop_array[t_ind, max_m+1 + m]+1), c = 'k', edgecolors = 'k')
        ax1.scatter(center_positions_x[m], center_positions_y[m], s = scale*np.log(pop_array[t_ind, 1 + m]+1), c = 'k', edgecolors = 'k')

    for key, value in phages_dict.items():

        size_phage = pop_array[t_ind, max_m + 1 + key]
        size_bac = pop_array[t_ind, 1 + key]

        if size_phage > 0 or size_bac > 0:

            #print(key, value["parents"])

            # possibly also check if multiple mutations have happened at that time? Then plot two lines?
            # i.e. len(np.unique(parent_ids[:parent_ind]))

            shortest_distance = np.min(distance_matrix[key, :m_init])
            angle_chain = []
            total_distance = 0

            phage = key
            parent_phage = phage
            time2 = time

            while parent_phage > m_init-1: # continue following back
                mutation_ts = phages_dict[phage]["mutation_times"] # get all the mutations that have happened
                possible_parent_inds = np.where(np.array(mutation_ts) < time2/(g*c0))[0] # get all the indices for mutations before time time2
                possible_parent_phages, sortinds = np.unique(np.array(phages_dict[phage]["parents"])[possible_parent_inds], return_index = True) # unique parent phages

                possible_parent_phages = np.array(possible_parent_phages, dtype = 'int')[np.argsort(sortinds)] # get possible parents in the order in which they happened
                #parent_pop_sizes = pop_array[t_ind, max_m +1 : 2*max_m + 1][possible_parent_phages] # get phage populations for parents
                #parent_phage = possible_parent_phages[np.nonzero(parent_pop_sizes)[0][0]]
                parent_phage = possible_parent_phages[0]
                parent_ind = np.sort(sortinds)[0]
                distance_from_parent = phages_dict[phage]["parent_distance"][parent_ind]
                mutation_pos = phages_dict[phage]["mutation_position"][parent_ind]
                mutation_pos = mutation_pos[0] # in case of double mutant - this is crude, FIX THIS
                angle_chain.append(mutation_pos)
                total_distance += distance_from_parent
                phage = parent_phage
                time2 = mutation_ts[int(possible_parent_inds[-1])]*g*c0
                initial_phage_parent = parent_phage

            mutation_ts = phages_dict[key]["mutation_times"]
            possible_parent_inds = np.where(np.array(mutation_ts) < time/(g*c0))[0] # get all the indices for mutations before time time2
            possible_parent_phages, sortinds = np.unique(np.array(phages_dict[key]["parents"])[possible_parent_inds], return_index = True) # unique parent phages
            possible_parent_phages = np.array(possible_parent_phages, dtype = 'int')[np.argsort(sortinds)] # get possible parents in the order in which they happened

            parent_phage = possible_parent_phages[0]
            parent_ind = np.sort(sortinds)[0]
            mutation_pos = phages_dict[key]["mutation_position"][parent_ind]
            mutation_pos = mutation_pos[0] # in case of double mutant - this is crude, FIX THIS
            
            if len(angle_chain) < 2:
                parent_angle = 0
                angle = angle_chain[-1] * (360/L) * np.pi / 180

            else:
                # calculate angle from angle_chain
                parent_angle = angle_chain[-1] * (360/L) * np.pi / 180
                angle = parent_angle - (spread/2 )* np.pi / 180 + angle_chain[-2]*(spread/30)*np.pi / 180 

                for a in angle_chain[::-1][2:]:
                    parent_angle = angle
                    angle = parent_angle - (spread/2 )* np.pi / 180 + a*(spread/30)*np.pi / 180 
                    #print(parent_angle, angle)

            if parent_angle == 0:
                colour = colours1[int(mutation_pos)]
            else:
                colour_centre = parent_angle/(2*np.pi)
                colours2 = cm.gist_rainbow(np.linspace(colour_centre-0.5/total_distance,colour_centre+0.5/total_distance,30))
                colour = colours2[mutation_pos]

            initial_phage_pos_x = center_positions_x[initial_phage_parent]
            initial_phage_pos_y = center_positions_y[initial_phage_parent]
      
            ax0.scatter(length*total_distance*np.cos(angle) + initial_phage_pos_x, 
                        length*total_distance*np.sin(angle) + initial_phage_pos_y, s = scale*np.log(size_phage+1),
                           c = colour.reshape(1,-1), edgecolors='k')

            ax1.scatter(length*total_distance*np.cos(angle) + initial_phage_pos_x, 
                        length*total_distance*np.sin(angle) + initial_phage_pos_y, s = scale*np.log(size_bac+1),
                           c = colour.reshape(1,-1), edgecolors='k')

            # connect with lines
            size_parent_phage = pop_array[t_ind, max_m + 1 + parent_phage]
            size_parent_bac = pop_array[t_ind, 1 + parent_phage]
            
            if size_parent_phage > 0:
                ax0.plot([length*(total_distance-1)*np.cos(parent_angle) + initial_phage_pos_x, length*total_distance*np.cos(angle) + initial_phage_pos_x], 
                     [length*(total_distance-1)*np.sin(parent_angle) + initial_phage_pos_y, 
                      length*total_distance*np.sin(angle) + initial_phage_pos_y], 'k-', alpha = transparency)  

            if size_bac > 0 and size_parent_bac > 0:         
                ax1.plot([length*(total_distance-1)*np.cos(parent_angle) + initial_phage_pos_x, 
                      length*total_distance*np.cos(angle) + initial_phage_pos_x], 
                     [length*(total_distance-1)*np.sin(parent_angle)+ initial_phage_pos_y, 
                      length*total_distance*np.sin(angle)+ initial_phage_pos_y],
                         'k-', alpha = transparency)

axs[0,0].annotate(text = "%s generations" %times[0], xy = (-9, 8.5), fontsize = small_fontsize)
axs[0,1].annotate(text = "%s generations" %times[1], xy = (-9, 8.5), fontsize = small_fontsize)
axs[0,2].annotate(text = "%s generations" %times[2], xy = (-9, 8.5), fontsize = small_fontsize)
axs[0,1].annotate(text = "Colour = protospacer\nsequence", xy = (-9.5, -box_size*0.95), fontsize = small_fontsize)
axs[1,1].annotate(text = "Colour = spacer sequence", xy = (-9, -box_size*0.9), fontsize = small_fontsize)

axs[0,1].annotate(text = "New phage\nmutant", xy = (-0.1, 3.6), xytext = (-8, -1), 
                  arrowprops=dict(facecolor='black', shrink=0.05), fontsize = small_fontsize)


axs[0,0].set_ylabel("Phages", fontsize = big_fontsize)
axs[1,0].set_ylabel("Bacteria", fontsize = big_fontsize)

plt.tight_layout()
plt.subplots_adjust(wspace = 0.05)
plt.savefig("fig_1B_%s.svg" %timestamp)
plt.savefig("fig_1B_%s.pdf" %timestamp)
# -

# ## Panels C and D - bacteria and phage clone trajectories

# +
# get theoretical population sizes using predicted m

B = 170
pv = 0.02
f = 0.3
R = 0.04
L = 30
m_init_select = 1

row = grouped_data[(grouped_data['C0'].values == c0) &
            (grouped_data['mu'].values == mu) &
            (grouped_data['e'].values == e) & 
                (grouped_data['B'].values == B) & 
             (grouped_data['eta'].values == eta) &
             (grouped_data['m_init'] == m_init_select) &
            (grouped_data['pv_type'] == 'binary')]

nb_pred = float(row['nb_pred_recursive'])
nv_pred = float(row['nv_pred_recursive'])
C_pred = float(row['C_pred_recursive'])
nu_pred = float(row['nu_pred_recursive'])
nb0_pred = (1-nu_pred)*nb_pred

nvi_ss_pred = nvi_steady_state(nb_pred, nv_pred, C_pred, nb0_pred, f, g, c0, e, alpha, B, mu, pv, R, eta)
nbi_ss_pred = nbi_steady_state(nb_pred, f, g, c0, e, alpha, B, mu, pv)


# +
t_ss = gen_max / 5 # time in bacterial generations at which sim is assumed to be in steady state

F = f*g*c0
r = R*g*c0

# start index for steady-state time
t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

# extract clone details for bacteria and phages
nbi = pop_array[t_ss_ind:, 1 : max_m + 1].toarray()
nvi = pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1].toarray()
t_all = pop_array[t_ss_ind:, -1].toarray().flatten()

# extract individual bacteria and phage trajectories
(nvi_trajectories, nbi_trajectories, t_trajectories, nbi_acquisitions, phage_size_at_acquisition,  
        trajectory_lengths, trajectory_lengths_small, trajectory_lengths_large, trajectory_extinct, 
                 acquisition_rate, phage_identities) = get_trajectories(pop_array, nvi, nbi, f, g, c0, R, eta, 
                                                      alpha, e, pv, B, mu, max_m, m_init, t_ss_ind, 
                    split_small_and_large = True, size_cutoff = nvi_ss_pred,
                     aggressive_trim = True, aggressive_trim_length = 1200, return_fitness = False)
    
# -

# ### Plot

# +
## Plot clone-specific trajectories
fig, axs = plt.subplots(2,1, figsize = (6.5, 3))

N = 1 # plot every Nth point
plotted = 0

colours = cm.Dark2(np.linspace(0,1, 11))

for i in tqdm(range(nbi.shape[0])):
    
    #if i not in plot_inds:
    #    continue
        
    if not np.any(nbi[:,i]):
        continue
        
    # if nvi trajectory doesn't go extinct, continue
    if nvi[-1,i] > 0:
        continue
        
    # if nvi or nbi trajectories don't start at 0 after steady-state, continue
    if nvi[0,i] > 0:
        continue
    if nbi[0,i] > 0:
        continue
    
    #print(i)
    #axs[0].plot((t_all*g*c0)[int(N/2):-(int(N/2)-1)], running_mean(nvi[:,i], N))
    axs[0].plot(t_all[::N]*g*c0, nvi[::N,i], linewidth = 1, color = colours[plotted])
    axs[1].plot(t_all[::N]*g*c0, nbi[::N,i], linewidth = 1, color = colours[plotted])
    plotted +=1 
    
    if plotted > 10:
        break
    
    
axs[0].axhline(nvi_ss_pred,  linestyle = ':', color = 'k', label = 'Mean clone size')
axs[1].axhline(nbi_ss_pred,  linestyle = ':', color = 'k')
axs[0].set_xlim(2000,4000)
axs[1].set_xlim(2000,4000)
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].legend(loc='lower right')
axs[0].set_ylabel("Phage\nclone size")
axs[1].set_ylabel("Bacteria\nclone size")
axs[1].set_xlabel("Time (bacterial generations)")
axs[0].set_xticks([])

plt.tight_layout()
plt.savefig("fig1_CD_%s.svg" %timestamp)
plt.savefig("fig1_CD_%s.pdf" %timestamp)
# -

# ## Panel E

# +
axis_fontsize = 12

# define parameters
B = 170
#pv = 0.02
R = 0.04
c0 = 10**4
alpha = 2*10**-2/c0
g = 1/(42*c0)
eta = 10**-3
mu = 10**-7
e = 0.95
f = 0.3

# get simulation data
data_subset = grouped_data[(grouped_data['C0'] == c0)
         & (grouped_data['eta'] == eta)
         & (grouped_data['m_init'] == 1)
         & (grouped_data['B'] == B)
         & (grouped_data['pv_type'] == 'binary')
         & (grouped_data['mu'] < mu*1.1)
         & (grouped_data['mu'] > mu*0.9)
         & (grouped_data['e'] == e)
         & (grouped_data['f'] == f)
         & (grouped_data['pv'] != 0.02)]

#data_subset = data_subset.iloc[np.argmax(data_subset['e_effective']['nanmean'])]

data_subset = data_subset.sort_values(by = 'pv')

# +
# phage unable to persist past this point
pv_critical = (1/B)*(g*f/((1-f)*alpha) + 1)

pv_vals_low = np.logspace(-4, np.log10(pv_critical), 10)
pv_vals_high = np.logspace(np.log10(pv_critical)+0.001, 0, 50)

nb_low = [c0*(1-f)]*len(pv_vals_low)
C_low = [c0*f]*len(pv_vals_low)
# -

# ### Plot

# +
colours = ["#FFB000", "rebeccapurple", "lightseagreen" ]

fig, ax = plt.subplots(figsize = (4,3.2))

ax.plot(pv_vals_low, C_low, color = colours[0], label = "Nutrients")
ax.plot(pv_vals_low, [0]*len(pv_vals_low), color = colours[1], label = "Phages")
ax.plot(pv_vals_low , nb_low, color = colours[2], label = "Bacteria")

#ax.plot(pv_vals_high, C_pred, color = colours[0], label = "Nutrients")
#ax.plot(pv_vals_high, nv_pred, color = colours[1], label = "Phages")
#ax.plot(pv_vals_high, nb_pred, color = colours[2], label = "Bacteria")


ax.errorbar(data_subset['pv'], data_subset['mean_C']['nanmean'], marker = 's', 
            yerr = data_subset['mean_C']['nanstd'], linestyle = "none",
           color = colours[0], mec = 'k', zorder = 3)
ax.errorbar(data_subset['pv'], data_subset['mean_nv']['nanmean'], marker = 'v', 
            yerr = data_subset['mean_nv']['nanstd'], linestyle = "none",
           color = colours[1], mec = 'k', zorder = 3)
ax.errorbar(data_subset['pv'], data_subset['mean_nb']['nanmean'], marker = 'o', 
            yerr = data_subset['mean_nb']['nanstd'], linestyle = "none",
           color = colours[2], mec = 'k', zorder = 3,
          label = "Simulation\ndata")


# plot a few curves for constant e


for e in [0.2, 0.7, 0.95]:

    nb, nv, C, nu = analytic_steady_state_vec(pv_vals_high, e, B, R, eta, f, c0, g, alpha)
    ax.plot(pv_vals_high, C, color = colours[0], linestyle = '-', alpha = e, linewidth = 2)
    ax.plot(pv_vals_high, nv, color = colours[1], linestyle = '-', alpha = e, linewidth = 2)
    ax.plot(pv_vals_high, nb, color = colours[2], linestyle = '-', alpha = e, linewidth = 2)

ax.axvline(pv_critical, linestyle = '--', color = 'k')

#ax.axvline(1, linestyle = '--', color = 'k')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Probability of phage success $p_v$", fontsize = axis_fontsize)
ax.set_ylabel(r"Population size", fontsize = axis_fontsize)

ax.fill_between([10**-4, pv_critical], y1 =0, y2 = 10**6, color = 'lightgrey', alpha = 0.6)
#ax.fill_between([1, 1.4], y1 =0, y2 = 10**6, color = 'darkgrey', alpha = 0.6)

ax.annotate("Phages\nextinct", xy = (1*10**-3, 1.4*10**5))
#ax.annotate("Bacteria\nextinct", xy = (1.04, 2*10**5))

ax.annotate(r'Increasing $e$',
            xy=(8*10**-1, 3*10**5), xycoords='data',
            xytext=(4*10**-2, 1.6*10**4), textcoords='data',
            arrowprops=dict(facecolor='black', arrowstyle="->"))

ax.set_ylim(4*10**2,4*10**5)
ax.set_xlim(4*10**-4, 10*10**-1)

ax.legend(fontsize = 9, loc = 'lower left', bbox_to_anchor = (-0.01,0.405))
plt.tight_layout()
plt.savefig("fig1_E.pdf")
plt.savefig("fig1_E.svg")
# -

# ## Panel F

# +
# load dataframe with extinction information
extinction_df = pd.read_csv("../data/extinction_df.csv", index_col = 0)

extinction_df_grouped = extinction_df.groupby(['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv', 'm_init', 'pv_type', 'theta'])[['mean_m', 
        'mean_nb', 'mean_nu',
       'mean_C', 'mean_nv', 'e_effective', 
       'bac_extinct', 'phage_extinct', 'end_time']].agg([np.nanmean, np.nanstd, 'count']).reset_index()

# +
# define parameters
pv = 0.02
c0 = 10**3
alpha = 2*10**-2/c0
g = 1/(42*c0)
eta = 10**-3

# get simulation data
data_subset = extinction_df_grouped[(extinction_df_grouped['C0'] == c0)
         & (extinction_df_grouped['eta'] == eta)
         & (extinction_df_grouped['m_init'] == 1)
         & (extinction_df_grouped['B'] == B)
         & (extinction_df_grouped['f'] == f)
         & (extinction_df_grouped['pv'] == pv)
         & (extinction_df_grouped['pv_type'] == 'binary')
         & (extinction_df_grouped['mu'] > 3*10**-8)]

data_subset = data_subset.sort_values(by = ('e_effective', 'nanmean'))

# +
e_eff_vals = np.arange(0,1, 0.01)

nb_list = []
nv_list = []

for e in e_eff_vals:
    nb, nv, C, nu = analytic_steady_state(pv, e, B, R, eta, f, c0, g, alpha)
    nb_list.append(nb)
    nv_list.append(nv)
# -

# ### Plot

# +
fig, ax = plt.subplots(figsize = (3.4,3.2))

ax.errorbar(data_subset['e_effective']['nanmean'], data_subset['mean_nv']['nanmean'] , 
            xerr = data_subset['e_effective']['nanstd'],
            yerr = data_subset['mean_nv']['nanstd'] ,
         linestyle = 'none', color = 'k', zorder = 0, linewidth = 1,
            alpha = 0.8)

sc = ax.scatter(data_subset['e_effective']['nanmean'], data_subset['mean_nv']['nanmean'], marker = 'v', 
          s= 90, edgecolor = 'k', linewidth = 0.2, c = data_subset['phage_extinct']['nanmean'],
               label = "Phages", alpha = 0.7)

ax.errorbar(data_subset['e_effective']['nanmean'], data_subset['mean_nb']['nanmean'], 
            xerr = data_subset['e_effective']['nanstd'],
            yerr = data_subset['mean_nb']['nanstd'],
         linestyle = 'none', color = 'k', zorder = 0, linewidth = 1,
            alpha = 0.8)
ax.scatter(data_subset['e_effective']['nanmean'], data_subset['mean_nb']['nanmean'] , marker = 'o', 
          s= 90, edgecolor = 'k',  linewidth = 0.2, c = data_subset['bac_extinct']['nanmean'],
          label = "Bacteria", alpha = 0.7)


ax.plot(e_eff_vals, np.array(nv_list), color = 'k', zorder = 0, label = "Theory")

ax.plot(e_eff_vals, np.array(nb_list), color = 'k', zorder = 0)

cb = plt.colorbar(sc)
cb.ax.set_ylabel("Fraction of simulations\nwhere extinction occurs", fontsize = axis_fontsize)
ax.set_yscale('log')
ax.set_xlabel("Average immunity", fontsize = axis_fontsize)
#ax.set_ylabel(r"Average population size (scaled by $C_0$)")
ax.set_xlim(0,1)

ax.legend()
plt.tight_layout()
plt.savefig("fig1_F.pdf")
plt.savefig("fig1_F.svg")
# -

# ## Supplementary figures



grouped_data_multisample = extinction_df_grouped[extinction_df_grouped['bac_extinct']['count'] > 3]

# +
mu_vals = list(np.unique(extinction_df_grouped['mu']))
c0_vals = list(np.unique(extinction_df_grouped['C0']))
eta_vals = list(np.unique(extinction_df_grouped['eta']))
e_vals = list(np.unique(extinction_df_grouped['e']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd', 'X', 'h']
colours = sns.color_palette("hls", len(c0_vals))
#colours = cm.rainbow_r(np.linspace(0.1,1, len(c0_vals)))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

# +
colour = 'mu'
shape = 'e'
#line = 'eta'

colour_label = '\mu'
shape_label = 'e'
#line_label = '\eta'
eta_select = 10**-2

data_subset = extinction_df_grouped[
         (extinction_df_grouped['B'] == B)
         & (extinction_df_grouped['f'] == 0.3)
         & (extinction_df_grouped['pv'] == pv)
         & (extinction_df_grouped['pv_type'] == 'binary')
         & (extinction_df_grouped['m_init'] == 1)
         & (extinction_df_grouped['eta'] == eta_select)
         & (extinction_df_grouped['mu'] > 4*10**-8)
        & (extinction_df_grouped['C0'] < 10**5)]

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))

for i in range(len(np.sort(data_subset[colour].unique()))):
    legend_elements.append(Line2D([0], [0], marker='o', 
                                  label='$%s = %s$' %(colour_label, 
                                round(np.sort(data_subset[colour].unique())[i],7)),
                          markerfacecolor=colours[i], markeredgecolor = 'k', markersize = 10, linestyle = "None"))


fig, axs = plt.subplots(1,2, figsize = (10,4))


for group in data_subset.groupby([colour, shape]):
    data = group[1].sort_values(by = 'mu')
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)

    axs[0].errorbar(data['mean_nv']['nanmean'], data['phage_extinct']['nanmean'],
               linestyle = "none",
                xerr = data['mean_nv']['nanstd'],
               yerr = data['phage_extinct']['nanstd'],
                color = colours[colour_ind], marker = markerstyles[shape_ind], 
               elinewidth = 0.3, mec = 'k', alpha = 0.6, markersize = 10)
    
    axs[1].errorbar(data['mean_nb']['nanmean'], data['bac_extinct']['nanmean'],
               linestyle = "none",
                xerr = data['mean_nb']['nanstd'],
               yerr = data['bac_extinct']['nanstd'],
                color = colours[colour_ind], marker = markerstyles[shape_ind], 
               elinewidth = 0.3, mec = 'k', alpha = 0.6, markersize = 10)

axs[0].set_xscale('log')
axs[1].set_xscale('log')

axs[0].set_ylabel("Probability of phage population extinction")
axs[0].set_xlabel("Simulation mean phage population size")
axs[1].set_ylabel("Probability of bacteria population extinction")
axs[1].set_xlabel("Simulation mean bacteria population size")
axs[0].set_xlim(1*10**3, 8*10**5)
axs[1].set_xlim(3.5*10**1, 10**3)
axs[1].legend(handles = legend_elements, ncol =2, fontsize = 8)
plt.tight_layout()
plt.savefig("extinction_prob_vs_pop_size_eta_%s.pdf" %eta_select)
# -

colours_rgba = []
for c in colours:
    colours_rgba.append(matplotlib.colors.to_rgba(c))

# +
colour = 'eta'
shape = 'e'
#line = 'eta'

colour_label = '\eta'
shape_label = 'e'
#line_label = '\eta'

c0_select = 1000
e_select = 0.95
eta_select = 10**-4

data_subset = grouped_data_multisample [
         (grouped_data_multisample['B'] == B)
         & (grouped_data_multisample['f'] == 0.3)
         & (grouped_data_multisample['pv'] == 0.02)
         & (grouped_data_multisample['pv_type'] == 'binary')
       & (grouped_data_multisample['bac_extinct']['nanmean'] > 0)
         & (grouped_data_multisample['phage_extinct']['nanmean'] == 0) # bac extinction, not phage extinction
         & (grouped_data_multisample['mu'] > 3*10**-8)
        #& (extinction_df_grouped['C0'] == c0_select)
       # & (extinction_df_grouped['e'] == e_select)
      #& (extinction_df_grouped['eta'] == eta_select)
        & (grouped_data_multisample['m_init'] == 1)]

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))

for i in range(len(np.sort(data_subset[colour].unique()))):
    legend_elements.append(Line2D([0], [0], marker='o', 
                                  label='$%s = %s$' %(colour_label, 
                                round(np.sort(data_subset[colour].unique())[i],5)),
                          markerfacecolor=colours[i], markeredgecolor = 'k', markersize = 10, linestyle = "None"))


fig, ax = plt.subplots(figsize = (5,4))


for group in data_subset.groupby([colour, shape]):
    
    data = group[1].sort_values(by = 'mu')
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    # add transparency based on extinction probability
    color = colours_rgba[colour_ind]
    colors = np.tile(color,(len(data),1))
    colors[:, 3] = data['bac_extinct']['nanmean']

    ax.errorbar(data['mu'], data['end_time']['nanmean'],
                yerr = data['end_time']['nanstd'],
               linestyle = "-",
                color = colours[colour_ind], 
               elinewidth = 0.3)
    
    ax.scatter(data['mu'], data['end_time']['nanmean'],
                color = colors, marker = markerstyles[shape_ind], 
               edgecolor = 'k', s = 60)


ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Phage mutation rate")
ax.set_ylabel(r"Mean population extinction time")
ax.legend(handles = legend_elements, ncol =2, fontsize = 8)
plt.tight_layout()
plt.savefig("Extinction_time_vs_mu_bac_extinct.pdf")

# +
colour = 'C0'
shape = 'e'
#line = 'eta'

colour_label = 'C_0'
shape_label = 'e'
#line_label = '\eta'

c0_select = 1000
e_select = 0.95
eta_select = 10**-2

data_subset = grouped_data_multisample [
         (grouped_data_multisample['B'] == B)
         & (grouped_data_multisample['f'] == 0.3)
         & (grouped_data_multisample['pv'] == 0.02)
         & (grouped_data_multisample['pv_type'] == 'binary')
       & (grouped_data_multisample['phage_extinct']['nanmean'] > 0)
         #& (grouped_data_multisample['bac_extinct']['nanmean'] == 0) # bac extinction, not phage extinction
         & (grouped_data_multisample['mu'] > 3*10**-8)
     & (extinction_df_grouped['eta'] == eta_select)
        #& (extinction_df_grouped['C0'] == c0_select)
       # & (extinction_df_grouped['e'] == e_select)
      #& (extinction_df_grouped['eta'] == eta_select)
        & (grouped_data_multisample['m_init'] == 1)]

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))

for i in range(len(np.sort(data_subset[colour].unique()))):
    legend_elements.append(Line2D([0], [0], marker='o', 
                                  label='$%s = %s$' %(colour_label, 
                                round(np.sort(data_subset[colour].unique())[i],5)),
                          markerfacecolor=colours[i], markeredgecolor = 'k', markersize = 10, linestyle = "None"))


fig, ax = plt.subplots(figsize = (5,4))


for group in data_subset.groupby([colour, shape]):
    
    data = group[1].sort_values(by = 'mu')
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    # add transparency based on extinction probability
    color = colours_rgba[colour_ind]
    colors = np.tile(color,(len(data),1))
    colors[:, 3] = data['phage_extinct']['nanmean']

    ax.errorbar(data['mu'], data['end_time']['nanmean'],
                yerr = data['end_time']['nanstd'],
               linestyle = "-",
                color = colours[colour_ind], 
               elinewidth = 0.3)
    
    ax.scatter(data['mu'], data['end_time']['nanmean'],
                color = colors, marker = markerstyles[shape_ind], 
               edgecolor = 'k', s = 60)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Phage mutation rate")
ax.set_ylabel(r"Mean population extinction time")
ax.legend(handles = legend_elements, ncol =2, fontsize = 8)
plt.tight_layout()
plt.savefig("Extinction_time_vs_mu_phage_extinct_eta_%s.pdf" %eta_select)

# +
colour = 'm_init'
shape = 'e'
#line = 'eta'

colour_label = 'm_{init}'
shape_label = 'e'
#line_label = '\eta'

c0_select = 300
e_select = 0.95
eta_select = 10**-2

data_subset = grouped_data_multisample [
         (grouped_data_multisample['B'] == B)
         & (grouped_data_multisample['f'] == 0.3)
         & (grouped_data_multisample['pv'] == 0.02)
         & (grouped_data_multisample['pv_type'] == 'binary')
       & (grouped_data_multisample['phage_extinct']['nanmean'] > 0)
         #& (grouped_data_multisample['bac_extinct']['nanmean'] == 0) # bac extinction, not phage extinction
         & (grouped_data_multisample['mu'] > 3*10**-8)
     & (extinction_df_grouped['eta'] == eta_select)
        & (extinction_df_grouped['C0'] == c0_select)]
       # & (extinction_df_grouped['e'] == e_select)
      #& (extinction_df_grouped['eta'] == eta_select)
        #& (grouped_data_multisample['m_init'] == 1)]

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))

for i in range(len(np.sort(data_subset[colour].unique()))):
    legend_elements.append(Line2D([0], [0], marker='o', 
                                  label='$%s = %s$' %(colour_label, 
                                int(np.sort(data_subset[colour].unique())[i])),
                          markerfacecolor=colours[i], markeredgecolor = 'k', markersize = 10, linestyle = "None"))


fig, ax = plt.subplots(figsize = (5,4))


for group in data_subset.groupby([colour, shape]):
    
    data = group[1].sort_values(by = 'mu')
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    # add transparency based on extinction probability
    color = colours_rgba[colour_ind]
    colors = np.tile(color,(len(data),1))
    colors[:, 3] = data['phage_extinct']['nanmean']

    ax.errorbar(data['mu'], data['end_time']['nanmean'],
                yerr = data['end_time']['nanstd'],
               linestyle = "-",
                color = colours[colour_ind], 
               elinewidth = 0.3)
    
    ax.scatter(data['mu'], data['end_time']['nanmean'],
                color = colors, marker = markerstyles[shape_ind], 
               edgecolor = 'k', s = 60)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Phage mutation rate")
ax.set_ylabel(r"Mean population extinction time")
ax.legend(handles = legend_elements, ncol =2, fontsize = 8)
plt.tight_layout()
plt.savefig("Extinction_time_vs_mu_phage_extinct_eta_%s_c0_%s.pdf" %(eta_select, c0_select))
