# -*- coding: utf-8 -*-
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

# # Establishment and extinction figure (Figure 3)
#
# Code for Figure 3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import matplotlib
import matplotlib.cm as cm


# %matplotlib inline

# from https://stackoverflow.com/a/53191379
################### Function to truncate color map ###################
def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


def T_ext_e0(g,c0,f,B,pv,alpha,eta,m):
    """
    Approximate nv as nv without CRISPR (nu = 0, e = 0)
    """
        
    nv = c0*y_fn_nu(0, f, pv*alpha/g, pv, 0, B, R, eta)
    
    return 2*(B*pv-1)*(1+np.log(m))*nv/(f*B*pv*(B-1)*m)


# +
def nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta):
    
    """nv without CRISPR is setting nu and e to zero"""
    nu = 0
    e = 0
    
    return c0*y_fn_nu(nu, f, pv*alpha/g, pv, e, B, R, eta)

def Aterm(f,g,alpha,pv,B):
    """
    A > 1 is phage existence criterion (without CRISPR)
    """
    
    return (1-f)*(B*pv-1)*alpha/(f*g)
def nuapprox_small_e(f,g,c0,alpha,pv,B,R,eta,e,m):
    """
    Assume nu = -d/c
    """
    r = R*g*c0
    A = Aterm(f,g,alpha,pv,B)
    nv = nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta)
    
    return 1 / (1 + r/(eta*(1-pv)*alpha*nv) - e*pv/(m*eta*(1-pv)) + (A*B*pv*e/m) /((A-1)*(B*pv - 1)))


# +
from sim_analysis_functions import (find_nearest, load_simulation,find_file)

from spacer_model_plotting_functions import (nbi_steady_state, nvi_steady_state, 
                                             get_trajectories, interpolate_trajectories, y_fn_nu)

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

all_data = pd.read_csv("../data/all_data.csv", index_col = 0)

# +
## select parameters
mu_select = 3*10**-6
e_select = 0.95
c0_select = 10**4
eta_select = 10**-3

all_data_subset = all_data[(all_data['C0'] == c0_select)
         #(all_data['mean_nu'] > 0.6)
        & (all_data['eta'] == eta_select)
        & (all_data['e'] == e_select)
        & (all_data['m_init'] == 1)
        & (all_data['mu'] < mu_select*1.1)
        & (all_data['mu'] > mu_select*0.9)]
# -

top_folder = "../data/"
timestamps = list(all_data_subset['timestamp'])
sub_folders = list(all_data_subset['folder_date'])

timestamps

# ## Load data

check_folder = "./"

# +
# load simulations

B = 170
pv = 0.02
f = 0.3
R = 0.04
L = 30

c0_list = []
g_list = []
eta_list = []
mu_list = []
m_init_list = []
max_m_list = []
alpha_list = []
e_list = []
gen_max_list = []
all_phages_list = []
nvi_trajectories_list = []
nbi_trajectories_list = []
t_trajectories_list = []
trajectory_lengths_list = []
mutation_times_list = []
phage_identity_list = []
trajectory_extinct_list = []
theta_list = []

timestamps_list = []

pop_array_list = []

for i, timestamp in tqdm(enumerate(timestamps)):
    
    sub_folder = top_folder + "/%s" %sub_folders[i]
    folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, sub_folder)

    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
     max_m, mutation_times, parent_list, all_phages = load_simulation(folder, timestamp, return_parents = True);
    
    F = f*g*c0
    r = R*g*c0
    
    try: #load pre-calculated trajectory info
        with open('%s/trajectories_%s.pickle' %(check_folder, timestamp), 'rb') as handle:
            trajectory_info = pickle.load(handle)
        #print("loaded trajectories: %s" %timestamp)
        
    except: # get trajectories  
        t_ss = gen_max / 5 # time in bacterial generations at which sim is assumed to be in steady state
        t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)
        
        nvi = pop_array[t_ss_ind:, max_m + 1: 2*max_m+1].toarray()
        nbi = pop_array[t_ss_ind:, 1: max_m+1].toarray()

        (nvi_trajectories, nbi_trajectories, t_trajectories, nbi_acquisitions, phage_size_at_acquisition,  
            trajectory_lengths, trajectory_extinct, acquisition_rate, phage_identities) = get_trajectories(pop_array, 
                        nvi, nbi, f, g, c0, R, eta, alpha, e, pv, B, mu, max_m, m_init, t_ss_ind, return_fitness = False,
                            trim_at_max_size = True, aggressive_trim_length = 1200)

        # make object to save
        trajectory_info = [nvi_trajectories, nbi_trajectories, t_trajectories, 
                           trajectory_lengths, trajectory_extinct, phage_identities]

        # pickle trajectory info
        with open('trajectories_%s.pickle' %timestamp, 'wb') as handle:
            pickle.dump(trajectory_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("trajectories saved: %s" %timestamp)
        
    # make list of lists
    nvi_trajectories_list.append(trajectory_info[0])
    nbi_trajectories_list.append(trajectory_info[1])
    t_trajectories_list.append(trajectory_info[2])
    trajectory_lengths_list.append(trajectory_info[3])
    phage_identity_list.append(trajectory_info[5])
    trajectory_extinct_list.append(trajectory_info[4])
    
    c0_list.append(c0)
    g_list.append(g)
    eta_list.append(eta)
    mu_list.append(mu)
    m_init_list.append(m_init)
    max_m_list.append(max_m)
    alpha_list.append(alpha)
    e_list.append(e)
    gen_max_list.append(gen_max)
    all_phages_list.append(all_phages)
    theta_list.append(theta)
    mutation_times_list.append(mutation_times)
    
    pop_array_list.append(pop_array)
# -

# ### Combine all trajectories from multiple simulations

# +
nvi_trajectories_collected = []
nbi_trajectories_collected = []
t_trajectories_collected = []
trajectory_extinct_collected = []
trajectory_lengths_collected = []

for ind, timestamp in enumerate(timestamps):
    nvi_trajectories_collected += nvi_trajectories_list[ind]
    nbi_trajectories_collected += nbi_trajectories_list[ind]
    t_trajectories_collected += t_trajectories_list[ind]
    trajectory_extinct_collected += trajectory_extinct_list[ind]
    trajectory_lengths_collected += trajectory_lengths_list[ind]

# +
# get theoretical population sizes using predicted m

B = 170
pv = 0.02
f = 0.3
R = 0.04
L = 30

F = f*g*c0
r = R*g*c0

#t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

# doing .toarray() is slow and memory-intensive, so do it once per simulation
#nvi = pop_array[t_ss_ind:, max_m + 1: 2*max_m+1].toarray()
#nbi = pop_array[t_ss_ind:, 1 : max_m+1].toarray()
#nb = np.array(np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)).flatten()
#nv = np.array(np.sum(pop_array[t_ss_ind:, max_m + 1: 2*max_m+1], axis = 1)).flatten()
#t_all = pop_array[t_ss_ind:, -1].toarray().flatten()

m_init_select = 1

row = grouped_data[(grouped_data['C0'].values == c0_select) &
            (grouped_data['mu'].values > mu_select*0.9) &
            (grouped_data['mu'].values < mu_select*1.1) &
            (grouped_data['e'].values == e_select) & 
             (grouped_data['eta'].values == eta_select) &
             (grouped_data['m_init'] == m_init_select) &
            (grouped_data['B'] == B) &
            (grouped_data['pv_type'] == 'binary')]

nb_pred = float(row['nb_pred_recursive'])
nv_pred = float(row['nv_pred_recursive'])
C_pred = float(row['C_pred_recursive'])
nu_pred = float(row['nu_pred_recursive'])
nb = float(row['mean_nb']['nanmean'])
nv = float(row['mean_nv']['nanmean'])
C = float(row['mean_C']['nanmean'])
nu = float(row['mean_nu']['nanmean'])
nb0 = (1-nu)*nb
nb0_pred = (1-nu_pred)*nb_pred
c0 = float(row['C0'])
g = 1/(42*c0)
alpha = 2*10**-2 / c0
e = float(row['e'])
eta = float(row['eta'])
mu = float(row['mu'])

nvi_ss_pred = nvi_steady_state(nb_pred, nv_pred, C_pred, nb0_pred, f, g, c0, e, alpha, B, mu, pv, R, eta)
nbi_ss_pred = nbi_steady_state(nb_pred, f, g, c0, e, alpha, B, mu, pv)

# -

# ## Clone size and fitness

# +
# interpolate clone sizes - include all clones, but put nan if trajectory has gone extinct (exclude from mean)

new_times = np.concatenate([np.arange(0,30,0.5), np.arange(30,100,5), np.arange(100,1000,50)])
#fitness_times = np.concatenate([np.arange(0.5,6,0.5), np.arange(6,25,2)])
nvi_interp = interpolate_trajectories(nvi_trajectories_collected, t_trajectories_collected, new_times, g, c0)
nbi_interp = interpolate_trajectories(nbi_trajectories_collected, t_trajectories_collected, new_times, g, c0)

# +
# interpolate trajectories for plot

L = 30 # protospacer length
P0 = np.exp(-mu*L)
delta = f*g*c0 + alpha*nb_pred*(1-pv)
beta = alpha*pv*nb_pred
s = (beta*(B*P0 - 1) - delta)

# WARNING: make sure this spacing is larger than the main interpolation spacing in new_times above.
new_times2 = np.arange(0,500, 10) 
nvi_mean = np.interp(new_times2, new_times, np.nanmean(nvi_interp, axis = 1))
nbi_mean = np.interp(new_times2, new_times, np.nanmean(nbi_interp, axis = 1))
nvi_std = np.interp(new_times2, new_times, np.nanstd(nvi_interp, axis = 1))
nbi_std = np.interp(new_times2, new_times, np.nanstd(nbi_interp, axis = 1))

# with only trajectories that have a spacer get acquired
#fitness_interp_large = fitness_interp[:, np.array(acquisition_inds)] 
#nbi_fitness_interp_large = nbi_fitness_interp[:, np.array(acquisition_inds)]
new_fitness = np.gradient(np.nanmean(nvi_interp, axis = 1), new_times) / np.nanmean(nvi_interp, axis = 1)
new_bac_fitness = np.gradient(np.nanmean(nbi_interp, axis = 1), new_times) / np.nanmean(nbi_interp, axis = 1)
fitness_mean = np.interp(new_times2, new_times, new_fitness)
bac_fitness_mean = np.interp(new_times2, new_times, new_bac_fitness)
#nbi_fitness_mean = np.interp(new_times2, new_times, np.nanmean(nbi_fitness_interp_large, axis = 1))

# +
# get mean establishment time

size_cutoff = nvi_steady_state(nb, nv, C, nb0, 
                 f, g, c0, e, alpha, B, mu, pv, R, eta)
t_trajectories_endpoints = []
t_establish = []
for i, nvi in enumerate(nvi_trajectories_collected):
    if np.any(nvi > size_cutoff):
        t_i = t_trajectories_collected[i]
        trajectory_large = t_i[np.where(nvi >= size_cutoff)[0][0]:]
        t_establish.append(trajectory_large[0] - t_i[0]) # respective to trajectory start
            
        if trajectory_extinct_collected[i] == True: # include only if trajectory goes extinct
            large_trajectory_length = trajectory_lengths_collected[i] - (trajectory_large[0] - t_i[0])*g*c0
            t_trajectories_endpoints.append(large_trajectory_length)
    
t_establish = np.array(t_establish)*g*c0
mean_establishment_time = np.mean(t_establish)

# theoretical establishment timescale: ln2 / s0
s0 = alpha*B*pv*np.mean(nb)*np.exp(-mu*L) - F - alpha*np.mean(nb)
phage_establishment_timescale = (np.log(2)/s0)*g*c0
# -

# ## Establishment vs avg immunity, time to extinction

# +
mu_vals = list(np.unique(grouped_data['mu']))
c0_vals = list(np.unique(grouped_data['C0']))
eta_vals = list(np.unique(grouped_data['eta']))
e_vals = list(np.unique(grouped_data['e']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd']
colours = sns.color_palette("hls", len(c0_vals))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

# +
grouped_data_multisample = grouped_data[grouped_data['mean_m']['count'] > 2]
markersize = 7

colour = 'C0'
shape = 'mu'
line = 'eta'

colour_label = 'C_0'
shape_label = '\mu'
line_label = '\eta'

#e_select = 0.5
B_select = 170
m_init_select = 1
data_subset = grouped_data_multisample[(grouped_data_multisample['C0'] < 10**6) # currently only 2 values here
                                      & (grouped_data_multisample['m_init'] == m_init_select)
                                       & (grouped_data_multisample['mu'] > 4*10**-8)
                                         & (grouped_data_multisample['B'] == B_select)
                                        #& (grouped_data_multisample['eta'] != 10**-4)
                                        #& (grouped_data_multisample['eta'] != 10**-3)
                                        #& (grouped_data_multisample['mu'] < 10**-4)
                                        #& (grouped_data_multisample['mean_m']['nanmean'] >=1)
                                       & (grouped_data_multisample['f'] == 0.3)
                                       & (grouped_data_multisample['pv'] == 0.02)
                                       & (grouped_data_multisample['pv_type'] == 'binary')]

data_subset["pred_e_eff"] = data_subset['e'] / data_subset['pred_bac_m_recursive']

legend_elements = []
shapevals = np.sort(data_subset[shape].unique())
for i in range(len(shapevals)):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(shapevals[i], 2+int(np.abs(np.log10(shapevals[i]))))),
                          markerfacecolor='grey', markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

#for i in range(len(np.sort(data_subset[colour].unique()))):
#    legend_elements.append(Line2D([0], [0], marker='o', 
#                                  label='$%s = %s$' %(colour_label, 
#                                int(np.sort(data_subset[colour].unique())[i])),
#                          markerfacecolor=colours[i], markeredgecolor = 'none', markersize = markersize, linestyle = "None"))

# -

# ## Plot

# +
##### Axes layout #######

fig = plt.figure(figsize = (11.5,7.8))
title_fontsize = 16
absolute_left = 0.06
left_fitness = 0.64
bottom_fitness = 0.56
top_extinct = 0.44
absolute_bottom = 0.08
absolute_top = 0.95
absolute_right = 0.94

# establishment half
gs_clonesize = gridspec.GridSpec(1,1)
gs_clonesize.update(left=absolute_left, right=0.5, bottom = bottom_fitness, top = absolute_top)
ax1 = plt.subplot(gs_clonesize[0])
#ax1 = plt.subplot(gs_clonesize[1])
ax1b = ax1.twinx()

gs_fitness = gridspec.GridSpec(4,2)
gs_fitness.update(left=left_fitness, right=absolute_right, bottom = bottom_fitness, top = absolute_top, wspace=0.29, 
                  hspace = 0.1)

gs_est = gridspec.GridSpec(1,1)
gs_est.update(left=absolute_left, right=0.37, bottom = absolute_bottom, top = top_extinct)
ax3 = plt.subplot(gs_est[0])

#gs2.update(left=0.22, right=0.8, bottom = 0.6, wspace=0.1)
#ax3 = plt.subplot(gs2[-1,-1])

hspace_extinct = 0.1
left_extinct = 0.675
gs_ext = gridspec.GridSpec(1,2)
gs_ext.update(left=0.43, right=0.92,top = top_extinct, bottom = absolute_bottom, wspace=0.15)

#ax1 = plt.subplot(gs1[:-1, :])
ax4 = plt.subplot(gs_ext[0])
ax5 = plt.subplot(gs_ext[1])
#ax8 = plt.subplot(gs_T_ext[0])
#ax9 = plt.subplot(gs_T_ext[1])

# colorbar
gs_cbar = gridspec.GridSpec(1,1)
gs_cbar.update(left=0.93, right=absolute_right, bottom = absolute_bottom, top = top_extinct)
ax_cbar = plt.subplot(gs_cbar[0])
#cmap = sns.color_palette("hls", len(c0_vals))
#cmap = ListedColormap(sns.color_palette('hls', len(c0_vals)).as_hex())
cmap = sns.color_palette("hls", as_cmap=True)
# figure out how much to truncate colormap to end at 10^6
extent = np.log10(3*10**6) - np.log10(300) # log-range of c0 values for original color mapping
new_extent = np.log10(10**6) - np.log10(300) # want to end at 10^6 instead
fraction = (extent - new_extent)/extent
new_cmap =  truncate_colormap(cmap, 0, 1-fraction, n=100)

cbar = fig.colorbar(cm.ScalarMappable(norm= matplotlib.colors.LogNorm(vmin=min(c0_vals), vmax=10**6), cmap=new_cmap),
             cax=ax_cbar, orientation='vertical', label='Nutrient concentration $C_0$')
cbar.ax.fontsize=16


width = 0.365
height = 0.36

ax1.set_title("A", fontsize = title_fontsize, loc = 'left')
ax3.set_title("C", fontsize = title_fontsize, loc = 'left')
ax4.set_title("D", fontsize = title_fontsize, loc = 'left')
ax4.set_title("Bacteria")
ax5.set_title("E", fontsize = title_fontsize, loc = 'left')
ax5.set_title("Phage")


### Plot bacteria and phage extinction

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
            
    bac_ext_std = data['mean_bac_extinction_time']['nanstd'] / phage_establishment_timescale

    yerr = np.stack([np.zeros(bac_ext_std.shape), bac_ext_std])

    ax4.errorbar(data['pred_bac_extinction_time_nodrift'] / phage_establishment_timescale, 
                data['mean_bac_extinction_time']['nanmean'] / phage_establishment_timescale,
                yerr = yerr,
                c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")
    
    phage_ext_std = data['mean_large_trajectory_length_nvi_ss']['nanstd'] / phage_establishment_timescale

    yerr = np.stack([np.zeros(phage_ext_std.shape), phage_ext_std])


    ax5.errorbar(data['mean_T_backwards_nvi_ss_nbi_ss_recursive'] / phage_establishment_timescale,
                data['mean_large_trajectory_length_nvi_ss']['nanmean'] / phage_establishment_timescale,
                yerr = yerr,
                c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")

B = 170
# c0 cancels out
c0 = 10**4
alpha = 2*10**-2/c0
g = 1/(42*c0)
B2 = (B*pv-1)*alpha/g

ax5.set_yscale('log')
ax5.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xscale('log')
ref_line, = ax4.plot([10**-1, 10**5], [10**-1, 10**5], 'k', label = r'$y = x$')
ref_line, = ax5.plot([10**-1, 10**5], [10**-1, 10**5], 'k', label = r'$y = x$')
ax4.set_xlim(10 / phage_establishment_timescale, 2*10**4 / phage_establishment_timescale)
ax4.set_ylim(15 / phage_establishment_timescale, 2*10**4 / phage_establishment_timescale)
ax5.set_xlim(90 / phage_establishment_timescale, 7*10**4 / phage_establishment_timescale)
ax5.set_ylim(100 / phage_establishment_timescale, 2.1*10**4 / phage_establishment_timescale)
ax4.set_xlabel("Predicted neutral time to extinction\n(Phage establishment timescale)")
ax4.set_ylabel("Measured time to extinction")
ax5.set_xlabel("Predicted neutral time to extinction\n(Phage establishment timescale)")
#ax5.set_ylabel("Measured phage time to extinction")

#ax4.legend(handles=legend_elements, loc='lower right', ncol = 1, fontsize = 9)

### Plot clone fitness

# growth rate for clones conditioned on survival
points = ax1.plot(new_times2 / phage_establishment_timescale, fitness_mean, alpha = 0.8, marker = 'v', linestyle = ':', 
                 markersize = 5, color = 'orangered', label = "Phage")

#ax1.plot(new_times2 / phage_establishment_timescale, bac_fitness_mean, alpha = 0.8, marker = 'o', linestyle = ':', 
#                 markersize = 5, color = 'lightseagreen', label = "Bacteria")

nvi_fitness_std = np.nanstd(np.gradient(nvi_interp, new_times,  axis = 0), axis = 1) / np.nanmean(nvi_interp, axis = 1)
nvi_fitness_std = np.interp(new_times2, new_times, nvi_fitness_std)
#bac_fitness_std = np.nanstd(np.gradient(nbi_interp, new_times,  axis = 0), axis = 1) / np.nanmean(nbi_interp, axis = 1)
#bac_fitness_std = np.interp(new_times2, new_times, bac_fitness_std)
ax1.fill_between(new_times2 / phage_establishment_timescale, nvi_fitness_std + fitness_mean, y2 = fitness_mean - nvi_fitness_std, 
                color = 'orangered', alpha = 0.1)
#ax1.fill_between(new_times2 / phage_establishment_timescale, bac_fitness_mean + bac_fitness_std, y2 = bac_fitness_mean - bac_fitness_std, 
#                color = 'lightseagreen', alpha = 0.1)

#ax1.axvline(mean_establishment_time / phage_establishment_timescale, linestyle = '-.', color = 'k',
#           label = 'Mean phage\nestablishment time')
ax1b.axvline(mean_establishment_time / phage_establishment_timescale, linestyle = '-.', color = 'k',
           label = 'Mean phage\nestablishment time')

# mean phage clone size
ax1b.plot(new_times2 / phage_establishment_timescale, nvi_mean,
             color = 'rebeccapurple', label = 'Phage', marker = 'v', markersize = 5,
        linestyle = '--', linewidth = 1, alpha = 1)
ax1b.fill_between(new_times2 / phage_establishment_timescale, nvi_std + nvi_mean, y2 = nvi_mean - nvi_std, color = 'rebeccapurple', 
                  alpha = 0.1)

# mean bacteria clone size
ax1b.plot(new_times2 / phage_establishment_timescale, nbi_mean, color = 'lightseagreen', 
          label = 'Bacteria', marker = 'o', markersize = 5,
        linestyle = '--', linewidth = 1, alpha = 1)
ax1b.fill_between(new_times2 / phage_establishment_timescale, nbi_std + nbi_mean, y2 = nbi_mean - nbi_std, color = 'lightseagreen', alpha = 0.1)

ax1.set_xlim(0,300 / phage_establishment_timescale)
ax1b.set_xlim(0,300 / phage_establishment_timescale)
#ax1b.set_xticks([])
#ax.set_ylim(np.min(fitness_mean)*2, np.max(fitness_mean)*1.2)
ax1.set_ylim(np.abs(np.nanmin(fitness_mean[~np.isinf(fitness_mean)])/2), np.nanmax(fitness_mean[~np.isinf(fitness_mean)])*1.2)
#ax.set_yscale('log')
ax1b.set_yscale('log')
ax1b.set_ylim(8*10**-1, np.nanmax(nvi_mean)*1.8)

# colour of clone size left axis
ax1.spines['left'].set_color(points[0].get_color())
ax1.yaxis.label.set_color(points[0].get_color())
ax1.tick_params(axis='y', colors=points[0].get_color())

ax1b.annotate("High fitness\nregime", xy = (0,0), xytext = (0.28,0.6), xycoords = 'axes fraction')
ax1b.annotate("Low fitness\nregime", xy = (0,0), xytext = (0.7,0.6), xycoords = 'axes fraction')  

ax1b.axhline(nvi_ss_pred,  linestyle = ':', color = 'rebeccapurple', label = 'Mean clone size')
ax1b.axhline(nbi_ss_pred,  linestyle = ':', color = 'lightseagreen')

ax1.set_xlabel("Time since phage mutation\n(phage establishment timescale)")
ax1.set_ylabel("Phage clone growth rate (per capita)")
ax1b.set_ylabel("Average clone size")

ax1b.legend(loc = 'lower right', bbox_to_anchor = (1, 0.02), fontsize = 9)
#ax1b.legend(loc = 'lower right', bbox_to_anchor= (1, 0.58))

# clone fitness histograms

times = [0.5, 4.5, 10, 300]

for i, time in enumerate(times):
    t_ind = find_nearest(new_times, time)

    gradient_subset = np.gradient(nvi_interp, new_times, axis = 0)[t_ind]
    gradient_subset_per_capita = gradient_subset / nvi_interp[t_ind]
    
    bac_gradient_subset = np.gradient(nbi_interp, new_times, axis = 0)[t_ind]
    bac_gradient_subset_per_capita = bac_gradient_subset / nbi_interp[t_ind]
    #gradient_subset = gradient_subset[~np.isnan(gradient_subset)]

    ax2a = plt.subplot(gs_fitness[i, 0])
    ax2b = plt.subplot(gs_fitness[i, 1])
    
    bins = ax2a.hist(gradient_subset_per_capita, bins = np.linspace(-3, 3, 20), color = 'rebeccapurple')
    bins = ax2b.hist(bac_gradient_subset_per_capita, bins = np.linspace(-3, 3, 20), color = 'lightseagreen')
    
    ax2a.axvline(0, color = 'k', linestyle = '--')
    ax2b.axvline(0, color = 'k', linestyle = '--')

    #ax2a.set_ylabel("# of clones")
    ax2a.annotate("Time\n%s" %round(new_times[t_ind] / phage_establishment_timescale, 2), 
                  xy = (0,0), xytext = (0.05,0.58), xycoords = 'axes fraction') 
    ax2a.set_xlim(-2.1, 2.1)
    ax2b.set_xlim(-2.1, 2.1)
    
    #ax2a.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #ax2a.ticklabel_format(style='plain')
    
    if i < len(times) - 1:
        ax2a.set_xticks([])
        ax2b.set_xticks([])

    if i == 0:
        ax2a.set_title("Phage")
        ax2b.set_title("Bacteria")
        ax2b.set_yticks([])

# add common labels
#fig.text(left_fitness*0.96, 0.96, "C", fontsize = title_fontsize)
ax2a = plt.subplot(gs_fitness[0, 0])
ax2a.set_title("B", loc = "left", fontsize = title_fontsize)
fig.text((left_fitness + 0.98) / 2, bottom_fitness*0.88, 
         'Per capita clone growth rate\nper bacterial generation', ha='center')
fig.text(left_fitness*0.91, (1-bottom_fitness*0.45), '# of clones', va='center', rotation='vertical')

####### Establishment vs avg immunity

data_subset2 = data_subset[(data_subset['eta'] != 10**-3)
                        & (data_subset['eta'] != 10**-5)]

for group in data_subset2.groupby([colour, shape, line]):
    data = group[1].sort_values(by = 'm_init')

    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]

    colour_ind = list(np.sort(data_subset2[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset2[shape].unique())).index(shape_variable)

    P_est_std = ((data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean']) 
            * np.sqrt((data['establishment_rate_nvi_ss']['nanstd'] / data['establishment_rate_nvi_ss']['nanmean'])**2 
            + (data['measured_mutation_rate']['nanstd'] / data['measured_mutation_rate']['nanmean'])**2 ))

    yerr = np.stack([np.zeros(P_est_std.shape), P_est_std])

    xerr = np.stack([np.zeros(data['e_effective']['nanstd'].shape), data['e_effective']['nanstd']])

    ax3.errorbar(  data['e_effective']['nanmean'], 
            data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'],
                yerr = yerr, 
                xerr = xerr,
            c = colours[colour_ind], 
       alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")
    
    


# plot theoretical lines
approxs = []
for group in data_subset2.groupby([line]):
    data = group[1].sort_values(by = "pred_e_eff")
    #data = data[~np.isnan(data['predicted_establishment_fraction_recursive'])]
    line_variable = group[0]
    line_ind = list(np.sort(data_subset2[line].unique())).index(line_variable)
    
    e = data['e']

    ax3.plot(e/data['pred_bac_m_recursive'], 
        data['predicted_establishment_fraction_recursive'],
        linestyle = '-', color = 'k')
    
    # calculate approximation
    
    c0 = data['C0']
    g = 1/(42*c0)
    alpha = 2*10**-2 / c0
    eta = data['eta']
    e = data['e']
    m = data['pred_bac_m_recursive']
    r = R*g*c0
    
    nuapprox_e0 = nuapprox_small_e(f,g,c0,alpha,pv,B,R,eta,0,m)
    nuapprox_small_eta = nuapprox_small_e(f,g,c0,alpha,pv,B,R,eta,e,m)
    
    P_est_e0 = 2*e*nuapprox_e0/(m*(B-1))
    P_est_small_eta = 2*e*nuapprox_small_eta/(m*(B-1))
    
    t, = ax3.plot((e/data['pred_bac_m_recursive'])[P_est_small_eta > 0], 
        P_est_small_eta[P_est_small_eta > 0],
        linestyle = linestyles[line_ind], color = 'blue', alpha = 0.8, label = r"Small $e_{eff}$")
    approxs.append(t)
e_over_m_vals = np.logspace(-3,0,50)

t2, = ax3.plot(e_over_m_vals, 2*e_over_m_vals/(B-1), linestyle = "-", color = 'green', alpha = 0.8, label = r"Large $\nu$")  # nu = 1

ax3.annotate(r"$\eta = %s$" %eta_vals[1], xy = (0,0), xytext = (0.45,0.14), xycoords = 'axes fraction')    
ax3.annotate(r"$\eta = %s$" %eta_vals[-1], xy = (0,0), xytext = (0.25,0.7), xycoords = 'axes fraction') 

ax3.set_yscale('log')
ax3.set_xscale('log')

xmin = 4.3*10**-3
xmax = 1.*10**0

ax3.set_xlim(xmin, xmax)

ax3.set_ylim(5*10**-7, 3.5*10**-2)


ax3.set_xlabel(r"Average bacterial immunity")
ax3.set_ylabel("Phage establishment probability")  

#legend_elements2 = []
legend_elements.append(Line2D([0], [0], label = 'Theory', linestyle = '-', color = 'k'))
legend_elements.append(approxs[0])
legend_elements.append(t2)

ax3.legend(handles = legend_elements, ncol = 1, fontsize = 8, loc = 'lower right')

#### Save

plt.savefig("establishment_extinction_figure_c0_%s_eta_%s_e_%s_mu_%s.pdf" %(c0_select, eta_select, e_select, mu_select))
plt.savefig("establishment_extinction_figure_c0_%s_eta_%s_e_%s_mu_%s.svg" %(c0_select, eta_select, e_select, mu_select))
# -

# ## Presentation plots

# +
colours_1 = ['#1f78b4', '#b2df8a' ] # from colorbrewer
colours_2 = ['#08519c','#3182bd','#6baed6','#bdd7e7'] # for crossreactivity

fig, axs = plt.subplots(1,2, figsize = (9,4.5))

ax3 = axs[0]
ax3b = axs[1]

####### Establishment vs avg immunity

data_subset2 = data_subset[(data_subset['eta'] != 10**-3)
                        & (data_subset['eta'] != 10**-5)]

for group in data_subset2.groupby([colour, shape, line]):
    data = group[1].sort_values(by = 'm_init')

    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]

    colour_ind = list(np.sort(data_subset2[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset2[shape].unique())).index(shape_variable)
    line_ind = list(np.sort(data_subset2[line].unique())).index(line_variable)

    P_est_std = ((data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean']) 
            * np.sqrt((data['establishment_rate_nvi_ss']['nanstd'] / data['establishment_rate_nvi_ss']['nanmean'])**2 
            + (data['measured_mutation_rate']['nanstd'] / data['measured_mutation_rate']['nanmean'])**2 ))

    yerr = np.stack([np.zeros(P_est_std.shape), P_est_std])

    xerr = np.stack([np.zeros(data['e_effective']['nanstd'].shape), data['e_effective']['nanstd']])

    ax3.errorbar(  data['e_effective']['nanmean'], 
            data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'],
                yerr = yerr, 
                xerr = xerr,
            c = colours_1[line_ind], 
       alpha = 0.85, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")

# plot theoretical lines
for group in data_subset2.groupby([line]):
    data = group[1].sort_values(by = "pred_e_eff")
    #data = data[~np.isnan(data['predicted_establishment_fraction_recursive'])]
    line_variable = group[0]
    line_ind = list(np.sort(data_subset2[line].unique())).index(line_variable)
    
    e = data['e']

    ax3.plot(e/data['pred_bac_m_recursive'], 
        data['predicted_establishment_fraction_recursive'],
        linestyle = '-', color = 'k')

ax3.annotate("Spacer acquisition\n" + r"probability $ = %s$" %eta_vals[1], xy = (0,0), xytext = (0.48,0.18), xycoords = 'axes fraction')    
ax3.annotate("Spacer acquisition\n" + r"probability $ = %s$" %eta_vals[-1], xy = (0,0), xytext = (0.1,0.7), xycoords = 'axes fraction') 
ax3b.annotate("Spacer acquisition\n" + r"probability $ = %s$" %eta_vals[1], xy = (0,0), xytext = (0.1,0.8), xycoords = 'axes fraction') 

ax3.set_yscale('log')
ax3.set_xscale('log')

xmin = 4.3*10**-3
xmax = 1.*10**0

ax3.set_xlim(xmin, xmax)

ax3.set_ylim(5*10**-7, 4*10**-2)
ax3b.set_ylim(5*10**-7, 4*10**-2)

ax3.set_xlabel("Average immunity")
ax3.set_ylabel("Phage establishment probability")  

### Establishment with crossreactivity
pv_types = ['binary', 'exponential', 'exponential_025', 'theta_function']

eta_select2 = 10**-4
mu_select = 10**-6
m_init_select = 1
data_subset3 = grouped_data[(grouped_data['m_init'] == m_init_select)
                                    & (grouped_data['eta'] == eta_select2)
                                   & (grouped_data['mu'] == mu_select)
                                   #& (grouped_data['C0'] == 10000)
                                    & (grouped_data['mean_m']['nanmean'] >= 1)
                                    & (grouped_data['theta'] < 2)
                                    & (grouped_data['B'] == 170)
                                    & (grouped_data['f'] == 0.3)
                                   & (grouped_data['pv'] == 0.02)]
                                    #& (grouped_data['pv_type'] == 'exponential')]

data_subset3["pred_e_eff"] = data_subset3['e'] / data_subset3['pred_bac_m_recursive']

for group in data_subset3.groupby([colour, shape, line, 'pv_type']):
    data = group[1].sort_values(by = 'm_init')
    if len(data) > 0:
        colour_variable = group[0][0]
        shape_variable = group[0][1]
        line_variable = group[0][2]
        pv_type = group[0][3]

        colour_ind = list(np.sort(data_subset3[colour].unique())).index(colour_variable)
        shape_ind = list(np.sort(data_subset3[shape].unique())).index(shape_variable)
        line_ind = list(np.sort(data_subset3[line].unique())).index(line_variable)
        pv_ind = pv_types.index(pv_type)
        
    P_est_std = ((data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean']) 
            * np.sqrt((data['establishment_rate_nvi_ss']['nanstd'] / data['establishment_rate_nvi_ss']['nanmean'])**2 
            + (data['measured_mutation_rate']['nanstd'] / data['measured_mutation_rate']['nanmean'])**2 ))

    yerr = np.stack([np.zeros(P_est_std.shape), P_est_std])

    #if np.any(data['pv_type'] == 'exponential'):
    ax3b.errorbar(data['e_effective']['nanmean'], 
                data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'],
                    xerr = data['e_effective']['nanstd'], 
                  yerr = yerr,
                c = colours_2[pv_ind], 
           alpha = 0.85, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")

# connect sim data points with line
for group in data_subset3.groupby('pv_type'):
    data = group[1].sort_values(by = ('e_effective', 'nanmean'))
    pv_type = group[0]
    
    # keep only nonzero establishment rates for the line
    data = data[data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'] > 0]
    
    #linestyle by pv type
    line_ind = ['binary', 'exponential', 'exponential_025', 'theta_function'].index(pv_type)
    
    ax3b.plot( data['e_effective']['nanmean'], 
            data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'],
            linestyle = linestyles[line_ind], color = 'grey', alpha = 0.5, label = pv_type)
    

# plot theoretical line
data = data_subset3[data_subset3['pv_type'] == 'binary'].sort_values(by = "pred_e_eff")
e = data['e']

t, = ax3b.plot(e/data['pred_bac_m_recursive'], 
    data['predicted_establishment_fraction_recursive'],
    linestyle = '-', color = 'k', linewidth = 2, zorder = 2, label = 'Theory')

ax3b.annotate("Increasing\ncross-reactivity",
            xy=(3*10**-1, 5*10**-6), xycoords='data',
            xytext=(1.5*10**-2, 1*10**-4), textcoords='data',
            arrowprops=dict(facecolor='grey', arrowstyle="->"))

ax3b.set_xscale('log')
ax3b.set_yscale('log')


ax3b.set_yticks([])
#ax3b.set_ylabel("Establishment probability")
ax3b.set_xlabel("Average immunity")

plt.tight_layout()
plt.savefig("p_est_vs_average_immunity_presentation.png", dpi = 350)
# -

# ## Supplementary plots

# +
# set logm to False to drop the 1 + lnm term
logm = True

fig, axs = plt.subplots(2,2, figsize = (6,6), facecolor = 'white')

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    for i, row in data.iterrows():
        
        eta_ind = eta_vals.index(float(row['eta']))
        
            
        m = row['mean_m']['nanmean']
        nv = row['mean_nv']['nanmean']
        nb = row['mean_nb']['nanmean']
        c0 = row['C0']

        yerr = row['mean_large_trajectory_length_nvi_ss']['nanstd'] / phage_establishment_timescale
        yerr = np.stack([np.zeros(yerr.shape), yerr])
        yerr= yerr[:, np.newaxis]

        if logm == True:
            x = (1+np.log(m))*nv/m
        elif logm == False:
            x = nv/m
        axs.flatten()[eta_ind].errorbar(x, row['mean_large_trajectory_length_nvi_ss']['nanmean'] / phage_establishment_timescale,
            yerr = yerr,
            #xerr = xerr,
            c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")

        
xvals = np.arange(10**3, 5*10**6, 10**4)
B = 170
# c0 cancels out
c0 = 10**4
alpha = 2*10**-2/c0
g = 1/(42*c0)
B2 = (B*pv-1)*alpha/g

for i, ax in enumerate(axs.flatten()):
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1.2*10**3, 2*10**6)
    ax.set_ylim(2, 6*10**2)
    
    ax.set_title(r"$\eta = %s$" %eta_vals[i])
    
    t1, = ax.plot(xvals, (xvals * 2*(1-1/(B*pv))/(f*(B-1))) /phage_establishment_timescale, 
                'k-', label = "Theory")
    
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,1].set_yticks([])

if logm == True:
    label = r"$n_V\frac{(1 + \ln m)}{m}$"
else:
    label = r"$\frac{n_V}{m}$"
    
    
axs[1,0].set_xlabel(label, fontsize = 13)
axs[1,1].set_xlabel(label, fontsize = 13)
axs[0,0].set_ylabel("Phage mean time to extinction\n(bacterial generations)")
axs[1,0].set_ylabel("Phage mean time to extinction\n(bacterial generations)")
axs[1,0].legend(handles = legend_elements[:7], ncol = 1, fontsize = 7)
axs[1,1].legend(handles = legend_elements[7:-1], ncol = 1, fontsize = 7)
plt.tight_layout()

plt.savefig("phage_extinction_eta_split_logm_%s.svg" %logm)
plt.savefig("phage_extinction_eta_split_logm_%s.pdf" %logm)

# +
# set logm to False to drop the 1 + lnm term
logm = True

fig, axs = plt.subplots(2,2, figsize = (6,6), facecolor = 'white')

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    for i, row in data.iterrows():
        
        eta_ind = eta_vals.index(float(row['eta']))
        
            
        m = row['mean_m']['nanmean']
        nv = row['mean_nv']['nanmean']
        nb = row['mean_nb']['nanmean']
        c0 = row['C0']

        yerr = row['mean_bac_extinction_time']['nanstd'] / phage_establishment_timescale
        yerr = np.stack([np.zeros(yerr.shape), yerr])
        yerr= yerr[:, np.newaxis]

        if logm == True:
            x = (1+np.log(m))*nb/m
        else:
            x = nb/m
        
        axs.flatten()[eta_ind].errorbar(x, row['mean_bac_extinction_time']['nanmean'] / phage_establishment_timescale,
                    yerr = yerr,
                    c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")
        
xvals = np.arange(10**0, 5*10**6, 10**4)
B = 170
# c0 cancels out
c0 = 10**4
alpha = 2*10**-2/c0
g = 1/(42*c0)
B2 = (B*pv-1)*alpha/g

for i, ax in enumerate(axs.flatten()):
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1.2*10**1, 2*10**4)
    ax.set_ylim(0.2, 6*10**2)
    
    ax.set_title(r"$\eta = %s$" %eta_vals[i])
    
    eta = eta_vals[i]
    
    tilde_nv = nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta)
    t2, = ax.plot(xvals, xvals*g*c0/(f*g*c0 + alpha*pv*tilde_nv) / phage_establishment_timescale, 'k-', 
                   label = "Theory")
    
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,1].set_yticks([])

if logm == True:
    label = r"$n_B\frac{(1 + \ln m)}{m}$"
else:
    label = r"$\frac{n_B}{m}$"
    
    
axs[1,0].set_xlabel(label, fontsize = 13)
axs[1,1].set_xlabel(label, fontsize = 13)
axs[0,0].set_ylabel("Bacteria mean time to extinction\n(bacterial generations)")
axs[1,0].set_ylabel("Bacteria mean time to extinction\n(bacterial generations)")
axs[1,0].legend(handles = legend_elements[:7], ncol = 1, fontsize = 7)
axs[1,1].legend(handles = legend_elements[7:-1], ncol = 1, fontsize = 7)
plt.tight_layout()

plt.savefig("bacteria_extinction_eta_split_logm_%s.svg" %logm)
plt.savefig("bacteria_extinction_eta_split_logm_%s.pdf" %logm)
# -

# ### Bacteria extinction including drift term

# +
### Plot bacteria and phage extinction

colour = 'C0'
shape = 'eta'

colour_label = "C_0"
shape_label = "\eta"

legend_elements = []
shapevals = np.sort(data_subset[shape].unique())
for i in range(len(shapevals)):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(shapevals[i], 2+int(np.abs(np.log10(shapevals[i]))))),
                          markerfacecolor='grey', markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

for i in range(len(np.sort(data_subset[colour].unique()))):
    legend_elements.append(Line2D([0], [0], marker='o', 
                                  label='$%s = %s$' %(colour_label, 
                                int(np.sort(data_subset[colour].unique())[i])),
                          markerfacecolor=colours[i], markeredgecolor = 'none', markersize = markersize, linestyle = "None"))

fig, ax = plt.subplots()

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
            
    bac_ext_std = data['mean_bac_extinction_time']['nanstd'] / phage_establishment_timescale

    yerr = np.stack([np.zeros(bac_ext_std.shape), bac_ext_std])

    ax.errorbar(data['pred_bac_extinction_time_numeric'], 
                data['mean_bac_extinction_time']['nanmean'],
                yerr = yerr,
                c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = markersize, linestyle = "None")
    
ax.plot([10**0, 10**5], [10**0, 10**5], color = 'k')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Predicted bacteria time to extinction")
ax.set_ylabel("Measured bacteria mean time to extinction")
ax.set_xlim(5*10**0,)
ax.set_ylim(10**1, 2*10**4)
ax.legend(handles = legend_elements[:-1], ncol = 2, fontsize = 8)
plt.savefig("bac_extinction_time_numeric.pdf")
# -

# ## Bacterial acquisition time

# matches the mathematica definition except for regularization - just need to multiply by gamma(0)
from scipy.special import gammaincc
from scipy.special import gamma
# these don't work if the first argument is zero, but can use a really small positive number to get close

# +
def mean_acquisition_time_v1(acquisition, s0):
    eulergamma = 0.577216
    a = 10**-20 # arbitrary very small number to evaluate gamma functions
    return (1/s0)*np.exp(acquisition/s0)*(eulergamma + gammaincc(a,acquisition/s0)*gamma(a) + np.log(acquisition/s0))

def mean_acquisition_time(acquisition, s0):
    a = 10**-20 # arbitrary very small number to evaluate gamma functions
    return (1/s0)*np.exp(acquisition/s0)*(gammaincc(a,acquisition/s0)*gamma(a))

def P_acquisition(acquisition, s0, t):
    """
    Probability of first acquisition happening at time t
    """
    return acquisition*np.exp(s0*t)*np.exp((-acquisition/s0)*(np.exp(s0*t)-1))


# +
mu_vals = list(np.unique(grouped_data['mu']))
c0_vals = list(np.unique(grouped_data['C0']))
eta_vals = list(np.unique(grouped_data['eta']))
e_vals = list(np.unique(grouped_data['e']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd', 'X', 'h']
colours = sns.color_palette("hls", len(c0_vals))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

# +
c0_select = 10**3
eta_select = 10**-5

data_subset = grouped_data[(grouped_data['m_init'] == 1)
            # &(grouped_data['C0'] == c0_select)
            & (grouped_data['mu'] > 4*10**-9)
             #& (grouped_data['e'] == e_select)  
            # & (grouped_data['eta'] == eta_select) 
            & (grouped_data['B'] == 170)
            & (grouped_data['pv'] == 0.02)
            & (grouped_data['f'] == 0.3)
            & (grouped_data['pv_type'] == 'binary')]

# +
colour = 'C0'
shape = 'e'
#line = 'eta'

colour_label = 'C_0'
shape_label = 'e'
#line_label = '\eta'

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


# +
fig, axs = plt.subplots(2,2, figsize = (8,6), facecolor = 'white')

e_select = 0.1
mu_select = 10**-6
c0_select = 10**5

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    for i, row in data.iterrows():
        
        eta_ind = eta_vals.index(float(row['eta']))
        
        nvi_ss = row['mean_nv']['nanmean'] / row['rescaled_phage_m']['nanmean']

        axs.flatten()[eta_ind].errorbar(nvi_ss, row['mean_size_at_acquisition']['nanmean'],
            yerr = row['mean_size_at_acquisition']['nanstd'],
            #xerr = xerr,
            c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")
        
    


for i, ax in enumerate(axs.flatten()):
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(8*10**2, 9*10**6)
    ax.set_ylim(5*10**1, 10**6)
    
    ax.set_title(r"$\eta = %s$" %eta_vals[i])
    
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,1].set_yticks([])

axs[1,0].set_xlabel("Deterministic mean phage clone size")
axs[1,1].set_xlabel("Deterministic mean phage clone size")
axs[0,0].set_ylabel("Mean phage clone size at\ntime of first acquisition")
axs[1,0].set_ylabel("Mean phage clone size at\ntime of first acquisition")
axs[1,0].legend(handles = legend_elements[:6], ncol = 1, fontsize = 7)
axs[1,1].legend(handles = legend_elements[6:], ncol = 1, fontsize = 7)
plt.tight_layout()
plt.savefig("phage_clone_size_at_acquisition_%s.pdf" %shape)

# +
fig, ax = plt.subplots(figsize = (8,6), facecolor = 'white')

e_select = 0.1
mu_select = 10**-6
c0_select = 10**5

for group in data_subset.groupby([colour, shape]):
    data = group[1]

    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    for i, row in data.iterrows():
        
        eta_ind = eta_vals.index(float(row['eta']))
        
        c0 = row['C0']
        eta = row['eta']
        alpha = 2*10**-2/c0
        g = 1/(42*c0)
    
        nb = row['mean_nb']['nanmean']
        nu = row['mean_nu']['nanmean']
        nb0 = nb*(1-nu)

        acquisition = float(alpha*eta*(1-pv)*nb0)
        s0 = float(alpha*nb*(B*pv-1) - f*g*c0)
        
        acq_time = mean_acquisition_time(acquisition, s0)
        
        est_size = np.exp(s0*acq_time)
        
        ax.errorbar(est_size, row['mean_size_at_acquisition']['nanmean'],
            yerr = row['mean_size_at_acquisition']['nanstd'],
            #xerr = xerr,
            c = colours[colour_ind], 
           alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")
        
ax.plot([10**1, 10**6], [10**1, 10**6], color = 'k')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(3*10**0, 5*10**5)
ax.set_ylim(3*10**0, 5*10**5)
ax.legend(handles = legend_elements, ncol = 2)

ax.set_ylabel("Mean phage clone size at time of first acquisition")
ax.set_xlabel("Predicted phage clone size at mean time of first acquisition")

plt.tight_layout()

plt.savefig("measured_vs_predicted_phage_size_at_acquisition.pdf")
# -

# ### Mean time to extinction vs. bacterial establishment

# +
fig, ax = plt.subplots(figsize = (7,6))

for group in grouped_data_multisample.groupby(['C0', 'eta', 'e', 'm_init', 'pv_type']):
    data = group[1].sort_values(by = 'mu')
    if group[0][3] != 1: # use m_init == 1
        continue
    if group[0][4] != 'binary': # plot only binary pv here
        continue
        
    # remove results with long equilibration times
    #data = data[data['mean_T_backwards_nvi_ss_nbi_ss_recursive'] < 2*10**4]

    #if group[0][1] > 10**-5:
    #    continue
        
    #if group[0][0] > 1000:
    #    continue
    
    #if not np.all(data['e'] == 0.95):
    #    continue
    
    c0 = group[0][0]
    eta = group[0][1]
        
    c0_ind = list(grouped_data_multisample['C0'].unique()).index(c0)
    eta_ind = list(np.sort(grouped_data_multisample['eta'].unique())).index(eta)
    e = group[0][2]
    e_ind = list(np.sort(grouped_data_multisample['e'].unique())).index(e)

    for i, row in data.iterrows():
        
        #if float(row['mu']) < 10**-6:
        #    continue
            
        #nvi_ss = row['mean_nv']['nanmean'] / row['rescaled_phage_m']['nanmean']
        
        #nvi_ss = row['mean_size_at_acquisition']['nanmean'] # phage clone size at spacer acquisition
        
        nvi_ss = nvi_steady_state(row['mean_nb']['nanmean'], row['mean_nv']['nanmean'], row['mean_C']['nanmean'], 
                                       row['mean_nb']['nanmean']*(1-row['mean_nu']['nanmean']), 
                                       f, g, c0, e, alpha, B, float(row['mu']), 
                                          pv, R, eta)
        
        F = f*g*c0
        alpha = 2*10**-2/c0
        g = 1/(42*c0)
        beta = row['mean_nb']['nanmean']*alpha*pv
        delta = F + alpha*row['mean_nb']['nanmean']*(1-pv)
        s = beta*(B-1) - delta
        
        #nvi_ss = B*(s + delta)/(2*s) # establishment phage clone size
        
        #freq = nvi_ss / row['mean_nv']['nanmean']
        #mean_T_backwards_large_phage_clone = 2*row['mean_nv']['nanmean']*freq*(1-np.log(freq))*g*c0/((B-1)**2 * beta + delta)

        D = (nvi_ss*(2*10**-2)/row['C0']*row['eta']*(1-row['mean_nu']['nanmean'])
                 *row['mean_nb']['nanmean']*(1-pv))*g*c0


        if row['rescaled_phage_m']['nanmean'] > 1.2*row['mean_m']['nanmean']:
            ax.errorbar(1/(D),
                      #  mean_T_backwards_large_phage_clone,
                    row['mean_large_trajectory_length_nvi_ss']['nanmean'],
                    #yerr =  row['mean_large_trajectory_length_nvi_ss']['nanstd'],
                    c = colours[c0_ind], 
                   alpha = 0.5, marker = markerstyles[eta_ind], mec ='r', mew = 3, markersize = 8, linestyle = "None")
        else:
            #continue
            ax.errorbar(1/(D),
                       # mean_T_backwards_large_phage_clone,
                    row['mean_large_trajectory_length_nvi_ss']['nanmean'],
                    #yerr =  row['mean_large_trajectory_length_nvi_ss']['nanstd'],
                    c = colours[c0_ind], 
               alpha = 0.5, marker = markerstyles[eta_ind], mec ='k', markersize = 8, linestyle = "None")

ref_line, = ax.plot([10**2, 10**5], [10**2, 10**5], 'k', label = r'$y = x$')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_xlim(0.06, 40)
ax.set_ylim(7*10**1, 1.6*10**4)
ax.legend(handles=legend_elements, loc='lower left', ncol = 2, fontsize = 9)
ax.set_xlabel("Bacterial spacer acquisition timescale (bacterial generations)\n" + r"$=1/(\alpha  \eta (1-p_V) n_B^0 {n_V^i}^* g C_0)$", fontsize = 12)
ax.set_ylabel("Mean time to extinction for large phage clones", fontsize = 12)
plt.tight_layout()
plt.savefig("Phage_extinction_vs_bacteria_acquisition.pdf") 
# -

# ### Phage fitness vs. mu/eta

# +
fig, ax = plt.subplots(figsize = (7,5))

for group in grouped_data_multisample.groupby(['C0', 'eta', 'e', 'm_init', 'pv_type']):
    data = group[1].sort_values(by = 'mu')
    if group[0][3] != 1: # use m_init == 1
        continue
    if group[0][4] != 'binary': # plot only binary pv here
        continue
        
    # remove results with long equilibration times
    #data = data[data['mean_T_backwards_nvi_ss_nbi_ss_recursive'] < 2*10**4]

    #if group[0][1] > 10**-5:
    #    continue
        
    #if group[0][0] > 1000:
    #    continue
    
    #if not np.all(data['e'] == 0.95):
    #    continue
    
    c0 = group[0][0]
    eta = group[0][1]
        
    c0_ind = list(grouped_data_multisample['C0'].unique()).index(c0)
    eta_ind = list(np.sort(grouped_data_multisample['eta'].unique())).index(eta)
    e = group[0][2]
    e_ind = list(np.sort(grouped_data_multisample['e'].unique())).index(e)

    F = f*g*c0
    alpha = 2*10**-2/c0
    g = 1/(42*c0)

    for i, row in data.iterrows():
        
        D = (row['mean_size_at_acquisition']['nanmean']*((2*10**-2)/row['C0'])*row['eta']*(1-row['mean_nu']['nanmean'])
                 *row['mean_nb']['nanmean']*(1-pv))*g*c0

        s0 = alpha*pv*row['mean_nb']['nanmean']*(B-1) - f*g*c0 - alpha*(1-pv)*row['mean_nb']['nanmean']
        d0 = f*g*c0 + alpha*(1-pv)*row['mean_nb']['nanmean']

        N_est = (B*(s0 + d0))/(2*s0)

        if row['rescaled_phage_m']['nanmean'] > 1.2*row['mean_m']['nanmean']:
            ax.errorbar(row['eta'] / row['mu'], 
                        row['fitness_discrepancy']['nanmean'] /  row['fitness_at_mean_acquisition']['nanmean'],
                    c = colours[c0_ind], 
                   alpha = 0.5, marker = markerstyles[eta_ind], mec ='r', mew = 3, markersize = 8, linestyle = "None")
        else:
            #continue
            ax.errorbar(row['eta'] / row['mu'], 
                        row['fitness_discrepancy']['nanmean'] / row['fitness_at_mean_acquisition']['nanmean'] ,
                    c = colours[c0_ind], 
               alpha = 0.5, marker = markerstyles[eta_ind], mec ='k', markersize = 8, linestyle = "None")

ax.axhline(1, linestyle = '--', color = 'k')
#ref_line, = ax.plot([10**0, 10**4], [10**0, 10**4], 'k', label = r'$y = x$')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.axhline(1, linestyle = '--', color = 'k')
#ax.axvline(1, linestyle = '--', color = 'k')
#ax.set_xlim(0.06, 40)
#ax.set_ylim(0.7, 40)
#ax.set_xlim(-1, 45)
#ax.set_ylim(-1, 45)
ax.legend(handles=legend_elements, loc='upper right', ncol = 2, fontsize = 9)
ax.set_ylabel("Ratio of fitness at mutation time to\nfitness at mean acquisition time", fontsize = 12)
ax.set_xlabel(r"$\eta / \mu$", fontsize = 12)
plt.tight_layout()
plt.savefig("phage_fitness_ratio_mutation_to_mean_acquisition_vs_eta_over_mu.pdf")
# -

# ### Plot probability of first acquisition at time t

# +
c0_select = 10**4
e_select = 0.95
mu_select = 10**-5

data_subset = grouped_data[(grouped_data['m_init'] == 1)
             &(grouped_data['C0'] == c0_select)
            & (grouped_data['mu'] == mu_select)
             & (grouped_data['e'] == e_select)  
            # & (grouped_data['eta'] == eta_select) 
            & (grouped_data['B'] == 170)
            & (grouped_data['pv'] == 0.02)
            & (grouped_data['f'] == 0.3)
            & (grouped_data['pv_type'] == 'binary')]

# +
alpha = 2*10**-2/c0_select
g=1/(42*c0_select)

c0 = float(c0_select)
g = float(g)

t = np.arange(0,1000000,10)
times = []
sizes = []
sizes_peak = []

fig, ax = plt.subplots(figsize = (6,4))

for i, row in data_subset.iterrows():
    eta = row['eta']
    
    nb = row['mean_nb']['nanmean']
    nu = row['mean_nu']['nanmean']
    nb0 = nb*(1-nu)

    acquisition = float(alpha*eta*(1-pv)*nb0)
    s0 = float(alpha*nb*(B*pv-1) - f*g*c0)
    
    p, = ax.plot(t*g*c0, P_acquisition(acquisition, s0, t), label = r'$\eta = %s$' %float(eta)) 
    
    ind = np.argmax(P_acquisition(acquisition, s0, t))
    
    est_phage_size = np.exp(s0*t[ind])
    sizes_peak.append(est_phage_size)
    
    
    if float(eta) == 10**-2:
        label = "Mean time"
    else:
        label = ""
    ax.axvline(mean_acquisition_time(acquisition, s0)*g*c0, color = p.get_color(), linestyle = '--', linewidth = 1, label = label)
    
    times.append(mean_acquisition_time(acquisition, s0))
    sizes.append(np.exp(s0*mean_acquisition_time(acquisition, s0)))
    
ax.set_ylabel("Probability")
ax.set_xlabel("Time (bacterial generations)")
ax.legend()
ax.set_ylim(0,)
ax.set_xlim(0,1800)
plt.tight_layout()
plt.savefig("time_of_first_spacer_acquisition_C0_%s_mu_%s_e_%s.pdf" %(c0_select, mu_select, e_select))

# +
fig, ax = plt.subplots(figsize = (5,3.5))

ax.plot(data_subset['eta'], data_subset['mean_size_at_acquisition']['nanmean'], marker = 'o', label = 'measured')
ax.plot(data_subset['eta'], sizes, marker = 's', label = 'predicted (mean)')
ax.plot(data_subset['eta'], sizes_peak, marker = 's', label = 'predicted (mode)')

ax.set_yscale('log')
ax.set_xscale('log')

ax.set_ylabel("Mean phage clone size at\ntime of first acquisition")
ax.set_xlabel(r"Spacer acquisition probability $\eta$")
ax.legend()
plt.tight_layout()
plt.savefig("phage_clone_size_at_acquisition_vs_eta_C0_%s_mu_%s_e_%s.pdf" %(c0_select, mu_select, e_select))
