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

# # Speed, turnover, time shift analysis
#
# Code for Figure 7

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from matplotlib.lines import Line2D
import seaborn as sns
#import pickle
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import decomposition
import matplotlib.cm as cm
from scipy.interpolate import interp1d

from sim_analysis_functions import load_simulation, find_file

from spacer_model_plotting_functions import find_nearest, fraction_remaining, calculate_speed, paez_espino_to_array, Guerrero_to_array


# from https://stackoverflow.com/a/53191379
################### Function to truncate color map ###################
def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap

def v_approx(g,c0,f,B,eta,e,alpha,pv,R,mu,L):
    r = R*g*c0
    return alpha*B*f*(e*mu*eta*L*(1-pv)/(2*alpha**2*B**2*r))**(1/3)


def fraction_remaining_paez_espino(bac_array, interp_times):
    """
    Calculate the fraction of bacterial spacer types remaining at time t
    as a function of the time delay (interp_times).
    
    Inputs:
    bac_array : array of interpolated time points and bacteria spacer abundances.
        Each column is a time point and the rows are bacterial types
        This is the transpose of the pop_array format.
    interp_times : time spacing in bacterial generations for interpolation
        
    Returns:
    turnover_array : square array of dimension len(interp_times) with each row
        being the fraction of bacterial spacers remaining at each time delay. 
        Increasing row number is a later starting point for the calculation.
        Time delays after the simulation endpoint are padded with np.nan.
    interp_times - t_ss : time shift axis in bacterial generations 
    """
    
    fraction_list = []

    for i in range(len(interp_times)):
        num_remaining = np.count_nonzero(bac_array[np.where(bac_array[:, i] > 0)[0], i:], axis = 0)
        fraction_list.append(np.append(num_remaining / num_remaining[0], [np.nan]*i))
        
    turnover_array = np.stack(fraction_list)
    
    return turnover_array


def get_PCA_data(data, pop_array, gen_max, max_m, ram_skip = 4, time_skip = 5, n_components = 2):
    """
    Perform PCA on bacteria and phage clone abundances.
    
    Inputs:
    data : row from all_data corresponding to timestamp
    pop_array : array of simulation data for timestamp
    gen_max : end time of simulation in bacterial generations
    max_m : total number of unique clones in simulation (array size)
    ram_skip : numer of save points to skip by in subsample
    time_skip : timestep in bacterial generations to subsample
    n_components : number of PCA components
    
    Returns:
    Phage : array of phage clone abundances transformed into PCA coordinates
    Bac : array of bacteria spacer abundances transformed into PCA coordinates
    """
    
    ## PCA calculation
    t_ss = gen_max / 5
    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

    timepoints = pop_array[t_ss_ind::ram_skip, -1].toarray().flatten()*g*c0

    nvi = pop_array[t_ss_ind::ram_skip, max_m+1 : 2*max_m + 1]
    nbi = pop_array[t_ss_ind::ram_skip, 1 : max_m + 1]

    nvi = nvi.toarray()
    nbi = nbi.toarray()

    phage_array = np.array(nvi) #  an array where each row abundance of the ith phage
    bac_array = np.array(nbi)

    # interpolate population sizes to have consistent time spacing
    f_bac = interp1d(timepoints, bac_array,  axis = 0)
    f_phage = interp1d(timepoints, phage_array, axis = 0)

    # subsample the simulation times
    end_time = float(data['mean_large_trajectory_length_nvi_ss'])*4 + t_ss # 4 times the mean phage clone extinction time
    #end_time = t_ss + 2500
    #end_time = timepoints[-1]
    new_times = np.arange(timepoints[0], end_time, time_skip)[:-2]

    # create interpolated array
    phage_array = f_phage(new_times)
    bac_array = f_bac(new_times)

    # normalize so that each time point abundance vector is 1
    # don't do the L2 norm - doesn't really make sense

    phage_array_norm = phage_array / np.sum(phage_array, axis = 1)[:,None]
    bac_array_norm = bac_array / np.sum(bac_array, axis = 1)[:,None]

    # PCA
    bac_array_norm = np.nan_to_num(bac_array_norm)

    pca = decomposition.PCA(n_components = n_components)
    pca.fit(phage_array_norm)

    Phage = pca.transform(phage_array_norm)
    Bac = pca.transform(bac_array_norm)
    
    return Phage, Bac, new_times



# %matplotlib inline

# +
all_data = pd.read_csv("../data/all_data.csv", index_col = 0)

top_folder = "../data/"

# +
c0_select = 10**4
mu_select = 10**-5
e_select = 0.95
m_init_select = 1

all_data_subset = all_data[(all_data['C0'] == c0_select) 
        & (all_data['mu'] == mu_select)
        & (all_data['e'] == e_select)
        & (all_data['m_init'] == m_init_select)
        & (all_data['pv'] == 0.02)
        & (all_data['f'] == 0.3)
        & (all_data['B'] == 170)
        & (all_data['pv_type'] == 'binary')]

all_data_subset = all_data_subset.loc[all_data_subset[['C0', 'e', 'mu', 'eta', 'm_init']].drop_duplicates().index]
all_data_subset = all_data_subset.sort_values(by = 'eta')

timestamps = list(all_data_subset['timestamp'])
sub_folders = list(all_data_subset['folder_date'])

# +
## Load data for PCA plot and turnover

c0_list = []
g_list = []
eta_list = []
mu_list = []
m_init_list = []
max_m_list = []
alpha_list = []
e_list = []
gen_max_list = []

timestamps_list = []

pop_array_list = []
all_phages_list = []

for i, timestamp in tqdm(enumerate(timestamps)):
    
    sub_folder = top_folder + "%s" %sub_folders[i]
    
    folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, sub_folder)

    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
     max_m, mutation_times, all_phages = load_simulation(folder, timestamp);
    
    t_ss = gen_max / 5
    
    c0_list.append(c0)
    g_list.append(g)
    eta_list.append(eta)
    mu_list.append(mu)
    m_init_list.append(m_init)
    max_m_list.append(max_m)
    alpha_list.append(alpha)
    e_list.append(e)
    gen_max_list.append(gen_max)
    
    timestamps_list.append(timestamp)
    pop_array_list.append(pop_array)
    all_phages_list.append(all_phages)

# -

# ### PCA plot

timestamp_select = '2019-02-25T14:24:15.599257'
i = timestamps_list.index(timestamp_select)
pop_array = pop_array_list[i]
c0 = c0_list[i]
g = g_list[i]
eta = eta_list[i]
mu = mu_list[i]
max_m = max_m_list[i]
m_init = m_init_list[i]
alpha = alpha_list[i]
e = e_list[i]
gen_max = gen_max_list[i]
all_phages = np.array(all_phages_list[i])

data = all_data[all_data['timestamp'] == timestamp_select]
Phage, Bac, new_times = get_PCA_data(data, pop_array, gen_max, max_m)

# ## paez_espino turnover

threshold = 0.15
wild_type = True
pam = "perfect"
paez_espino_data = pd.read_csv("../data/PaezEspino2015/%s_PAM/banfield_data_combined_type_%s_wt_%s.csv" %(pam,
                                                                            1-threshold, wild_type), index_col = 0)
grouping = ['type_%s' %(1-threshold), 'CRISPR']
#grouping = 'type'
bac_wide_filtered, phage_wide_filtered = paez_espino_to_array(paez_espino_data, grouping, norm = True)

time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
time_points = np.arange(0,len(time_points_in_days))

# interpolate the bacteria spacer values
# remove the last column which is a sum column
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])

step = 3
interp_times_paez_espino = np.arange(np.min(time_points_in_days), np.max(time_points_in_days) + step, step)

bac_array = f(interp_times_paez_espino)
turnover_array_paez_espino = fraction_remaining_paez_espino(bac_array, interp_times_paez_espino)

# From previous paper: "Time in
# generations for the experimental data is time in days ×6.64, assuming exponential growth between
# daily 100-fold dilutions."

# ## Burstein turnover

# grouped with 85% similarity also
burstein_spacers = pd.read_csv("../data/Burstein2016/Burstein2016_spacers_with_type.csv", index_col = 0)

# increasing numerical accession is also increasing time point
accessions = burstein_spacers['accession'].unique()
# from fig 1, Castelle2015
timepoints_in_days = [-2, 7, 20, 34, 65, 92]
timepoints = np.arange(0, len(accessions), 1)
accession_to_time = dict(zip(accessions, timepoints))
burstein_spacers['timepoint'] = burstein_spacers['accession'].map(accession_to_time)

burstein_spacers_grouped = burstein_spacers.groupby(['timepoint', 'query_id', 'type'])['spacer_sequence'].count().reset_index()

# +
burstein_wide = burstein_spacers_grouped.pivot_table(index = ['query_id','type'], columns = 'timepoint', values = 'spacer_sequence',
                                  fill_value = 0, margins = True, aggfunc = 'sum').reset_index()

# remove the last column which is a sum column
burstein_array = burstein_wide.drop(['type', 'query_id'], axis = 1).iloc[:-1,:-1]
burstein_array = np.array(burstein_array)
# -

# interpolate the bacteria spacer values
f = interp1d(timepoints_in_days,burstein_array)

step = 9
interp_times_burstein = np.arange(np.min(timepoints_in_days), np.max(timepoints_in_days) , step)

burstein_array_interp = f(interp_times_burstein)
turnover_burstein = fraction_remaining_paez_espino(burstein_array_interp, interp_times_burstein)

# ### Guerrero turnover

pam = "perfect"
folder = "../data"
wild_type = True
phage_only = False
grouping = ['type_%s' %(1-threshold), 'crispr']
df_combined = pd.read_csv("%s/Guerrero2021/%s_PAM/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,
                                                             pam, 1-threshold, wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)

# +
datapath = "../data/Guerrero2021"
metadata = pd.read_csv("%s/SraRunTable.txt" %datapath)

accessions = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
d = pd.to_datetime(dates)
dates_int = np.array((d[1:] - d[:-1]).days)
time_points_in_days = np.concatenate([[0], dates_int])
time_points_in_days = np.cumsum(time_points_in_days)

time_points = np.arange(0, len(dates), 1)
# -

# interpolate the bacteria spacer values
# remove the last column which is a sum column
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])

step = 14
interp_times_Guerrero = np.arange(np.min(time_points_in_days), np.max(time_points_in_days), step)

# + tags=[]
bac_array = f(interp_times_Guerrero)
turnover_array_Guerrero = fraction_remaining_paez_espino(bac_array, interp_times_Guerrero)
# -

phage_array = f_phage(interp_times_Guerrero)
turnover_array_phage_Guerrero = fraction_remaining_paez_espino(phage_array, interp_times_Guerrero)

# ### Speed 

grouped_data = all_data.groupby(['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv', 'm_init', 'pv_type', 'theta', 'gen_max'])[['mean_m',
       'mean_phage_m', 'mean_large_phage_m', 'mean_large_phage_size',
       'rescaled_phage_m', 'mean_nu', 'mean_nb',
       'mean_nv', 'mean_C', 'e_effective',
       'fitness_discrepancy', 'mean_size_at_acquisition',
       'std_size_at_acquisition', 'fitness_at_90percent_acquisition',
       'fitness_at_mean_acquisition', 'fitness_at_first_acquisition',
       'num_bac_acquisitions', 'mean_bac_acquisition_time',
       'median_bac_acquisition_time', 'first_bac_acquisition_time',
       'mean_large_trajectory_length_nvi_ss', 'mean_trajectory_length',
       'mean_T_backwards_nvi_ss', 'mean_bac_extinction_time',
       'mean_bac_extinction_time_phage_present', 'establishment_rate_nvi_ss',
       'turnover_speed', 'predicted_establishment_fraction',
       'measured_mutation_rate', 'mean_establishment_time', 'bac_speed_mean',
       'bac_speed_std', 'phage_speed_mean', 'phage_speed_std',
       'bac_spread_mean', 'bac_spread_std', 'phage_spread_mean',
       'phage_spread_std', 'net_phage_displacement', 'net_bac_displacement',
       'max_phage_displacement', 'max_bac_displacement',
       'bac_phage_distance_mean', 'bac_phage_distance_std',
       'time_to_reach_bac_0.25', 'time_to_reach_phage_0.25',
       'time_to_reach_bac_0.5', 'time_to_reach_phage_0.5',
       'time_to_reach_bac_1', 'time_to_reach_phage_1', 'time_to_reach_bac_2',
       'time_to_reach_phage_2', 'time_to_reach_bac_5', 'time_to_reach_phage_5',
       'time_to_reach_bac_10', 'time_to_reach_phage_10',
       'time_to_reach_bac_15', 'time_to_reach_phage_15', 'pred_num_establishments',
       'measured_num_establishments', 'measured_num_establishments_8000',
       'time_to_full_turnover_phage', 'time_to_full_turnover_bac',
       'phage_abundance_speed_mean', 'phage_abundance_speed_std',
       'bac_abundance_speed_mean', 'bac_abundance_speed_std',
       'bac_clan_number', 'phage_clan_number', 'bac_clan_number_std',
       'phage_clan_number_std', 'bac_clan_size', 'phage_clan_size',
       'bac_clan_size_std', 'phage_clan_size_std']].agg([np.nanmean, np.nanstd,
                                                                            'count']).reset_index()

# +
c0_vals = list(np.unique(all_data['C0']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd']
colours_2 = sns.color_palette("hls", len(c0_vals))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

# +
colour = 'C0'
shape = 'eta'
line = 'mu'

colour_label = 'C_0'
shape_label = '\eta'
line_label = '\mu'

e_select = 0.95
m_init_select = 1
#eta_select = 0.01
data_subset = grouped_data[(grouped_data['m_init'] == m_init_select)
                                       & (grouped_data['mu'] > 4*10**-8)
                                       & (grouped_data['pv_type'] == 'binary')
                                       & (grouped_data['B'] == 170)
                                      & (grouped_data['f'] == 0.3)
                                    & (grouped_data['pv'] == 0.02)]

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))
# -

# ## Plot

# +
fig = plt.figure(figsize = (10,6))

absolute_left = 0.075
absolute_right = 0.93
absolute_bottom = 0.11
absolute_top = 0.95
middle_top = 0.57
middle_bottom = 0.42

title_fontsize = 16
legend_fontsize = 8
label_fontsize = 10

# clone size distributions
gs_pca= gridspec.GridSpec(1,1)
gs_pca.update(left=absolute_left, right=0.3, bottom = middle_top, top = absolute_top, wspace = 0.3)

ax_pca = plt.subplot(gs_pca[0])

gs_speed = gridspec.GridSpec(1,2)
gs_speed.update(left=0.48, right=0.91, bottom = middle_top, top = absolute_top, wspace = 0.1)
ax_speed1 = plt.subplot(gs_speed[0])
ax_speed2 = plt.subplot(gs_speed[1])

gs_turnover_sim = gridspec.GridSpec(1,1)
gs_turnover_sim.update(left=absolute_left, right=0.38, bottom = absolute_bottom, top = middle_bottom, wspace = 0.1)
ax_sim = plt.subplot(gs_turnover_sim[0])

gs_turnover = gridspec.GridSpec(1,3)
gs_turnover.update(left=0.48, right=absolute_right, bottom = absolute_bottom, top = middle_bottom, wspace = 0.1)

ax = plt.subplot(gs_turnover[0])
ax1 = plt.subplot(gs_turnover[1])
ax2 = plt.subplot(gs_turnover[2])

gs_cbar = gridspec.GridSpec(1,1)
gs_cbar.update(left=0.92, right=absolute_right, bottom = middle_top, top = absolute_top)
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


ax_pca.set_title("A", loc = "left", fontsize = title_fontsize)
ax_speed1.set_title("B", loc = "left", fontsize = title_fontsize)
ax_speed2.set_title("C", loc = "left", fontsize = title_fontsize)
ax_sim.set_title("D", loc = "left", fontsize = title_fontsize)
ax.set_title("E", loc = "left", fontsize = title_fontsize)
ax1.set_title("F", loc = "left", fontsize = title_fontsize)
ax2.set_title("G", loc = "left", fontsize = title_fontsize)

##### PCA PLOT
# static version
skip = 5
time_select = 3800

num_points = 5 # number of points to highlight
colours = cm.YlGnBu(np.linspace(0,1, len(new_times[::skip])))[::-1]
red_colours = cm.Reds(np.linspace(0.4, 1, num_points))[::-1]
pattern = '.'


ax_pca.scatter(Bac[::skip,0], Bac[::skip,1], c = colours, marker = 'o', s = 60, label = "Bacteria", alpha = 0.6)
ax_pca.plot(Bac[::skip,0], Bac[::skip,1], linestyle = ':', linewidth = 1, color = 'k', alpha = 0.4)


ax_pca.scatter(Phage[::skip,0], Phage[::skip,1], c = colours, marker = 'v', s = 60, label = "Phage", alpha = 0.6)
ax_pca.plot(Phage[::skip,0], Phage[::skip,1], linestyle = '--', linewidth = 1, color = 'k', alpha = 0.4)

# highlight a few times
ind = find_nearest(new_times[::skip], time_select)

ax_pca.scatter(Bac[::skip, 0][ind:ind+num_points], Bac[::skip, 1][ind:ind+num_points], marker = 'o', s = 60, c = "none", ec= red_colours, linewidth =2)
ax_pca.scatter(Phage[::skip, 0][ind:ind+num_points], Phage[::skip, 1][ind:ind+num_points], marker = 'v', s = 60, c = "none", ec = red_colours, linewidth =2)

# add colorbar
divider = make_axes_locatable(ax_pca)
cax = divider.append_axes('right', size='5%', pad=0.05)
cmap = cm.YlGnBu_r
norm = matplotlib.colors.Normalize(vmin=new_times[0], vmax=new_times[-1])

cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('Time (bacterial generations)')

ax_pca.set_xlabel("PC1")
ax_pca.set_ylabel("PC2")
ax_pca.legend(fontsize = legend_fontsize)


##### SIMULATION TURNOVER

#colours = ['slateblue', 'darkorange']
colours = cm.viridis(np.linspace(0,0.9, len(timestamps_list)))

count = 0
for timestamp in timestamps_list:
    j = timestamps_list.index(timestamp)
    print(j)
    
    max_m = max_m_list[j]
    c0 = c0_list[j]
    g = g_list[j]
    alpha = alpha_list[j]
    mu = mu_list[j]
    eta = eta_list[j]
    e = e_list[j]
    pop_array = pop_array_list[j]
    gen_max = gen_max_list[j]
    #if mu < 10**-7:
    #    continue
    
    t_ss = gen_max / 5
    
    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)
    
    turnover_array, interp_times = fraction_remaining(pop_array, t_ss, t_ss_ind, g, c0, gen_max, max_m)
    
    speed, start_ind = calculate_speed(turnover_array, interp_times)
    print(speed)

    ax_sim.scatter(interp_times, np.nanmean(turnover_array, axis = 0), s = 10,
              c = colours[count])
    
    # plot dummy point for legend
    ax_sim.scatter([-1], np.nanmean(turnover_array, axis = 0)[0], s = 50, label = "$\eta= %s$" %round(eta,6),
              c = colours[count])

    if count == len(timestamps_list) - 1:
        label =  "Standard deviation\nacross simulation"
    else:
        label = ""
    ax_sim.fill_between(interp_times, np.nanmean(turnover_array, axis = 0)
                    + np.nanstd(turnover_array, axis = 0),                    
                        y2 = np.nanmean(turnover_array, axis = 0) 
                    - np.nanstd(turnover_array, axis = 0), color = colours[count],
                    alpha = 0.1, label = label)
    
    # plot turnover rate
    if count == len(timestamps_list) - 1:
        label =  "Turnover rate"
    else:
        label = ""

    count += 1
ax_sim.set_ylim(0,1.01)
ax_sim.set_xlim(-40,6000)
ax_sim.legend(fontsize = legend_fontsize)
ax_sim.set_xlabel("Time delay (bacterial generations)")
ax_sim.set_ylabel("Fraction of bacterial clones\nremaining")

### EXPERIMENT TURNOVER

ax.plot(interp_times_paez_espino*6.64, np.nanmean(turnover_array_paez_espino, axis = 0), color = 'k', linewidth = 2)
ax.fill_between(interp_times_paez_espino*6.64, 
                y1 = np.nanmean(turnover_array_paez_espino, axis = 0) - np.nanstd(turnover_array_paez_espino, axis = 0),
               y2 = np.nanmean(turnover_array_paez_espino, axis = 0) + np.nanstd(turnover_array_paez_espino, axis = 0),
               alpha = 0.4, color = 'grey')

ax.set_ylim(0,1.0)
ax.set_xlim(0, 235*6.64)
ax.set_xlabel("Time delay\n(bacterial generations)")
ax.set_ylabel("Fraction of spacer types\nremaining")



ax1.plot(interp_times_burstein + 2, np.nanmean(turnover_burstein, axis = 0), color = 'k', linewidth = 2)
ax1.fill_between(interp_times_burstein + 2, y1 = np.nanmean(turnover_burstein, axis = 0) - np.nanstd(turnover_burstein, axis = 0),
               y2 = np.nanmean(turnover_burstein, axis = 0) + np.nanstd(turnover_burstein, axis = 0),
               alpha = 0.4, color = 'grey')

ax1.set_ylim(0,1)
ax1.set_xlim(0, 88)
ax1.set_yticks([])
ax1.set_xlabel("Time delay (days)")

ax2.plot(interp_times_Guerrero, np.nanmean(turnover_array_Guerrero, axis = 0), color = 'k', linewidth = 2)
ax2.fill_between(interp_times_Guerrero, 
                 y1 = np.nanmean(turnover_array_Guerrero, axis = 0) - np.nanstd(turnover_array_Guerrero, axis = 0),
               y2 = np.nanmean(turnover_array_Guerrero, axis = 0) + np.nanstd(turnover_array_Guerrero, axis = 0),
               alpha = 0.4, color = 'grey')

ax2.set_ylim(0,1.0)
ax2.set_yticks([])
ax2.set_xlabel("Time delay (days)")

#### SPEED

for group in data_subset.groupby([colour, shape, line]):
    data = group[1].sort_values(by = 'm_init')
    
    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    x = data['max_phage_displacement']['nanmean']
    dx = data['max_phage_displacement']['nanstd']
    
    y = data['measured_num_establishments']['nanmean']
    dy = data['measured_num_establishments']['nanstd']
    
    xerr = np.stack([np.zeros(data['mean_large_trajectory_length_nvi_ss']['nanstd'].shape), 
                     data['mean_large_trajectory_length_nvi_ss']['nanstd']]) 
    yerr = data['max_phage_displacement']['nanstd'] / (data['gen_max'] - data['gen_max']/5)
    yerr = np.stack([np.zeros(yerr.shape), yerr])
    
    ax_speed1.errorbar(data['mean_large_trajectory_length_nvi_ss']['nanmean'],  
                data['max_phage_displacement']['nanmean'] / (data['gen_max']- data['gen_max']/5),
            xerr = xerr,
            yerr = yerr,
            c = colours_2[colour_ind], 
       alpha = 0.7, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")
    

# linear regression
x_y = pd.DataFrame()
x_y['x'] = data_subset['mean_large_trajectory_length_nvi_ss']['nanmean']
x_y['y'] = data_subset['max_phage_displacement']['nanmean'] / (data_subset['gen_max']
                                                               - data_subset['gen_max']/5)

x_y = x_y.dropna(axis = 0)

T_ext_vals = np.arange(10**2, 10**5, 10**1)
t, = ax_speed1.plot(T_ext_vals, 1/T_ext_vals, color ='k', linestyle = '--', label = r'$y = \frac{1}{T_{ext}}$')

m_vals = np.arange(0.5, 50, 0.5)

ax_speed1.legend(handles=[t], loc='upper right', ncol = 1, fontsize = legend_fontsize)

ax_speed1.set_yscale('log')
ax_speed1.set_xscale('log')
ax_speed1.set_xlim(2*10**1, 1.5*10**4)
ax_speed1.set_ylim(3.2*10**-5, 3*10**-3)
ax_speed1.set_xlabel(r"Mean phage time to extinction")
ax_speed1.set_ylabel("Phage speed (mutational\ndistance per generation)")

B = 170
R = 0.04
pv = 0.02
f = 0.3
L = 30

for group in data_subset.groupby([colour, shape, line]):
    data = group[1].sort_values(by = 'm_init')
    
    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    c0 = data['C0']
    eta = data['eta']
    e = data['e']   
    g = 1/(42*c0)
    alpha = 2*10**-2/c0
    mu = data['mu']
    
    if np.any(eta < 10**-3):
        alphaval = 0.3
    else:
        alphaval = 0.6
    
    e = data['e']   
    g = 1/(42*c0)
    alpha = 2*10**-2/c0
    mu = data['mu']

    yerr = data['max_phage_displacement']['nanstd'] / (data['gen_max'] - data['gen_max']/5)
    yerr = np.stack([np.zeros(yerr.shape), yerr])
    
    v = v_approx(g,c0,f,B,eta,e,alpha,pv,R,mu,L)
    
    ax_speed2.errorbar(v,  
                data['max_phage_displacement']['nanmean'] / (data['gen_max'] - data['gen_max']/5),
                yerr = yerr,
            c = colours_2[colour_ind], 
       alpha = alphaval, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")
    

t, = ax_speed2.plot([0,0.05], [0,0.05], color ='k', linestyle = '--', label = "Approximate\ntheory")
#legend_elements.append(t)

ax_speed2.legend(handles=[t], loc='lower right', ncol = 2, fontsize = legend_fontsize)
ax_speed1.legend(handles=legend_elements, loc = 'lower left', fontsize = legend_fontsize)

ax_speed1.set_xlim(4*10**1, 2*10**4)
ax_speed2.set_xlim(5*10**-6, 2*10**-2)
ax_speed1.set_ylim(3*10**-6, 4*10**-3)
ax_speed2.set_ylim(3*10**-6, 4*10**-3)
ax_speed2.axes.get_yaxis().set_ticklabels([])

ax_speed2.set_yscale('log')
ax_speed2.set_xscale('log')
ax_speed2.set_xlabel(r"$\alpha f B \left(\frac{e \mu \eta L (1-p_V)}{2\alpha^2 B^2 r}\right)^\frac{1}{3}$", fontsize = 13)

plt.savefig("speed_and_turnover.pdf")
plt.savefig("speed_and_turnover.svg")
# -

# ## Supplementary figures

# +
## PCA plot for simulation with cross-reactivity
timestamp = "2021-09-12T20:20:54.807443" # exponential cross-reactivity
data = all_data[all_data['timestamp'] == timestamp]
sub_folder = top_folder + "%s" %data['folder_date'].values[0]

folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, sub_folder)

f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
 max_m, mutation_times, all_phages = load_simulation(folder, timestamp);

t_ss = gen_max / 5
# -

Phage, Bac, new_times = get_PCA_data(data, pop_array, gen_max, max_m)

# +
## PCA plot

fig, ax_pca = plt.subplots(figsize = (5,3.8))

# static version
skip = 5
time_select = 2230

num_points = 5 # number of points to highlight
colours = cm.YlGnBu(np.linspace(0,1, len(new_times[::skip])))[::-1]
red_colours = cm.Reds(np.linspace(0.4, 1, num_points))[::-1]
pattern = '.'


ax_pca.scatter(Bac[::skip,0], Bac[::skip,1], c = colours, marker = 'o', s = 60, label = "Bacteria", alpha = 0.6)
ax_pca.plot(Bac[::skip,0], Bac[::skip,1], linestyle = ':', linewidth = 1, color = 'k', alpha = 0.4)


ax_pca.scatter(Phage[::skip,0], Phage[::skip,1], c = colours, marker = 'v', s = 60, label = "Phage", alpha = 0.6)
ax_pca.plot(Phage[::skip,0], Phage[::skip,1], linestyle = '--', linewidth = 1, color = 'k', alpha = 0.4)

# highlight a few times
ind = find_nearest(new_times[::skip], time_select)

ax_pca.scatter(Bac[::skip, 0][ind:ind+num_points], Bac[::skip, 1][ind:ind+num_points], marker = 'o', s = 60, c = "none", ec= red_colours, linewidth =2)
ax_pca.scatter(Phage[::skip, 0][ind:ind+num_points], Phage[::skip, 1][ind:ind+num_points], marker = 'v', s = 60, c = "none", ec = red_colours, linewidth =2)

# add colorbar
divider = make_axes_locatable(ax_pca)
cax = divider.append_axes('right', size='5%', pad=0.05)
cmap = cm.YlGnBu_r
norm = matplotlib.colors.Normalize(vmin=new_times[0], vmax=new_times[-1])

cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('Time (bacterial generations)')

ax_pca.set_xlabel("PC1")
ax_pca.set_ylabel("PC2")
ax_pca.legend()
plt.tight_layout()
plt.savefig("PCA_2D_%s.pdf" %timestamp)
plt.savefig("PCA_2D_%s.svg" %timestamp)
# -


