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

# # Time shift plots
#
# Plot e effective as a function of time shift between phage and bacterial populations

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.interpolate import interp1d
from matplotlib import gridspec
import urllib
from Bio import Entrez
from scipy.stats import wilcoxon

from sim_analysis_functions import find_nearest
from sim_analysis_functions import load_simulation, find_file
from spacer_model_plotting_functions import e_effective_shifted

def Guerrero_to_array(df_combined, grouping):
    """
    Create an array of bacteria and phage types at each time point
    
    Inputs:
    df_combined : a dataframe with columns 'time_point', 'count_bac', 'count_phage', 'count_bac_normalized', 'count_phage_normalized', and grouping
    grouping : column name for the grouping variable, the spacer type label (i.e. 'type_0.85'), or a list of column names (i.e . ['type_0.85', 'CRISPR'])
    
    Returns:
    bacteria and phage arrays where columns are time points and rows are types, values are normalized counts,
        with a sum column and sum row on the right and bottom respectively.
    """
    # reshape to wide so that each spacer type is represented over time
    # fill NaNs with 0 since they're 0 abundance for that type on that day
    # the columns are the times and the rows are the types
    all_wide = df_combined.pivot_table(index = grouping, columns = 'time_point', values = ['count_bac', 'count_phage'],
                                  fill_value = 0, margins = True, aggfunc = 'sum').reset_index()

    #all_wide_normalized = df_combined.pivot_table(index = grouping, columns = 'time_point', 
    #                                              values = ['count_bac_normalized', 'count_phage_normalized'],
    #                                  fill_value = 0, margins = True, aggfunc = 'sum').reset_index()

    # this way the bacteria and phage will have the same type labels and number of types
    #bac_wide = all_wide_normalized['count_bac_normalized']
    #phage_wide = all_wide_normalized['count_phage_normalized']
    
    # unnormalized version
    bac_wide = all_wide['count_bac']
    phage_wide = all_wide['count_phage']

    #keep only types that have more than 1 appearance in total
    bac_wide_filtered = bac_wide.copy()
    bac_wide_filtered[all_wide['count_bac']['All'] <= 1] = 0

    phage_wide_filtered = phage_wide.copy()
    phage_wide_filtered[all_wide['count_phage']['All'] <= 1] = 0

    # normalize to a constant total fraction at each time point
    bac_wide_normalized = bac_wide_filtered.iloc[:,: ] / bac_wide_filtered.iloc[-1, :]
    #bac_wide_normalized['type'] = bac_wide_filtered['type']

    phage_wide_normalized = phage_wide_filtered.iloc[:, :] / phage_wide_filtered.iloc[-1,:]
    #phage_wide_normalized['type'] = phage_wide_filtered['type']

    # drop the sum row at the bottom
    bac_wide = bac_wide.iloc[:-1]
    phage_wide = phage_wide.iloc[:-1]
    bac_wide_filtered = bac_wide_filtered.iloc[:-1]
    phage_wide_filtered = phage_wide_filtered.iloc[:-1]
    all_wide = all_wide.iloc[:-1]
    bac_wide_normalized = bac_wide_normalized.iloc[:-1]
    phage_wide_normalized = phage_wide_normalized.iloc[:-1]
    
    return bac_wide, phage_wide
    #return bac_wide_filtered, phage_wide_filtered
    #return bac_wide_normalized, phage_wide_normalized


def paez_espino_to_array(df_combined, grouping, norm = True):
    """
    Create an array of bacteria and phage types at each time point
    
    Inputs:
    df_combined : a dataframe with columns 'time_point', 'count_bac', 'count_phage', 'count_bac_normalized', 'count_phage_normalized', and grouping
    grouping : column name for the grouping variable, the spacer type label (i.e. 'type_0.85'), or a list of column names (i.e . ['type_0.85', 'CRISPR'])
    
    Returns:
    bacteria and phage arrays where columns are time points and rows are types, values are normalized counts,
        with a sum column and sum row on the right and bottom respectively.
    """
    # reshape to wide so that each spacer type is represented over time
    # fill NaNs with 0 since they're 0 abundance for that type on that day
    # the columns are the times and the rows are the types
    all_wide = df_combined.pivot_table(index = grouping, columns = 'time_point', values = ['count_bac', 'count_phage'],
                                  fill_value = 0, margins = True, aggfunc = 'sum').reset_index()

    all_wide_normalized = df_combined.pivot_table(index = grouping, columns = 'time_point', 
                                                  values = ['count_bac_normalized', 'count_phage_normalized'],
                                      fill_value = 0, margins = True, aggfunc = 'sum').reset_index()

    if norm == True:
        # this way the bacteria and phage will have the same type labels and number of types
        bac_wide = all_wide_normalized['count_bac_normalized']
        phage_wide = all_wide_normalized['count_phage_normalized']
    
    elif norm == False:
        bac_wide = all_wide['count_bac']
        phage_wide = all_wide['count_phage']

    #keep only types that have more than 1 appearance in total
    bac_wide_filtered = bac_wide.copy()
    bac_wide_filtered[all_wide['count_bac']['All'] <= 1] = 0

    phage_wide_filtered = phage_wide.copy()
    phage_wide_filtered[all_wide['count_phage']['All'] <= 1] = 0

    # normalize to a constant total fraction at each time point
    bac_wide_normalized = bac_wide_filtered.iloc[:,: ] / bac_wide_filtered.iloc[-1, :]
    #bac_wide_normalized['type'] = bac_wide_filtered['type']

    phage_wide_normalized = phage_wide_filtered.iloc[:, :] / phage_wide_filtered.iloc[-1,:]
    #phage_wide_normalized['type'] = phage_wide_filtered['type']

    # drop the sum row at the bottom
    bac_wide_filtered = bac_wide_filtered.iloc[:-1]
    phage_wide_filtered = phage_wide_filtered.iloc[:-1]
    all_wide = all_wide.iloc[:-1]
    bac_wide_normalized = bac_wide_normalized.iloc[:-1]
    phage_wide_normalized = phage_wide_normalized.iloc[:-1]
    
    return bac_wide_filtered, phage_wide_filtered
    #return bac_wide_normalized, phage_wide_normalized


# ## Load simulation data

# load table of results, skip simulations that are already in results
all_data = pd.read_csv("../data/all_data.csv", index_col = 0)

speed_and_slope = all_data

speed_and_slope.shape

# ## Load simulations for plotting

# +
top_folder = "../data/"
# this will only work with these specific parameters; if other parameters are desired, data must be downloaded from Dryad
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
max_shift = 3100 # largest time shift to use to calculate memory length
slope_width = 3
skip = 5

c0_list = []
g_list = []
eta_list = []
mu_list = []
m_init_list = []
max_m_list = []
alpha_list = []
e_list = []
gen_max_list = []

pop_array_list = []
e_eff_mean_past_list = []
e_eff_mean_future_list = []
e_eff_std_past_list = []
e_eff_std_future_list = []
interp_times_list = []
peak_time_list = []

# iterate through timestamps
for i, timestamp in tqdm(enumerate(timestamps)):

    sub_folder = top_folder + "/%s" %sub_folders[i]
    
    folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, top_folder)
    
    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
             max_m, mutation_times, all_phages = load_simulation(folder, timestamp);
    

    t_ss = gen_max / 5
    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)
    
    # get time shift data
    timepoints = pop_array[t_ss_ind-1::skip,-1].toarray().flatten()*g*c0
    timepoints = timepoints - timepoints[0] # shift to 0 start time
    interp_times = np.arange(t_ss, gen_max, skip) - t_ss

    nbi = pop_array[t_ss_ind-1::skip, 1: max_m+1].toarray()
    nvj = pop_array[t_ss_ind-1::skip, max_m+1 : 2*max_m+1].toarray()
    
    try:
        interp_fun_nbi = interp1d(timepoints, nbi, kind='linear', axis = 0)
        interp_fun_nvj = interp1d(timepoints, nvj, kind='linear', axis = 0)
    except MemoryError:
        continue
    
    nbi_interp = interp_fun_nbi(interp_times)
    nvj_interp = interp_fun_nvj(interp_times)
    
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
    
    c0_list.append(c0)
    g_list.append(g)
    eta_list.append(eta)
    mu_list.append(mu)
    m_init_list.append(m_init)
    max_m_list.append(max_m)
    alpha_list.append(alpha)
    e_list.append(e)
    gen_max_list.append(gen_max)
    pop_array_list.append(pop_array)
    
    e_eff_mean_past_list.append(e_eff_mean_past)
    e_eff_mean_future_list.append(e_eff_mean_future)
    e_eff_std_past_list.append(e_eff_std_past)
    e_eff_std_future_list.append(e_eff_std_future)
    interp_times_list.append(interp_times)
    peak_time_list.append(peak_time)
# -

# ## Paper figure: time shift from simulations, compare with experimental and natural data

# +
# downloaded from NCBI
metadata = pd.read_csv("../data/Guerrero2021/SraRunTable.txt")

accessions = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
d = pd.to_datetime(dates)
dates_int = np.array((d[1:] - d[:-1]).days)
time_points_in_days = np.concatenate([[0], dates_int])
time_points_in_days = np.cumsum(time_points_in_days)

time_points = np.arange(0, len(dates), 1)

# +
phageDC56 = "https://github.com/GuerreroCRISPR/Gordonia-CRISPR/raw/master/phage_DC-56.fa"
phageDS92 = "https://github.com/GuerreroCRISPR/Gordonia-CRISPR/raw/master/phage_DS-92.fa"

# phage DC-56
f = urllib.request.urlopen(phageDC56)
phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome:
    row = row.decode('utf-8')
    if row[0] != ">":
        phage_genome_seq += row.strip()
        
# count number of perfect PAMs in the reference genome
cr1_pams_DC56 = (phage_genome_seq.count('GTT')
            + phage_genome_seq.count('AAC'))

# phage DS-92
f = urllib.request.urlopen(phageDS92)
phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome:
    row = row.decode('utf-8')
    if row[0] != ">":
        phage_genome_seq += row.strip()
        
# count number of perfect PAMs in the reference genome
cr1_pams_DS92 = (phage_genome_seq.count('GTT')
            + phage_genome_seq.count('AAC'))

num_protospacers_guerrero = round((cr1_pams_DC56 + cr1_pams_DS92)/2)

# +
accession = "NC_007019" # Streptococcus phage 2972

f = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome[1:]:
    phage_genome_seq += row.strip()
    
# count number of perfect PAMs in the reference genome
cr3_pams = (phage_genome_seq.count('GGCG')
            + phage_genome_seq.count('GGAG')
            + phage_genome_seq.count('GGTG')
            + phage_genome_seq.count('GGGG')
            + phage_genome_seq.count('CCCC')
            + phage_genome_seq.count('CGCC')
            + phage_genome_seq.count('CTCC')
            + phage_genome_seq.count('CACC'))
            
cr1_pams = (phage_genome_seq.count('AGAAT')
            + phage_genome_seq.count('AGAAA')
            + phage_genome_seq.count('TTTCT')
            + phage_genome_seq.count('ATTCT'))

num_protospacers_paez_espino = cr3_pams + cr1_pams # Paez-Espino2013, 233 CRISPR1 protospacers

# +
# memory length
# load grouped data directly
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

R = 0.04
L = 30

grouped_data_multisample = grouped_data[grouped_data['mean_m']['count'] > 2]


c0_select = 10**4
e_select = 0.95
mu_select = 10**-5
eta_select = 10**-3

eta_list = [10**-5, 10**-4, 10**-3, 10**-2]

data_subset = grouped_data[(grouped_data['C0'] == c0_select)
                           & (grouped_data['e'] == e_select)
                           #& (grouped_data['mu'] < mu_select*1.1)
                           & (grouped_data['mu'] > 4*10**-9)
                           #& (grouped_data['eta'] == eta_select)
                           & (grouped_data['B'] == 170)
                           & (grouped_data['m_init'] == 1)        
                           & (grouped_data['pv'] == 0.02)
                           & (grouped_data['f'] == 0.3)
                           & (grouped_data['pv_type'] == 'binary')]

# +
# varying eta
#colours = sns.color_palette("viridis", 4)
colours = cm.viridis(np.linspace(0,0.9, len(timestamps)))
eta_vals = [10**-5, 10**-4, 10**-3, 10**-2]
slope_width = 3

bottom = 0.08
top = 0.95
hspace = 0.4

fig = plt.figure(figsize = (10,6))

gs_time_shift = gridspec.GridSpec(2,2)
gs_time_shift.update(left=0.08, right=0.71, bottom = bottom, top = top, hspace = hspace, wspace =0.28)
axs = np.array([[plt.subplot(gs_time_shift[0,0]), plt.subplot(gs_time_shift[0,1])],
       [plt.subplot(gs_time_shift[1,0]), plt.subplot(gs_time_shift[1,1])]])

gs_memory= gridspec.GridSpec(2,1)
gs_memory.update(left=0.8, right=0.97, bottom = bottom, top = top, wspace=0.29, hspace = hspace)

#fig, axs = plt.subplots(2,3, figsize = (9,6))

timestamps_plot = timestamps
for timestamp in timestamps_plot:

    j = timestamps.index(timestamp)
    
    #if e_list[j] == 0.1:
    #    continue

    interp_times = interp_times_list[j]
    #nvj_interp = nvj_interp_list[j]
    #nbi_interp = nbi_interp_list[j]
    
    e_eff_mean_past = e_eff_mean_past_list[j][~np.isnan(e_eff_mean_past_list[j])] / e_list[j]
    e_eff_std_past = e_eff_std_past_list[j][~np.isnan(e_eff_std_past_list[j])] / e_list[j]
    e_eff_mean_future = e_eff_mean_future_list[j][~np.isnan(e_eff_mean_future_list[j])] / e_list[j]
    e_eff_std_future = e_eff_std_future_list[j][~np.isnan(e_eff_std_future_list[j])] / e_list[j]
    
    ind = find_nearest(interp_times, peak_time_list[j])
    axs[0,0].plot([-peak_time_list[j], -peak_time_list[j]], [e_eff_mean_past[ind]-0.02, e_eff_mean_past[ind] + 0.02], 
                 color = 'k', linestyle = "-")
    
    if j == 0:
        axs[0,0].annotate(text = r"$\tau^*$", xy = (-peak_time_list[j]-12, e_eff_mean_past[ind]-0.06), fontsize = 14, 
                          xycoords = 'data')
        
        # plot slope around 0
        plot_width = 10
        slope = (e_eff_mean_future[slope_width] - e_eff_mean_past[slope_width]) / interp_times[slope_width*2]
        axs[0,0].plot([-interp_times[plot_width], interp_times[plot_width]], 
                    np.array([-interp_times[plot_width], interp_times[plot_width]])*slope + e_eff_mean_past[0], 'k:',
                     linewidth = 2, zorder = 5)
    
    for ax in [axs[0,0], axs[1,0]]:
    
        ax.plot(-interp_times[:max_shift],  e_eff_mean_past, 
                            label = r"$\eta = %s$" %round(eta_list[j],7), 
                color = colours[eta_vals.index(eta_list[j])], linestyle = '-', linewidth = 3)

        ax.plot(interp_times[:max_shift], e_eff_mean_future, 
                color = colours[eta_vals.index(eta_list[j])],
                  linestyle = '-', linewidth = 3)

        std_dev = ax.fill_between(-interp_times[:max_shift], e_eff_mean_past + e_eff_std_past,                    
                        y2 = e_eff_mean_past - e_eff_std_past, alpha = 0.05, color = colours[eta_vals.index(eta_list[j])])

        col = std_dev.get_facecolor()

        ax.fill_between(interp_times[:max_shift], e_eff_mean_future + e_eff_std_future,                    
                        y2 = e_eff_mean_future - e_eff_std_future, alpha = 0.1, color = col)

for ax in [axs[0,0], axs[1,0]]:
    ax.annotate(text = "Past phages", xy = (0.05, 0.9), fontsize = 12, xycoords = 'axes fraction')
    ax.annotate(text = "Future phages", xy = (0.55, 0.9), fontsize = 12, xycoords = 'axes fraction')

    #ax.set_yticks([])
    #ax.set_yscale('log')
    ax.set_ylim(0, 0.33)
    ax.axvline(ymin = 0, ymax = 1, linestyle = '--', color = 'k', linewidth =1)
    #ax.legend(loc = 'upper right')

    ax.set_xlabel("Time shift (bacterial generations)")
    
axs[0,0].set_ylabel("Average immunity")    

axs[0,0].set_xlim(-200, 200)
axs[1,0].set_xlim(-1500, 1500)
axs[0,0].legend(fontsize = 9.5, loc = 'upper right', bbox_to_anchor = (1,0.88))

# guerrero data

threshold = 0.15
wild_type = True
phage_only = False
pam = "perfect"
folder = "../data"

# trims: none, cut off high part, cut off low part, cut off both
# high part starts at 13, ends at index 20 (start from 21)
# low part starts at index 43 (do to -17)

start_ind = 0
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
step = 14
#step = np.mean(np.diff(time_points_in_days))
#step = 1
time_points_in_days = np.concatenate([[0], dates_int])
time_points_in_days = np.cumsum(time_points_in_days)

time_points = np.arange(0, len(dates), 1)
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

grouping = ['type_%s' %(1-threshold), 'crispr']
df_combined = pd.read_csv("%s/Guerrero2021/%s_PAM/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,
                                                                pam,1-threshold, wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)

# interpolate the bacteria spacer values
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])

# interpolated version
# each row is the time series for that numbered spacer type
# the columns correspond to the times in time_min_spacing
bac_interp = f(time_min_spacing)
phage_interp = f_phage(time_min_spacing)

nbi_interp = bac_interp.T
nvj_interp = phage_interp.T
e=1

# bac_interp and phage_interp are the same as nbi and nvi, just the shape is transposed
avg_immunity_past, avg_immunity_past_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'past')
avg_immunity_future, avg_immunity_future_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'future')

avg_immunity_mean = np.concatenate([avg_immunity_past, avg_immunity_future])
avg_immunity_std = np.concatenate([avg_immunity_past_std, avg_immunity_future_std])
# this is the number of generations per day based on 100-fold serial dilution and exponential growth
times = np.concatenate([-(time_min_spacing - time_min_spacing[0]), time_min_spacing - time_min_spacing[0]] )
avg_immunity_mean = avg_immunity_mean[np.argsort(times)]
avg_immunity_std = avg_immunity_std[np.argsort(times)]
times = times[np.argsort(times)]

axs[1,1].scatter(times, avg_immunity_mean*num_protospacers_guerrero,  marker = 'o', color = 'k', 
           label = "%s" %int((1-threshold)*100) + r"% similarity")
axs[1,1].fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers_guerrero, 
                y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers_guerrero, 
                color = 'k', alpha = 0.05)

# statistical annotations
axs[1,1].plot([2,2,500,500], [0.55, 0.66, 0.66, 0.3], linewidth=1, color='k')
axs[1,1].annotate(text = "**", xy = (220, 0.67), fontsize = 12, xycoords = 'data')
axs[1,1].plot([-2,-2,-500,-500], [0.55, 0.68, 0.68, 0.35], linewidth=1, color='k')
axs[1,1].annotate(text = "ns", xy = (-320, 0.7), fontsize = 12, xycoords = 'data')

axs[1,1].plot([2,2,200,200], [0.55, 0.56, 0.56, 0.4], linewidth=1, color='k')
axs[1,1].annotate(text = "**", xy = (20, 0.57), fontsize = 12, xycoords = 'data')
axs[1,1].plot([-2,-2,-200,-200], [0.55, 0.56, 0.56, 0.35], linewidth=1, color='k')
axs[1,1].annotate(text = "**", xy = (-150, 0.57), fontsize = 12, xycoords = 'data')

axs[1,1].axvline(ymin = 0, ymax = 1, linestyle = '--', color = 'k', linewidth =1)
axs[1,1].set_ylim(0,1)
axs[1,1].set_xlim(-1010, 1010)
axs[1,1].set_xlabel("Time shift (days)")

#### paez_espino data
time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
time_points = np.arange(0,len(time_points_in_days))

start_ind = 1
stop_ind = -3 # cut off the last np.abs(stop_ind) -1 points
step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

threshold = 0.15

grouping = ['type_%s' %(1-threshold), 'CRISPR']
df_combined = pd.read_csv("%s/PaezEspino2015/%s_PAM/banfield_data_combined_type_%s_wt_%s.csv" %(folder,
                                                                        pam,1-threshold, wild_type), index_col = 0)
bac_wide_filtered, phage_wide_filtered = paez_espino_to_array(df_combined, grouping, norm = False)

# interpolate the bacteria spacer values
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])

# interpolated version
# each row is the time series for that numbered spacer type
# the columns correspond to the times in time_min_spacing
bac_interp = f(time_min_spacing)
phage_interp = f_phage(time_min_spacing)

nbi_interp = bac_interp.T
nvj_interp = phage_interp.T
e=1

# bac_interp and phage_interp are the same as nbi and nvi, just the shape is transposed
avg_immunity_past, avg_immunity_past_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'past')
avg_immunity_future, avg_immunity_future_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'future')


avg_immunity_mean = np.concatenate([avg_immunity_past, avg_immunity_future])
avg_immunity_std = np.concatenate([avg_immunity_past_std, avg_immunity_future_std])
# this is the number of generations per day based on 100-fold serial dilution and exponential growth
times = np.concatenate([-(time_min_spacing - time_min_spacing[0]), time_min_spacing - time_min_spacing[0]] )*6.64
avg_immunity_mean = avg_immunity_mean[np.argsort(times)]
avg_immunity_std = avg_immunity_std[np.argsort(times)]
times = times[np.argsort(times)]


axs[0,1].scatter(times, avg_immunity_mean*num_protospacers_paez_espino,  marker = 'o', color = 'k', 
           label = "%s" %int((1-threshold)*100) + r"%")

#print((avg_immunity_mean*num_protospacers)[-1])
axs[0,1].fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers_paez_espino, 
                y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers_paez_espino, 
                color = 'k', alpha = 0.1)
axs[0,1].set_ylim(0, 0.15)
axs[0,1].set_xlabel("Time shift (bacterial generations)")
axs[0,1].axvline(0, linestyle = ':', color = 'k')
axs[1,0].set_ylabel("Average immunity")
axs[0,1].set_xlim(-1420, 1420)

## memory length ------------------------

ax = plt.subplot(gs_memory[0])
ax1 = plt.subplot(gs_memory[1])

# plot memory length vs eta and mu

colours = cm.viridis(np.linspace(0,1, len(data_subset)))[::-1]

#ax = axs[0,2]
#ax1 = axs[1,2]

data_eta = data_subset[(data_subset['mu'] < mu_select*1.1)
                    & (data_subset['mu'] > mu_select*0.9)].sort_values(by = 'eta')

data_mu = data_subset[data_subset['eta'] == eta_select].sort_values(by = 'mu')

ax.errorbar(data_eta['eta'],  
                data_eta['peak_immunity']['nanmean'],
            yerr = data_eta['peak_immunity']['nanstd'],
            c = colours[eta_list.index(eta_select)], 
       alpha = 0.9, marker = 'o', mec ='k', markersize = 8, linestyle = "-")


ax1.errorbar(data_mu['mu'],  
                data_mu['peak_immunity']['nanmean'],
            yerr = data_mu['peak_immunity']['nanstd'],
            c = colours[-1], 
       alpha = 0.9, marker = 's', mec ='k', markersize = 8, linestyle = "-")
    

#ax.legend(handles=legend_elements, loc='upper right', ncol = 2, fontsize = 7.8)

ax.set_xscale('log')
ax1.set_xscale('log')
#ax.set_xlim(1*10**-1, 2*np.max(data_subset['mean_m']['nanmean']))
#ax.set_ylim(10**-2, 8*10**1)
#ax.set_yscale('log')

ax.set_xlabel(r"Spacer acquisition probability $\eta$")
ax1.set_xlabel(r"Phage mutation rate $\mu$")
ax.set_ylabel(r"Memory length $\tau^*$" +"\n(bacterial generations)")
ax1.set_ylabel(r"Memory length $\tau^*$" + "\n(bacterial generations)")

### -----------------------------------

axs[0,0].set_title("A", loc = "left", fontsize = 16)
axs[0,1].set_title("C", loc = "right", fontsize = 16)
axs[1,0].set_title("B", loc = "left", fontsize = 16)
axs[1,1].set_title("D", loc = "right", fontsize = 16)
ax.set_title("E", loc = "left", fontsize = 16)
ax1.set_title("F", loc = "left", fontsize = 16)

axs[0,0].set_title("Simulation")
axs[0,1].set_title("Experiment")

plt.savefig("time_shift_figure_mu_%s_c0_%s_e_%s.pdf" %(mu_select, c0_select, e_select))
plt.savefig("time_shift_figure_mu_%s_c0_%s_e_%s.svg" %(mu_select, c0_select, e_select))
# -

# ## Supplementary figures

# ### Guerrero significance testing

# +
start_ind = 0
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
step = 14
#step = np.mean(np.diff(time_points_in_days))
#step = 1
time_points_in_days = np.concatenate([[0], dates_int])
time_points_in_days = np.cumsum(time_points_in_days)

time_points = np.arange(0, len(dates), 1)
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

grouping = ['type_%s' %(1-threshold), 'crispr']
df_combined = pd.read_csv("%s/Guerrero2021/%s_PAM/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,
                                                            pam,1-threshold, wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)

# interpolate the bacteria spacer values
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])

# interpolated version
# each row is the time series for that numbered spacer type
# the columns correspond to the times in time_min_spacing
bac_interp = f(time_min_spacing)
phage_interp = f_phage(time_min_spacing)

nbi_interp = bac_interp.T
nvj_interp = phage_interp.T
e=1

# bac_interp and phage_interp are the same as nbi and nvi, just the shape is transposed
avg_immunity_past, avg_immunity_past_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'past')
avg_immunity_future, avg_immunity_future_std = e_effective_shifted(1, bac_interp.T, phage_interp.T, 
                                                               max_shift = len(time_min_spacing), direction = 'future')

# +
i = find_nearest(time_min_spacing, 200)

e_effective_0 = np.array((e*np.sum(nbi_interp * nvj_interp, axis = 1).flatten()/
                        (np.array(np.sum(nvj_interp, axis = 1).flatten()) 
                         * np.array(np.sum(nbi_interp, axis = 1).flatten()))))

e_effective_past = np.array((e*np.sum(nbi_interp[i:] * nvj_interp[:-i], axis = 1).flatten()/
                    (np.array(np.sum(nvj_interp[:-i], axis = 1).flatten()) 
                     * np.array(np.sum(nbi_interp[i:], axis = 1).flatten()))))

e_effective_future = np.array((e*np.sum(nbi_interp[:-i] * nvj_interp[i:], axis = 1).flatten()/
                            (np.array(np.sum(nvj_interp[i:], axis = 1).flatten()) 
                             * np.array(np.sum(nbi_interp[:-i], axis = 1).flatten()))))
# -

# compare e_effective_0[i:] with e_effective_past - each bacteria abundance matched with a time shift
# this is the right direction to match the plot, from the bacteria perspective
# are the time-shifted overlaps for each bacterial abundance lower than the overlap at present? 
# comparing nbi cde | nvi cde with nbi cde/nvi abc
print(wilcoxon(e_effective_0[i:], e_effective_past, alternative = 'greater'))

# are the future overlaps for each phage abundance lower than the overlap at present? 
print(wilcoxon(e_effective_0[:-i], e_effective_past, alternative = 'greater'))

# are the future overlaps for each bacteria abundance lower than the overlap at present?
print(wilcoxon(e_effective_0[:-i], e_effective_future, alternative = 'greater'))

# are the past overlaps for each phage abundance lower than the overlap at present?
# comparing nbi 
print(wilcoxon(e_effective_0[i:], e_effective_future, alternative = 'greater'))

# +
bac_past_stats = []
bac_future_stats = []
phage_past_stats = []
phage_future_stats = []
num_points = []

for t in range(14, 1000, 14):
    i = find_nearest(time_min_spacing, t)

    e_effective_0 = np.array((e*np.sum(nbi_interp * nvj_interp, axis = 1).flatten()/
                            (np.array(np.sum(nvj_interp, axis = 1).flatten()) 
                             * np.array(np.sum(nbi_interp, axis = 1).flatten()))))

    e_effective_past = np.array((e*np.sum(nbi_interp[i:] * nvj_interp[:-i], axis = 1).flatten()/
                        (np.array(np.sum(nvj_interp[:-i], axis = 1).flatten()) 
                         * np.array(np.sum(nbi_interp[i:], axis = 1).flatten()))))

    e_effective_future = np.array((e*np.sum(nbi_interp[:-i] * nvj_interp[i:], axis = 1).flatten()/
                                (np.array(np.sum(nvj_interp[i:], axis = 1).flatten()) 
                                 * np.array(np.sum(nbi_interp[:-i], axis = 1).flatten()))))
    
    bac_past_stats.append(wilcoxon(e_effective_0[i:], e_effective_past, alternative = 'greater')[1])
    bac_future_stats.append(wilcoxon(e_effective_0[:-i], e_effective_future, alternative = 'greater')[1])
    phage_past_stats.append(wilcoxon(e_effective_0[i:], e_effective_future, alternative = 'greater')[1])
    phage_future_stats.append(wilcoxon(e_effective_0[:-i], e_effective_past, alternative = 'greater')[1])
    
    num_points.append(len(e_effective_past))

# +
fig, ax = plt.subplots(figsize = (6,4))
#ax = axs[0]
#ax1 = axs[1]
ax1 = ax.twinx()
ax.plot(-np.array(range(14, 1000, 14)), bac_past_stats, label = "Bacteria overlap with past phages")
ax.plot(range(14, 1000, 14), bac_future_stats, label = "Bacteria overlap with future phages")
ax.plot(range(14, 1000, 14), phage_past_stats, label = "Phage overlap with past bacteria")
ax.plot(-np.array(range(14, 1000, 14)), phage_future_stats, label = "Phage overlap with future bacteria")

ax.axhline(0.05, color = 'k', linestyle = ":", label = "$p=0.05$")

ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Wilcoxon signed-rank p-value")
ax.set_xlim(-1000,1000)
ax.legend()
ax1.plot(range(14, 1000, 14), num_points, color = 'k', linewidth = 1.5, linestyle = "--", label = "Number of data points")
ax1.plot(-np.array(range(14, 1000, 14)), num_points, color = 'k', linewidth = 1.5, linestyle = "--")
ax1.legend()
ax1.set_ylabel("Number of time points included in comparison")
ax.set_yscale('log')
plt.tight_layout()
plt.savefig("Guerrero_wilcoxon_p_values.pdf")
