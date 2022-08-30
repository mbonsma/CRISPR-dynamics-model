# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
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

# # Make dataframe for simulations that go extinct

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

from sim_analysis_functions import find_nearest, load_simulation

from spacer_model_plotting_functions import get_clone_sizes

# %matplotlib inline

# ## Load simulation data

# +
search_folders = ["2019-02-07", "2019-05-07", "2019-05-08", "2019-06-24", 
                 "2021-09-09", "2021-02-01", "2021-02-19", "2021-05-25", "2020-09-15", "2021-06-11",
                 "2021-08-26", "2021-09-08", "2021-09-13", "2021-11-16", "2021-11-18", "2021-11-19"]
search_folders_run = ["2019-03-14", "2019-05-14"]

top_folder = "/media/madeleine/My Passport/Data/results"

timestamps_list = []
folders_list = []

for f in search_folders:
    folder = "%s/%s" %(top_folder, f)
    # folder_list = !find "$folder" -type f -path "{folder}/serialjobdir*/pop_array*" | rev | cut -c45- | rev
    # timestamps = !find "$folder" -type f -path "{folder}/serialjobdir*/pop_array*" | rev | cut -d'_' -f1 | cut -c9- | rev
    
    timestamps_list += timestamps
    folders_list += folder_list
    print("%s: %s" %(f, len(timestamps)))
    
for f in search_folders_run:
    folder =  "%s/%s" %(top_folder, f)
    # folder_list = !find "$folder" -type f -path "{folder}/run*/serialjobdir*/pop_array*" | rev | cut -c45- | rev
    # timestamps = !find "$folder" -type f -path "{folder}/run*/serialjobdir*/pop_array*" | rev | cut -d'_' -f1 | cut -c9- | rev
    
    timestamps_list += timestamps
    folders_list += folder_list
    print("%s: %s" %(f, len(timestamps)))

# +
params_20190207 = pd.read_csv("%s/2019-02-07/params_list.txt" %top_folder, 
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# this covers 2019-05-07, 2019-05-08, 2019-05-14, and 2021-02-19 as well since these are taken from the 2019-03-14 list
params_20190314 = pd.read_csv("%s/2019-03-14/params_list_shuffled.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# mutation rate 3 x 10^-8 and below
params_20200915 = pd.read_csv("%s/2020-09-15/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# exponential pv: covers 2021-02-01, 2021-09-08 and 2021-09-09 as well
params_20190624 = pd.read_csv("%s/2019-06-24/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# changing B
params_20210525 = pd.read_csv("%s/2021-05-25/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# changing f
params_20211116 = pd.read_csv("%s/2021-11-16/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# changing pv
params_20211118 = pd.read_csv("%s/2021-11-18/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# changing pv
params_20211119 = pd.read_csv("%s/2021-11-19/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max"], sep = ' ')

# theta function pv, has an extra column for theta
params_20210611 = pd.read_csv("%s/2021-06-11/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max", "theta"], sep = ' ')

# theta function pv, has an extra column for theta
params_20210826 = pd.read_csv("%s/2021-08-26/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max", "theta"], sep = ' ')

# theta function pv, has an extra column for theta
params_20210913 = pd.read_csv("%s/2021-09-13/params_list.txt" %top_folder,            
            names = ["B", "C0", "g", "f", "R", "eta", "pv", "alpha", "e", "mu",
           "L", "epsilon", "m_init", "max_save", "gen_max", "theta"], sep = ' ')
# -

# load table of results, skip simulations that are already in results
all_data = pd.read_csv("all_data.csv", index_col = 0)
all_data = all_data.drop_duplicates()

all_data.shape

extinction_df = pd.read_csv("extinction_df.csv", index_col = 0)
print(extinction_df.shape)

# ## Add extinction info to df

# +
n_snapshots = 50 # number of points to sample (evenly) to get population averages

c0_list = []
g_list = []
eta_list = []
mu_list = []
m_init_list = []
max_m_list = []
alpha_list = []
e_list = []
B_list = []
f_list = []
pv_list = []
end_time_list = []
bac_extinct_list = []
gen_max_list = []
pv_type_list = []
phage_extinct_list = []
folder_date_list = []
mean_m_list = []
mean_nb_list = []
mean_nv_list = []
mean_C_list = []
mean_nu_list = []
theta_list = []
e_effective_list = []
timestamps = []

exponential_pv_dates = ["2019-06-24", "2021-09-09"]
exponential_pv_025_dates = ["2021-02-01", "2021-09-08"]
theta_pv_dates = ["2021-06-11", "2021-08-26", "2021-09-13"]

for i, timestamp in tqdm(enumerate(timestamps_list)):

    if np.sum(all_data['timestamp'].isin([timestamp])) > 0: # then this timestamp is not extinct
        continue
        
    if np.sum(extinction_df['timestamp'].isin([timestamp])) > 0: # then this timestamp has been analyzed
        continue
    
    #if i < 5800:
    #    continue
    #if i > 2000:
    #    break
        
    folder = folders_list[i]
    
    # regex to match a year beginning with 20
    folder_date = re.findall("20[0-9][0-9]-[0-1][0-9]-[0-3][0-9]", folder) 

    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
         max_m, mutation_times, all_phages = load_simulation(folder, timestamp);

    #if m_init > 1:
    #    continue
        
    # check for extinction:
    last_tp = pop_array[-1].toarray().flatten()
    if not np.any(last_tp[:max_m+1] > 0):
        bac_extinct = 1
    else:
        bac_extinct = 0
    if not np.any(last_tp[max_m+1:2*max_m+1] > 0):
        phage_extinct = 1
    else:
        phage_extinct = 0

    if phage_extinct == 0 and bac_extinct == 0:
        continue # analyze this one with the simulation_stats script
        
    if any(x in folder for x in exponential_pv_dates): # then this is a new pv sim
        pv_type = 'exponential'
    elif any(x in folder for x in exponential_pv_025_dates):  # then this is a new pv sim with rate 0.25
        pv_type = 'exponential_025'
    elif any(x in folder for x in theta_pv_dates): # then this is theta function pv
        pv_type = 'theta_function'
    else:
        pv_type = 'binary'
    
    t_ss = gen_max / 5 # minimun t_ss = 2000, otherwise gen_max/5
    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)
    
    if t_ss  >= last_tp[-1]*g*c0 - 10: # no steady-state data to work with
        mean_m = np.nan
        mean_nb = np.nan
        mean_nv = np.nan
        mean_C = np.nan
        mean_nu = np.nan
        e_effective = np.nan
    else:
        # get mean_m
        (mean_m, mean_phage_m, mean_large_phage_m, mean_large_phage_size, Delta_bac, Delta_phage, 
             mean_nu, e_effective) = get_clone_sizes(pop_array, c0, e, max_m, t_ss_ind, pv_type, theta, 
                                                     all_phages, size_cutoff=1)
        
        # get mean population sizes
        
        nv = np.sum(pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1], axis = 1)
        nb = np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)
        nb0 = pop_array[t_ss_ind:, 0]
        C = pop_array[t_ss_ind:, -2]

        mean_nb = np.mean(nb[::int(len(nb)/n_snapshots)])
        mean_nv = np.mean(nv[::int(len(nb)/n_snapshots)])
        mean_C = np.mean(C[::int(len(nb)/n_snapshots)])

    pv_type_list.append(pv_type)
    
    folder_date_list.append(folder_date[0])
    timestamps.append(timestamp)
    bac_extinct_list.append(bac_extinct)
    phage_extinct_list.append(phage_extinct)
    end_time_list.append(last_tp[-1]*g*c0)
    c0_list.append(c0)
    eta_list.append(eta)
    f_list.append(f)
    pv_list.append(pv)
    m_init_list.append(m_init)
    e_list.append(e)
    mu_list.append(mu)
    B_list.append(B)
    mean_m_list.append(mean_m)
    mean_nu_list.append(mean_nu)
    mean_nb_list.append(mean_nb)
    mean_nv_list.append(mean_nv)
    mean_C_list.append(mean_C)
    e_effective_list.append(e_effective)
    gen_max_list.append(gen_max)
    theta_list.append(theta)
# -

df = pd.DataFrame()

df['C0'] = c0_list
df['mu'] = mu_list
df['eta'] = eta_list
df['e'] = e_list
df['B'] = B_list
df['f'] = f_list
df['pv'] = pv_list
df['m_init'] = m_init_list
df['mean_m'] = mean_m_list
df['mean_nb'] = mean_nb_list
df['mean_nu'] = mean_nu_list
df['mean_C'] = mean_C_list
df['mean_nv'] = mean_nv_list
df['e_effective'] = e_effective_list
df['pv_type'] = pv_type_list
df['theta'] = theta_list
df['gen_max'] = gen_max_list
df['bac_extinct'] = bac_extinct_list
df['phage_extinct'] = phage_extinct_list
df['timestamp'] = timestamps
df['end_time'] = end_time_list
df['folder_date'] = folder_date_list


# everything except theta pv
all_params = pd.concat([params_20190207, params_20190314, params_20200915, params_20190624, params_20210525, params_20211116]) 

all_params = all_params.drop_duplicates()
all_params = all_params.drop(['max_save', 'gen_max'], axis = 1) # drop these columns because data will come from df

theta_params = pd.concat([params_20210611, params_20210826, params_20210913])
theta_params = theta_params.drop_duplicates()
theta_params = theta_params.drop(['max_save', 'gen_max'], axis = 1)

all_params['theta'] = 0 # set theta = 0 for all other simulations

all_params = pd.concat([all_params, theta_params])

# add mean_m to dataframe by joining on parameters that vary
all_params = all_params.merge(df, on = ['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv', 'm_init', 'theta'])

# assuming extinct simulations are skipped in all_data
all_data['bac_extinct'] = 0
all_data['phage_extinct'] = 0
all_data['end_time'] = all_data['gen_max']

result = pd.concat([df, all_data], join = 'inner')

result.shape

result.columns

extinction_df.columns == result.columns

# add new data to original df
result = pd.concat([extinction_df, result]).reset_index()

result = result.drop("index", axis = 1)

result = result.drop_duplicates()

result = result.drop_duplicates(subset="timestamp")

result.shape

result.to_csv("extinction_df.csv")
