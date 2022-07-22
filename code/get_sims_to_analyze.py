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

# # Get a list of unprocessed simulations to calculate stats 

import pandas as pd
import numpy as np
from tqdm import tqdm

from sim_analysis_functions import load_simulation

# +
# list of folders to get simulation list from
search_folders = ["2019-02-07", "2019-05-07", "2019-05-08", "2019-06-24", 
                 "2021-09-09", "2021-02-01", "2021-02-19", "2021-05-25", "2020-09-15", "2021-06-11",
                 "2021-08-26", "2021-09-08", "2021-09-13", "2021-11-16", "2021-11-18", "2021-11-19"]

search_folders_run = ["2019-03-14", "2019-05-14"]

top_folder = "/media/madeleine/My Passport/Data/results"

timestamps_list = []
folders_list = []

for f in search_folders:
    folder = "%s/%s" %(top_folder, f)
    # folder_list = !find "$folder" -type f -path "{folder}/serialjobdir*/pop_array*" | rev | cut -c45- | cut -d" " -f1,2 | rev
    # timestamps = !find "$folder" -type f -path "{folder}/serialjobdir*/pop_array*" | rev | cut -d'_' -f1 | cut -c9- | rev

    timestamps_list += timestamps
    folders_list += folder_list
    print("%s: %s" %(f, len(timestamps)))
    
for f in search_folders_run:
    folder =  "%s/%s" %(top_folder, f)
    # folder_list = !find "$folder" -type f -path "{folder}/run*/serialjobdir*/pop_array*" | rev | cut -c45- | cut -d" " -f1,2 | rev
    # timestamps = !find "$folder" -type f -path "{folder}/run*/serialjobdir*/pop_array*" | rev | cut -d'_' -f1 | cut -c9- | rev
  
    timestamps_list += timestamps
    folders_list += folder_list
    print("%s: %s" %(f, len(timestamps)))

# +
# load table of results, skip simulations that are already in results
all_data = pd.read_csv("all_data.csv", index_col = 0)
all_data = all_data.drop_duplicates()
print(all_data.shape)

# load extinction df to skip extinct simulations
extinction_df = pd.read_csv("extinction_df.csv", index_col = 0)
print(extinction_df.shape)

# this selects simulations where either phage or bacteria are extinction
extinction_df_extinct = extinction_df[np.any(extinction_df[['phage_extinct', 'bac_extinct']] == 1, axis = 1)]
# -

# ## Get list of un-analyzed simulations

# +
# skip simulations that are already in all_data results file

unanalyzed_timestamps = []
unanalyzed_folders = []

for i, timestamp in tqdm(enumerate(timestamps_list)):
    if np.sum(extinction_df_extinct['timestamp'].isin([timestamp])) > 0: # then this timestamp has been analyzed
        continue
    
    if np.sum(all_data['timestamp'].isin([timestamp])) > 0: # then this timestamp has been analyzed
        continue
        
    folder = folders_list[i]
        
    # load simulation
    try:
        f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
             max_m, mutation_times, all_phages = load_simulation(folder, timestamp)
    except FileNotFoundError:
        continue

    # check for extinction: skip extinct simulations
    last_tp = pop_array[-1].toarray().flatten()
    if not np.any(last_tp[:max_m+1] > 0):
        continue
    if not np.any(last_tp[max_m+1:2*max_m+1] > 0):
        continue

    unanalyzed_timestamps.append(timestamp)
    unanalyzed_folders.append(folder)
# -

np.savetxt("to_analyze.txt", unanalyzed_folders, fmt = "%s")

# bash command to trim excess folder info
# !cat to_analyze.txt | cut -d"/" -f 7- > to_analyze_temp.txt
# !mv to_analyze_temp.txt to_analyze.txt


