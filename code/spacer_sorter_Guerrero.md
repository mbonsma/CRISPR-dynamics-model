---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: spacer_phage
    language: python
    name: spacer_phage
---

# Spacer_sorter for Guerrero 2021 data

Take BLAST results from spacer_finder.ipynb, sort and aggregate.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
import os
from Bio import SeqIO
from tqdm import tqdm
import Levenshtein # for calculating sequence similarity
from sklearn.cluster import AgglomerativeClustering # for clustering spacers
import matplotlib.cm as cm
```

```python
%matplotlib inline
```

```python
def count_paired_ends(all_spacers, spacer_subject_ids, unique_sp, spacer_counts, seq_type = str, 
                      use_PAMs = False, PAM_seqs_5 = None, PAM_seqs_3 = None, unique_sp_PAM_5 = None, unique_sp_PAM_3 = None):
    """
    Some paired-end reads overlap, and this means that some spacers end up double-counted.
    This function takes the spacer sequences detected and decrements the spacer count for each unique 
    sequence if that sequence shows up on both reads of a paired read. 
    It's possible that this is not perfect and that some double-counts may be legitimate 
    (the way to tell would be to check the total read overlap besides the spacer), but I expect those cases to 
    be very infrequent.
    
    Caution: the change to spacer_counts happens in-place.
    
    Inputs:
    all_spacers : full list of spacers detected from a fasta file
    spacer_subject_ids : list of read headers from deteted spacers
    unique_sp : list of unique spacer sequences
    spacer_counts : list of spacer counts corresponding to unique_sp
    seq_type : whether spacer sequences are given by an integer label or the actual sequence itself
    use_PAMs : whether to group using separate PAM sequences as well
    PAM_seqs : if use_PAMs == True, list of PAM sequences corresponding to all_spacers
    unique_sp_PAM : if use_PAMs == True, list of PAM sequences corresponding to unique_sp
    
    Returns:
    spacer_counts : the spacer_counts list after decrementing the double-counted spacers. 
    """
    
    # data frame with read ID and spacer sequence
    reads_seqs = pd.DataFrame()
    reads_seqs["read_id"] = spacer_subject_ids
    reads_seqs["spacer_sequence"] = all_spacers
    if use_PAMs == True:
        reads_seqs["PAM_region_5"] = PAM_seqs_5
        reads_seqs["PAM_region_3"] = PAM_seqs_3
    
    reads_seqs[['pair_id', 'direction']] = reads_seqs['read_id'].str.rsplit('.',1, expand = True) # get just the pair id
    
    # if the count in the 'direction' column is >1, then that spacer is showing up on both ends of a pair 
    # and should not be double-counted
    if use_PAMs == False:
        grouped = reads_seqs.groupby(['pair_id', 'spacer_sequence'])['direction'].count().reset_index()
    else: 
        grouped = reads_seqs.groupby(['pair_id', 'spacer_sequence', 'PAM_region_5', 'PAM_region_3'])['direction'].count().reset_index()
        
    if seq_type == str:
        # remove trailing \n from spacer sequences
        unique_sp_strip = []
        for seq in unique_sp:
            unique_sp_strip.append(seq.strip())

        unique_sp = unique_sp_strip
    
    # double counts are any that show up more than once 
    grouped_double_count = grouped[grouped['direction'] > 1]
    
    # this data frame has for each sequence the number of times it was double-counted
    if use_PAMs == False:
        double_counts = grouped_double_count.groupby('spacer_sequence')[['pair_id']].count().reset_index()
    else:
        double_counts = grouped_double_count.groupby(['spacer_sequence', 'PAM_region_5', 'PAM_region_3'])[['pair_id']].count().reset_index()
        
    double_counts = double_counts.rename(columns={"pair_id": "count"})
    
    # decrement spacer_counts according to double_counting
    if use_PAMs == False:
        for i, row in double_counts.iterrows():
            ind = unique_sp.index(row['spacer_sequence'])
            spacer_counts[ind] -= row['count']
            
    else:
        counts_df = pd.DataFrame()
        counts_df['spacer_sequence'] = unique_sp
        counts_df['PAM_region_5'] = unique_sp_PAM_5
        counts_df['PAM_region_3'] = unique_sp_PAM_3
        counts_df['spacer_count'] = spacer_counts
        # merge with double counts to do subtraction
        counts_df = counts_df.merge(double_counts, on = ['spacer_sequence', 'PAM_region_5', 'PAM_region_3'], how = 'left').fillna(0)
        counts_df['new_count'] = counts_df['spacer_count'] - counts_df['count']
        spacer_counts = counts_df['new_count']

    if np.any(spacer_counts[spacer_counts < 1]): # make sure nothing weird happened giving a negative count
        print("Warning! a spacer count less than 1 was generated") 
    
    return spacer_counts
```

```python
def load_wild_type_spacers(datapath, cr):
    """
    Returns a list of wild-type spacers from the gordonia reference genomes
    
    Inputs
    ----------
    datapath : path to text files with list of spacers
    cr : either "CR1" or "CR2" for CRISPR1 or CRISPR2
    """
    wild_type_spacers_ref = []

    with open ("%s/gordonia_%s_ref_spacers.txt" %(datapath, cr), "r") as f:
        wild_type_spacers_ref_unstripped = f.readlines()
        for sp in wild_type_spacers_ref_unstripped:
            if sp[0] != '>':
                wild_type_spacers_ref.append(sp.rstrip())
    
    return wild_type_spacers_ref
```

## Process protospacers

The row indices of the below dataframes correspond to the query id in the protospacer data:
i.e. query_id = '331_12552' means spacer type 331, matches spacer_types_bac_unique.loc[12552] or spacer_types_bac.loc[12552]

```python tags=[]
datapath = "results/2022-03-23"
spacer_types_bac_CR1 = pd.read_csv("%s/spacer_types_Guerrero2021_CR1.csv" %(datapath))
spacer_types_bac_CR2 = pd.read_csv("%s/spacer_types_Guerrero2021_CR2.csv" %(datapath))

#spacer_types_bac_unique = spacer_types_bac[['sequence', 'type']].drop_duplicates()
#spacer_types_bac_all_previous = pd.read_csv("results/2019-09-27/spacer_types_MOI_2b.csv") 
```

## Process protospacers

Count unique spacer-PAM-query combinations for each accession

```python
phage_only = False
```

```python
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
#query = "%s/gordonia_%s_repeat.fasta" %(datapath, cr)

with open ('%s/../SRR_Acc_List.txt' %(datapath), 'r') as f:
    accessions = f.readlines()
    
accession_list = []
for accession in accessions:
    accession_list.append(accession.rstrip())

if phage_only == True:
    protospacer_datapath = "%s/protospacers_phage_only" %datapath
else:
    protospacer_datapath = "%s/protospacers" %datapath

for accession in accession_list:
    unique_protospacers = []
    unique_PAMs_5 = []
    unique_PAMs_3 = []
    counts = []
    query_ids = []
    
    # iterate through sub-files for each accession
    for fn in tqdm(glob("%s/%s*_protospacers.txt" %(protospacer_datapath, accession))):
        protospacers = pd.read_csv(fn)
        
        if len(protospacers) == 0:
            continue
            
        protospacers[['spacer_type','sequence_id', 'crispr']] = protospacers['query_id'].str.split('_', expand=True)
        protospacers['sequence_id'] = protospacers['sequence_id'].astype('int')
        spacers = list(protospacers['sequence'])
        
        # group by sequence and PAM region to count
        protospacers_grouped = protospacers.groupby(['sequence', 'PAM_region_5', 'PAM_region_3',
                                                     'query_id']).count().reset_index().sort_values(by = 'query_id', ascending = False)
        protospacers_grouped = protospacers_grouped.rename(columns = {"subject_id": "count"})

        # grouped by spacer and PAM
        unique_spacers = list(protospacers_grouped['sequence'])
        # 5' region is to the left, 3' region is to the right
        unique_PAM_seqs_5 = list(protospacers_grouped['PAM_region_5'])
        unique_PAM_seqs_3 = list(protospacers_grouped['PAM_region_3'])
        
        # count unique spacer and PAM combinations, removing paired-end overlap
        protospacers = protospacers.fillna("")
        sp_count = count_paired_ends(list(protospacers['sequence']), list(protospacers['subject_id']), unique_spacers, 
                                     list(protospacers_grouped['count']), seq_type = str, 
                                  use_PAMs = True, PAM_seqs_5 = list(protospacers['PAM_region_5']), PAM_seqs_3 = list(protospacers['PAM_region_3']),
                                     unique_sp_PAM_5 = unique_PAM_seqs_5,
                                    unique_sp_PAM_3 = unique_PAM_seqs_3)
        
        unique_protospacers += unique_spacers
        unique_PAMs_5 += unique_PAM_seqs_5
        unique_PAMs_3 += unique_PAM_seqs_3
        query_ids += list(protospacers_grouped['query_id'])
        counts += list(sp_count)
        
    # combine sub-lists into list for each time point
    accession_df = pd.DataFrame()
    accession_df['sequence'] = unique_protospacers
    accession_df['PAM_region_5'] = unique_PAMs_5
    accession_df['PAM_region_3'] = unique_PAMs_3
    accession_df['query_id'] = query_ids
    accession_df['count'] = counts
    
    # group by sequence, PAM, and query - sum over entire day
    accession_df = accession_df.groupby(['sequence', 'PAM_region_5', 
                                         'PAM_region_3', 'query_id']).sum('count').reset_index().sort_values(by = 'count', ascending = False)
    if phage_only == True:
        accession_df.to_csv("%s_protospacers_phage_only.txt" %accession, index = None)
    else:
        accession_df.to_csv("%s_protospacers.txt" %accession, index = None)
```

```python
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
metadata = pd.read_csv("%s/../SraRunTable.txt" %datapath)

accessions = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
```

```python
# concatenate accession_df files
df_list = []
for i, accession in tqdm(enumerate(accessions)):
    if phage_only == True:
        accession_df = pd.read_csv("%s_protospacers_phage_only.txt" %accession)
    else:
        accession_df = pd.read_csv("%s_protospacers.txt" %accession)
    accession_df['date'] = dates[i]
    accession_df['time_point'] = i
    df_list.append(accession_df)
    
all_df = pd.concat(df_list)
if phage_only == True:
    all_df.to_csv("protospacer_counts_phage_only.txt", index = None)
else:
    all_df.to_csv("protospacer_counts.txt", index = None)
```

```python
# create PAM list for webLogo
# DO WITH FULL DATASET AT END
#pams = all_df[all_df['PAM_region'].str.len() == 10][['PAM_region', 'count']]

max_length = 4
# by selecting just n nucleotides, we are guaranteed to have max 4^n  unique sequences
#accession_df['PAM_5_short'] = accession_df['PAM_region_5'].str.slice(max_length,) # get just the last half, expecting a short one
#accession_df['PAM_3_short'] = accession_df['PAM_region_3'].str.slice(0,max_length) # get just the first half, expecting a short one
#pams_5 = accession_df[accession_df['PAM_5_short'].str.len() == max_length][['PAM_5_short', 'count']]
#pams_3 = accession_df[accession_df['PAM_3_short'].str.len() == max_length][['PAM_3_short', 'count']]

all_df['PAM_5_short'] = all_df['PAM_region_5'].str.slice(-max_length,) # get just the last n, expecting a short one
all_df['PAM_3_short'] = all_df['PAM_region_3'].str.slice(0,max_length) # get just the first n, expecting a short one
pams_5 = all_df[all_df['PAM_5_short'].str.len() == max_length][['PAM_5_short', 'count']]
pams_3 = all_df[all_df['PAM_3_short'].str.len() == max_length][['PAM_3_short', 'count']]

pams_5 = pams_5.groupby('PAM_5_short').sum().reset_index().sort_values(by = 'count')
pams_3 = pams_3.groupby('PAM_3_short').sum().reset_index().sort_values(by = 'count')

pams_3.to_csv("pams_3_phage_only_%s.csv" %phage_only, sep = "\t", index = None)
pams_5.to_csv("pams_5_phage_only_%s.csv" %phage_only, sep = "\t", index = None)

pams_5.sort_values(by = 'count', ascending = False).head(100).to_csv("pams_5_head_phage_only_%s.csv" %phage_only, sep = "\t", index = None)
pams_3.sort_values(by = 'count', ascending = False).head(100).to_csv("pams_3_head_phage_only_%s.csv" %phage_only, sep = "\t", index = None)
```

```python
# make a list with duplicates to use weblogo

max_file_len = 10000

pams_5 = pams_5[~pams_5['PAM_5_short'].str.contains('N')]
pams_3 = pams_3[~pams_3['PAM_3_short'].str.contains('N')]

# we just want relative frequencies, so rescale counts to be within the max file length
pams_5['count_rescaled'] = round(pams_5['count'] / (pams_5['count'].sum() / max_file_len)).astype('int')
pams_3['count_rescaled'] = round(pams_3['count'] / (pams_3['count'].sum() / max_file_len)).astype('int')

pams_5 = pams_5[pams_5['count'] > 0]
pams_3 = pams_3[pams_3['count'] > 0]
```

```python
# save list to file to associate with distance matrix
with open ("pams_5_expanded_phage_only_%s.txt" %phage_only, 'w') as f:
    for i, row in pams_5.iterrows():
        for num in range(row['count_rescaled']):
            f.write(str(row['PAM_5_short'].strip()) + "\n")
            
# save list to file to associate with distance matrix
with open ("pams_3_expanded_phage_only_%s.txt" %phage_only, 'w') as f:
    for i, row in pams_3.iterrows():
        for num in range(row['count_rescaled']):
            f.write(str(row['PAM_3_short'].strip()) + "\n")
```

I am also picking up 5'GTT as the PAM on the 5' end, so the way I've saved it, this means the last 3 nucleotides in 'PAM_region_5' should be AAC. 


## Clustering all spacers and protospacers with different thresholds

In Paez-Espino 2015, spacers were grouped that shared >85% in length and > 85% sequence identity. Here, I use the Levenshtein ratio as the metric of edit distance between sequences, then cluster using AgglomerativeClustering from scipy with the 'average' linkage criterion. (The 'complete' linkage criterion, which guarantees a minimum similarity between all members of a cluster, is very sensitive to which sequences are present.)

Here, we cluster all detected spacer and protospacer sequences with different thresholds in order to examine the effect of similarity on average immunity.


### Iterate through a few different thresholds

```python
# spacers and protospacers detected in spacer_finder.ipynb

spacer_types_bac_CR1 = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_CR1.csv")
spacer_types_bac_CR2 = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_CR2.csv")

if phage_only == True:
    spacer_types_phage_all = pd.read_csv("results/2022-03-29/protospacer_counts_phage_only.txt")
else:
    spacer_types_phage_all = pd.read_csv("results/2022-03-29/protospacer_counts.txt")
spacer_types_phage_all[["spacer_type", "sequence_id", "crispr"]] = spacer_types_phage_all['query_id'].str.split('_', 
                                                                                            expand = True)
```

```python tags=[]
# load wild-type spacers
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
wild_type_spacers_CR1 = load_wild_type_spacers(datapath, "CR1")
wild_type_spacers_CR2 = load_wild_type_spacers(datapath, "CR2")
```

```python
cr = "CR2"

if cr == "CR1":
    wild_type_spacers = wild_type_spacers_CR1
    spacer_types_phage = spacer_types_phage_all[spacer_types_phage_all['crispr'] == 'CR1']
    spacer_types_bac = spacer_types_bac_CR1
elif cr == "CR2":
    wild_type_spacers = wild_type_spacers_CR2
    spacer_types_phage = spacer_types_phage_all[spacer_types_phage_all['crispr'] == 'CR2']
    spacer_types_bac = spacer_types_bac_CR2
```

```python
# remove wild-type spacers so we can add them back at the begining

unique_spacers_all = list(set(list(spacer_types_phage['sequence']) + list(spacer_types_bac['sequence'])))
unique_spacers_no_wildtype = []

for sp in unique_spacers_all:
    if sp in wild_type_spacers:
        continue
    unique_spacers_no_wildtype.append(sp)
    
# add wild-type spacers to start of list 
unique_spacers = wild_type_spacers + unique_spacers_no_wildtype

# save list to file to associate with distance matrix
with open ('unique_spacers_all_%s_phage_only_%s.txt' %(cr, phage_only), 'w') as f:
    for seq in unique_spacers:
        f.write(str(seq.strip()) + "\n")
```

```python
# create distance matrix using the Levenshtein similarity ratio
# this is slow for a long list of spacers, just a fact of life
# just do this ONCE per set of spacers, then reload
 
distance_matrix = np.zeros((len(unique_spacers), len(unique_spacers)))

for i, spacer1 in tqdm(enumerate(unique_spacers)):
    for j, spacer2 in enumerate(unique_spacers):
        if i <= j:
            distance_matrix[i,j] = 1 - Levenshtein.ratio(str(spacer1.strip()), str(spacer2.strip()))

# matrix is triangular, make it symmetric
# since the diagonals are zero, don't need to worry about double-counting them
distance_matrix = np.array(distance_matrix, dtype = 'float16')
distance_matrix = distance_matrix + distance_matrix.T
```

```python
#distance_matrix = np.array(distance_matrix, dtype = 'float16') # make array smaller by reducing float size
# float16 has max +- 65500, ~4 decimal places, good enough for this purpose: np.finfo(np.float16)
np.savez_compressed("distance_matrix_all_%s_phage_only_%s" %(cr, phage_only), distance_matrix)
```

## Agglomerative clustering with different thresholds

```python
cr = "CR2"
```

```python
with np.load('results/2022-03-29/distance_matrix_all_%s_phage_only_%s.npz' %(cr, phage_only)) as data:
    distance_matrix = data['arr_0']
    # data.files is how I know it's called 'arr_0'
```

```python
# load associated spacer list
with open ('results/2022-03-29/unique_spacers_all_%s_phage_only_%s.txt' %(cr, phage_only), 'r') as f:
    unique_spacers = f.readlines()
```

```python
thresholds = np.arange(0.01, 0.16, 0.01)
thresholds = np.concatenate([[0.005], thresholds])
```

```python
distance_matrix.shape
```

```python
# this takes a decently long time with lots of data as well
# also takes a lot of RAM

if cr == "CR1":
    wild_type_spacers = wild_type_spacers_CR1
else:
    wild_type_spacers = wild_type_spacers_CR2

for threshold in tqdm(thresholds):
    fit = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage='average',
                             affinity='precomputed').fit(distance_matrix)
    wild_type_labels = fit.labels_[:len(wild_type_spacers)]
    
    spacer_types = pd.DataFrame()
    spacer_types['sequence'] = unique_spacers
    spacer_types['type'] = fit.labels_
    spacer_types['sequence'] = spacer_types['sequence'].astype(str)
    spacer_types['sequence'] = spacer_types['sequence'].str.strip()
    spacer_types = spacer_types.drop_duplicates()
    
    spacer_types.to_csv("spacer_protospacer_types_%s_similarity_%s.csv" %(cr, 1-threshold), index=None)
    # drop any spacers that are grouped with wild-type spacers
    spacer_types_df_new = spacer_types[~spacer_types['type'].isin(wild_type_labels)]
    spacer_types_df_new.to_csv("spacer_protospacer_types_%s_wildtype_removed_similarity_%s.csv" %(cr, 1-threshold), index=None)
```

```python

```
