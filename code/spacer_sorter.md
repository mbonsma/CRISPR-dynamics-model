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

```python

```

# Spacer_sorter

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
                      use_PAMs = False, PAM_seqs = None, unique_sp_PAM = None):
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
        reads_seqs["PAM_region"] = PAM_seqs
    
    reads_seqs[['pair_id', 'direction']] = reads_seqs['read_id'].str.rsplit('.',1, expand = True) # get just the pair id
    
    # if the count in the 'direction' column is >1, then that spacer is showing up on both ends of a pair 
    # and should not be double-counted
    if use_PAMs == False:
        grouped = reads_seqs.groupby(['pair_id', 'spacer_sequence'])['direction'].count().reset_index()
    else: 
        grouped = reads_seqs.groupby(['pair_id', 'spacer_sequence', 'PAM_region'])['direction'].count().reset_index()
        
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
        double_counts = grouped_double_count.groupby(['spacer_sequence', 'PAM_region'])[['pair_id']].count().reset_index()
        
    double_counts = double_counts.rename(columns={"pair_id": "count"})
    
    # decrement spacer_counts according to double_counting
    if use_PAMs == False:
        for i, row in double_counts.iterrows():
            ind = unique_sp.index(row['spacer_sequence'])
            spacer_counts[ind] -= row['count']
            
    else:
        counts_df = pd.DataFrame()
        counts_df['spacer_sequence'] = unique_sp
        counts_df['PAM_region'] = unique_sp_PAM
        counts_df['spacer_count'] = spacer_counts
        # merge with double counts to do subtraction
        counts_df = counts_df.merge(double_counts, on = ['spacer_sequence', 'PAM_region'], how = 'left').fillna(0)
        counts_df['new_count'] = counts_df['spacer_count'] - counts_df['count']
        spacer_counts = counts_df['new_count']

    if np.any(spacer_counts[spacer_counts < 1]): # make sure nothing weird happened giving a negative count
        print("Warning! a spacer count less than 1 was generated") 
    
    return spacer_counts
```

```python
def load_wild_type_spacers(datapath, cr):
    """
    Returns a list of wild-type spacers from the S. thermophilus reference genome
    and from blasting against accession SRR1873863 (control data with no phage)
    
    Inputs
    ----------
    datapath : path to text files with list of spacers
    cr : either "CR1" or "CR3" for CRISPR1 or CRISPR3
    """
    wild_type_spacers = []
    wild_type_spacers_ref = []

    with open ('%s/SRR1873863_%s_spacers_unique.txt' %(datapath, cr), 'r') as f:
        wild_type_spacers_unstripped = f.readlines()
        for sp in wild_type_spacers_unstripped:
            wild_type_spacers.append(sp.rstrip())

    # load spacers from reference genome
    with open ("%s/NZ_CP025216_%s_spacers.txt" %(datapath,cr), "r") as f:
        wild_type_spacers_ref_unstripped = f.readlines()
        for sp in wild_type_spacers_ref_unstripped:
            wild_type_spacers_ref.append(sp.rstrip())
            
    # add reference genome wild-type spacers and keep only unique sequences
    wild_type_spacers += wild_type_spacers_ref
    wild_type_spacers = list(set(wild_type_spacers))
    
    return wild_type_spacers
```

## Process protospacers

The row indices of the below dataframes correspond to the query id in the protospacer data:
i.e. query_id = '331_12552' means spacer type 331, matches spacer_types_bac_unique.loc[12552] or spacer_types_bac.loc[12552]


CRISPR3 PAM: GGNG
CRISPR1 PAM: AGAAW (W = A or T)

```python tags=[]
cr = "CR1"

if cr == "CR1":
    datapath = "results/2021-11-05"
    spacer_types_bac = pd.read_csv("%s/spacer_types_MOI_2b.csv" %datapath)
elif cr == "CR3":
    datapath = "results/2022-02-03"
    spacer_types_bac = pd.read_csv("%s/spacer_types_MOI_2b_%s.csv" %(datapath, cr))

spacer_types_bac_unique = spacer_types_bac[['sequence', 'type']].drop_duplicates()
#spacer_types_bac_all_previous = pd.read_csv("results/2019-09-27/spacer_types_MOI_2b.csv") 
```

## Process protospacers

Count unique spacer-PAM-query combinations for each accession

```python
fn
```

```python
accessions = ["SRR1873837", "SRR1873838", "SRR1873839", "SRR1873840", "SRR1873841", "SRR1873842",
             "SRR1873843", "SRR1873844", "SRR1873845", "SRR1873846", "SRR1873847", "SRR1873848", "SRR1873849"]
protospacer_datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232/protospacers_2022_%s" %cr

timepoints = np.arange(0,13)

for accession in accessions:
    unique_protospacers = []
    unique_PAMs = []
    counts = []
    query_ids = []
    
    # iterate through sub-files for each accession
    for fn in tqdm(glob("%s/%s_process/%s*_protospacers_fast.txt" %(protospacer_datapath, accession, accession))):
        timepoint = accessions.index(accession)
        protospacers = pd.read_csv(fn)
        
        protospacers[['spacer_type','sequence_id']] = protospacers['query_id'].str.split('_', expand=True)
        protospacers['sequence_id'] = protospacers['sequence_id'].astype('int')
        spacers = list(protospacers['sequence'])
        
        # group by sequence and PAM region to count
        protospacers_grouped = protospacers.groupby(['sequence', 'PAM_region', 
                                                     'query_id']).count().reset_index().sort_values(by = 'query_id', ascending = False)
        protospacers_grouped = protospacers_grouped.rename(columns = {"subject_id": "count"})

        # grouped by spacer and PAM
        unique_spacers = list(protospacers_grouped['sequence'])
        unique_PAM_seqs = list(protospacers_grouped['PAM_region'])
        
        # count unique spacer and PAM combinations, removing paired-end overlap
        sp_count = count_paired_ends(list(protospacers['sequence']), list(protospacers['subject_id']), unique_spacers, 
                                     list(protospacers_grouped['count']), seq_type = str, 
                                  use_PAMs = True, PAM_seqs = list(protospacers['PAM_region']), unique_sp_PAM = unique_PAM_seqs)
        
        unique_protospacers += unique_spacers
        unique_PAMs += unique_PAM_seqs
        query_ids += list(protospacers_grouped['query_id'])
        counts += list(sp_count)
        
    # combine sub-lists into list for each time point
    accession_df = pd.DataFrame()
    accession_df['sequence'] = unique_protospacers
    accession_df['PAM_region'] = unique_PAMs
    accession_df['query_id'] = query_ids
    accession_df['count'] = counts
    
    # group by sequence, PAM, and query - sum over entire day
    accession_df = accession_df.groupby(['sequence', 'PAM_region', 'query_id']).sum('count').reset_index().sort_values(by = 'count', ascending = False)
    accession_df.to_csv("%s_protospacers.txt" %accession, index = None)
```

```python
# concatenate accession_df files
df_list = []
for accession in accessions:
    accession_df = pd.read_csv("%s_protospacers.txt" %accession)
    accession_df['time_point'] = accessions.index(accession)
    df_list.append(accession_df)
    
all_df = pd.concat(df_list)
all_df.to_csv("protospacer_counts_%s.txt" %cr, index = None)
```

## Clustering all spacers and protospacers with different thresholds

In Paez-Espino 2015, spacers were grouped that shared >85% in length and > 85% sequence identity. Here, I use the Levenshtein ratio as the metric of edit distance between sequences, then cluster using AgglomerativeClustering from scipy with the 'average' linkage criterion. (The 'complete' linkage criterion, which guarantees a minimum similarity between all members of a cluster, is very sensitive to which sequences are present.)

Here, we cluster all detected spacer and protospacer sequences with different thresholds in order to examine the effect of similarity on average immunity.


### Iterate through a few different thresholds

```python
# spacers and protospacers detected in spacer_finder.ipynb
spacer_types_bac_cr1 = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b.csv")
spacer_types_phage_cr1 = pd.read_csv("results/2022-02-08/protospacer_counts_CR1.txt")
spacer_types_bac_cr3 = pd.read_csv("results/2022-02-03/spacer_types_MOI_2b_CR3.csv")
spacer_types_phage_cr3 = pd.read_csv("results/2022-02-08/protospacer_counts_CR3.txt")
```

```python
# load wild-type spacers
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
wild_type_spacers_CR1 = load_wild_type_spacers(datapath, "CR1")
wild_type_spacers_CR3 = load_wild_type_spacers(datapath, "CR3")
```

```python
cr = "CR3"

if cr == "CR1":
    wild_type_spacers = wild_type_spacers_CR1
    spacer_types_phage = spacer_types_phage_cr1
    spacer_types_bac = spacer_types_bac_cr1
else:
    wild_type_spacers = wild_type_spacers_CR3
    spacer_types_phage = spacer_types_phage_cr3
    spacer_types_bac = spacer_types_bac_cr3
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
with open ('unique_spacers_all_%s.txt' %(cr), 'w') as f:
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
distance_matrix = distance_matrix + distance_matrix.T
```

```python
distance_matrix = np.array(distance_matrix, dtype = 'float16') # make array smaller by reducing float size
# float16 has max +- 65500, ~4 decimal places, good enough for this purpose: np.finfo(np.float16)
np.savez_compressed("distance_matrix_all_%s" %(cr), distance_matrix)
```

## Agglomerative clustering with different thresholds

```python
cr = "CR3"
```

```python
with np.load('results/2022-02-09/distance_matrix_all_%s.npz' %cr) as data:
    distance_matrix = data['arr_0']
    # data.files is how I know it's called 'arr_0'
```

```python
# load associated spacer list
with open ('results/2022-02-09/unique_spacers_all_%s.txt' %cr, 'r') as f:
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
    wild_type_spacers = wild_type_spacers_CR3

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
    
    spacer_types.to_csv("spacer_protospacer_types_MOI_2b_%s_similarity_%s.csv" %(cr, 1-threshold), index=None)
    # drop any spacers that are grouped with wild-type spacers
    spacer_types_df_new = spacer_types[~spacer_types['type'].isin(wild_type_labels)]
    spacer_types_df_new.to_csv("spacer_protospacer_types_MOI_2b_%s_wildtype_removed_similarity_%s.csv" %(cr, 1-threshold), index=None)
```

```python

```
