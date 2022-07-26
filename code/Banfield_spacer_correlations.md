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

# Time-shift overlap and spacer correlations from Paez-Espino et al. 2015


## Data overview

Illumina metagenomic sequencing of samples containing a mixture of bacteria and phages. 

## Analysis overview

### Data processing

**Spacers**
1. Search all data for matches to the CRISPR2 repeat (GTTTTTGTACTCTCAAGATTTAAGTAACTGTACAAC) (`spacer_finder.ipynb`)
2. Get a list of wild-type CRISPR spacers by either looking at the S. thermophilus reference genome or by looking for spacers in the control data with no phage. (`spacer_finder.ipynb`)
3. Get spacers from the reads by looking at the repeat hits and taking either 30 nt on either side of the repeat hit or the sequence between two repeat hits. ((`spacer_finder.ipynb` - `extract_spacers`). New in 2021: blast results are filtered to either full 36 nt repeat alignment or partial alignment if the repeat is at the edge of a read. If a partial alignment exists on a read with a full alignment match, it is kept to find the spacer in between. For `SRR*_spacers_exact.txt`, only exact matches to the repeat were kept to compare better with the method of Paez-Espino2015.
4. Remove spacers that are double-counted from being on an overlapping segment of paired-end reads. (`spacer_sorter.ipynb` - `count_paired_ends`)
5. Create list of all unique spacers (`spacer_sorter.ipynb` - saved as `unique_spacers.txt`).
6. Create distance matrix using Levenshtein ratio (simple edit distance) for each pairwise spacer combination (`spacer_sorter.ipynb`).
7. Group spacers using Agglomerative clustering using an 85% sequence identity threshold using the Levenshtein ratio as distance (`spacer_sorter.ipynb`). 
8. Create dataframe where each unique sequence is listed along with its assigned type and count at each time point (`spacer_sorter.ipynb` - saved as "spacer_types_MOI_2b.csv").
9. Also create the same dataframe but remove all spacer sequences that clustered together with the wild-type spacers (`spacer_sorter.ipynb` - saved as "spacer_types_MOI_2b_wildtype_removed.csv").

**Protospacers**
1. Assign reads to either bacteria or phage: blast all reads against S. thermophilus genome and blast all reads against phage genome (`spacer_finder.ipynb`).
2. Make fasta files for all the unique spacer sequences (`spacer_finder.ipynb` - saved as "unique_spacers.fasta" and "unique_spacers_no_wildtype.fasta").
3. Blast all unique spacer sequences against all reads (done on scinet - `blast_setup_niagara.sh` and `blast_submit_script_niagara.sh`).
4. Process protospacer sequences on scinet with `process_protospacers_niagara.py`.

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from Bio.Seq import Seq
from Bio import Entrez
```

```python
from spacer_model_plotting_functions import e_effective_shifted
```

```python
%matplotlib inline
```

```python
def remove_poly_nucleotide_spacers(df, cutoff = 10):
    """
    Takes a dataframe where 'sequence' is one of the columns with spacer sequences, 
    finds sequences that contain long strings of a single letter - these are assumed
    to be sequencing errors. 
    
    Returns the dataframe with those sequences removed.
    """
    
    weird_seqs = []

    for i, row in tqdm(df[['sequence']].drop_duplicates().iterrows()):
        seq = np.array(list(row['sequence']))

        # assign numbers
        seq[seq == 'A'] = 0
        seq[seq == 'C'] = 1
        seq[seq == 'G'] = 2
        seq[seq == 'T'] = 3
        seq[seq == 'N'] = 4

        seq = np.array(seq, dtype = int)

        condition = np.diff(seq) == 0

        runs = np.diff(np.where(np.concatenate(([condition[0]],
                                         condition[:-1] != condition[1:],
                                         [True])))[0])[::2]

        if len(runs) > 0:
            if np.max(runs) > cutoff:
                weird_seqs.append(row['sequence'])

    print("total unique sequences: " + str(len(df[['sequence']].drop_duplicates())))
    print("sequences to remove: " + str(len(weird_seqs)))
    
    inds = df[df['sequence'].isin(weird_seqs)].sort_values(by='count', ascending=False).index
    weird_types = df[df['sequence'].isin(weird_seqs)]['type'].unique()

    # remove types associated with sequences that have long poly runs
    df = df[~df['type'].isin(weird_types)]
    
    return df

```

```python
def compute_overlap(tp1, tp2, df_combined, phage_ahead = True, case = 'presence'):
    """
    Compute the spacer overlap between bacteria and phage at different time points.
    
    If phage_ahead == True, the time points will be ordered so that the phage data frame is from the "future"
    from the perspective of the bacteria (i.e. i2 > i1).
    
    If case = 'abundance', then use the normalized abundance
    
    """
    
    if case == 'presence':
        bac_column = 'bac_presence'
        phage_column = 'phage_presence'
    elif case == 'abundance':
        bac_column = 'count_bac_normalized'
        phage_column = 'count_phage_normalized'
    else:
        print("invalid option for 'case'")
    
    if phage_ahead == True:
        i1 = np.min([tp1, tp2])
        i2 = np.max([tp1, tp2])
    else:
        i1 = np.max([tp1, tp2])
        i2 = np.min([tp1, tp2])
    
    df_bac = df_combined[df_combined['time_point'] == i1]
    df_phage = df_combined[df_combined['time_point'] == i2]
        
    df = df_bac[['type', bac_column]].merge(df_phage[['type', phage_column]], on = 'type', how = 'outer')
        
    overlap = (np.sum(df[bac_column] * df[phage_column]) 
               / (np.sum(df[bac_column]) * np.sum(df[phage_column])))
    
    return overlap
    
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

```python
# spacers and protospacers detected in spacer_finder.ipynb
spacer_types_bac_cr1 = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b.csv")
spacer_types_bac_cr1_no_wildtype = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b_wildtype_removed.csv")
spacer_types_phage_cr1 = pd.read_csv("results/2022-02-08/protospacer_counts_CR1.txt")
spacer_types_bac_cr3 = pd.read_csv("results/2022-02-03/spacer_types_MOI_2b_CR3.csv")
spacer_types_bac_cr3_no_wildtype = pd.read_csv("results/2022-02-03/spacer_types_MOI_2b_wildtype_removed_CR3.csv")
spacer_types_phage_cr3 = pd.read_csv("results/2022-02-08/protospacer_counts_CR3.txt")
```

## Inspecting spacers from CR1 and CR3

```python
# overlap between new spacers and the cr1 spacers: very minimal, only a few types that appear in both from the clustering
spacer_types_bac_cr3[spacer_types_bac_cr3['sequence'].isin(spacer_types_bac_cr1['sequence'])].drop_duplicates('type')
```

Only a few sequences shared between types. What about the reverse complement?

```python
spacer_types_bac_cr3['sequence_revcomp'] = spacer_types_bac_cr3['sequence'].apply(Seq).apply(lambda x: x.reverse_complement()).apply(str)
```

```python
spacer_types_bac_cr3[spacer_types_bac_cr3['sequence_revcomp'].isin(spacer_types_bac_cr1['sequence'])].drop_duplicates('type')
```

Also very few match the reverse complement, so we just need to blast them again anyway. 

```python
spacer_types_bac_cr3 = spacer_types_bac_cr3.drop(columns = 'sequence_revcomp')
```

## Add different grouping thresholds

```python
# load all the thresholds and combine into one dataframe
thresholds = np.arange(0.01, 0.16, 0.01)
thresholds = np.concatenate([[0.005], thresholds])

for threshold in tqdm(thresholds):
    cr = "CR3"
    spacer_types_threshold = pd.read_csv("results/2022-02-09/spacer_protospacer_types_MOI_2b_%s_similarity_%s.csv" %(cr, 1-threshold))
    spacer_types_threshold = spacer_types_threshold.rename(columns = {"type": "type_%s" %(1-threshold)})
    spacer_types_bac_cr3 = spacer_types_bac_cr3.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    spacer_types_phage_cr3 = spacer_types_phage_cr3.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    
    cr = "CR1"
    spacer_types_threshold = pd.read_csv("results/2022-02-09/spacer_protospacer_types_MOI_2b_%s_similarity_%s.csv" %(cr, 1-threshold))
    spacer_types_threshold = spacer_types_threshold.rename(columns = {"type": "type_%s" %(1-threshold)})
    spacer_types_bac_cr1 = spacer_types_bac_cr1.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    spacer_types_phage_cr1 = spacer_types_phage_cr1.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
```

```python
# note that 'type' and 'type_085' are different because in the type_0.85 version, protospacers were also included.
print(spacer_types_bac_cr3['type_0.85'].nunique())
print(spacer_types_bac_cr3['type'].nunique())
```

```python
# not all types match perfectly: 102 type_0.85 types map to multiple 'type' types, and 126 'type' types map to multiple 'type_0.85'
type_comparison = spacer_types_bac_cr3.groupby(['type', 'type_0.85'])['count'].sum().reset_index()

multi_counted_type_085 = type_comparison['type_0.85'].value_counts()[type_comparison['type_0.85'].value_counts() > 1].index
multi_counted_type = type_comparison['type'].value_counts()[type_comparison['type'].value_counts() > 1].index
```

```python
# all the pairs of types that map to multiple other types
len(type_comparison[(type_comparison['type'].isin(multi_counted_type))
               | (type_comparison['type_0.85'].isin(multi_counted_type_085))])
```

## Preprocessing: remove spacers with long poly-nucleotide sequences

```python
spacer_types_phage_cr3[["type","query_seq"]] = spacer_types_phage_cr3['query_id'].str.split('_', expand = True)
spacer_types_phage_cr1[["type","query_seq"]] = spacer_types_phage_cr1['query_id'].str.split('_', expand = True)
```

```python
# preprocessing: remove spacers that have long poly-N sequences
spacer_types_bac_cr3 = remove_poly_nucleotide_spacers(spacer_types_bac_cr3)
spacer_types_bac_cr1 = remove_poly_nucleotide_spacers(spacer_types_bac_cr1)
spacer_types_bac_cr3_no_wildtype = remove_poly_nucleotide_spacers(spacer_types_bac_cr3_no_wildtype)
spacer_types_bac_cr1_no_wildtype = remove_poly_nucleotide_spacers(spacer_types_bac_cr1_no_wildtype)
spacer_types_phage_cr3 = remove_poly_nucleotide_spacers(spacer_types_phage_cr3)
spacer_types_phage_cr1 = remove_poly_nucleotide_spacers(spacer_types_phage_cr1)
```

## PAM processing

Version 1: keeping perfect PAM only


CRISPR3 PAM: GGNG
CRISPR1 PAM: AGAAW (W = A or T)


There's clearly some spacers where the different PAMs associated with the same sequence are actually the same PAM but cutoff. Other times where this isn't the case. How to decide? Maybe only group them if they're shorter, clearly could be the start of a valid PAM, and a longer one in the group does have a valid PAM? 
If there are multiple long ones in the group and not all of them have the PAM, then treat them individually again?

How to count: group by individual sequence and PAM, or just classify PAM as present or absent?


```python
# only sequences containing the perfect PAM
spacer_types_phage_cr3_PAM = spacer_types_phage_cr3[(spacer_types_phage_cr3['PAM_region'].str.contains('GGGG')) 
             | (spacer_types_phage_cr3['PAM_region'].str.contains('GGCG')) 
             | (spacer_types_phage_cr3['PAM_region'].str.contains('GGTG')) 
             | (spacer_types_phage_cr3['PAM_region'].str.contains('GGAG'))]
```

```python tags=[]
spacer_types_phage_cr1_PAM = spacer_types_phage_cr1[(spacer_types_phage_cr1['PAM_region'].str.contains('AGAAT')) 
             | (spacer_types_phage_cr1['PAM_region'].str.contains('AGAAA'))]
```

### PAM processing version 2: keep PAMs that are subsets of a perfect PAM

- Iterate through each unique sequence
- find the subset of PAM regions that contain a perfect PAM
- then find all PAM sequences that are subsets of those perfect PAM sequences
- then keep all spacer-PAM combinations from the perfect + subset list

```python
def keep_partial_PAM(df, cr = "CR1"):
    """
    Keep PAMs that are a subset of a sequence that contains a perfect PAM for the same sequence.
    For instance, if one PAM region is TGGCGTAGGG, and the sequence TGGC is present as a PAM region,
    this is a good candidate for a true PAM that got cut off at the end of a read. 
    Because this is meant to capture read end effects, only check if the sequences are present
    at the start of a PAM region, not just anywhere in the region.
    
    Steps:
    - Iterate through each unique sequence
    - find the subset of PAM regions that contain a perfect PAM
    - then find all PAM sequences that are subsets of those perfect PAM sequences
    - then keep all spacer-PAM combinations from the perfect + subset list
    
    This is unfortunately extremely slow
    
    Inputs:
    ------
    df : a dataframe of protospacer sequences with at least the columns 'sequence' and 'PAM_region'
    cr : CRISPR locus, either CR1 or CR3
    
    Returns :
    spacer_types_PAM_partial : df, filtered to include only perfect PAMs and subsets of perfect PAMs
    """
    spacer_types_PAM_partial = pd.DataFrame()

    spacer_types_phage_unique = df.drop_duplicates(['sequence', 'PAM_region'])

    for sequence, row in tqdm(spacer_types_phage_unique.groupby('sequence')):

        # get subset of perfect pams
        if cr == "CR3":
            perfect_pams = row[(row['PAM_region'].str.contains('GGGG')) 
                     | (row['PAM_region'].str.contains('GGCG')) 
                     | (row['PAM_region'].str.contains('GGTG')) 
                     | (row['PAM_region'].str.contains('GGAG'))]
        elif cr == "CR1":
            perfect_pams = row[(row['PAM_region'].str.contains('AGAAT')) 
                        | (row['PAM_region'].str.contains('AGAAA'))]
        else:
            break

        # get a list of all pams that are a substring of a perfect pam
        # accepted_pams will also include all the perfect pams
        # accepted_pams is particular to a particular sequence
        accepted_pams = []
        for pam in row['PAM_region']:
            if pam == 'N' or pam == 'NN' or pam == 'NNN':
                continue
            if np.any(perfect_pams['PAM_region'].str.startswith(pam)): #use startswith instead of contains
                accepted_pams.append(pam)

        df2 = df[(df['sequence'] == sequence)
                                    & (df['PAM_region'].isin(accepted_pams))]
        spacer_types_PAM_partial = pd.concat([spacer_types_PAM_partial, df2])
    
    return spacer_types_PAM_partial
```

```python tags=[]
# this is super slow: only run once

#spacer_types_phage_cr3_PAM_partial = keep_partial_PAM(spacer_types_phage_cr3, cr = "CR3")
#spacer_types_phage_cr1_PAM_partial = keep_partial_PAM(spacer_types_phage_cr1, cr = "CR1")
#spacer_types_phage_cr3_PAM_partial.to_csv("spacer_types_phage_cr3_PAM_partial.csv", index = None)
#spacer_types_phage_cr1_PAM_partial.to_csv("spacer_types_phage_cr1_PAM_partial.csv", index = None)
```

<!-- #region tags=[] -->
## Count total number of reads matching either phage or bacteria genome per sample

There are 1000000 reads per split genome file; this counts the number that matched to the bacteria genome or the CRISPR repeat and the number that matched to the phage genome. 

Count for each time point:
- [x] Total that match the bacteria genome or the CRISPR repeat
- [x] Total that match the phage genome
- [x] Total that match both
- [x] Total that match neither
<!-- #endregion -->

```python
#datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
#!python get_read_counts.py "datapath"
```

```python
total_reads_df = pd.read_csv("total_reads_normalization.csv", index_col = 0)

total_reads_df['total_reads'] = total_reads_df['total_reads'].astype('int')
total_reads_df['total_matching_neither'] = total_reads_df['total_matching_neither'].astype('int')
```

```python
# the total of phage + bac + neither - both should be 1
(total_reads_df['num_bac_reads'] + total_reads_df['num_phage_reads'] + total_reads_df['total_matching_neither']
 - total_reads_df['total_matching_both']) / total_reads_df['total_reads']
```

### Choose whether to include partial PAMs or perfect PAMs only

```python
pam = "perfect"
#pam = "partial"
#pam = "no" # use all protospacer matches regardless of PAM

folder = "results/%s_PAM" %pam
```

## Overlap: both CR1 and CR3

```python
spacer_types_phage_cr3_PAM_partial = pd.read_csv("results/2022-02-10/spacer_types_phage_cr3_PAM_partial.csv")
spacer_types_phage_cr1_PAM_partial = pd.read_csv("results/2022-02-10/spacer_types_phage_cr1_PAM_partial.csv")
```

```python
spacer_types_bac_cr1['CRISPR'] = 1
spacer_types_bac_cr3['CRISPR'] = 3

if pam == "perfect":
    spacer_types_phage_cr1_PAM['CRISPR'] = 1
    spacer_types_phage_cr3_PAM['CRISPR'] = 3
elif pam == "partial":
    spacer_types_phage_cr1_PAM_partial['CRISPR'] = 1
    spacer_types_phage_cr3_PAM_partial['CRISPR'] = 3
elif pam == "no":
    spacer_types_phage_cr1['CRISPR'] = 1
    spacer_types_phage_cr3['CRISPR'] = 3

#spacer_types_bac_cr3 = spacer_types_bac_cr3.drop(columns='sequence_revcomp')
```

```python
spacer_types_bac = pd.concat([spacer_types_bac_cr1, spacer_types_bac_cr3])
if pam == "perfect":
    spacer_types_phage = pd.concat([spacer_types_phage_cr1_PAM, spacer_types_phage_cr3_PAM])
elif pam == "partial":
    spacer_types_phage = pd.concat([spacer_types_phage_cr1_PAM_partial, spacer_types_phage_cr3_PAM_partial])
elif pam == "no":
    spacer_types_phage = pd.concat([spacer_types_phage_cr1, spacer_types_phage_cr3])
```

Types that are present in both CR1 and CR3 should be fine - differentiated in the phage data based on PAM


### Choose whether to include wildtype

```python
# load wild-type spacers
datapath = "/media/madeleine/My Passport1/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
wild_type_spacers_CR1 = load_wild_type_spacers(datapath, "CR1")
wild_type_spacers_CR3 = load_wild_type_spacers(datapath, "CR3")

# these are all the sequences that are in the wild-type list and their type identifiers
spacer_types_bac_wt = spacer_types_bac[(((spacer_types_bac['CRISPR'] == 1) 
                & (spacer_types_bac['sequence'].isin(wild_type_spacers_CR1)))
                 | ((spacer_types_bac['CRISPR'] == 3) 
                & (spacer_types_bac['sequence'].isin(wild_type_spacers_CR3))))]

spacer_types_phage_wt = spacer_types_phage[(((spacer_types_phage['CRISPR'] == 1) 
                & (spacer_types_phage['sequence'].isin(wild_type_spacers_CR1)))
                 | ((spacer_types_phage['CRISPR'] == 3) 
                & (spacer_types_phage['sequence'].isin(wild_type_spacers_CR3))))]
```

```python
wild_type = True # False: don't include wild-type
```

### plot a few of the largest types over time

```python
def remove_wild_type(spacer_types, spacer_types_wt, grouping):
    """
    Remove sequences that cluster with wild-type spacers for a particular grouping threshold
    """
    
    # these are the sequences that cluster with the wild-type for this particular grouping
    wt_spacers = spacer_types_wt[[grouping, 'CRISPR']].drop_duplicates().merge(spacer_types, on = [grouping, 'CRISPR'], how = 'left')

    wt_spacers_cr3 = wt_spacers[wt_spacers['CRISPR'] == 3]['sequence'].drop_duplicates()
    wt_spacers_cr1 = wt_spacers[wt_spacers['CRISPR'] == 1]['sequence'].drop_duplicates()

    # keep only sequences that didn't cluster with wild-type
    df = spacer_types[~(((spacer_types['sequence'].isin(wt_spacers_cr1))
                     & (spacer_types['CRISPR'] == 1))
                     | ((spacer_types['sequence'].isin(wt_spacers_cr3)
                     & (spacer_types['CRISPR'] == 3))))]
    
    return df
```

```python
grouping = 'type_0.95'

if wild_type == False:
    df = remove_wild_type(spacer_types_bac, spacer_types_bac_wt, grouping)
    df_phage = remove_wild_type(spacer_types_phage, spacer_types_phage_wt, grouping)
else:
    df = spacer_types_bac
    df_phage = spacer_types_phage
    
top_bac_types = df.groupby([grouping, 'CRISPR'])['count'].sum().reset_index().sort_values(by = 'count', ascending = False)[:10]
top_phage_types = df_phage.groupby([grouping, 'CRISPR'])['count'].sum().reset_index().sort_values(by = 'count', ascending = False)[:10]
```

```python
fig, axs = plt.subplots(2,1, figsize = (5,7))

ax = axs[0]
ax1 = axs[1]

for ind in top_bac_types.index:
    
    bac_subset = spacer_types_bac[(spacer_types_bac[grouping] == top_bac_types.loc[ind][grouping])
                             & (spacer_types_bac['CRISPR'] == top_bac_types.loc[ind]['CRISPR'])].groupby('time_point')['count'].sum()
    ax.plot(bac_subset.index, list(bac_subset))
    
    phage_subset = spacer_types_phage[(spacer_types_phage[grouping] == top_bac_types.loc[ind][grouping])
                             & (spacer_types_phage['CRISPR'] == top_bac_types.loc[ind]['CRISPR'])].groupby('time_point')['count'].sum()
    ax1.plot(phage_subset.index, list(phage_subset))
    
for ind in top_phage_types.index:
    
    bac_subset = spacer_types_bac[(spacer_types_bac[grouping] == top_phage_types.loc[ind][grouping])
                             & (spacer_types_bac['CRISPR'] == top_phage_types.loc[ind]['CRISPR'])].groupby('time_point')['count'].sum()
    ax.plot(bac_subset.index, list(bac_subset))
    
    
    phage_subset = spacer_types_phage[(spacer_types_phage[grouping] == top_phage_types.loc[ind][grouping])
                             & (spacer_types_phage['CRISPR'] == top_phage_types.loc[ind]['CRISPR'])].groupby('time_point')['count'].sum()
    ax1.plot(phage_subset.index, list(phage_subset))
    
ax.set_yscale('log')
ax1.set_yscale('log')

ax.set_ylabel("Bacteria clone size")
ax1.set_ylabel("Phage clone size")

plt.tight_layout()

plt.savefig("top_10_clones_%s_PAM_wt_%s.pdf" %(pam, wild_type))
```

### Are clone sizes correlated over the course of the experiment?

```python
grouping = 'type_0.95'

# group by time point, spacer type, and CRISPR locus - this groups different sequences that have been assigned the same type
bac_types_grouped = spacer_types_bac.groupby(['time_point', grouping, 'CRISPR'])[['count']].sum().reset_index()
phage_types_grouped = spacer_types_phage.groupby(['time_point', grouping, 'CRISPR'])[['count']].sum().reset_index()

df_combined = bac_types_grouped.merge(phage_types_grouped, on = (grouping, 'time_point', 'CRISPR'), suffixes = ('_bac', '_phage'),
                                        how = 'outer')

df_combined = df_combined.fillna(0)

df_combined_all_time = df_combined.groupby([grouping, 'CRISPR'])[['count_bac', 'count_phage']].sum().reset_index()
```

```python
fig, ax = plt.subplots()
ax.scatter(df_combined_all_time['count_bac'], df_combined_all_time['count_phage'], alpha = 0.1)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Bacteria clone size")
ax.set_ylabel("Phage clone size")

plt.tight_layout()
```

```python
from scipy.stats import pearsonr
```

```python
# according to pearsonr, not correlated. 
# basically there are a bunch of large protospacers that are not targeted by bacteria
pearsonr(df_combined_all_time['count_bac'], df_combined_all_time['count_phage'])
```

### Make combined data frame

```python
#grouping = 'type'
#grouping = 'sequence'
#grouping = 'type_0.99'
presence_cutoff = 2 # how many detections to count spacer type as "present"
```

```python
for threshold in tqdm(thresholds):
    grouping = 'type_%s' %(1-threshold)
    
    if wild_type == False:
        df_bac = remove_wild_type(spacer_types_bac, spacer_types_bac_wt, grouping)
        df_phage = remove_wild_type(spacer_types_phage, spacer_types_phage_wt, grouping)
    else:
        df_bac = spacer_types_bac
        df_phage = spacer_types_phage

    # group by time point, spacer type, and CRISPR locus - this groups different sequences that have been assigned the same type
    bac_types_grouped = df_bac.groupby(['time_point', grouping, 'CRISPR'])[['count']].sum().reset_index()
    phage_types_grouped = df_phage.groupby(['time_point', grouping, 'CRISPR'])[['count']].sum().reset_index()

    df_combined = bac_types_grouped.merge(phage_types_grouped, on = (grouping, 'time_point', 'CRISPR'), suffixes = ('_bac', '_phage'),
                                            how = 'outer')

    df_combined = df_combined.fillna(0)

    # convert spacer counts to presence-absence
    df_combined['bac_presence'] = np.where(df_combined['count_bac'] >= presence_cutoff, 1, 0)
    df_combined['phage_presence'] = np.where(df_combined['count_phage'] >= presence_cutoff, 1, 0)

    # normalize count data
    df_combined['count_bac_normalized'] = 0
    df_combined['count_phage_normalized'] = 0
    for tp in np.arange(0,13):
        df_tp = df_combined[df_combined['time_point'] == tp]
        bac_norm = total_reads_df[total_reads_df['time_point'] == tp]['num_bac_reads']
        phage_norm = total_reads_df[total_reads_df['time_point'] == tp]['num_phage_reads']

        df_combined.loc[df_combined['time_point'] == tp, 'count_bac_normalized'] = df_tp['count_bac'] / float(bac_norm)
        df_combined.loc[df_combined['time_point'] == tp, 'count_phage_normalized'] = df_tp['count_phage'] / float(phage_norm)

    df_combined.to_csv("results/%s_PAM/banfield_data_combined_%s_wt_%s.csv" %(pam, grouping, wild_type))
```

Might want to think more carefully about normalization: normalizing to the total number of reads is one way to compare between time points, but it would do different things to bacteria and phage based on the size of their genomes... maybe there's a better way?

If the reads are evenly spread across the genome, then we can weight by the genome size as well to compare bacteria and phage on equal footing. 

Should I compare total reads to total population size? If they track nicely, then maybe we don't need to normalize at all?

```python
# size of genomes in base pairs
s_therm_genome = 1877654
phage_genome = 35201
```

```python
s_therm_genome / phage_genome
```

## Quality checks

Figure 1B from Paez-Espino et al. 2015, showing total number of unique spacers in each CRISPR type over time in the MOI2B experiment.

![](results/2021-10-13/Paez-Espino2015_1B.jpg)

![](results/2021-10-13/Paez-Espino2015_S3.jpg)


According to [CRISPRdb](https://crisprcas.i2bc.paris-saclay.fr/MainDb/StrainList) for Streptococcus thermophilus DGCC 7710:

- CRISPR1: GTTTTTGTACTCTCAAGATTTAAGTAACTGTACAAC
- CRISPR2: GATATAAACCTAATTACCTCGAGAGGGGACGGAAAC
- CRISPR3: GGATCACCCCCGCGTGTGCGGGAAAAAC
- CRISPR4: GTTTTGGAACCATTCGAAACAACACAGCTCTAAAAC

```python
# brown CRISPR1 columns from Figure 1
banfield_counts_cr1 = [86.12903225806454
, 71.61290322580648
, 153.87096774193552
, 169.35483870967744
, 120
, 163.54838709677415
, 99.6774193548387
, 84.19354838709677
, 106.45161290322581
, 33.87096774193551
, 30.967741935483872
, 15.483870967741936]

banfield_counts_cr3 = [29.032258064516157
, 31.935483870967758
, 79.35483870967741
, 91.93548387096774
, 91.93548387096774
, 120.96774193548387
, 63.8709677419355
, 57.096774193548384
, 120.96774193548387
, 71.61290322580649
, 80.3225806451613
, 39.67741935483873]
```

```python
spacer_types_bac_exact = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b_wildtype_removed_exact.csv")
```

```python
grouping = "type"
df_bac_exact = spacer_types_bac_exact.groupby(['time_point', grouping])['count'].sum().reset_index()
```

```python
time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]

threshold = 1

fig, ax = plt.subplots()
ax.bar(np.unique(df_bac_exact['time_point']), 
       df_bac_exact[df_bac_exact['count'] > threshold].groupby('time_point')[grouping].count(), label = 'My counts')
ax.plot(np.unique(df_bac_exact['time_point'])[1:], banfield_counts_cr1, marker = 'o', color = 'r', label = 'Banfield counts')

ax.set_xticks(np.unique(df_bac_exact['time_point']))
ax.set_xticklabels(time_points_in_days);

ax.legend()
plt.tight_layout()
plt.savefig("Banfield_counts_comparison_exact_threshold_%s_grouping_%s.pdf" %(threshold, grouping))
```

This is good, the exact match version with removing single counts matches their results quite closely, close enough that I feel confident proceeding with analysis based on my version.

My version also looks pretty good with a threshold of 1. 

```python
threshold = 0.15
grouping = 'type_%s' %(1-threshold)
cutoff = 1

df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)

crisprs = [1,3]

time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
fig, axs = plt.subplots(2,1, figsize = (5,7))

for i, crispr in enumerate(crisprs):
    df_bac = df_combined[(df_combined['count_bac'] > cutoff)
                        & (df_combined['CRISPR'] == crispr)]
    if crispr == 1:
        banfield_counts = banfield_counts_cr1
    elif crispr == 3:
        banfield_counts = banfield_counts_cr3
    
    axs[i].bar(np.unique(df_bac['time_point']), df_bac.groupby('time_point')[grouping].count(), label = 'Re-processed counts')
    axs[i].plot(np.unique(df_bac['time_point'])[1:], banfield_counts, marker = 'o', color = 'r', label = 'Experiment reported counts')

    axs[i].set_xticks(np.unique(df_bac['time_point']))
    axs[i].set_xticklabels(time_points_in_days)
    
    axs[i].set_ylabel("Unique spacer types")

axs[0].legend()
axs[0].set_title("CRISPR1 locus")
axs[1].set_title("CRISPR3 locus")
plt.tight_layout()
plt.savefig("Banfield_counts_comparison_cutoff_%s_grouping_%s_%s_PAM_wt_%s.pdf" %(cutoff, grouping, pam, wild_type))
```

Note that they only searched for exact matches to the repeat, whereas I kept some partial matches where the were at the edge of a read and allowed for some mismatches. I'm not sure if their clustering was done on the entire time series or just the partial time series, and how exactly they did the clustering.


## Absolute overlap

```python
def type_overlap(df_combined, cutoff = 1):
    """
    Count the number of types in bacteria and phage and the number of shared types
    """
    
    overlap_list = []
    bac_types = []
    phage_types = []
    for group in df_combined.groupby(['time_point']):
        tp = group[0]
        data = group[1]

        num_shared = len(data[(data['count_bac'] > cutoff) 
                    & (data['count_phage'] > cutoff)])

        num_bac = len(data[(data['count_bac'] > cutoff)])
        num_phage = len(data[(data['count_phage'] > cutoff)])


        overlap_list.append(num_shared)
        bac_types.append(num_bac)
        phage_types.append(num_phage)
        
    return overlap_list, bac_types, phage_types
```

```python
time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
colours = cm.viridis(np.linspace(0,1, len(thresholds)))[::-1]
fig, axs = plt.subplots(2,2, figsize = (10,8))
cutoff = 0

for i, threshold in enumerate(thresholds):
    grouping = 'type_%s' %(1-threshold)
    df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
    
    shared_types, bac_types, phage_types = type_overlap(df_combined, cutoff = cutoff)
    
    axs[0,0].plot(np.unique(df_combined['time_point']), shared_types, 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[0,1].plot(np.unique(df_combined['time_point']), np.array(phage_types) / np.array(bac_types), 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[1,0].plot(np.unique(df_combined['time_point']), bac_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
    
    axs[1,1].plot(np.unique(df_combined['time_point']), phage_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
for ax in axs.flatten():
    ax.set_xticklabels(time_points_in_days)
    
axs[0,0].set_ylabel("Number of shared spacer types")
axs[0,1].set_ylabel("Ratio of phage types to bacteria types")
axs[1,0].set_ylabel("Number of bacteria types")
axs[1,1].set_ylabel("Number of phage types")
axs[1,0].set_xlabel("Time (days)")
axs[1,1].set_xlabel("Time (days)")
axs[0,0].legend(ncol = 2)
axs[0,0].set_yscale('log')

plt.tight_layout()
plt.savefig("unique_types_and_overlap_cr1_cr3_all_cutoff_%s_%s_PAM_wt_%s.pdf" %(cutoff, pam, wild_type))
```

```python
fig, axs = plt.subplots(3,1, figsize = (6,8))

ax = axs[0]
ax1 = axs[1]
ax2 = axs[2]

ax.plot(total_reads_df['time_point'], total_reads_df['num_phage_reads'], label = "Phage")
ax.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads'], label = "Bacteria")

ax1.plot(total_reads_df['time_point'], total_reads_df['num_phage_reads'] / phage_genome, label = "Phage")
ax1.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads']/ s_therm_genome, label = "Bacteria")


ax2.plot(total_reads_df['time_point'], total_reads_df['num_phage_reads'] / total_reads_df['total_reads'], label = "Phage")
ax2.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads']/ total_reads_df['total_reads'], label = "Bacteria")


#axb.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads'] )
#ax.set_yscale('log')
ax.legend()

ax2.set_xlabel("Time (days)")
ax.set_ylabel("Total reads")
ax1.set_ylabel("Total reads / genome size")
ax1.set_yscale('log')
ax2.set_ylabel("Fraction of total reads")

for ax in axs:
    ax.set_xticklabels(time_points_in_days)

plt.tight_layout()
plt.savefig("Banfield_read_totals.pdf")
```

Interestingly the number of shared types has a peak at an intermediate threshold: about 94% similarity. Tradeoff between total number of types (increases as similarity cutoff increases) and likelihood that types are shared (increases as similarity cutoff decreases).

```python
time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
colours = cm.viridis(np.linspace(0,1, len(thresholds)))[::-1]
fig, ax = plt.subplots(1,1, figsize = (6,6))

for i, threshold in enumerate(thresholds):
    grouping = 'type_%s' %(1-threshold)
    df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
    
    shared_types, bac_types, phage_types = type_overlap(df_combined, cutoff = 1)
    
    ax.plot(np.unique(df_combined['time_point']), shared_types, 
            label = "%s" %int((1-threshold)*100) + r"%: " + "%s shared" %np.sum(shared_types), 
            color = colours[i], linewidth = 3, marker = 'o')
    
    print("%s" %int((1-threshold)*100) + r"%:" + "%s" %np.sum(shared_types[1:-3]) + " shared types")
ax.set_xticklabels(time_points_in_days);

ax.set_ylabel("Number of shared spacer types")

ax.set_xlabel("Time (days)")
ax.set_ylim(0,520)
ax.legend(ncol = 2)
#ax.set_yscale('log')

plt.tight_layout()
plt.savefig("Banfield_shared_types_%s_PAM_wt_%s.pdf" %(pam, wild_type))
```

## Time shift

Caution: the time intervals are all weird, so averaging over the same delay will be tricky.


Do I need to re-think my average immunity calculation to account for multiple spacers? Basically right now I'm treating a single spacer as a single organism, but in reality multiple spacers and protospacers would hitchhike along. Average immunity should really be on the organism level. Think about this. It's probably ok from the bacteria side of things since most bacteria get one new spacer over the course of the experiment. 

```python
time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
time_points = np.arange(0,len(time_points_in_days))
```

```python
def banfield_to_array(df_combined, grouping):
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

    # this way the bacteria and phage will have the same type labels and number of types
    bac_wide = all_wide_normalized['count_bac_normalized']
    phage_wide = all_wide_normalized['count_phage_normalized']

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
```

```python
accession = "NC_007019" # Streptococcus phage 2972

f = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome[1:]:
    phage_genome_seq += row.strip()
```

```python
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
```

```python
# is there a pam that overlaps the start?
# no...
print(phage_genome_seq[-5:] + phage_genome_seq[:5])

```

```python tags=[]
num_protospacers = cr3_pams + cr1_pams # Paez-Espino2013, 233 CRISPR1 protospacers
```

```python
start_ind = 1
stop_ind = -3 # cut off the last np.abs(stop_ind) -1 points
if stop_ind == -1:
    step = np.min(np.diff(time_points_in_days[start_ind: stop_ind]))
else:
    step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)
#time_min_spacing = np.arange(time_points_in_days[0], time_points_in_days[-1] + step, step)
```

```python
thresholds_subset = np.arange(0.01, 0.16, 0.01)
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]
```

```python tags=[]
fig, ax = plt.subplots(figsize = (10,8))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'CRISPR']
    df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
    bac_wide_filtered, phage_wide_filtered = banfield_to_array(df_combined, grouping)
    
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

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.1)

#ax.set_xlim(-1550, 1550)
#ax.set_ylim(0.075, 0.175)
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (bacterial generations)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Banfield_both_cr_start_trim_%s_end_trim_%s_%s_PAM_wt_%s.pdf" %(start_ind, np.abs(stop_ind) - 1, pam, wild_type))
```

Make a grid of the start and end trim amounts

```python
thresholds_subset = np.arange(0.01, 0.16, 0.02)
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

max_start_trim = 4
max_end_trim = 4

start_trims = np.arange(0,max_start_trim)
end_trims = np.arange(0,max_end_trim)

fig, axs = plt.subplots(max_start_trim, max_end_trim, figsize = (3*max_start_trim, 2.5*max_end_trim))

for i, start_ind in enumerate(start_trims):
    for j, end in tqdm(enumerate(end_trims)):
        stop_ind = -1*end - 1 # cut off the last np.abs(stop_ind) -1 points
        if stop_ind == -1:
            step = np.min(np.diff(time_points_in_days[start_ind: stop_ind]))
        else:
            step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
        # remove the first and last time point for interpolation - no shared types on the first day
        time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

        for k, threshold in enumerate(thresholds_subset):
            grouping = ['type_%s' %(1-threshold), 'CRISPR']
            df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
            bac_wide_filtered, phage_wide_filtered = banfield_to_array(df_combined, grouping)

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

            axs[i,j].scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[k], 
                       label = "%s" %int((1-threshold)*100) + r"%")
            axs[i,j].fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                            y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                            color = colours[k], alpha = 0.1)

        axs[i,j].set_xlim(-1550, 1550)
        
        if pam == "perfect":
            if wild_type == True:
                axs[i,j].set_ylim(0.0, 0.18)
                ymax = 0.155
                x = 100
            elif wild_type == False:
                axs[i,j].set_ylim(0.0, 1.3)
                ymax = 1.1
                x = 100
        elif pam == "partial":
            if wild_type == True:
                axs[i,j].set_ylim(0.0, 0.26)
                ymax = 0.21
                x = -1300
            elif wild_type == False:
                axs[i,j].set_ylim(0.0, 1.3)
                ymax = 1.1
                x = -1300
                
        axs[i,j].axvline(0, linestyle = ':', color = 'k')
        
        axs[i,j].annotate("Trim %s from start,\n%s from end" %(start_ind, end),
               xy = (x, ymax), xycoords = 'data', fontsize = 8)

for i in range(max_start_trim):
    axs[i,0].set_ylabel("Average overlap between\nbacteria and phage")
for j in range(max_end_trim):    
    axs[-1,j].set_xlabel("Time shift (bacterial generations)")
    
axs[-1,-1].legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,0.85), fontsize = 8)
plt.tight_layout()
plt.savefig("Time_shift_Banfield_both_cr_%s_PAM_wt_%s.pdf" %(pam, wild_type))
```

```python
## plot for paper
#thresholds_subset = np.arange(0.01, 0.17, 0.02)
thresholds_subset = np.concatenate([[0.01], np.arange(0.03, 0.16, 0.03)])
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

start_ind = 1
stop_ind = -3 # cut off the last np.abs(stop_ind) -1 points
step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)


fig, ax = plt.subplots(figsize = (4.7,3.5))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'CRISPR']
    df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
    bac_wide_filtered, phage_wide_filtered = banfield_to_array(df_combined, grouping)
    
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

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"%")
    
    #print((avg_immunity_mean*num_protospacers)[-1])
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.1)

    #ax.annotate("%s" %int((1-threshold)*100) + r"%",
    #           xy = (1200, annotation_vals[i]), xycoords = 'data', fontsize = 8)
    
#ax.annotate('Decreasing similarity\nthreshold',
#            xy=(1000, 0.58), xycoords='data',
#            xytext=(20, 0.07), textcoords='data',
#            arrowprops=dict(facecolor='black', arrowstyle="->"))

#ax.set_xlim(-1200, 1400)
#ax.set_ylim(0, 0.08)
#ax.legend(ncol = 2, loc = 'lower left', bbox_to_anchor = (0, 0.5) )
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (bacterial generations)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Banfield_all_groups_small_start_trim_%s_end_trim_%s_%s_PAM_wt_%s.pdf" %(start_ind, 
                                                                                   np.abs(stop_ind) - 1, pam, wild_type))
```

### Presentation version

```python
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

start_ind = 1
stop_ind = -3 # cut off the last np.abs(stop_ind) -1 points
step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)


fig, ax = plt.subplots(figsize = (4.7,3.5))

threshold = 0.15

grouping = ['type_%s' %(1-threshold), 'CRISPR']
df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder,1-threshold, wild_type), index_col = 0)
bac_wide_filtered, phage_wide_filtered = banfield_to_array(df_combined, grouping)

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


ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
           label = "%s" %int((1-threshold)*100) + r"%")

#print((avg_immunity_mean*num_protospacers)[-1])
ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                color = colours[0], alpha = 0.1)

ax.annotate(s = "Past phages", xy = (-1000, 0.13), fontsize = 12)
ax.annotate(s = "Future phages", xy = (200, 0.13), fontsize = 12)


ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (bacterial generations)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Banfield_all_groups_small_start_trim_%s_end_trim_%s_%s_PAM_wt_%s_threshold_%s_presentation.png" %(start_ind, 
                                                                                   np.abs(stop_ind) - 1, pam, wild_type, threshold), dpi = 300)
```

In my model for accounting for bacterial array length, the number of spacers in total (44) means that average immunity works out to 1 regardless of the starting value using my toy model theory.

It's possible that the effective array length is much shorter in that there aren't protospacers for a bunch of older spacers. Can I see this in the data?

```python
cr = "CR3"
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
# load spacers from reference genome
wild_type_spacers_ref = []
with open ("%s/NZ_CP025216_%s_spacers.txt" %(datapath,cr), "r") as f:
    wild_type_spacers_ref_unstripped = f.readlines()
    for sp in wild_type_spacers_ref_unstripped:
        wild_type_spacers_ref.append(sp.rstrip())
```

```python
# from control data

wild_type_spacers = []
with open ('%s/SRR1873863_%s_spacers_unique.txt' %(datapath, cr), 'r') as f:
    wild_type_spacers_unstripped = f.readlines()
    for sp in wild_type_spacers_unstripped:
        wild_type_spacers.append(sp.rstrip())
```

```python
# all the reference spacers are in the full reference set, so don't need to worry about the reverse complement thing
for sp in wild_type_spacers_ref:
    print(sp in wild_type_spacers)
```

```python
# iterate through spacers, get phage matches for each one
#grouping = "type_0.97" # this allows for basically one letter difference
#grouping = "type_0.94" # this allows for two letters difference
#grouping = "type_0.9"
grouping = "type_0.85"

if cr == "CR1":
    crispr = 1
elif cr == "CR3":
    crispr = 3
    
total_spacer_counts = []
total_protospacer_counts = []
for sp in wild_type_spacers_ref:
    spacer_type = float(spacer_types_bac[(spacer_types_bac['sequence'] == sp)
                                   & (spacer_types_bac['CRISPR'] == crispr)][grouping].unique())
    
    spacer_count = spacer_types_bac[(spacer_types_bac[grouping] == spacer_type) 
                 & (spacer_types_bac['CRISPR'] == crispr)]['count'].sum()
    
    protospacer_count = spacer_types_phage[(spacer_types_phage[grouping] == spacer_type) 
                 & (spacer_types_phage['CRISPR'] == crispr)]['count'].sum()
    
    total_spacer_counts.append(spacer_count)
    total_protospacer_counts.append(protospacer_count)
```

Even at 85% similarity, there are NO protospacer matches to the wild-type spacers for CR1 and only a single protospacer match to CR3. This means the phage has escaped all the wild-type spacers and only the new spacers matter.  

```python
unique_protospacers = []
thresholds = np.arange(0.01, 0.16, 0.01)
thresholds = np.concatenate([[0.005], thresholds])

for threshold in tqdm(thresholds):
    grouping = 'type_%s' %(1-threshold)
    
    spacer_types_phage_no_wt = remove_wild_type(spacer_types_phage, spacer_types_phage_wt, grouping)
    
    unique_protospacers.append(spacer_types_phage_no_wt['sequence'].nunique())
```

```python
fig, ax = plt.subplots(figsize = (5,3))

ax.plot(1-thresholds, unique_protospacers, marker = 'o', 
       label = "Removing sequences\ngrouped with wild-type")
ax.axhline(spacer_types_phage['sequence'].nunique(), linestyle = '--', color = 'k', 
          label = "Total number of sequences")

ax.legend()
ax.set_ylim(15300, 16850)
ax.set_xlim(0.85, 1)
ax.set_ylabel("Number of unique\nprotospacer sequences")
ax.set_xlabel("Similarity grouping threshold")
plt.tight_layout()
plt.savefig("unique_protospacers_vs_grouping_%s_PAM.pdf" %pam)
```

```python

```

## Population size

```python
phage_pop_size_MOI2 = pd.read_csv("results/2021-05-10/MOI2b_phage_pop_size.csv", sep = ",")
bac_pop_size_MOI2 = pd.read_csv("results/2021-05-10/MOI2b_bac_pop_size.csv", sep = ",")
```

```python
phage_pop_size_MOI2 = phage_pop_size_MOI2.sort_values(by = 'Date')
bac_pop_size_MOI2 = bac_pop_size_MOI2.sort_values(by = 'Date')
```

```python
phage_pop_interp = interp1d(phage_pop_size_MOI2['Date'], phage_pop_size_MOI2['Phage population size'])
phage_pop_size_interp = phage_pop_interp(time_points_in_days)

bac_pop_interp = interp1d(bac_pop_size_MOI2['Date'], bac_pop_size_MOI2['Bacteria population size'])
bac_pop_size_interp = bac_pop_interp(time_points_in_days)
```

```python
fig, ax = plt.subplots()

ax.plot(phage_pop_size_MOI2['Date'], phage_pop_size_MOI2['Phage population size'], marker = 'o', 
       color = 'rebeccapurple', label = "Phage", alpha = 0.6)
ax.plot(time_points_in_days, phage_pop_size_interp, marker = 's', linestyle = 'None', 
       color = 'indigo', mec = 'k')

ax.plot(bac_pop_size_MOI2['Date'], bac_pop_size_MOI2['Bacteria population size'], marker = 'o',
       color = 'lightseagreen', label = "Bacteria", alpha = 0.6)
ax.plot(time_points_in_days, bac_pop_size_interp, marker = 's', linestyle = 'None', color = 'darkcyan',
       mec = 'k', label = "Interpolated number at sequencing date")


ax.legend()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Plaque or colony forming units")
ax.set_yscale('log')
ax.set_xlim(-10, 250)
plt.savefig("Banfield_population_size.pdf")
```

### Does average immunity correlate with population size?

Average immunity correlates strongly with (log) phage population size, and you can see average immunity trending higher over time. 

```python
from scipy.stats import linregress
```

```python
from spacer_model_plotting_functions import effective_e
```

```python
thresholds = np.arange(0.01, 0.16, 0.01)
colours = cm.viridis(np.linspace(0,0.9, len(effective_e_list[1:])))
```

```python
for threshold in thresholds:
    fig, ax = plt.subplots(1,1, figsize = (4,3))
    ax.plot(time_points_in_days, effective_e_list*num_protospacers, marker = 'o', color = 'k')
    
    ax.set_xlabel("Time point (days)")
    ax.set_ylabel("Average immunity")
    
    plt.tight_layout()
    
    plt.savefig("average_immunity_vs_time_grouping_%s_%s_PAM_wt_%s.pdf" %(round(1-threshold,3), pam, wild_type))
    plt.close()
```

```python
for threshold in thresholds:

    grouping = ['type_%s' %(round(1-threshold,3)), 'CRISPR']
    df_combined = pd.read_csv("%s/banfield_data_combined_type_%s_wt_%s.csv" %(folder, round(1-threshold,3), wild_type), index_col = 0)
    bac_wide_filtered, phage_wide_filtered = banfield_to_array(df_combined, grouping)

    effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)

    fig, axs = plt.subplots(2,1, figsize = (5,7))

    axs[0].scatter(effective_e_list[1:]*num_protospacers, phage_pop_size_interp[1:], c = colours, s = 60)
    axs[1].scatter(effective_e_list[1:]*num_protospacers, bac_pop_size_interp[1:], c = colours, s = 60)
    
    for i, e_eff in enumerate(effective_e_list[1:]):
        axs[0].plot(effective_e_list[1+i:3+i]*num_protospacers, phage_pop_size_interp[1+i:3+i], color = colours[i], linestyle = '--', linewidth = 1, alpha = 0.5)
        axs[1].plot(effective_e_list[1+i:3+i]*num_protospacers, bac_pop_size_interp[1+i:3+i], color = colours[i], linestyle = '--', linewidth = 1, alpha = 0.5)

    # calculate pearson r
    p_r, p_p = pearsonr(effective_e_list[1:]*num_protospacers, np.log(phage_pop_size_interp[1:]))
    b_r, b_p = pearsonr(effective_e_list[1:]*num_protospacers, bac_pop_size_interp[1:])

    result_phage = linregress(effective_e_list[1:]*num_protospacers, np.log(phage_pop_size_interp[1:]))
    result_bac = linregress(effective_e_list[1:]*num_protospacers, bac_pop_size_interp[1:])

    xvals = np.arange(np.min(effective_e_list[1:]*num_protospacers), np.max(effective_e_list[1:]*num_protospacers),0.01)
    axs[0].plot(xvals, np.exp(xvals*result_phage.slope + result_phage.intercept), color = 'r', linestyle = '-',
               label = "Linear fit to log-\ntransformed data,\n" + r" $R=%s$, $p=%s$" %(round(p_r, 2), round(p_p,4)))
    axs[1].plot(xvals, xvals*result_bac.slope + result_bac.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=%s$" %(round(b_r, 2), round(b_p,2)))

    # for colours
    axs[0].scatter(list(effective_e_list)[1]*num_protospacers, phage_pop_size_interp[1], color = colours[0], s = 60, label = "Early time")
    axs[0].scatter(list(effective_e_list)[-1]*num_protospacers, phage_pop_size_interp[-1], color = colours[-1], s = 60, label = "Late time")   
    
    axs[0].set_yscale('log')
    #axs[1].set_yscale('log')

    axs[0].set_ylabel("Phage population size")
    #axs[0].set_xlabel("Same-day average immunity")
    axs[1].set_ylabel("Bacteria population size")
    axs[1].set_xlabel("Same-day average immunity")
    
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("pop_size_vs_avg_immunity_grouping_%s_%s_PAM_wt_%s.pdf" %(round(1-threshold,3), pam, wild_type))

```

```python
pearsonr(effective_e_list[1:], phage_pop_size_interp[1:])
```

```python
pearsonr(effective_e_list[1:], np.log(phage_pop_size_interp[1:]))
```

```python
pearsonr(effective_e_list[1:], bac_pop_size_interp[1:])
```

```python
pearsonr(effective_e_list[1:], np.log(bac_pop_size_interp[1:]))
```

```python
# it's very hard to relate the measured population size to the DNa because of all the vagaries of extraction, library prep, etc. 
sampled_fraction = 0.01
total_reads_df['num_phage_reads']* 100 / (phage_pop_size_interp * phage_genome * sampled_fraction) 
```

```python
# the interpolated version matches the original version pretty well
# still the question of normalization, but if we let the counts stand as they are and proceed...
plt.plot(time_min_spacing, np.sum(bac_interp*phage_interp, axis = 0)/(np.sum(bac_interp, axis = 0) * np.sum(phage_interp, axis = 0)))
plt.plot(time_points_in_days, np.sum(bac_wide_filtered.iloc[:, :-1]*phage_wide_filtered.iloc[:, :-1], 
                axis = 0)/(np.sum(bac_wide_filtered.iloc[:, :-1], axis = 0) * np.sum(phage_wide_filtered.iloc[:, :-1], axis = 0)))
plt.plot(time_min_spacing, np.sum(bac_interp.T * phage_interp.T, axis = 1)/(np.sum(bac_interp.T, axis = 1) 
                                                                            * np.sum(phage_interp.T, axis =1)))
```

## Simulating the effect of longer arrays on average immunity


The average overlap calculation depends on knowing the number of spacers and protospacers per organism: basically, by considering all spacers and protospacers as belonging to separate organisms, we underestimate average immunity. For example, if phages have 3 protospacers each, a bacterium needs only one of the three to be immune to it. 

Table S2: "Average length of the CRISPR loci by time point. CRISPR locus expansion was calculated by dividing the difference between the total and the wild-type locus length by the length of the corresponding spacer-repeat segment. Locus length was calculated using the repeat-containing reads divided by the host coverage by time point and CRISPR locus."

For CRISPR1 (the data I use), the average lengths are around 4 spacers. This wouldn't give a straight factor of 4 boost to average immunity though, since some new spacers would target the same phage as others in the locus, not boosting average immunity. 

My simple theory is that the initial overlap assuming each spacer is a single organism (regardless of the number of protospacers) gives a "base fraction" that determines what the effect is of adding more spacers. Let's call the intial single-spacer overlap $1-C$: $C = 1-a_1$. Then the average immunity $a$ if all bacteria have $n$ spacers is:

$$a_{n+1} = 1- (1 - a_n)C$$

Plugging in $a_1 = C$, we get

$$a_n = 1 - C^n$$

Now we'll explore this theory for a few different assumptions of how the spacers and arrays are distributed


```python
def count_matches(phages, spacers):
    """
    count the total number of phage matches from any of the spacers in spacers
    """
    presence_counter = 0
    for p in phages:
        present = 0
        for sp in spacers:
            if sp in p:
                present = 1
        presence_counter += present
        
    return presence_counter
```

```python
def generate_spacers(dist, spacer_seqs, num_spacers = 420, beta = 6):
    """
    Generate a set of bacteria spacers drawn from the `spacer_seqs` list with distribution `dist`.
    
    Inputs:
    dist: either 'exponential' or 'uniform'
    spacer_seqs: a list of possible sequences (letters of the alphabet here)
    num_spacers: total number of spacers to sample
    beta: parameter for the exponential distribution. Beta is the distribution mean.
    
    Returns:
    spacers_total: list of all the spacer sequences
    """
    
    # generate set of bacteria spacers
    spacers_total = []
    for n in range(num_spacers):
        # from exponential distribution
        if dist == 'exponential':
            ind = int(np.random.exponential(beta))
            while ind >= len(spacer_seqs):
                ind = int(np.random.exponential(beta))
            spacers_total.append(spacer_seqs[ind])
        elif dist == 'uniform':
        # uniform distribution
            spacers_total.append(np.random.choice(spacer_seqs))
            
    return spacers_total
```

```python
import string
spacer_seqs = list(string.ascii_uppercase)
```

```python
# generate phage sequences
# series of 5 unique letters randomly drawn from the alphabet
phages = []
num_phages = 50
for n in range(num_phages):
    phages.append(list(np.random.choice(spacer_seqs, 5, replace = False)))
```

```python
def create_arrays(array_size, spacers_total, length_dist, sigma = 2):
    """
    Randomly split the set of `spacers_total` into arrays of mean length `array_size` sampled from `length_dist` distribution.
    
    Inputs:
    array_size : characteristic array length
    spacers_total : set of total spacers
    length_dist : distribution from which to draw array lengths: either 'constant', 'gaussian', or 'exponential'
    sigma : scale parameter for the `gaussian` length_dist
    
    Returns:
    arrays: list of the returned arrays    
    """
    
    inds = np.arange(0,len(spacers_total),1)
    
    arrays = []
    for i in range(int(len(spacers_total) / array_size)):
        if length_dist == 'constant':
            size = array_size
        elif length_dist == 'gaussian':
            size = 0
            while size < 1:
                size = round(np.random.normal(loc = array_size, scale = sigma))
        elif length_dist == 'exponential':
            size = 0
            while size < 1:
                size = round(np.random.exponential(array_size))
        else:
            print("Invalid length distribution")
            
        if size > len(inds):
            choice = inds
        else:
            choice = np.random.choice(inds, size=size, replace=False)
        spacers = np.array(spacers_total)[choice]
    
        arrays.append(list(spacers))

        # delete the sampled spacers
        for j in choice:
            to_delete = np.where(inds == j)[0]
            inds = np.delete(inds, to_delete)
        
        if len(inds) == 0: # if all the spacers have been sampled
            break
            
    return arrays
```

Make a figure with different simulated scenarios:

```python
# distribution of 
dists = ['uniform', 'exponential']
array_sizes = np.arange(1, 8, 1)
n_iter = 50
length_dists = ['constant', 'gaussian', 'exponential']
num_spacers = 420
    
fig, axs = plt.subplots(2,3, figsize = (8,5))

# iterate over spacer distributions
for i, dist in enumerate(dists):
    spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)

    # iterate over array length distributions
    for j, length_dist in enumerate(length_dists):
        mean_avg_immunity = []
        std_avg_immunity = []
        
        for array_size in tqdm(array_sizes):
            avg_immunity_vals = []

            # simulate array splitting n_iter times
            for n in range(n_iter): 
                avg_immunity = 0
                arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
                for spacers in arrays:
                    overlap = count_matches(phages, spacers)
                    avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

                avg_immunity_vals.append(avg_immunity)
                
            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))
            
        # initial average immunity:
        C = 1-mean_avg_immunity[0]
        axs[i,j].errorbar(array_sizes, np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), label = "Simulated spacer\ndistributions")

        axs[i,j].plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
        
        axs[i,j].set_title("%s spacers,\n%s array length" %(dist, length_dist))
        
        axs[i,j].set_ylim(0.1, 0.8)

axs[0,0].set_ylabel("Average immunity")
axs[1,0].set_ylabel("Average immunity")
axs[1,1].set_xlabel("Mean bacterial array length")

axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[0,2].set_xticklabels([])

#ax.set_xlabel("Mean bacteria array length")
axs[0,0].legend()
plt.tight_layout()

plt.savefig("simulated_avg_immunity_vs_array_length_full_nphage_%s_nbac_%s.pdf" %(num_phages, num_spacers))
```

```python
spacer_seqs = list(string.ascii_uppercase)[:6]
```

```python
# generate phage sequences
# series of 5 unique letters randomly drawn from the alphabet
phages = []
num_phages = 100
for n in range(num_phages):
    phages.append(list(np.random.choice(spacer_seqs, 5, replace = False)))
```

```python
# sort protospacers to calculate phage diversity
phages_list = []

for phage in phages:
    phage.sort()
    phages_list.append("".join(phage))
```

```python
len(set(phages_list))
```

```python
# Single plot
dist = 'exponential'
length_dist = 'gaussian'

array_sizes = np.arange(1, 8, 1)
n_iter = 50
num_spacers = 1260

mean_avg_immunity = []
std_avg_immunity = []
mean_diversity = []
std_diversity = []
total_arrays = [] 

spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)

for array_size in tqdm(array_sizes):
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    # simulate array splitting n_iter times
    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))
        
        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
    
    mean_diversity.append(np.mean(diversity))
    std_diversity.append(np.std(diversity))
    mean_avg_immunity.append(np.mean(avg_immunity_vals))
    std_avg_immunity.append(np.std(avg_immunity_vals))
    total_arrays.append(np.mean(num_arrays))  

```

```python
# plot of diversity vs average immunity for fixed total number of spacers but variable array size


# initial average immunity:
C = 1-mean_avg_immunity[0]

fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar(np.array(mean_avg_immunity), mean_diversity,  marker = 'o', 
           xerr = np.array(std_avg_immunity), label = "Simulated spacer distributions")
ax.plot(np.array(mean_avg_immunity), total_arrays, marker = 'o', label = "Total population size")

#ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_xlabel("Average immunity")
ax.set_ylabel("Bacterial diversity")
ax.legend()
plt.tight_layout()
```

```python

# initial average immunity:
C = 1-mean_avg_immunity[0]

fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar(array_sizes, np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), label = "Simulated spacer distributions")

ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_ylabel("Average immunity")
ax.set_xlabel("Mean bacteria array length")
ax.legend()
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_array_length_nphage_%s_nbac_%s_%s_array_size_%s_spacers.pdf" %(num_phages, num_spacers, length_dist, dist))
```

## Exploring the relationship between diversity and average immunity as a function of array size

In the previous stuff, I fix the total set of spacers and protospacers, then shuffle them around differently. 

Here, I want to increase the number of spacers (which would increase diversity) for a fixed array size, then try different phage array sizes. 

"At some point, even though phages have multiple protospacers, bacteria will be dividing their eggs among too many baskets and average immunity will go down."

```python
# generate phage sequences
# series of 5 unique letters randomly drawn from the alphabet
phages = []
num_phages = 100
for n in range(num_phages):
    phages.append(list(np.random.choice(spacer_seqs, 2, replace = False)))
```

```python
# sort protospacers to calculate phage diversity
phages_list = []

for phage in phages:
    phage.sort()
    phages_list.append("".join(phage))
```

```python
len(set(phages_list))
```

```python
array_size = 1 # bacteria array size - want this to be low to be in the low-bacterial array size limit, but high enough that can get >26 values of diversity

dist = 'exponential'
length_dist = 'gaussian'
#length_dist = 'constant'
n_iter = 75
num_spacers = array_size * 400
phage_array_length = 1
diversity_vals_list = np.arange(phage_array_length,27, 3)
```

```python
mean_avg_immunity = []
std_avg_immunity = []
mean_diversity = []
std_diversity = []
total_arrays = [] 


for i, d in tqdm(enumerate(diversity_vals_list)):
    # make a set of spacers to arrange into arrays
    spacer_seqs = list(string.ascii_uppercase)[:d]
    spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)
    
    # generate phage sequences
    # series of x unique letters randomly drawn from the alphabet
    # use the same spacer sequences that bacteria get
    phages = []
    num_phages = 100
    for n in range(num_phages):
        phages.append(list(np.random.choice(spacer_seqs, phage_array_length, replace = False)))
    
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    # simulate array splitting n_iter times
    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))
        
        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
        
    mean_diversity.append(np.mean(diversity))
    std_diversity.append(np.std(diversity))
    mean_avg_immunity.append(np.mean(avg_immunity_vals))
    std_avg_immunity.append(np.std(avg_immunity_vals))
    total_arrays.append(np.mean(num_arrays))  

```

```python
mean_avg_immunity = []
std_avg_immunity = []
mean_diversity = []
std_diversity = []
total_arrays = [] 


for i, d in tqdm(enumerate(diversity_vals_list)):
    # make a set of spacers to arrange into arrays
    spacer_seqs = list(string.ascii_uppercase)[:d]
    spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)
    
    # generate phage sequences
    # series of x unique letters randomly drawn from the alphabet
    # use the same spacer sequences that bacteria get
    phages = []
    num_phages = 100
    for n in range(num_phages):
        phages.append(list(np.random.choice(spacer_seqs, phage_array_length, replace = False)))
    
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    # simulate array splitting n_iter times
    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))
        
        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
        
    mean_diversity.append(np.mean(diversity))
    std_diversity.append(np.std(diversity))
    mean_avg_immunity.append(np.mean(avg_immunity_vals))
    std_avg_immunity.append(np.std(avg_immunity_vals))
    total_arrays.append(np.mean(num_arrays))  
```

```python
mean_avg_immunity = []
std_avg_immunity = []
mean_diversity = []
std_diversity = []
total_arrays = [] 


for i, d in tqdm(enumerate(diversity_vals_list)):
    # make a set of spacers to arrange into arrays
    spacer_seqs = list(string.ascii_uppercase)[:d]
    spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)
    
    # generate phage sequences
    # series of x unique letters randomly drawn from the alphabet
    # use the same spacer sequences that bacteria get
    phages = []
    num_phages = 100
    for n in range(num_phages):
        phages.append(list(np.random.choice(spacer_seqs, phage_array_length, replace = False)))
    
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    # simulate array splitting n_iter times
    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))
        
        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
        
    mean_diversity.append(np.mean(diversity))
    std_diversity.append(np.std(diversity))
    mean_avg_immunity.append(np.mean(avg_immunity_vals))
    std_avg_immunity.append(np.std(avg_immunity_vals))
    total_arrays.append(np.mean(num_arrays))  
```

```python
fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar( np.array(mean_avg_immunity), mean_diversity,   marker = 'o', 
           xerr = np.array(std_avg_immunity),
            yerr = std_diversity, label = "Simulated spacer distributions")
#ax.plot(total_arrays,np.array(mean_avg_immunity),  marker = 'o', label = "Total population size")

#ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_xlabel("Average immunity")
ax.set_ylabel("Bacterial diversity")
#ax.set_xscale('log')
#ax.set_yscale('log')
```

If the phage diversity is also increasing as bacterial diversity increases, average immunity will go down as diversity increases, provided the diversity is larger than the bacterial array size.

```python
fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar( mean_diversity,  np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), 
            xerr = std_diversity, label = "Simulated spacer distributions")
#ax.plot(total_arrays,np.array(mean_avg_immunity),  marker = 'o', label = "Total population size")

#ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_ylabel("Average immunity")
ax.set_xlabel("Bacterial diversity")
ax.set_yscale('log')
ax.set_xscale('log')

plt.savefig("simulated_avg_immunity_vs_diversity_array_size_%s_phage_array_size_%s_length_dist_%s_log.pdf" %(array_size, phage_array_length, length_dist))
```

#### Panel plot - varying bacteria & phage array length

```python
array_sizes = [1,2,4,7] # bacteria array size - want this to be low to be in the low-bacterial array size limit, but high enough that can get >26 values of diversity

dist = 'exponential'
#length_dist = 'gaussian'
length_dist = 'constant'
n_iter = 50
phage_array_lengths = [1,2,4,7,10]
#num_spacers = array_size * 400
num_spacers = 1000
num_phages = 100
```

```python
fig, axs = plt.subplots(len(array_sizes), len(phage_array_lengths), figsize = (20,15))

for i, array_size in enumerate(array_sizes):
    for j, phage_array_length in enumerate(phage_array_lengths):
        diversity_vals_list = np.arange(phage_array_length,27, 3)
        
        mean_avg_immunity = []
        std_avg_immunity = []
        mean_diversity = []
        std_diversity = []
        total_arrays = [] 

        for k, d in tqdm(enumerate(diversity_vals_list)):
            # make a set of spacers to arrange into arrays
            spacer_seqs = list(string.ascii_uppercase)[:d]
            spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)

            # generate phage sequences
            # series of x unique letters randomly drawn from the alphabet
            # use the same spacer sequences that bacteria get
            phages = []
            for n in range(num_phages):
                phages.append(list(np.random.choice(spacer_seqs, phage_array_length, replace = False)))

            avg_immunity_vals = []
            diversity = []
            num_arrays = []

            # simulate array splitting n_iter times
            for n in range(n_iter): 
                avg_immunity = 0
                arrays_concat = []
                arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
                for spacers in arrays:
                    overlap = count_matches(phages, spacers)
                    avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

                    # calculate bacterial diversity: number of unique genotypes
                    spacers.sort()
                    arrays_concat.append("".join(spacers))

                diversity.append(len(set(arrays_concat)))
                avg_immunity_vals.append(avg_immunity)
                num_arrays.append(len(arrays))

            mean_diversity.append(np.mean(diversity))
            std_diversity.append(np.std(diversity))
            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))
            total_arrays.append(np.mean(num_arrays))  
            
        axs[i,j].errorbar( mean_diversity,  np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), 
            xerr = std_diversity, label = "Simulated spacer distributions")
        #ax.plot(total_arrays,np.array(mean_avg_immunity),  marker = 'o', label = "Total population size")

        #ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
        #plt.yscale('log')

        
        axs[i,j].set_title("%s spacers, %s protospacers" %(array_size, phage_array_length))
        #ax.set_yscale('log')
        #ax.set_xscale('log')

for i in range(len(array_sizes)):
    axs[i,0].set_ylabel("Average immunity")
for j in range(len(phage_array_lengths)):
    axs[-1,j].set_xlabel("Bacterial diversity")
    
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_diversity_%s.pdf" %length_dist)
```

```python

```

### Exploring the effect of array size on time shifted average immunity

This is a bit trickier because you need some kind of model for how spacers change over time. But maybe with some very simple assumptions we can still say something? Maybe try taking data from one of our simulations and distributing it in arrays differently? I think taking simulation data won't work, since spacer lifetimes are set already and the overlap shouldn't change length if we group them differently...

Want some way to say that A -> B -> C gives different memory length than XA -> AB -> BC

Maybe spacer lifetime is (array_length) x base lifetime if arrays are longer? 

```python
beta = 6
spacer_seqs = np.arange(0, 100, 1)
num_spacers = 1000
spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = beta)
```

```python
# create arrays
array_size = 30
arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
```

```python
# flatten array list to get histogram of spacer appearances

spacers_flat = [item for sublist in arrays for item in sublist]
```

```python
fig, ax = plt.subplots()
ax.hist(spacers_total, density = True, bins = 50, label = "Spacers to sample from", alpha = 0.5)

ax.hist(spacers_flat, density = True,  bins = 50, label = "Sampled spacers", alpha = 0.5)

ax.plot(spacer_seqs,(1/beta)*np.exp(-spacer_seqs/beta), label = "Exponential generating distribution")

ax.legend()
```
