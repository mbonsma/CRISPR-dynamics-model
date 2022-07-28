---
jupyter:
  jupytext:
    cell_metadata_json: true
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

# Time-shift overlap and spacer correlations from Guerrero et al 2021

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from scipy.stats import pearsonr
from scipy.stats import linregress
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

```

```python
from spacer_model_plotting_functions import e_effective_shifted, effective_e
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

```python
phage_only = False
```

```python
# spacers and protospacers detected in spacer_finder.ipynb
# spacers and protospacers detected in spacer_finder.ipynb

spacer_types_bac_CR1 = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_CR1.csv")
spacer_types_bac_CR2 = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_CR2.csv")

if phage_only == True:
    spacer_types_phage_all = pd.read_csv("results/2022-03-30/protospacer_counts_phage_only.txt")
else:
    spacer_types_phage_all = pd.read_csv("results/2022-03-31/protospacer_counts.txt")
spacer_types_phage_all[["spacer_type", "sequence_id", "crispr"]] = spacer_types_phage_all['query_id'].str.split('_', 
                                                                                            expand = True)

spacer_types_phage_CR1 = spacer_types_phage_all[spacer_types_phage_all['crispr'] == 'CR1']
spacer_types_phage_CR2 = spacer_types_phage_all[spacer_types_phage_all['crispr'] == 'CR2']
```

## Inspecting spacers from CR1 and CR3

```python
# overlap between new spacers and the cr1 spacers: very minimal, only a few types that appear in both from the clustering
spacer_types_bac_CR2[spacer_types_bac_CR2['sequence'].isin(spacer_types_bac_CR1['sequence'])].drop_duplicates('type')
```

```python
from Bio.Seq import Seq
spacer_types_bac_CR2['sequence_revcomp'] = spacer_types_bac_CR2['sequence'].apply(Seq).apply(lambda x: x.reverse_complement()).apply(str)
```

```python
spacer_types_bac_CR2[spacer_types_bac_CR2['sequence_revcomp'].isin(spacer_types_bac_CR1['sequence'])].drop_duplicates('type')
```

```python
spacer_types_bac_CR2 = spacer_types_bac_CR2.drop(columns = 'sequence_revcomp')
```

## Add different grouping thresholds

```python
# load all the thresholds and combine into one dataframe
thresholds = np.arange(0.01, 0.16, 0.01)
thresholds = np.concatenate([[0.005], thresholds])

for threshold in tqdm(thresholds):
    cr = "CR2"
    spacer_types_threshold = pd.read_csv("results/2022-03-29/spacer_protospacer_types_%s_phage_only_%s_similarity_%s.csv" %(cr, 
                                                                                                        phage_only, 1-threshold))
    spacer_types_threshold = spacer_types_threshold.rename(columns = {"type": "type_%s" %(1-threshold)})
    spacer_types_bac_CR2 = spacer_types_bac_CR2.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    spacer_types_phage_CR2 = spacer_types_phage_CR2.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    
    cr = "CR1"
    spacer_types_threshold = pd.read_csv("results/2022-03-29/spacer_protospacer_types_%s_phage_only_%s_similarity_%s.csv" %(cr, 
                                                                                                        phage_only, 1-threshold))
    spacer_types_threshold = spacer_types_threshold.rename(columns = {"type": "type_%s" %(1-threshold)})
    spacer_types_bac_CR1 = spacer_types_bac_CR1.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
    spacer_types_phage_CR1 = spacer_types_phage_CR1.merge(spacer_types_threshold, on = ('sequence'), how = 'left')
```

```python
# note that 'type' and 'type_085' are different because in the type_0.85 version, protospacers were also included.
print(spacer_types_bac_CR2['type_0.85'].nunique())
print(spacer_types_bac_CR2['type'].nunique())
```

```python
# not all types match perfectly: 102 type_0.85 types map to multiple 'type' types, and 126 'type' types map to multiple 'type_0.85'
type_comparison = spacer_types_bac_CR2.groupby(['type', 'type_0.85'])['count'].sum().reset_index()

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
if phage_only == False: # there are no CR2 protospacers for phage_only == True
    spacer_types_phage_CR2[["type","query_seq", "crispr"]] = spacer_types_phage_CR2['query_id'].str.split('_', expand = True)

spacer_types_phage_CR1[["type","query_seq", "crispr"]] = spacer_types_phage_CR1['query_id'].str.split('_', expand = True)
```

```python
# preprocessing: remove spacers that have long poly-N sequences
spacer_types_bac_CR2 = remove_poly_nucleotide_spacers(spacer_types_bac_CR2)
spacer_types_bac_CR1 = remove_poly_nucleotide_spacers(spacer_types_bac_CR1)
#spacer_types_bac_CR2_no_wildtype = remove_poly_nucleotide_spacers(spacer_types_bac_CR2_no_wildtype)
#spacer_types_bac_CR1_no_wildtype = remove_poly_nucleotide_spacers(spacer_types_bac_CR1_no_wildtype)
spacer_types_phage_CR1 = remove_poly_nucleotide_spacers(spacer_types_phage_CR1)
if phage_only == False:
    spacer_types_phage_CR2 = remove_poly_nucleotide_spacers(spacer_types_phage_CR2)
```

## PAM processing

Version 1: keeping perfect PAM only


CRISPR1 PAM for phage DC-56: GTT

"Figure S7: WebLogo for the PAM consensus sequence predicted for phage DC-56
required for CRISPR-1"


There's clearly some spacers where the different PAMs associated with the same sequence are actually the same PAM but cutoff. Other times where this isn't the case. How to decide? Maybe only group them if they're shorter, clearly could be the start of a valid PAM, and a longer one in the group does have a valid PAM? 
If there are multiple long ones in the group and not all of them have the PAM, then treat them individually again?

How to count: group by individual sequence and PAM, or just classify PAM as present or absent?


```python tags=[]
# only sequences containing the perfect PAM
spacer_types_phage_CR1_PAM = spacer_types_phage_CR1[(spacer_types_phage_CR1['PAM_region_5'].str.slice(-5,).str.contains('AAC'))] 
spacer_types_phage_CR2_PAM = spacer_types_phage_CR2 # no PAM for type 3
```

<!-- #region {"tags": []} -->
## Count total number of reads matching either phage or bacteria genome per sample

There are 1000000 reads per split genome file; this counts the number that matched to the bacteria genome or the CRISPR repeat and the number that matched to the phage genome. 

Count for each time point:
- [x] Total that match the bacteria genome or the CRISPR repeat
- [x] Total that match the phage genome
- [x] Total that match both
- [x] Total that match neither
<!-- #endregion -->

```python
#datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
#!python get_read_counts_Guerrero.py "$datapath"
```

```python tags=[]
total_reads_df = pd.read_csv("total_reads_normalization_Guerrero.csv", index_col = 0)

total_reads_df['total_reads'] = total_reads_df['total_reads'].astype('int')
total_reads_df['total_matching_neither'] = total_reads_df['total_matching_neither'].astype('int')
```

```python tags=[]
# the total of phage + bac + neither - both should be 1
np.all((total_reads_df['num_bac_reads_base'] + total_reads_df['num_phage_reads_base'] + total_reads_df['total_matching_neither']
 - total_reads_df['total_matching_both']) / total_reads_df['total_reads'] == 1)
```

### Choose whether to include partial PAMs or perfect PAMs only

```python
pam = "perfect"
#pam = "no" # use all protospacer matches regardless of PAM

folder = "results/%s_PAM" %pam
```

## Overlap: both CR1 and CR3

```python
spacer_types_bac_CR1['crispr'] = 1
spacer_types_bac_CR2['crispr'] = 2
```

```python
if pam == "perfect":
    spacer_types_phage_CR1_PAM['crispr'] = 1
    spacer_types_phage_CR2_PAM['crispr'] = 2
elif pam == "no":
    spacer_types_phage_CR1['crispr'] = 1
    spacer_types_phage_CR2['crispr'] = 2
```

```python
spacer_types_bac = pd.concat([spacer_types_bac_CR1, spacer_types_bac_CR2])
if pam == "perfect":
    spacer_types_phage = pd.concat([spacer_types_phage_CR1_PAM, spacer_types_phage_CR2_PAM])
elif pam == "no":
    spacer_types_phage = pd.concat([spacer_types_phage_CR1, spacer_types_phage_CR2])
```

Types that are present in both CR1 and CR3 should be fine - differentiated in the phage data based on PAM


### Choose whether to include wildtype

```python
# load wild-type spacers
datapath = "/media/madeleine/WD_BLACK/Blue hard drive/Data/Guerrero2021/data"
wild_type_spacers_CR1 = load_wild_type_spacers(datapath, "CR1")
wild_type_spacers_CR2 = load_wild_type_spacers(datapath, "CR2")

# these are all the sequences that are in the wild-type list and their type identifiers
spacer_types_bac_wt = spacer_types_bac[(((spacer_types_bac['crispr'] == 1) 
                & (spacer_types_bac['sequence'].isin(wild_type_spacers_CR1)))
                 | ((spacer_types_bac['crispr'] == 2) 
                & (spacer_types_bac['sequence'].isin(wild_type_spacers_CR2))))]

spacer_types_phage_wt = spacer_types_phage[(((spacer_types_phage['crispr'] == 1) 
                & (spacer_types_phage['sequence'].isin(wild_type_spacers_CR1)))
                 | ((spacer_types_phage['crispr'] == 2) 
                & (spacer_types_phage['sequence'].isin(wild_type_spacers_CR2))))]
```

```python
wild_type = True # False: don't include wild-type
```

## Spacer abundance distributions

```python
def cumulative_sum(array,reverse = False):
    # do cumulative sum
    if reverse == False:
        cumul = np.cumsum(array)
    elif reverse == True:
        array_to_sum = np.fliplr([array])[0]
        cumul_flipped = np.cumsum(array_to_sum)
        cumul = np.fliplr([cumul_flipped])[0]
        
    # normalize distribution   
    cumul = cumul/np.amax(cumul)
    return cumul
```

```python
time_points = np.sort(spacer_types_bac['time_point'].unique())
spacers_grouped = spacer_types_bac.groupby(['type', 'time_point', 'crispr'])['count'].sum().reset_index()
spacers_grouped_phage = spacer_types_phage.groupby(['spacer_type', 'time_point', 'crispr'])['count'].sum().reset_index()
```

```python

colours = cm.gist_rainbow(np.linspace(0,1, len(time_points)))
fig, axs = plt.subplots(1,2, figsize = (12,5))
ax = axs[0]
ax1 = axs[1]

ax.set_ylabel("Cumulative frequency")
ax.set_xlabel("Spacer abundance")

#ax1.set_ylabel("Cumulative frequency")
ax1.set_xlabel("Protospacer abundance")

for time_point in time_points:

    abundance_list = np.sort(spacers_grouped[spacers_grouped['time_point'] == time_point]['count'])
    abundance_hist, bins = np.histogram(abundance_list, 
                                         bins = np.arange(0, np.max(abundance_list)+1, 1))

    ax.plot(bins[:-1], cumulative_sum(abundance_hist, reverse = True), label = "Time %s" %time_point,
           color = colours[time_point], alpha = 0.6, linewidth = 3)
    
    
    abundance_list_phage = np.sort(spacers_grouped_phage[spacers_grouped_phage['time_point'] == time_point]['count'])
    abundance_hist_phage, bins = np.histogram(abundance_list_phage, 
                                         bins = np.arange(0, np.max(abundance_list_phage)+1, 1))

    ax1.plot(bins[:-1], cumulative_sum(abundance_hist_phage, reverse = True), label = "Time %s" %time_point,
           color = colours[time_point], alpha = 0.6, linewidth = 3)

# add colorbar

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)

cmap = cm.gist_rainbow
norm = matplotlib.colors.Normalize(vmin=0, vmax=59)

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', label='Time point')
    
ax.set_yscale('log')
ax.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax.set_title("Gordonia spacers")
ax1.set_title("Phage protospacers")
#ax.legend()
plt.tight_layout()
plt.savefig("Guerrero_cumulative_distributions_%s_pam_wt_%s.pdf" %(pam, wild_type))
```

### Rank-abundance version

```python
fig, axs = plt.subplots(1,2, figsize = (12, 5))

ax = axs[0]
ax1 = axs[1]

colours = cm.gist_rainbow(np.linspace(0,1, len(time_points)))
for time_point in time_points:    
    s = np.sort(spacers_grouped[spacers_grouped['time_point'] == time_point]['count'])
    
    rank_abund_simulation = np.fliplr([np.sort(s)])[0]/np.sum(s)

    x_axis = np.arange(1, len(s)+1, 1)
    x_axis = x_axis/len(np.nonzero(s)[0])
    
    ax.plot(x_axis, rank_abund_simulation, color = colours[time_point], alpha = 0.6, linewidth = 3)
    
    s2 = np.sort(spacers_grouped_phage[spacers_grouped_phage['time_point'] == time_point]['count'])

    rank_abund_simulation_phage = np.fliplr([np.sort(s2)])[0]/np.sum(s2)

    x_axis = np.arange(1, len(s2)+1, 1)
    x_axis = x_axis/len(np.nonzero(s2)[0])
    
    ax1.plot(x_axis, rank_abund_simulation_phage, color = colours[time_point], alpha = 0.6, linewidth = 3)
    
    
# add colorbar

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)

cmap = cm.gist_rainbow
norm = matplotlib.colors.Normalize(vmin=0, vmax=59)

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='vertical', label='Time point')

ax.set_yscale('log')
ax.set_xlim(0,1)
ax1.set_xlim(0,1)
ax1.set_yscale('log')
ax.set_ylabel("Spacer frequency")
ax.set_xlabel("Normalized abundance rank")
ax1.set_ylabel("Protospacer frequency")
ax1.set_xlabel("Normalized abundance rank")
ax.set_title("Gordonia spacers")
ax1.set_title("Phage protospacers")
plt.tight_layout()
plt.savefig("Guerrero_rank_abundance_distributions_%s_pam_wt_%s.pdf" %(pam, wild_type))
```

### Plot a few of the largest types over time

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
metadata = pd.read_csv("%s/../SraRunTable.txt" %datapath)

accessions = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
d = pd.to_datetime(dates)
dates_int = np.array((d[1:] - d[:-1]).days)
time_points_in_days = np.concatenate([[0], dates_int])
time_points_in_days = np.cumsum(time_points_in_days)

time_points = np.arange(0, len(dates), 1)
```

```python
grouping = 'type_0.95'

# group by time point, spacer type, and CRISPR locus - this groups different sequences that have been assigned the same type
bac_types_grouped = spacer_types_bac.groupby(['time_point', grouping, 'crispr'])[['count']].sum().reset_index()
phage_types_grouped = spacer_types_phage.groupby(['time_point', grouping, 'crispr'])[['count']].sum().reset_index()

df_combined = bac_types_grouped.merge(phage_types_grouped, on = (grouping, 'time_point', 'crispr'), suffixes = ('_bac', '_phage'),
                                        how = 'outer')

df_combined = df_combined.fillna(0)

df_combined_all_time = df_combined.groupby([grouping, 'crispr'])[['count_bac', 'count_phage']].sum().reset_index()
```

```python
grouping = 'type_0.95'

num_types = 15

df_combined['type_crispr'] = df_combined[grouping].astype('int').astype('str') + "_" + df_combined['crispr'].astype(str)

top_bac_types = df_combined.groupby(['type_crispr'])['count_bac'].sum().reset_index().sort_values(by = 'count_bac', ascending = False)[:num_types]
top_phage_types = df_combined.groupby(['type_crispr'])['count_phage'].sum().reset_index().sort_values(by = 'count_phage', ascending = False)[:num_types]
```

```python
subset_top = df_combined[(df_combined['type_crispr'].isin(top_bac_types['type_crispr']))
                         | (df_combined['type_crispr'].isin(top_phage_types['type_crispr']))].groupby(['time_point', 
                                                        'type_crispr'])[['count_bac', 'count_phage']].sum().reset_index()

subset_top_pivot = subset_top.pivot_table(index = 'type_crispr', columns = 'time_point', values = ['count_bac', 'count_phage'],
                                  fill_value = 0, margins = True, aggfunc = 'sum').reset_index()
```

```python
# sort by max counts
subset_pivot = subset_top_pivot.iloc[:-1].sort_values(by = [('count_bac', 'All'), ('count_phage', 'All')], ascending = False)
```

```python
bac_subset = subset_pivot['count_bac'].iloc[:, :-1]
phage_subset = subset_pivot['count_phage'].iloc[:, :-1]
```

```python
colours_bac = cm.viridis(np.linspace(0,0.92, num_types))
colours_phage = cm.plasma(np.linspace(0,0.9, num_types))

colours = np.concatenate([colours_bac, colours_phage])

fig, axs = plt.subplots(2,2, figsize = (10,7))

ax = axs[0,0]
ax1 = axs[1,0]
ax2 = axs[0,1]
ax3 = axs[1,1]

for i in range(num_types*2):
    ax.plot(time_points_in_days, bac_subset.iloc[i], color = colours[i])
    ax1.plot(time_points_in_days, phage_subset.iloc[i], color = colours[i])
    
ax2.stackplot(time_points_in_days, bac_subset / np.sum(bac_subset, axis = 0), colors = colours, edgecolor = 'k', linewidth = 0.5)
ax3.stackplot(time_points_in_days, 
              phage_subset / np.sum(phage_subset, axis = 0), colors = colours, edgecolor = 'k', linewidth = 0.5)
ax.set_yscale('log')
ax1.set_yscale('log')

ax.set_ylabel("Bacteria clone size")
ax1.set_ylabel("Phage clone size")
ax2.set_ylabel("Bacteria clone frequency")
ax3.set_ylabel("Phage clone frequency")

ax2.set_ylim(0,1)
ax3.set_ylim(0,1)

for ax in axs.flatten():
    ax.set_xlim(0, np.max(time_points_in_days))

plt.tight_layout()

plt.savefig("top_%s_clones_%s_PAM_wt_%s_phage_only_%s.pdf" %(num_types, pam, wild_type, phage_only))
```

### Are clone sizes correlated over the course of the experiment?

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
# according to pearsonr, not correlated. 
# basically there are a bunch of large protospacers that are not targeted by bacteria
pearsonr(df_combined_all_time['count_bac'], df_combined_all_time['count_phage'])
```

### Make combined data frame

```python
#grouping = 'type'
#grouping = 'sequence'
#grouping = 'type_0.99'
presence_cutoff = 1 # how many detections to count spacer type as "present"
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
    bac_types_grouped = df_bac.groupby(['time_point', grouping, 'crispr'])[['count']].sum().reset_index()
    phage_types_grouped = df_phage.groupby(['time_point', grouping, 'crispr'])[['count']].sum().reset_index()
    
    phage_types_grouped[grouping] = phage_types_grouped[grouping].astype('int')

    df_combined = bac_types_grouped.merge(phage_types_grouped, on = (grouping, 'time_point', 'crispr'), suffixes = ('_bac', '_phage'),
                                            how = 'outer')

    df_combined = df_combined.fillna(0)

    # convert spacer counts to presence-absence
    df_combined['bac_presence'] = np.where(df_combined['count_bac'] >= presence_cutoff, 1, 0)
    df_combined['phage_presence'] = np.where(df_combined['count_phage'] >= presence_cutoff, 1, 0)

    df_combined.to_csv("results/%s_PAM/Guerrero_data_combined_%s_wt_%s_phage_only_%s.csv" %(pam, grouping, wild_type, phage_only))
```

Might want to think more carefully about normalization: normalizing to the total number of reads is one way to compare between time points, but it would do different things to bacteria and phage based on the size of their genomes... maybe there's a better way?

If the reads are evenly spread across the genome, then we can weight by the genome size as well to compare bacteria and phage on equal footing. 

Should I compare total reads to total population size? If they track nicely, then maybe we don't need to normalize at all?


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
    time_points = []
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
        time_points.append(tp)
        
    return overlap_list, bac_types, phage_types, time_points
```

```python
#time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
colours = cm.viridis(np.linspace(0,1, len(thresholds)))[::-1]
fig, axs = plt.subplots(2,2, figsize = (10,8))
cutoff = 0

for i, threshold in enumerate(thresholds):
    grouping = 'type_%s' %(1-threshold)
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
    
    shared_types, bac_types, phage_types, tps = type_overlap(df_combined, cutoff = cutoff)
    
    axs[0,0].plot(tps, shared_types, 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[0,1].plot(tps, np.array(phage_types) / np.array(bac_types), 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[1,0].plot(tps, bac_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
    
    axs[1,1].plot(tps, phage_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
for ax in axs.flatten():
    ax.set_xticks(time_points[::5])
    ax.set_xticklabels(time_points_in_days[::5])
    
axs[0,0].set_ylabel("Number of shared spacer types")
axs[0,1].set_ylabel("Ratio of phage types to bacteria types")
axs[1,0].set_ylabel("Number of bacteria types")
axs[1,1].set_ylabel("Number of phage types")
axs[1,0].set_xlabel("Time (days)")
axs[1,1].set_xlabel("Time (days)")
axs[0,0].legend(ncol = 2)
axs[0,0].set_yscale('log')

plt.tight_layout()
plt.savefig("unique_types_and_overlap_all_cutoff_%s_%s_PAM_wt_%s_phage_only_%s.pdf" %(cutoff, pam, wild_type, phage_only))
```

```python
metadata['accession'] = metadata['Run']
total_reads_df = total_reads_df.merge(metadata[['accession', 'Collection_date']], on = 'accession')
total_reads_df = total_reads_df.sort_values(by = 'Collection_date')
total_reads_df['time_point'] = np.arange(0,60)
```

```python tags=[]
fig, axs = plt.subplots(2,1, figsize = (6,6))

ax = axs[0]
ax1 = axs[1]

ax.plot(total_reads_df['time_point'], total_reads_df['num_phage_reads_final'], label = "Phage")
ax.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads_final'], label = "Bacteria")

ax1.plot(total_reads_df['time_point'], total_reads_df['num_phage_reads_final'] / total_reads_df['total_reads'], label = "Phage")
ax1.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads_final']/ total_reads_df['total_reads'], label = "Bacteria")


#axb.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads'] )
#ax.set_yscale('log')
ax.legend()

ax1.set_xlabel("Time (days)")
ax.set_ylabel("Total reads")
ax1.set_yscale('log')
ax.set_yscale('log')
ax1.set_ylabel("Fraction of total reads")

for ax in axs:
    ax.set_xticklabels(time_points_in_days)

plt.tight_layout()
plt.savefig("Guerrero_read_totals.pdf")
```

Interestingly the number of shared types has a peak at an intermediate threshold: about 94% similarity. Tradeoff between total number of types (increases as similarity cutoff increases) and likelihood that types are shared (increases as similarity cutoff decreases).


## Time shift

Caution: the time intervals are all weird, so averaging over the same delay will be tricky.

```python
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
```

Should I count just the phage DC-56 PAMs, or also phage DS-92? Maybe both and then average them?

```python
datapath = "/media/madeleine/WD_BLACK/Blue hard drive/Data/Guerrero2021/genomes"

# phage DC-56
with open ('%s/phage_DC-56.fa' %(datapath), 'r') as f:
    phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome:
    if row[0] != ">":
        phage_genome_seq += row.strip()
        
# count number of perfect PAMs in the reference genome
cr1_pams_DC56 = (phage_genome_seq.count('GTT')
            + phage_genome_seq.count('AAC'))

# phage DS-92
with open ('%s/phage_DS-92.fa' %(datapath), 'r') as f:
    phage_genome = f.readlines()
    
phage_genome_seq = ""
for row in phage_genome:
    if row[0] != ">":
        phage_genome_seq += row.strip()
        
# count number of perfect PAMs in the reference genome
cr1_pams_DS92 = (phage_genome_seq.count('GTT')
            + phage_genome_seq.count('AAC'))
```

```python
num_protospacers = round((cr1_pams_DC56 + cr1_pams_DS92)/2)
```

```python
#num_protospacers = cr1_pams_DC56
```

```python
num_protospacers
```

```python
start_ind = 0 # the 21st is the last point that has really high overlap
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
if stop_ind == -1:
    step = np.min(np.diff(time_points_in_days[start_ind: stop_ind]))
else:
    step = np.min(np.diff(time_points_in_days[start_ind: stop_ind + 1]))
    
    
step = 14
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
    grouping = ['type_%s' %(1-threshold), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
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

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.05)

#ax.set_xlim(-1550, 1550)
#ax.set_ylim(0, 1)
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Guerrero_both_cr_start_trim_%s_end_trim_%s_%s_PAM_wt_%s_phage_only_%s.pdf" %(start_ind, 
                                                                                                np.abs(stop_ind) - 1, pam, wild_type, phage_only))
```

## Turnover plots

```python
def fraction_remaining_banfield(bac_array, interp_times):
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
```

```python
threshold = 0.01
grouping = ['type_%s' %(1-threshold), 'crispr']
df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)
```

```python
# interpolate the bacteria spacer values
# remove the last column which is a sum column
f = interp1d(time_points_in_days,bac_wide_filtered.iloc[:,:-1])
f_phage = interp1d(time_points_in_days, phage_wide_filtered.iloc[:,:-1])
```

```python
step = 14
interp_times_Guerrero = np.arange(np.min(time_points_in_days), np.max(time_points_in_days), step)
```

```python tags=[]
bac_array = f(interp_times_Guerrero)
turnover_array = fraction_remaining_banfield(bac_array, interp_times_Guerrero)
```

```python
phage_array = f_phage(interp_times_Guerrero)
turnover_array_phage = fraction_remaining_banfield(phage_array, interp_times_Guerrero)
```

```python
fig, axs = plt.subplots(1,2, figsize = (4.7, 3.3))
ax = axs[0]
ax1 = axs[1]

ax.plot(interp_times_Guerrero, np.nanmean(turnover_array, axis = 0), color = 'k', linewidth = 2)
ax.fill_between(interp_times_Guerrero, y1 = np.nanmean(turnover_array, axis = 0) - np.nanstd(turnover_array, axis = 0),
               y2 = np.nanmean(turnover_array, axis = 0) + np.nanstd(turnover_array, axis = 0),
               alpha = 0.4, color = 'grey')

ax.set_ylim(0,1.0)
ax.set_xlabel("Time delay (days)")
ax.set_ylabel("Fraction of types remaining")

ax1.plot(interp_times_Guerrero, np.nanmean(turnover_array_phage, axis = 0), color = 'k', linewidth = 2)
ax1.fill_between(interp_times_Guerrero, y1 = np.nanmean(turnover_array_phage, axis = 0) - np.nanstd(turnover_array_phage, axis = 0),
               y2 = np.nanmean(turnover_array_phage, axis = 0) + np.nanstd(turnover_array_phage, axis = 0),
               alpha = 0.4, color = 'grey')

ax1.set_ylim(0,1)
ax1.set_xlim(0, np.max(time_points_in_days))
ax.set_xlim(0, np.max(time_points_in_days))
ax1.set_yticks([])
ax1.set_xlabel("Time delay (days)")
ax.set_title("Bacteria")
ax1.set_title("Phage")

plt.tight_layout()
plt.savefig("Guerrero_spacer_turnover_threshold_%s.pdf" %threshold)
```

### Align e_effective to the same value at time shift 0 (normalize to t 0)

```python
def e_effective_shifted_norm(e, nbi_interp, nvj_interp, max_shift = 1000, direction = 'past'):
    """
    Calculate the time-shifted average immunity between bacteria (nbi) and phage (nvj).
    
    Inputs:
    e : parameter e, spacer effectiveness
    nbi_interp : array of shape (timepoints, max_m) with interpolated bacteria clone abundances 
    with time going down the columns and clone identity going across rows.
    nvj_interp : array of shape (timepoints, max_m) with interpolated phage clone abundances 
    with time going down the columns and clone identity going across rows.
    max_shift : maximum time shift in index of nbi_interp or nvi_interp. 
    Max shift in generations = (interp_times[1] - interp_times[0])* max_shift
    direction : 'past' or 'future': whether to shift phages to the past or the future.

    """
    # overlap at concurrent times for entire timecourse
    e_effective_0 = np.array((e*np.sum(nbi_interp * nvj_interp, axis = 1).flatten()/
                            (np.array(np.sum(nvj_interp, axis = 1).flatten()) 
                             * np.array(np.sum(nbi_interp, axis = 1).flatten()))))

    e_eff_mean = [1]
    e_eff_std = [np.nanstd(e_effective_0) / np.nanmean(e_effective_0)]

    if direction == 'past':
        
        for i in range(1, max_shift):
            e_effective = np.array((e*np.sum(nbi_interp[i:] * nvj_interp[:-i], axis = 1).flatten()/
                                (np.array(np.sum(nvj_interp[:-i], axis = 1).flatten()) 
                                 * np.array(np.sum(nbi_interp[i:], axis = 1).flatten()))))

            # normalize to e_effective_0 for that nbi
            e_eff = e_effective / e_effective_0[i:]
            e_eff = e_eff[~np.isinf(e_eff)]
            e_eff_mean.append(np.nanmean(e_eff))
            e_eff_std.append(np.nanstd(e_eff))
        
    if direction == 'future':
        for i in range(1, max_shift):
            e_effective = np.array((e*np.sum(nbi_interp[:-i] * nvj_interp[i:], axis = 1).flatten()/
                                (np.array(np.sum(nvj_interp[i:], axis = 1).flatten()) 
                                 * np.array(np.sum(nbi_interp[:-i], axis = 1).flatten()))))
            
            # normalize to e_effective_0 for that nbi
            e_eff = e_effective / e_effective_0[:-i]
            e_eff = e_eff[~np.isinf(e_eff)]
            e_eff_mean.append(np.nanmean(e_eff))
            e_eff_std.append(np.nanstd(e_eff))
            
    return np.array(e_eff_mean), np.array(e_eff_std)
```

```python
fig, ax = plt.subplots(figsize = (10,8))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
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
    avg_immunity_past, avg_immunity_past_std = e_effective_shifted_norm(1, bac_interp.T, phage_interp.T, 
                                                                   max_shift = len(time_min_spacing), direction = 'past')
    avg_immunity_future, avg_immunity_future_std = e_effective_shifted_norm(1, bac_interp.T, phage_interp.T, 
                                                                   max_shift = len(time_min_spacing), direction = 'future')



    avg_immunity_mean = np.concatenate([avg_immunity_past, avg_immunity_future])
    avg_immunity_std = np.concatenate([avg_immunity_past_std, avg_immunity_future_std])
    # this is the number of generations per day based on 100-fold serial dilution and exponential growth
    times = np.concatenate([-(time_min_spacing - time_min_spacing[0]), time_min_spacing - time_min_spacing[0]] )
    avg_immunity_mean = avg_immunity_mean[np.argsort(times)]
    avg_immunity_std = avg_immunity_std[np.argsort(times)]
    times = times[np.argsort(times)]

    
    ax.scatter(times, avg_immunity_mean/avg_immunity_future[0],  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)/avg_immunity_future[0], 
                    y2 = (avg_immunity_mean + avg_immunity_std)/avg_immunity_future[0], 
                    color = colours[i], alpha = 0.1)

#ax.set_xlim(-1550, 1550)
#ax.set_ylim(0, 1)
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Guerrero_both_cr_start_trim_%s_end_trim_%s_%s_PAM_wt_%s_phage_only_%s_normed.pdf" %(start_ind, 
                                                                            np.abs(stop_ind) - 1, pam, wild_type, phage_only))#
```

Make a grid of the start and end trim amounts

```python
#thresholds_subset = np.arange(0.01, 0.16, 0.02)
thresholds_subset = [0.01, 0.05, 0.1, 0.15]
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

max_start_trim = 4
max_end_trim = 4

skip = 7

start_trims = np.arange(0,max_start_trim)*skip
end_trims = np.arange(0,max_end_trim)*skip

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
            grouping = ['type_%s' %(1-threshold), 'crispr']
            df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                                    wild_type, phage_only), index_col = 0)
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

            axs[i,j].scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[k], 
                       label = "%s" %int((1-threshold)*100) + r"%", s = 15)
            axs[i,j].fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                            y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                            color = colours[k], alpha = 0.1)

        axs[i,j].set_xlim(-500, 500)
        
        if pam == "perfect":
            axs[i,j].set_ylim(0.0, 0.7)
            ymax = 0.6
            x = 70
        elif pam == "no":
            axs[i,j].set_ylim(0.0, 1.7)
            ymax = 1.5
            x=100
                
        axs[i,j].axvline(0, linestyle = ':', color = 'k')
        
        axs[i,j].annotate("Trim %s from start,\n%s from end" %(start_ind, end),
               xy = (x, ymax), xycoords = 'data', fontsize = 8)

for i in range(max_start_trim):
    axs[i,0].set_ylabel("Average overlap between\nbacteria and phage")
    axs[i,1].set_yticks([])
    axs[i,2].set_yticks([])
    axs[i,3].set_yticks([])
for j in range(max_end_trim):    
    axs[-1,j].set_xlabel("Time shift (days)")
    axs[0,j].set_xticks([])
    axs[1,j].set_xticks([])
    axs[2,j].set_xticks([])
    
axs[-1,-1].legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,0.85), fontsize = 8)
plt.tight_layout()
plt.savefig("Time_shift_Guerrero_both_cr_%s_PAM_wt_%s_phage_only_%s.pdf" %(pam, wild_type, phage_only))
plt.savefig("Time_shift_Guerrero_both_cr_%s_PAM_wt_%s_phage_only_%s.png" %(pam, wild_type, phage_only), dpi = 100)
```

ADD TO SI:
- [x] plot for all the thresholds with all the data
- [x] reproduction of the total population size plot
- [x] plots of total types 

```python
## plot for paper
#thresholds_subset = np.arange(0.01, 0.17, 0.02)
#thresholds_subset = np.concatenate([[0.01], np.arange(0.03, 0.16, 0.03)])
thresholds_subset = [0.15]
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

# trims: none, cut off high part, cut off low part, cut off both
# high part starts at 13, ends at index 20 (start from 21)
# low part starts at index 43 (do to -17)

start_ind = 0
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
step = 14
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

fig, ax = plt.subplots(figsize = (4.7, 3.5))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
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

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.05)


ax.set_ylim(0,1)
ax.set_xlim(-800, 800)
#ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Guerrero_start_trim_%s_end_trim_%s_%s_PAM_wt_%s_phage_only_%s.pdf" %(start_ind, 
                                                                                   np.abs(stop_ind) - 1, pam, wild_type, phage_only))
```

### T-test (or similar) for two different points in the distribution: 0, +- 500

```python
from spacer_model_plotting_functions import find_nearest
```

```python
i = find_nearest(time_min_spacing, 500)
```

```python
len(e_effective_past)
```

```python
e_effective_0 = np.array((e*np.sum(nbi_interp * nvj_interp, axis = 1).flatten()/
                        (np.array(np.sum(nvj_interp, axis = 1).flatten()) 
                         * np.array(np.sum(nbi_interp, axis = 1).flatten()))))

e_effective_past = np.array((e*np.sum(nbi_interp[i:] * nvj_interp[:-i], axis = 1).flatten()/
                    (np.array(np.sum(nvj_interp[:-i], axis = 1).flatten()) 
                     * np.array(np.sum(nbi_interp[i:], axis = 1).flatten()))))

e_effective_future = np.array((e*np.sum(nbi_interp[:-i] * nvj_interp[i:], axis = 1).flatten()/
                            (np.array(np.sum(nvj_interp[i:], axis = 1).flatten()) 
                             * np.array(np.sum(nbi_interp[:-i], axis = 1).flatten()))))
```

```python
fig, ax = plt.subplots()
ax.hist(e_effective_0, bins = np.arange(0,0.01, 0.00005), alpha = 0.5, label = "Zero delay")
ax.hist(e_effective_future, bins = np.arange(0,0.01, 0.00005), alpha = 0.5, label = "500 days future")
ax.hist(e_effective_past, bins = np.arange(0,0.01, 0.00005), alpha = 0.5, label = "500 days past")

ax.set_xlabel("Average immunity")
ax.set_ylabel("Count")
ax.set_xlim(0,0.0013)
ax.legend()

plt.tight_layout()
plt.savefig("Guerrero_avg_immunity_distribution_comparison.pdf")
```

```python
from scipy.stats import wilcoxon
```

```python
# compare e_effective_0[i:] with e_effective_past - each bacteria abundance matched with a time shift
wilcoxon(e_effective_0[i:], e_effective_past, alternative = 'greater')
```

```python
wilcoxon(e_effective_0[:-i], e_effective_past, alternative = 'greater')
```

```python
wilcoxon(e_effective_0[:-i], e_effective_future, alternative = 'greater')
```

```python
wilcoxon(e_effective_0[i:], e_effective_future, alternative = 'greater')
```

### Permutation test

How about trying a permutation test: what happens if we shuffle time points?

```python
thresholds_subset = [0.15]
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

# trims: none, cut off high part, cut off low part, cut off both
# high part starts at 13, ends at index 20 (start from 21)
# low part starts at index 43 (do to -17)

start_ind = 0
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
step = 14
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

fig, ax = plt.subplots(figsize = (4.7, 3.5))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
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
    
    # randomize time points
    np.random.shuffle(nbi_interp)
    np.random.shuffle(nvj_interp)
    
    e=1

    # bac_interp and phage_interp are the same as nbi and nvi, just the shape is transposed
    avg_immunity_past, avg_immunity_past_std = e_effective_shifted(1, nbi_interp, nvj_interp, 
                                                                   max_shift = len(time_min_spacing), direction = 'past')
    avg_immunity_future, avg_immunity_future_std = e_effective_shifted(1, nbi_interp, nvj_interp, 
                                                                   max_shift = len(time_min_spacing), direction = 'future')



    avg_immunity_mean = np.concatenate([avg_immunity_past, avg_immunity_future])
    avg_immunity_std = np.concatenate([avg_immunity_past_std, avg_immunity_future_std])
    # this is the number of generations per day based on 100-fold serial dilution and exponential growth
    times = np.concatenate([-(time_min_spacing - time_min_spacing[0]), time_min_spacing - time_min_spacing[0]] )
    avg_immunity_mean = avg_immunity_mean[np.argsort(times)]
    avg_immunity_std = avg_immunity_std[np.argsort(times)]
    times = times[np.argsort(times)]

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.05)


ax.set_ylim(0,1)
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Guerrero_time_shift_bootstrapped.pdf")
```

```python
# shuffle time points but keep pairs together (i.e. shuffle nbi and nvi with the same randomizer)

thresholds_subset = [0.15]
colours = cm.cividis(np.linspace(0,0.7, len(thresholds_subset)))[::-1]

# trims: none, cut off high part, cut off low part, cut off both
# high part starts at 13, ends at index 20 (start from 21)
# low part starts at index 43 (do to -17)

start_ind = 0
stop_ind = -1 # cut off the last np.abs(stop_ind) -1 points
step = 14
#step = np.mean(np.diff(time_points_in_days))
#step = 1
# remove the first and last time point for interpolation - no shared types on the first day
time_min_spacing = np.arange(time_points_in_days[start_ind], time_points_in_days[stop_ind], step)

fig, ax = plt.subplots(figsize = (4.7, 3.5))

for i, threshold in enumerate(thresholds_subset):
    grouping = ['type_%s' %(1-threshold), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
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

    # shuffle time points
    inds = np.arange(0, len(time_min_spacing),1)
    np.random.shuffle(inds)
    
    # randomize time points
    nbi_interp = nbi_interp[inds]
    nvj_interp = nvj_interp[inds]
    
    e=1

    # bac_interp and phage_interp are the same as nbi and nvi, just the shape is transposed
    avg_immunity_past, avg_immunity_past_std = e_effective_shifted(1, nbi_interp, nvj_interp, 
                                                                   max_shift = len(time_min_spacing), direction = 'past')
    avg_immunity_future, avg_immunity_future_std = e_effective_shifted(1, nbi_interp, nvj_interp, 
                                                                   max_shift = len(time_min_spacing), direction = 'future')



    avg_immunity_mean = np.concatenate([avg_immunity_past, avg_immunity_future])
    avg_immunity_std = np.concatenate([avg_immunity_past_std, avg_immunity_future_std])
    # this is the number of generations per day based on 100-fold serial dilution and exponential growth
    times = np.concatenate([-(time_min_spacing - time_min_spacing[0]), time_min_spacing - time_min_spacing[0]] )
    avg_immunity_mean = avg_immunity_mean[np.argsort(times)]
    avg_immunity_std = avg_immunity_std[np.argsort(times)]
    times = times[np.argsort(times)]

    
    ax.scatter(times, avg_immunity_mean*num_protospacers,  marker = 'o', color = colours[i], 
               label = "%s" %int((1-threshold)*100) + r"% similarity")
    ax.fill_between(times, y1 = (avg_immunity_mean - avg_immunity_std)*num_protospacers, 
                    y2 = (avg_immunity_mean + avg_immunity_std)*num_protospacers, 
                    color = colours[i], alpha = 0.05)


ax.set_ylim(0,1)
ax.legend(ncol = 2, loc = 'upper right', bbox_to_anchor = (1,1) )
ax.axvline(0, linestyle = ':', color = 'k')
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Guerrero_time_shift_bootstrapped_matched_pairs.pdf")
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
ax.set_xlabel("Time shift (days)")
ax.set_ylabel("Average overlap between\nbacteria and phage")
plt.tight_layout()
plt.savefig("Time_shift_Banfield_all_groups_small_start_trim_%s_end_trim_%s_%s_PAM_wt_%s_threshold_%s_presentation.png" %(start_ind, 
                                                                                   np.abs(stop_ind) - 1, pam, wild_type, threshold), dpi = 300)
```

```python

```

In my model for accounting for bacterial array length, the number of spacers in total (44) means that average immunity works out to 1 regardless of the starting value using my toy model theory.

It's possible that the effective array length is much shorter in that there aren't protospacers for a bunch of older spacers. Can I see this in the data?

```python
cr = "CR3"
datapath = "/media/madeleine/My Passport1/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
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
phage_DS92_pop_size = pd.read_csv("results/2022-03-30/phage_DS-92_coverage.csv", sep = ",", names = ('time_point', 'coverage') )
phage_DC56_pop_size = pd.read_csv("results/2022-03-30/phage_DC-56_coverage.csv", sep = ",", names = ('time_point', 'coverage') )
bac_pop_size = pd.read_csv("results/2022-03-30/Gordonia_MAG_coverage.csv", sep = ",", names = ('time_point', 'coverage') )
```

```python
phage_DS92_pop_interp = interp1d(phage_DS92_pop_size['time_point'], phage_DS92_pop_size['coverage'], fill_value = 'extrapolate')
phage_DS92_pop_size_interp = phage_DS92_pop_interp(time_points + 1)

phage_DC56_pop_interp = interp1d(phage_DC56_pop_size['time_point'], phage_DC56_pop_size['coverage'], fill_value = 'extrapolate')
phage_DC56_pop_size_interp = phage_DC56_pop_interp(time_points + 1)

bac_pop_interp = interp1d(bac_pop_size['time_point'], bac_pop_size['coverage'], fill_value = 'extrapolate')
bac_pop_size_interp = bac_pop_interp(time_points + 1)
```

```python
fig, ax = plt.subplots(figsize = (8,3))

ax.plot(phage_DS92_pop_size['time_point'], phage_DS92_pop_size['coverage'], marker = 'o', 
       color = 'yellow', label = "Phage DS-92", alpha = 0.6)
ax.plot(phage_DC56_pop_size['time_point'], phage_DC56_pop_size['coverage'], marker = 'o', 
       color = 'green', label = "Phage DC-56", alpha = 0.6)

ax.plot(bac_pop_size['time_point'], bac_pop_size['coverage'], marker = 'o', 
       color = 'k', label = "Gordonia", alpha = 0.6)
#ax.plot(bac_pop_size_MOI2['Date'], bac_pop_size_MOI2['Bacteria population size'], marker = 'o',
#       color = 'lightseagreen', label = "Bacteria", alpha = 0.6)
#ax.plot(time_points_in_days, bac_pop_size_interp, marker = 's', linestyle = 'None', color = 'darkcyan',
#       mec = 'k', label = "Interpolated number at sequencing date")


ax.legend()
ax.set_xlabel("Time point")
ax.set_ylabel("Coverage")
#ax.set_yscale('log')
#ax.set_xlim(-10, 250)
plt.savefig("Guerrero_coverage.pdf")
```

```python
# coverage correlating with total reads by my detection?

fig, axs = plt.subplots(2,1, figsize = (6,6))

ax = axs[0]
ax1 = axs[1]

x_phage = total_reads_df.sort_values(by = 'time_point')['num_phage_reads_final']
x_bac = total_reads_df.sort_values(by = 'time_point')['num_bac_reads_final']
ax.scatter(x_phage, phage_DC56_pop_size_interp + phage_DS92_pop_size_interp,
          alpha = 0.7)
ax1.scatter(x_bac, bac_pop_size_interp, 
           alpha = 0.7)

phage_r, phage_p = pearsonr(x_phage, 
                            phage_DC56_pop_size_interp + phage_DS92_pop_size_interp,)
phage_fit = linregress(x_phage, phage_DC56_pop_size_interp + phage_DS92_pop_size_interp,)

ax.plot(x_phage, x_phage*phage_fit.slope +phage_fit.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=$" %round(phage_r, 2) + "{:.1e}".format(phage_p))

bac_r, bac_p = pearsonr(x_bac,  bac_pop_size_interp,)
bac_fit = linregress(x_bac, bac_pop_size_interp,)

ax1.plot(x_bac, x_bac*bac_fit.slope +bac_fit.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=$" %round(bac_r, 2) + "{:.1e}".format(bac_p))

ax.set_xlabel("Number of matched phage reads")
ax1.set_xlabel("Number of matched bacteria reads")
ax.set_ylabel("Phage coverage reported in study")
ax1.set_ylabel("Bacteria coverage reported in study")

ax.legend()
ax1.legend()

plt.tight_layout()
plt.savefig("Guerrero_coverage_vs_detected_reads.pdf")
```

### Does average immunity correlate with population size?

```python
from spacer_model_plotting_functions import effective_e
```

```python
thresholds = np.arange(0.01, 0.16, 0.01)
#colours = cm.viridis(np.linspace(0,0.9, len(effective_e_list[1:])))
```

```python
for threshold in thresholds:
    fig, ax = plt.subplots(1,1, figsize = (4,3))
    
    grouping = ['type_%s' %(round(1-threshold,3)), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                            wild_type, phage_only), index_col = 0)
    bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)
    effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)
    
    print("Mean: %s, std: %s, ratio: %s" %(round(np.nanmean(effective_e_list), 5), round(np.nanstd(effective_e_list), 5), 
                                           round(np.nanstd(effective_e_list) / np.nanmean(effective_e_list), 3)))
    
    ax.plot(time_points_in_days, effective_e_list*num_protospacers, marker = 'o', color = 'k',
           label = "Data")
    
    a_r, a_p = pearsonr(time_points_in_days, effective_e_list*num_protospacers)
    avg_immunity_fit = linregress(time_points_in_days, effective_e_list*num_protospacers)

    ax.plot(time_points_in_days, time_points_in_days*avg_immunity_fit.slope +avg_immunity_fit.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=%s$" %(round(a_r, 2), round(a_p,4)))
    
    ax.set_xlabel("Time point (days)")
    ax.set_ylabel("Average immunity")
    
    ax.set_ylim(-0.1,4)
    
    ax.legend()
    
    plt.tight_layout()
    
    plt.savefig("average_immunity_vs_time_grouping_%s_%s_PAM_wt_%s_phage_only_%s.pdf" %(round(1-threshold,3), pam, wild_type, phage_only))
    plt.close()
```

```python
np.nanstd(effective_e_list)
```

### What's going on between data points 13 and 20 that makes average immunity so high there? 

```python
threshold = 0.15
fig, ax = plt.subplots(1,1, figsize = (4,3))
    
grouping = ['type_%s' %(round(1-threshold,3)), 'crispr']
df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                        wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)
effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)

ax.plot(time_points_in_days, effective_e_list*num_protospacers, marker = 'o', color = 'k',
       label = "Data")

a_r, a_p = pearsonr(time_points_in_days, effective_e_list*num_protospacers)
avg_immunity_fit = linregress(time_points_in_days, effective_e_list*num_protospacers)

ax.plot(time_points_in_days, time_points_in_days*avg_immunity_fit.slope +avg_immunity_fit.intercept, color = 'r', linestyle = '-',
           label = "Linear fit,\n" + r" $R=%s$, $p=%s$" %(round(a_r, 2), round(a_p,4)))

high_vals = effective_e_list[effective_e_list*num_protospacers > 2].index

ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')

ax.set_xlabel("Time point (days)")
ax.set_ylabel("Average immunity")

ax.set_ylim(-0.1,4)

ax.legend()

plt.tight_layout()
plt.savefig("average_immunity_vs_time_highlighted_grouping_%s_%s_PAM_wt_%s_phage_only_%s.png" %(round(1-threshold,3), pam, wild_type, phage_only), dpi = 200)
```

```python
high_vals = effective_e_list[effective_e_list*num_protospacers > 1].index
```

I don't see anything overtly weird in the total number of types, the number of shared types, the coverage, etc. Maybe it's like both phage and bacteria population size are low there?

```python
fig, ax = plt.subplots()

ax.plot(time_points_in_days, np.sum(bac_wide_filtered.iloc[:,:-1], axis = 0), label = "Bacteria")
ax.plot(time_points_in_days, np.sum(phage_wide_filtered.iloc[:,:-1], axis = 0), label = "Phage")

ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')

ax.set_ylabel("Total spacers or protospacers")
ax.set_xlabel("Days")
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig("total_spacers_highlighted.png", dpi = 200)
```

```python
fig, axs = plt.subplots(2,2, figsize = (10,7))

axs[0,0].plot(time_points_in_days, np.sum(bac_wide.iloc[:-1,:-1], axis = 0), label = "Total spacers")
axs[0,0].plot(time_points_in_days, np.sum(phage_wide.iloc[:-1,:-1], axis = 0), label = "Total protospacers")
axs[0,1].plot(time_points_in_days, np.sum(bac_wide.iloc[:-1,:-1] * phage_wide.iloc[:-1,:-1], axis = 0), label = "Numerator")
axs[1,1].plot(time_points_in_days, (np.sum(bac_wide.iloc[:-1,:-1], axis = 0) * np.sum(phage_wide.iloc[:-1,:-1], axis = 0)), label = "Denominator")
axs[1,0].plot(time_points_in_days, effective_e_list, label = "Average immunity")

for ax in axs.flatten():
    ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
    ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')

    ax.legend()
    
#ax.set_yscale('log')

plt.tight_layout()

plt.savefig("average_immunity_components.png", dpi = 200)
```

```python
fig, ax = plt.subplots(figsize = (8,3))

ax.plot(time_points_in_days, phage_DS92_pop_size_interp, marker = 'o', 
       color = 'yellow', label = "Phage DS-92", alpha = 0.6)
ax.plot(time_points_in_days, phage_DC56_pop_size_interp, marker = 'o', 
       color = 'green', label = "Phage DC-56", alpha = 0.6)

ax.plot(time_points_in_days, bac_pop_size_interp, marker = 'o', 
       color = 'k', label = "Gordonia", alpha = 0.6)
#ax.plot(bac_pop_size_MOI2['Date'], bac_pop_size_MOI2['Bacteria population size'], marker = 'o',
#       color = 'lightseagreen', label = "Bacteria", alpha = 0.6)
#ax.plot(time_points_in_days, bac_pop_size_interp, marker = 's', linestyle = 'None', color = 'darkcyan',
#       mec = 'k', label = "Interpolated number at sequencing date")

ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')

ax.legend()
ax.set_yscale('log')
ax.set_xlabel("Time point")
ax.set_ylabel("Coverage")
plt.tight_layout()
plt.savefig("Guerrero_coverage_highlighted.png", dpi = 200)
```

```python
fig, axs = plt.subplots(2,1, figsize = (6,6))

ax = axs[0]
ax1 = axs[1]

ax.plot(time_points_in_days, total_reads_df.sort_values(by = 'time_point')['num_phage_reads_final'], label = "Phage")
ax.plot(time_points_in_days, total_reads_df.sort_values(by = 'time_point')['num_bac_reads_final'], label = "Bacteria")

ax1.plot(time_points_in_days, 
         total_reads_df.sort_values(by = 'time_point')['num_phage_reads_final'] / total_reads_df.sort_values(by = 'time_point')['total_reads'], 
         label = "Phage")
ax1.plot(time_points_in_days,
         total_reads_df.sort_values(by = 'time_point')['num_bac_reads_final']/ total_reads_df.sort_values(by = 'time_point')['total_reads'], 
         label = "Bacteria")


#axb.plot(total_reads_df['time_point'], total_reads_df['num_bac_reads'] )
#ax.set_yscale('log')
ax.legend()

ax1.set_xlabel("Time (days)")
ax.set_ylabel("Total reads")
ax1.set_yscale('log')
ax.set_yscale('log')
ax1.set_ylabel("Fraction of total reads")
    
for ax in axs:
    ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
    ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')

plt.tight_layout()
plt.savefig("Guerrero_read_totals_highlighted.png", dpi = 200)
```

```python
#time_points_in_days = [1, 4, 15, 65, 77, 104, 114, 121, 129, 187, 210, 224, 232]
colours = cm.viridis(np.linspace(0,1, len(thresholds)))[::-1]
fig, axs = plt.subplots(2,2, figsize = (10,8))
cutoff = 0

for i, threshold in enumerate(thresholds):
    grouping = 'type_%s' %(1-threshold)
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, wild_type, phage_only), index_col = 0)
    
    shared_types, bac_types, phage_types, tps = type_overlap(df_combined, cutoff = cutoff)
    
    # make sure tps is sorted
    if not np.all(tps[:-1] < tps[1:]):
        print("Warning: time points are not sorted! Don't use time_points_in_days")
    
    axs[0,0].plot(time_points_in_days, shared_types, 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[0,1].plot(time_points_in_days, np.array(phage_types) / np.array(bac_types), 
            label = "%s" %int((1-threshold)*100) + r"%", color = colours[i])
    
    axs[1,0].plot(time_points_in_days, bac_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
    
    axs[1,1].plot(time_points_in_days, phage_types, 
            label = "%s" %int((1-threshold)*100) + r"% similarity", color = colours[i])
    
for ax in axs.flatten():
    ax.axvline(time_points_in_days[high_vals[0]], color = 'k', linestyle = '--')
    ax.axvline(time_points_in_days[high_vals[-1]], color = 'k', linestyle = '--')
    
axs[0,0].set_ylabel("Number of shared spacer types")
axs[0,1].set_ylabel("Ratio of phage types to bacteria types")
axs[1,0].set_ylabel("Number of bacteria types")
axs[1,1].set_ylabel("Number of phage types")
axs[1,0].set_xlabel("Time (days)")
axs[1,1].set_xlabel("Time (days)")
axs[0,0].legend(ncol = 2)
axs[0,0].set_yscale('log')

plt.tight_layout()
plt.savefig("unique_types_and_overlap_all_highlighted_cutoff_%s_%s_PAM_wt_%s_phage_only_%s.png" %(cutoff, pam, wild_type, phage_only), dpi = 200)

```

### Synthetic data to explore the time shift peak

Try taking the time 0 shift average immunity, then sliding the same thing over and calculating average immunity as if that was the shift. I want to know, and I'm not sure how to do this: is the peak near 0 delay an artifact of the data structure? Would it appear without any genuine decay in average immunity if the data had a peak like that? 

Here's the idea: if each time point had the same average immunity to all past and future time points, but fewer points were included in the overall average as the delay increased, would we see a signal that looks like this based on the original average immunity shape? How would this change for different shapes? Obviously if average immunity was constant over time, the time shift would also be flat.

Another plot idea: take the average immunity with 0 delay for each point, correlate it with average immunity for the same point with a different delay - is it correlated? 

```python
threshold = 0.15
    
grouping = ['type_%s' %(round(1-threshold,3)), 'crispr']
df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                        wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)
effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)
```

```python
future_means = []
past_means = []
future_stds = []
past_stds = []

for i in range(1, len(effective_e_list)):
    future_means.append(np.nanmean(effective_e_list[:-i]))
    past_means.append(np.nanmean(effective_e_list[i:]))
    future_stds.append(np.nanstd(effective_e_list[:-i]))
    past_stds.append(np.nanstd(effective_e_list[i:]))
```

```python
fig, ax = plt.subplots()

ax.plot(time_points[1:], np.array(future_means)*num_protospacers, color = 'k')
ax.plot(-time_points[1:], np.array(past_means)*num_protospacers, color = 'k')
ax.scatter([0], [np.nanmean(effective_e_list)*num_protospacers], color = 'k')

ax.set_xlabel("Time shift")
ax.set_ylabel("Average immunity if constant in time")

plt.tight_layout()
plt.savefig("avg_immunity_Guerrero_constant_shift.pdf")
```

### Correlate average immunity delay for different time points

```python
threshold = 0.15
    
grouping = ['type_%s' %(round(1-threshold,3)), 'crispr']
df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                        wild_type, phage_only), index_col = 0)
bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)
effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)
    
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
```

```python

```

```python
shifts = [2,6,10]
effective_e_0 = effective_e(nbi_interp, nvj_interp, 
                                  all_phages=None, pv_type='binary', e=1, theta=None)*num_protospacers

fig, axs = plt.subplots(2,3, figsize = (12,10))

for i, shift in enumerate(shifts):
    effective_e_shifted_past = effective_e(nbi_interp[shift:], nvj_interp[:-shift], 
                                  all_phages=None, pv_type='binary', e=1, theta=None)*num_protospacers

    effective_e_shifted_future = effective_e(nbi_interp[:-shift], nvj_interp[shift:], 
                                  all_phages=None, pv_type='binary', e=1, theta=None)*num_protospacers
    
    axs[0, i].scatter(effective_e_0[shift:], effective_e_shifted_past) # comparing present bacteria with past phages
    
    axs[1, i].scatter(effective_e_0[:-shift], effective_e_shifted_future) # comparins present bacteria with future phages
    
    future_r, future_p = pearsonr(effective_e_0[:-shift], effective_e_shifted_future)
    past_r, past_p = pearsonr(effective_e_0[shift:], effective_e_shifted_past)
    
    future_fit = linregress(effective_e_0[:-shift], effective_e_shifted_future)
    past_fit = linregress(effective_e_0[shift:], effective_e_shifted_past)

    axs[0,i].plot(effective_e_0[shift:],effective_e_0[shift:]*past_fit.slope +past_fit.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=$" %round(past_r, 2) + "{:.1e}".format(past_p))
    
    axs[1,i].plot(effective_e_0[:-shift],effective_e_0[:-shift]*future_fit.slope +future_fit.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=$" %round(future_r, 2) + "{:.1e}".format(past_p))
    
    axs[0,i].legend()
    axs[1,i].legend()
    
    axs[0,i].set_title("Time shift = %s weeks" %(shift*2))
    
axs[1,1].set_xlabel("Zero shift average immunity")
axs[0,0].set_ylabel("Time shifted average immunity (past)")
axs[1,0].set_ylabel("Time shifted average immunity (future)")
    
plt.tight_layout()

plt.savefig("time_shift_correlations.pdf")
```

Note that we could do all these plots from the phage perspective as well; we expect them to be similar but they need not be in general (see Koskella papers).

Basically taking phage as the starting point and shifting bacteria to the past or the future. 


## Correlation of average immunity and population size

```python
phage_pop_size_interp = phage_DS92_pop_size_interp + phage_DC56_pop_size_interp
```

```python
log_phage = True

for threshold in thresholds:

    grouping = ['type_%s' %(round(1-threshold,3)), 'crispr']
    df_combined = pd.read_csv("%s/Guerrero_data_combined_type_%s_wt_%s_phage_only_%s.csv" %(folder,1-threshold, 
                                                                                            wild_type, phage_only), index_col = 0)
    bac_wide_filtered, phage_wide_filtered = Guerrero_to_array(df_combined, grouping)

    effective_e_list = effective_e(bac_wide_filtered.iloc[:,:-1].T, phage_wide_filtered.iloc[:,:-1].T, all_phages=None, pv_type='binary', e=1, theta=None)
    colours = cm.viridis(np.linspace(0,0.9, len(effective_e_list[1:])))
    
    fig, axs = plt.subplots(2,1, figsize = (5,7))

    axs[0].scatter(effective_e_list[1:]*num_protospacers, phage_pop_size_interp[1:], c = colours, s = 60)
    axs[1].scatter(effective_e_list[1:]*num_protospacers, bac_pop_size_interp[1:], c = colours, s = 60)
    
    for i, e_eff in enumerate(effective_e_list[1:]):
        axs[0].plot(effective_e_list[1+i:3+i]*num_protospacers, phage_pop_size_interp[1+i:3+i], color = colours[i], linestyle = '--', linewidth = 1, alpha = 0.5)
        axs[1].plot(effective_e_list[1+i:3+i]*num_protospacers, bac_pop_size_interp[1+i:3+i], color = colours[i], linestyle = '--', linewidth = 1, alpha = 0.5)

    # calculate pearson r
    # remove NaNs first
    e_pearson_list = effective_e_list[~np.isnan(effective_e_list)]
    phage_pearson = phage_pop_size_interp[~np.isnan(effective_e_list)]
    bac_pearson = bac_pop_size_interp[~np.isnan(effective_e_list)]
    
    if log_phage == True:
        p_r, p_p = pearsonr(e_pearson_list*num_protospacers, np.log(phage_pearson))
        result_phage = linregress(e_pearson_list*num_protospacers, np.log(phage_pearson))
    else:
        p_r, p_p = pearsonr(e_pearson_list*num_protospacers, phage_pearson)
        result_phage = linregress(e_pearson_list*num_protospacers, phage_pearson)
    
    b_r, b_p = pearsonr(e_pearson_list*num_protospacers, bac_pearson)
    result_bac = linregress(e_pearson_list*num_protospacers, bac_pearson)

    xvals = np.arange(np.min(effective_e_list*num_protospacers), np.max(effective_e_list*num_protospacers),0.01)
    
    if log_phage == True:
        axs[0].plot(xvals, np.exp(xvals*result_phage.slope + result_phage.intercept), color = 'r', linestyle = '-',
               label = "Linear fit to log-\ntransformed data,\n" + r" $R=%s$, $p=%s$" %(round(p_r, 2), round(p_p,4)))
        axs[0].set_yscale('log')
    else:
        axs[0].plot(xvals, xvals*result_phage.slope + result_phage.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=%s$" %(round(p_r, 2), round(p_p,4)))
    
    axs[1].plot(xvals, xvals*result_bac.slope + result_bac.intercept, color = 'r', linestyle = '-',
               label = "Linear fit,\n" + r" $R=%s$, $p=%s$" %(round(b_r, 2), round(b_p,4)))

    # for colours
    axs[0].scatter(list(effective_e_list)[1]*num_protospacers, phage_pop_size_interp[1], color = colours[0], s = 60, label = "Early time")
    axs[0].scatter(list(effective_e_list)[-1]*num_protospacers, phage_pop_size_interp[-1], color = colours[-1], s = 60, label = "Late time")   
    
    #axs[1].set_yscale('log')

    axs[0].set_ylabel("Phage coverage")
    #axs[0].set_xlabel("Same-day average immunity")
    axs[1].set_ylabel("Bacteria coverage")
    axs[1].set_xlabel("Same-day average immunity")
    
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("pop_size_vs_avg_immunity_grouping_%s_%s_PAM_wt_%s_phage_only_%s_log_phage_%s.pdf" %(round(1-threshold,3), 
                                                                                                     pam, wild_type, phage_only, log_phage))

```
