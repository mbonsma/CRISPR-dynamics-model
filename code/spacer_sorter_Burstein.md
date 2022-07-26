---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: spacer_phage
    language: python
    name: spacer_phage
---

# Burstein 2016 spacer sorter

- [ ] Load results of BLAST repeat list against metagenomic data
- [ ] Filter results to keep either high-quality matches or reads that have two matches (i.e. a spacer between two repeats)

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from Bio.Seq import Seq
from sklearn.cluster import AgglomerativeClustering # for clustering spacers
import Levenshtein # for calculating sequence similarity
```

```python
%matplotlib inline
```

```python
def get_read_from_index(subject_id, fasta_len = 400000):
    """
    Retrieves the header and sequence of a read based on the subject id sequence provided.
    Assumes reads are paired-end so that the read number is the line number / 4
    Don't use grep - just extract the line number based on the number modulo file length 
    and then do a check to make sure it matches the header - this will be faster than grep 
    Perhaps use grep if it doesn't match as a backup? Not sure if or why this would happen
    Example usage:
    get_read_from_index('SRR1658343.289344.1')
    
    """
    accession = subject_id.split(".")[0] # extract the accession number
    read_number = subject_id.split(".")[1] # extract the read number part of the subject id
    pair_number = subject_id.split(".")[2] # whether it's .1 or .2
    read_number = int(read_number) 
    pair_number = int(pair_number)

    # this returns the floor integer for the file that this read is in
    # so if the result is 0, then it's in SRR*aa, 1 is SRR*ab, etc. 
    # subtract 1 from read number because the reads start at 1 and not 0
    # the factor of 4 comes from the paired-end reads: each read number has .1 and .2 and a header line, 
    # so four lines in the fasta file per read number
    file_index = int((read_number -1 )/ (fasta_len/4))
    read_index = 4*read_number - fasta_len*file_index # line index of the read based on the number
    # if it's the first of the paired reads, subtract 2 lines from the line index
    if pair_number == 1:
        read_index -= 2

    # convert file_index to the alphabet suffix
    # ascii codes for letters a to z: numbers 97 to 122 
    
    if file_index >= 26*25: # then we're into z and a new naming system starts
        first_letter = 'z'
        new_file_index = file_index - 26*25
        second_digit = int(new_file_index / (26*25))
        third_digit = int(new_file_index / 26)
        fourth_digit = new_file_index % 26
        file_suffix = first_letter + chr(second_digit + 97) + chr(third_digit + 97) + chr(fourth_digit + 97)
    
    else:
        first_digit = int(file_index / 26)
        second_digit = file_index % 26
        file_suffix = chr(first_digit + 97) + chr(second_digit + 97)

    # construct the filename
    filename = "%s%s" %(accession, file_suffix)

    # access the read
    read = ! head -n "$read_index" "$datapath/$accession/$filename" | tail -n 2
    read_header = read[0]
    read_seq = read[1]

    # check if the extracted read matches the subject_id
    # this doesn't do anything except print a message
    # if it ever happens, perhaps use grep to search more widely
    if read_header.split(" ")[0] != ">" + subject_id:
        print("mismatched read: %s" %read_header) 
        
    return read_header, read_seq
```

```python
# there will either be 2 or 3 repeat matches
# i think it's easiest to detect if the match is reversed and then reverse the whole read and the hit locations? 
#instead of coding it all separately for each direction?

def extract_metagenome_spacers(data_subset, length_threshold = 0.85):
    
    """
    Given the BLAST output from blasting the CRISPR repeat against a fasta file, take that fasta file and the 
    BLAST output and extract the neighbouring spacers.
    
    Inputs:
    data_subset : a pandas dataframe of the BLAST results from blasting the repeat against fasta_file. This should contain 2 or three rows 
        of results for a single query and a single subject
    length_threshold : the minimum fractional spacer length that can be counted as a spacer
                      
    Returns:
    spacers : a list of all the spacers detected next to a CRISPR repeat
    read_header : the header of the matched read
    query_id : id of the repeat query that matched the read
    
    Possible improvements:
    - return the actual repeat sequence from the read as well
    - return the spacers in the order they are found in the read - not true for the 2-match cases
    """

    spacers = []

    read_header, read_seq = get_read_from_index(data_subset['subject_id'].unique()[0], fasta_len = 400000)

    read_length = len(read_seq)

    # check if reversed       
    if int(data_subset.iloc[0]['subject_start']) > int(data_subset.iloc[0]['subject_end']): # match is to reverse complement
        reverse = True    
        read_seq = str(Seq(read_seq).reverse_complement())

        # need to add 1 because of indexing change when flipping
        data_subset['final_start'] = read_length - data_subset['subject_start'] + 1       
        data_subset['final_end'] = read_length - data_subset['subject_end'] + 1

    else:
        reverse = False
        data_subset['final_start'] = data_subset['subject_start']        
        data_subset['final_end'] = data_subset['subject_end'] 
    
    # check if the repeat matches are overlapping, if so, exit and return None
    data_subset = data_subset.sort_values(by = 'final_start')
    if np.all(data_subset.iloc[1:]['final_start'].values < data_subset.iloc[:-1]['final_end'].values):
        return None

    if len(data_subset) == 2: # two matches

        # spacer in between
        spacer_start = min(data_subset['final_end'])
        spacer_end = max(data_subset['final_start'])
        spacer = read_seq[spacer_start : spacer_end - 1]
        spacers.append(spacer)

        canonical_spacer_length = len(spacer) # nothing better to go on really
        min_length = canonical_spacer_length * length_threshold

        # get spacer at start of read
        repeat_start = min(data_subset['final_start'])
        if repeat_start > min_length + 1:
            spacer_start = max(repeat_start - canonical_spacer_length - 1, 0)
            spacer = read_seq[spacer_start : repeat_start -1]
            spacers.append(spacer)

        # get spacer at end of read
        repeat_end = max(data_subset['final_end'])
        if read_length - repeat_end > min_length:
            spacer = read_seq[repeat_end : repeat_end + canonical_spacer_length]
            spacers.append(spacer)

    elif len(data_subset) == 3: 

        data_subset = data_subset.sort_values(by = 'final_start')
        for j in range(2):
            spacer_start = data_subset.iloc[j]['final_end']
            spacer_end = data_subset.iloc[j+1]['final_start']
            spacer = read_seq[spacer_start : spacer_end - 1]
            spacers.append(spacer)
    
    query_id = data_subset['query_id'].unique()[0]
    return spacers, read_header, query_id
    
```

```python
datapath = "/media/madeleine/WD_BLACK/Blue hard drive/Data/Burstein2016"
```

```python
# load detected CRISPR repeats from paper
Burstein_repeats = pd.read_csv("41467_2016_BFncomms10613_MOESM952_ESM.csv", skiprows = [0,1])
Burstein_repeats['query_id'] = Burstein_repeats['Genome bin'] + "_" + Burstein_repeats['Array location'] # make query id column
```

```python
# there are 197 genomes and 144 unique repeats
Burstein_repeats_unique = Burstein_repeats.drop_duplicates(subset = 'Repeat sequence')
```

```python
# these are the queries not included in the blast - duplicated repeats
duplicated_queries = Burstein_repeats[~Burstein_repeats['query_id'].isin(Burstein_repeats_unique['query_id'])]['query_id']
```

```python
# iterate through split fasta files and detect spacers

accession_list = ["SRR1658343", "SRR1658462", "SRR1658465", "SRR1658467", "SRR1658469", "SRR1658472"]
query_suffixes = ["1", "2", "3", "4"]
length_threshold = 0.85 # fractional alignment length to keep

i = 0

blast_data_concat = pd.DataFrame()

for accession in accession_list:
    
    blast_folder = "%s_repeat_blast_out" %accession

    all_spacers = []
    all_ids = []

    for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))): # folder with split fasta files
        for s in query_suffixes: # iterate through the repeat split files
        
            #if i > 300:
            #    break

            
            try:
                blast_data = pd.read_csv("%s/%s/%s_blast_%s.txt" %(datapath, blast_folder, fn, s), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])
            except FileNotFoundError:
                continue

            # remove duplicated queries - repeat sequence exactly the same
            blast_data = blast_data[~blast_data['query_id'].isin(duplicated_queries)]
            
            # merge repeat info and blast results
            blast_data_with_repeat = blast_data.merge(Burstein_repeats_unique, on = 'query_id')
            blast_data_with_repeat['repeat_length'] = blast_data_with_repeat['Repeat sequence'].str.len()
            blast_data_with_repeat['alignment_fraction'] = blast_data_with_repeat['alignment_length'] / blast_data_with_repeat['repeat_length']
            
            #print(len(blast_data))
            blast_data_concat = pd.concat([blast_data_concat, blast_data_with_repeat[blast_data_with_repeat['alignment_fraction'] > length_threshold]])
            
            i += 1
        
        ## ----- keep only high-quality matches ----- 
        
      
```

```python
# remove duplicates from changing the arrangement of the repeat blast files partway through
print(blast_data_concat.shape)
blast_data_concat = blast_data_concat.drop_duplicates()
print(blast_data_concat.shape)
```

```python
# count number of the same query hits for each read
count_hits_per_read = blast_data_concat.groupby(['query_id', 'subject_id'])['subject_start'].count().reset_index()

# queries and reads with multiple hits but less than 4 hits
multi_read_hits = count_hits_per_read[(count_hits_per_read['subject_start'] > 1)
                                      & (count_hits_per_read['subject_start'] < 4)]
```

On visual inspection, there look to be lots of repeat hits that overlap with each other, as well as cases of more than 3 hits. There are 6 cases with more than 3 hits; these are probably anomalous and can be removed. 

The 6 hits is a read with a bunch of repeats that all overlap each other by the same amount - I think this is not a real hit. Same for the next 5 hit one - a weird repeated region

```python
# check out the more than 3 hit cases just to be sure:
count_hits_per_read[count_hits_per_read['subject_start'] > 3]
```

```python
# test the function get_read_from_index()
get_read_from_index('SRR1658467.65987031.1', fasta_len = 400000)
```

```python
# make a column for alignment fraction * percent identity
blast_data_concat['alignment_identity'] = blast_data_concat['alignment_fraction'] * blast_data_concat['percent_identity']/100
```

```python tags=[]
# these are the blast results for multiple same-query hits to a read
blast_data_multi_hit = blast_data_concat[(blast_data_concat['query_id'] 
                                          + blast_data_concat['subject_id']).isin(multi_read_hits['query_id'] 
                                                                                  + multi_read_hits['subject_id'])]

# high quality: keep either PID * alignment fraction > threshold, 
# or multiple matches to the same repeat
blast_data_high_quality = blast_data_concat[((blast_data_concat['query_id'] 
                                          + blast_data_concat['subject_id']).isin(multi_read_hits['query_id'] 
                                                                                  + multi_read_hits['subject_id']))
                                            | (blast_data_concat['alignment_identity'] > length_threshold)]
```

```python
blast_data_multi_hit['accession'] = blast_data_multi_hit['subject_id'].str.split('.', expand = True)[0]
```

```python
blast_data_multi_hit.to_csv("burstein_blast_data.csv")
```

```python
blast_data_multi_hit = pd.read_csv("burstein_blast_data.csv", index_col = 0)
```

### Spacer sorter plan:

- [x] process blast results to get blast_data_high_quality: alignment length > 85% or repeat and has at least 2 matches to the same read. I'm keeping only 2 or more matches - this will ensure higher data quality and make sure I'm not at the end of an array taking the leader sequence or something. In my subset so far there are 3615 "high quality" matches of which 2830 are multiple query hits to the same read. This will also take care of cases where a single match is the best but there are also two pretty good matches that are probably better to keep. 
- [x] Iterate through subject_id of the blast results: in cases where multiple different repeats match the same read, take the best result? If two matches are exactly the same, then they are likely the reverse complement of each other?
    - [x] Instead of iterating through the fasta files (very slow), can I use grep or something to search the split file for the read? Identify the split file by the read ID? Each split file is 400000 lines long, and they are paired reads, so reads 1.1 to reads 100000.2 are in the split file
- [x] Extract the spacer sequence, the matched repeat sequence, the actual repeat sequence (from the read) and the read header for each match, put this in a dataframe
- [x] If there are 1 or 2 repeat matches, extract spacers from either side as well (reads are length 150 so 2 repeat matches might have space for a second or third spacer on either end - use the detected spacer length to infer)

```python
df = pd.read_csv("Burstein2016_spacers.csv", index_col = 0)
#df['subject_id'] = df['read_header'].str.split(' ', expand = True)[0].str[1:]
```

```python
# for each subject id, find the best matching query
# be careful about cases where two different queries match equally well - 
# if they really have the exact same fractional alignment and PID, they should be the exact same sequence
# also keep cases with two matches
# so keep either the single best match or a repeat with two matches?

triple_matches = []
single_matches = []
double_matches = []
length_threshold = 0.85

all_spacers = []
read_headers = []
query_ids = []

for group in tqdm(blast_data_multi_hit.groupby(['subject_id'])):
    subject_id = group[0]
    data = group[1]
    
    if np.sum(df['subject_id'].isin([subject_id])) > 0: # then this subject_id has already been analyzed
        continue

    # the query with the highest average alignment identity for all matches
    # guaranteed to be at least two matches since we removed single matches
    # otherwise do it the old way
    keep_query = data.groupby('query_id')['alignment_identity'].mean().reset_index().sort_values(
        by = 'alignment_identity', ascending = False).iloc[0]['query_id']
    
    # extract spacers from data subset
    data_subset = data[data['query_id'] == keep_query]
    
    results = extract_metagenome_spacers(data_subset, length_threshold = length_threshold)
    
    if results != None:
        spacers, read_header, query_id = results
    
        all_spacers+= spacers
        read_headers += [read_header]*len(spacers) 
        query_ids += [query_id]*len(spacers)
```

```python
new_df = pd.DataFrame()
new_df['spacer_sequence'] = all_spacers
new_df['read_header'] = read_headers
new_df['query_id'] = query_ids
```

```python
new_df['accession'] = new_df['read_header'].str.split('.', expand = True)[0].str.split('>', expand = True)[1]
new_df['subject_id'] = new_df['read_header'].str.split(' ', expand = True)[0].str[1:]


```

```python
result = pd.concat([df, new_df]).reset_index()
result = result.drop("index", axis = 1)

result = result.drop_duplicates()

result.to_csv("Burstein2016_spacers.csv")
```

```python
# spacer abundance histogram
fig, ax = plt.subplots()
ax.hist(df['spacer_sequence'].value_counts(), bins = 30)
ax.set_yscale('log')
```

## Cluster spacers into types

```python
df = pd.read_csv("Burstein2016_spacers.csv", index_col = 0)
```

```python
# create distance matrix using the Levenshtein similarity ratio

# make a distance matrix for each query separately
for group in tqdm(df.groupby('query_id')):
    query_id = group[0]
    data = group[1]
    
    unique_spacers = data['spacer_sequence'].dropna().unique()

    distance_matrix = np.zeros((len(unique_spacers), len(unique_spacers)))

    for i, spacer in enumerate(unique_spacers):
        for j, spacer in enumerate(unique_spacers):
            distance_matrix[i,j] = 1 - Levenshtein.ratio(str(unique_spacers[i].strip()), str(unique_spacers[j].strip()))
            
    # matrix is triangular, make it symmetric
    # since the diagonals are zero, don't need to worry about double-counting them
    distance_matrix = distance_matrix + distance_matrix.T

    distance_matrix = np.array(distance_matrix, dtype = 'float16') # make array smaller by reducing float size
    # float16 has max +- 65500, ~4 decimal places, good enough for this purpose: np.finfo(np.float16)
    np.savez_compressed("burstein_distance_matrix_%s" %query_id, distance_matrix)        
    
    # save list to file to associate with distance matrix
    with open ('burstein_unique_spacers_%s.txt' %query_id, 'w') as f:
        for seq in unique_spacers:
            f.write(str(seq.strip()) + "\n")
```

```python
df_list = []
threshold=0.15

for group in tqdm(df.groupby('query_id')):
    query_id = group[0]
    distance_matrix = np.load('burstein_distance_matrix_%s.npz' %query_id)['arr_0']
    with open ('burstein_unique_spacers_%s.txt' %query_id, 'r') as f:
        unique_spacers_list = f.readlines()
    
    unique_spacers = []
    for sp in unique_spacers_list:
        unique_spacers.append(sp.rstrip())
        
    spacer_types = pd.DataFrame()
    spacer_types['spacer_sequence'] = unique_spacers
    spacer_types['query_id'] = query_id
    
    # agglomerative clustering
    if len(unique_spacers) > 1:
        fit = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage='average',
                             affinity='precomputed').fit(distance_matrix)
        spacer_types['type'] = fit.labels_
    else:
        spacer_types['type'] = 0
    
    df_list.append(spacer_types)
```

```python
spacer_types = pd.concat(df_list)
```

```python
df = df.merge(spacer_types, on = ['spacer_sequence', 'query_id'])
```

```python
df.to_csv("Burstein2016_spacers_with_type_threshold_%s.csv" %threshold)
```

```python
df.groupby(['query_id', 'type'])['spacer_sequence'].count().reset_index().sort_values(by = 'spacer_sequence', ascending = False)
```

```python
unique_sequences_by_time = df.groupby('accession')['spacer_sequence'].unique().reset_index()
```

```python
# unique spacers from time point 1
print(len(unique_sequences_by_time.iloc[0]['spacer_sequence']))

# unique spacers from time point 2
print(len(unique_sequences_by_time.iloc[1]['spacer_sequence']))

# unique spacers from time point 3
print(len(unique_sequences_by_time.iloc[2]['spacer_sequence']))

# unique spacers shared between first two time points
print(len(set(unique_sequences_by_time.iloc[0]['spacer_sequence']) & set(unique_sequences_by_time.iloc[1]['spacer_sequence'])))
```
