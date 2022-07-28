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

# Spacer finder script for Guerrero2021 data


## Processing steps:

1. **Divide data file into smaller chunks for analysis**
- unzip fasta file of reads for each time point
- split fasta file into smaller files of 10^6 lines each: `split -l 1000000 SRR1873837.fasta SRR1873837`
- move split files into folder with same name as accession number

2. **Blast reads against CRISPR repeat**
- create folder "accession_repeat_blast_out"
- run code cell under header "[run BLAST to search for matches to repeat](#blast_cr1_repeat)"

3. **Detect spacers based on reads that match repeat**
- run code cell under header "[extract spacers](#extract_spacers)"

4. **[Blast reads against bacteria genome](#detect_protospacers)**

5. **[Blast reads against phage genomes](#phage_genome_blast)**

6. **[Cluster spacer sequences to assign type label](#cluster_spacers)**

7. **Blast all detected spacers against phage genome
- [get a list of all](#blast_protospacers) unique spacer sequences to use as queries
- [run blast on niagara](#blast_protospacers_niagara)

8. **Extract protospacers**
- [get a list of all reads](#get_bac_reads) that either match the bacteria genome or the CRISPR repeat
- run `process_protospacers_niagara.py` on scinet


This data is going to have a lot more overlap between paired ends by the looks of things. Many spot lengths are less than 500, which implies overlap, and anecdotally I see lots of overlap in extracted spacers so far. 

```python
from Bio.Blast.Applications import NcbiblastnCommandline
#help(NcbiblastnCommandline)

from Bio.Seq import Seq
from Bio import SeqIO
import os, math
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import Levenshtein # for calculating sequence similarity
from sklearn.cluster import AgglomerativeClustering # for clustering spacers
```

```python
def extract_spacers(fasta_file, blast_data, read_len = 250, spacer_len = 32, similarity_threshold = 0.85, progress_bar = True):
    """
    Given the BLAST output from blasting the CRISPR repeat against a fasta file, take that fasta file and the 
    BLAST output and extract the neighbouring spacers.
    
    Note: this does NOT return them in the same order they occur in in the read: the middle spacers are extracted first,
    followed by the first outside spacer and the second outside spacer, if applicable. This ordering is always the same, unless
    either the first or second outside spacer is too short to include, so it is possible to reconstruct the order after the fact
    if desired.
    
    For Gordonia, with repeat length 29, spacer length 32, and read length 250, we expect max 4 matches
    
    Inputs:
    fasta_file : the fasta file of reads corresponding to blast_data, i.e. SRR1873863.fasta. This is assumed to have reads
                      of length 250. 
    blast_data : a pandas dataframe of the BLAST results from blasting the repeat against fasta_file
    read_len : expected read length (default 250) to deal with edge effects
    spacer_len : the expected spacer length - the shortest spacer will be length >= similarity_threshold * spacer_len, 
                      the longest spacer will be length spacer_len unless the distance between two repeats on the 
                      same read is longer. In findRepeatCRISPR.py from the paper they assume spacer length = 30. Based on one 
                      of the reference genomes, I conclude it's about 32 (for CR1) and about 35 (for CR2).
                      I looked at a few middle-spacers in the extracted data, and their lengths seem quite variable but the mean
                      was still 35 for CR2.
                      
    Returns:
    spacers : a list of all the spacers detected next to a CRISPR repeat
    """
    
    min_length = round(spacer_len * similarity_threshold)
    
    reverse = False

    spacers = []
    ids = []

    if progress_bar == True:
        wrapper = tqdm
    if progress_bar == False:
        def wrapper(arg):
            return arg
    
    hit_ids = list(blast_data['subject_id'])
    
    for record in wrapper(SeqIO.parse(fasta_file, "fasta")):
        
        if record.id not in hit_ids:
            continue
        
        data = blast_data[blast_data['subject_id'] == record.id].sort_values(by = 'subject_start') # select matches from blast result

        read_len = len(record.seq) # use actual read length
        
        if len(data) == 1: # only one match

            subject_start = int(data['subject_start'])
            subject_end = int(data['subject_end'])

            # check if reversed
            if subject_start > subject_end: # match is to reverse complement
                reverse = True 
            else:
                reverse = False

            start = min(subject_start, subject_end)
            end = max(subject_start, subject_end)

            repeat = record.seq[start -1 : end]

            # get the spacer at the start of the read
            if start > min_length + 1: 
                spacer_start = max(start - spacer_len - 1, 0)
                spacer = record.seq[spacer_start : start -1]
                if reverse == True:
                    spacers.append(spacer.reverse_complement())
                else:
                    spacers.append(spacer)

                ids.append(str(record.id) + "_0")
                
            # get the spacer at the end of the read
            if read_len - end > min_length:
                spacer = record.seq[end : end + spacer_len]
                if reverse == True:
                    spacers.append(spacer.reverse_complement())
                else:
                    spacers.append(spacer)
                
                ids.append(str(record.id) + "_1")
        
        elif len(data) > 1:
        
            # get middle spacers
            if np.all(data['subject_start'] > data['subject_end']): # all reversed
                spacer_starts = list(data['subject_start'])[:-1]
                spacer_ends = list(data['subject_end'])[1:]
                reverse = True
            elif np.all(data['subject_start'] < data['subject_end']): # all forward
                spacer_starts = list(data['subject_end'])[:-1]
                spacer_ends = list(data['subject_start'])[1:]
                reverse = False
            else:
                print("mixture of directions: %s, %s" %(fasta_file, record.id)) # can't parse this properly, skip it
                continue

            # get middle spacers
            for i, spacer_start in enumerate(spacer_starts):
                if reverse == True: # reverse complement the matches
                    spacer = record.seq[spacer_start : spacer_ends[i] - 1].reverse_complement()
                elif reverse == False:
                    spacer = record.seq[spacer_start : spacer_ends[i] - 1]

                spacers.append(spacer)

                ids.append(str(record.id) + "_%s" %i)
        
            # get spacers at start and end, if long enough
            
            # spacer at start
            start = min(data.iloc[0]['subject_start'], data.iloc[0]['subject_end'])         
            
            if start > min_length + 1: # long enough
                spacer_start = max(start - spacer_len - 1, 0)
                spacer = record.seq[spacer_start : start -1]
                if reverse == True:
                    spacers.append(spacer.reverse_complement())
                else:
                    spacers.append(spacer)

                ids.append(str(record.id) + "_%s" %(i+1))
            
            # spacer at end
            end = max(data.iloc[-1]['subject_start'], data.iloc[-1]['subject_end'])   
            
            if read_len - end > min_length:
                spacer = record.seq[end : end + spacer_len]
                if reverse == True:
                    spacers.append(spacer.reverse_complement())
                else:
                    spacers.append(spacer)
                
                ids.append(str(record.id) + "_%s" %(i+2))
            
    return spacers, ids
```

```python
def extract_protospacer_from_read(data, read_seq):
    protospacers = []
    possible_PAMs_left = []
    possible_PAMs_right = []
    
    for i, row in data.iterrows():
        subject_start = int(row['subject_start'])
        subject_end = int(row['subject_end'])

        # check if reversed
        if subject_start > subject_end: # match is to reverse complement
            reverse = True 
        else:
            reverse = False

        start = min(subject_start, subject_end)
        end = max(subject_start, subject_end)

        protospacer = Seq(read_seq[start-1 : end])

        if reverse == True:
            protospacers.append(str(protospacer.reverse_complement()))
            PAM_start_right = max(start-11, 0) # in case this goes past the start of the read
            possible_PAMs_right.append(str(Seq(read_seq[PAM_start_right : start - 1]).reverse_complement())) # includes 3 extra nucleotides
            possible_PAMs_left.append(str(Seq(read_seq[end : end + 10]).reverse_complement()))
        else:
            protospacers.append(str(protospacer))
            PAM_start_left = max(start-11, 0) # in case this goes past the start of the read
            possible_PAMs_right.append(str(read_seq[end : end + 10])) # includes 3 extra nucleotides, will take as much as available if it hits the end of the read     
            possible_PAMs_left.append(str(Seq(read_seq[PAM_start_left : start - 1])))
                                      
    return protospacers, possible_PAMs_left, possible_PAMs_right
```

"Nucleotide fasta is always 5' to 3', as are all other formats." from https://www.biostars.org/p/151452/

This means the 5' end is on the left when you read from left to right, which means we expect the CR1 PAM to be on the *left* side of the spacer instead of the right side like in S. thermophilus, where the AGAAW PAM is on the 3' end of the sequence. 

```python
def extract_protospacers_by_fasta(fasta_file, protospacer_blast_data, queries_to_remove, spacer_len_CR1 = 32, 
                                  spacer_len_CR2 = 35, similarity_threshold = 0.85):
    
    """
    Iterate through fasta file
    """
    
    min_length_CR1 = round(spacer_len_CR1 * similarity_threshold) 
    min_length_CR2 = round(spacer_len_CR2 * similarity_threshold) 

    # remove alignments that are less than the minimum length
    protospacer_blast_data = protospacer_blast_data[((protospacer_blast_data['alignment_length'] > min_length_CR1)
                        & (protospacer_blast_data['crispr'] == 'CR1'))
                        | ((protospacer_blast_data['alignment_length'] > min_length_CR2)
                        & (protospacer_blast_data['crispr'] == 'CR2'))]

    # remove query sequences that have too many Ns
    protospacer_blast_data = protospacer_blast_data[~protospacer_blast_data['query_id'].isin(queries_to_remove['query_id'])]
    
    # if there are multiple hits from the same query spacer type, keep the lowest e-value one
    # sometimes there are two hits that are frame-shifted and one of them matches the PAM better - not much I can do about that
    # if there are two hits that are the same type on the same read but not in the same place, this will end up discarding one of them
    # in my tests this gets rid of ~2/3 of the matches
    sub_df = protospacer_blast_data.sort_values(by = ['subject_id','spacer_type', 'crispr', 
                                                      'evalue']).drop_duplicates(['subject_id','spacer_type', 'crispr'], keep = 'first')
    
    # iterate through fasta file and get sequences from read
    
    protospacer_list = []
    PAM_list_5 = []
    PAM_list_3 = []
    bac_seqs = []
    subject_ids = []
    
    print("Number of matched reads: %s" %len(sub_df.groupby(['subject_id'])))
    
    hit_ids = list(sub_df['subject_id'].unique())
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        
        if record.id not in hit_ids:
            continue
        
        data = sub_df[sub_df['subject_id'] == record.id] # select matches from blast result
        
        # iterate through matches to the read, get protospacer and possible PAM
        
        protospacers, possible_PAMs_left, possible_PAMs_right = extract_protospacer_from_read(data, str(record.seq))
        protospacer_list += protospacers
        PAM_list_5 += possible_PAMs_left
        PAM_list_3 += possible_PAMs_right
        bac_seqs += list(data['query_id'])
        subject_ids += list(data['subject_id'])
    
    return protospacer_list, PAM_list_5, PAM_list_3, bac_seqs, subject_ids
    
```

```python
def get_ambiguous_queries(datapath, queries, N_fraction = 0.3):
    """
    Get a data frame of queries that are > N_fraction of N nucleotides. These will be removed from the results.
    
    Inputs:
    datapath : path to query fasta file
    queries : fasta file with query sequences used for spacer blast
    N_fraction : fraction of the query that is the nucleotide N above which it will be included
    
    Returns:
    queries_to_remove : data frame with columns 'query_id' and 'spacer_sequence'
    """
    
    # get query sequences to remove because they have too many Ns
    query_ids = []
    query_seqs = []

    for record in SeqIO.parse(queries, "fasta"):
        id_str = record.id
        query_ids.append(id_str)
        query_seqs.append(str(record.seq))

    # data frame with read ID and spacer sequence
    queries_df = pd.DataFrame()
    queries_df["query_id"] = query_ids
    queries_df["spacer_sequence"] = query_seqs

    queries_to_remove = queries_df[queries_df['spacer_sequence'].str.count('N') 
                                   / queries_df['spacer_sequence'].str.len() > N_fraction]
    
    return queries_to_remove

```

```python
def get_bac_reads(datapath, accession, bac_blast_folder, alignment_cutoff = 70):
    
    """
    Get a list of reads that either matched the Gordonia MAGs or matched either of the CRISPR repeats. 
    Remove these from the phage spacer matches. 
    alignment_cutoff removes bacteria genome matches that have a shorter alignment than that value: could be aligned with a spacer 
    when it's actually on the phage genome
    """

    bac_blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, bac_blast_folder, accession), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])
    
    repeat_blast_data_CR1 = []
    repeat_blast_data_CR2 = []
    repeat_blast_folder_CR1 = "%s_CR1_repeat_blast_out" %accession
    repeat_blast_folder_CR2 = "%s_CR2_repeat_blast_out" %accession
    
    for fn in os.listdir("%s/%s" %(datapath, accession)): # folder with split fasta files
        repeat_blast_data_CR1.append(pd.read_csv("%s/%s/%s_blast.txt" %(datapath, repeat_blast_folder_CR1, fn), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"]))
        repeat_blast_data_CR2.append(pd.read_csv("%s/%s/%s_blast.txt" %(datapath, repeat_blast_folder_CR2, fn), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"]))
        
    repeat_blast_data_CR1_all = pd.concat(repeat_blast_data_CR1)
    repeat_blast_data_CR2_all = pd.concat(repeat_blast_data_CR2)
    
    # filter bac_blast_data to only good matches
    bac_blast_data = bac_blast_data[bac_blast_data['alignment_length'] > alignment_cutoff]
    # get e-value cutoff 
    cutoff = math.floor(np.log10(bac_blast_data.sort_values(by = 'evalue', ascending = False).iloc[0]['evalue']))
    bac_blast_data = bac_blast_data[bac_blast_data['evalue'] < 10**cutoff]
    
    bac_reads = list(np.unique(bac_blast_data['query_id']))
    bac_reads += list(np.unique(repeat_blast_data_CR1_all['subject_id']))
    bac_reads += list(np.unique(repeat_blast_data_CR2_all['subject_id']))
    
    return bac_reads


```

```python
def get_read_from_index(subject_id, datapath, fasta_len = 400000):
    """
    Retrieves the header and sequence of a read based on the subject id sequence provided.
    Assumes reads are paired-end so that the read number is the line number / 4
    Don't use grep - just extract the line number based on the number modulo file length 
    and then do a check to make sure it matches the header - this will be faster than grep 
    Perhaps use grep if it doesn't match as a backup? Not sure if or why this would happen
    
    fasta_len: the length of each sub-fasta file in number of lines (full length, not # reads)
    
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
    read = ! head -n "$read_index" "/$datapath/$accession/$filename" | tail -n 2
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

They detected two CRISPR arrays: one type-III and one type-I.  (SI Figure S4).


```python
repeatfwd = "GTGCTCCCCGCGCGAGCGGGGATGATCCC"
```

```python
repeatfwd_seq = Seq(repeatfwd)
repeatfwd_seq.reverse_complement()
```

```python
repeatfwd_CR2 = "ATCAAGAGCCATGTCTCGCTGAACAGCGAATTGAAAC"
```

```python
len(repeatfwd_CR2)
```

**BLAST Output format:**

http://www.metagenomics.wiki/tools/blast/blastn-output-format-6

Column headers:
qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore

| | | |
| -- | -- | -- |
| 1. | qseqid | query (e.g., gene) sequence id |
| 2. | sseqid | subject (e.g., reference genome) sequence id |
| 3. | pident | percentage of identical matches |
| 4. | length | alignment length |
| 5. | mismatch | number of mismatches |
| 6. | gapopen | number of gap openings |
| 7. | qstart | start of alignment in query |
| 8. | qend | end of alignment in query |
| 9. | sstart | start of alignment in subject |
| 10. | send | end of alignment in subject |
| 11. | evalue | expect value |
| 12. | bitscore | bit score |




## Get wild-type spacers
<a id='wild_type_spacers'></a>


### Method 1: using Gordonia assembly

Search for matches to the repeat sequence in the reference genome to get a list of wild-type spacers

```python
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
cr = "CR2" # or CR2
query = "%s/../gordonia_%s_repeat.fasta" %(datapath, cr)

filename = "%s/../genomes/gordonia_MAG.fa" %(datapath)
outfile = "%s/gordonia_%s_blast.txt" %(datapath, cr)

blastn_obj = NcbiblastnCommandline(query=query, subject=filename, 
                               evalue=10**-4, #num_descriptions = 100, 
                               max_target_seqs = 10000000, dust = "no", 
                               # see http://www.metagenomics.wiki/tools/blast/blastn-output-format-6 for outformat
                               outfmt = 6, 
                               task = "blastn",
                               out = outfile)

stdout, stderr = blastn_obj()
```

```python
# BLAST result
ref_genome_repeat_blast = pd.read_csv("%s/gordonia_%s_blast.txt" %(datapath, cr), sep = '\t', header = None,
                        names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                 "num_mismatches", "num_gapopen", "query_start", "query_end",
                                 "subject_start", "subject_end", "evalue", "bitscore"])
```

```python
ref_genome_repeat_blast
```

```python
# extract wild-type spacers

spacers = []
reverse = False
with open ('%s/gordonia_%s_ref_spacers.txt' %(datapath, cr), 'w') as f:
    for group in ref_genome_repeat_blast.groupby('subject_id'):
        subject_id = group[0]
        print(subject_id)
        data = group[1].sort_values(by = 'subject_start')
        if np.all(data['subject_start'] > data['subject_end']): # all reversed
            spacer_starts = list(data['subject_start'])[:-1]
            spacer_ends = list(data['subject_end'])[1:]
            reverse = True
        elif np.all(data['subject_start'] < data['subject_end']): # all forward
            spacer_starts = list(data['subject_end'])[:-1]
            spacer_ends = list(data['subject_start'])[1:]
            reverse = False
        else:
            print("mixture of directions")
        
        # get sequences
        for record in SeqIO.parse("%s/../genomes/gordonia_MAG.fa" %(datapath), "fasta"):
            if record.id != subject_id:
                continue

            for i, spacer_start in enumerate(spacer_starts):
                if reverse == True: # reverse complement the matches
                    spacer = record.seq[spacer_start : spacer_ends[i] - 1].reverse_complement()
                elif reverse == False:
                    spacer = record.seq[spacer_start : spacer_ends[i] - 1]

                if len(spacer) < 50: #write to file
                    f.write(">" + subject_id + "\n")
                    f.write(str(spacer) + "\n")
                    spacers.append(str(spacer))
```

```python
spacer_lengths = []
for seq in spacers:
    spacer_lengths.append(len(seq))
```

```python
np.mean(spacer_lengths)
```

## Run BLAST to search for matches to CRISPR repeat
<a id='blast_cr1_repeat'></a>

```python
#accession_list = ["SRR9261047", "SRR9261016", "SRR9261004", "SRR9260998", "SRR9260997"]
cr = "CR1"
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
query = "%s/gordonia_%s_repeat.fasta" %(datapath, cr)

with open ('%s/../SRR_Acc_List.txt' %(datapath), 'r') as f:
    accessions = f.readlines()
    
accession_list = []
for accession in accessions:
    accession_list.append(accession.rstrip())

for accession in accession_list:

    for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))):

        outfile = "%s/%s_%s_repeat_blast_out/%s_blast.txt" %(datapath, accession, cr, fn)
        filename = "%s/%s/%s" %(datapath, accession, fn)

        blastn_obj = NcbiblastnCommandline(query=query, subject=filename, 
                                       evalue=10**-4, #num_descriptions = 100, 
                                       max_target_seqs = 10000000, dust = "no", 
                                       # see http://www.metagenomics.wiki/tools/blast/blastn-output-format-6 for outformat
                                       outfmt = 6, 
                                       task = "blastn",
                                       out = outfile)

        stdout, stderr = blastn_obj()
```
## Extract spacers
<a id='extract_spacers'></a>

```python tags=[]
# iterate through split fasta files and detect spacers

with open ('%s/../SRR_Acc_List_temp.txt' %(datapath), 'r') as f:
    accessions = f.readlines()
    
accession_list = []
for accession in accessions:
    accession_list.append(accession.rstrip())
    
#accession_list = ["SRR1873863"] # control - no phage
accession_list = ["SRR9261025"]

if cr == "CR1":
    repeat_len = len(repeatfwd)
    spacer_len = 32
elif cr == "CR2":
    repeat_len = len(repeatfwd_CR2)
    spacer_len = 35
    
read_len = 250

datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"

for accession in accession_list:
    blast_folder = "%s_%s_repeat_blast_out" %(accession, cr)

    all_spacers = []
    all_ids = []

    for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))): # folder with split fasta files
        blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, blast_folder, fn), sep = '\t', header = None,
                            names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                     "num_mismatches", "num_gapopen", "query_start", "query_end",
                                     "subject_start", "subject_end", "evalue", "bitscore"])
        
        ## ----- keep only high-quality matches ----- 
        
        # alignments that are either the full length OR that hit the start or end of the read
        blast_data_high_quality = blast_data[(blast_data['alignment_length'] == repeat_len)
                    | ((blast_data['alignment_length'] < repeat_len)
                   & ((blast_data['subject_start'] == read_len)
                  | (blast_data['subject_start'] == 0)
                  | (blast_data['subject_end'] == read_len)
                   | (blast_data['subject_end'] == 0)))]

        # get read headers for short repeat matches
        short_matches = blast_data_high_quality[blast_data_high_quality['alignment_length'] < repeat_len]['subject_id']

        # get count of number of matches per read that contain short alignments
        single_matches = blast_data_high_quality[blast_data_high_quality['subject_id'].isin(
            short_matches)].groupby('subject_id')['alignment_length'].count().reset_index()

        # get read headers that only contain one match that is a short alignment
        reads_to_drop = single_matches[single_matches['alignment_length'] < 2]['subject_id']

        # remove those matches from the blast results
        blast_data_high_quality = blast_data_high_quality[~blast_data_high_quality['subject_id'].isin(reads_to_drop)]
        
        # add back in any matches from the same read as a match we're keeping
        # this should keep consistency in spacers that are identified between repeats
        blast_data_run = blast_data[blast_data['subject_id'].isin(blast_data_high_quality['subject_id'])]
        
        spacers, ids = extract_spacers("%s/%s/%s" %(datapath, accession, fn), blast_data_run, progress_bar = False, spacer_len = spacer_len)

        all_spacers += spacers
        all_ids += ids
            
    with open ('%s/%s_%s_spacers.txt' %(datapath, accession, cr), 'w') as f:
        for i, seq in enumerate(all_spacers):
            f.write(">" + str(all_ids[i]) + "\n")
            f.write(str(seq) + "\n")
    
```

## Detecting protospacers
<a id='detect_protospacers'></a>


### Blast all reads against Gordonia genome assemblies


This will take a long time - takes 12 minutes per split file, so about 24 files per accession means 12 days to run. No speedup from restricting number of hits

Difference locally vs niagara: need to specify task blastn, otherwise default is megablast. 

```python
# blast all reads against Gordonia genomes
accession_list = ["SRR9261044", "SRR9261045", "SRR9261047", "SRR9261016", "SRR9261004", "SRR9260998", "SRR9260997"]
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
filename = "%s/../genomes/gordonia_MAG.fa" %(datapath) # gordonia metagenome-assembled genomes (MAG)

for accession in accession_list:

    done_files = !ls -t "$datapath/$accession"_bac_genome_blast_out | tail -n +2 | cut -d'_' -f1 # exclude the most recent file - might be partially complete
    
    for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))):
        
        #if fn in done_files: # if already blasted, skip
        #    continue
        
        query = "%s/%s/%s" %(datapath, accession, fn)

        outfile = "%s/%s_bac_genome_blast_out/%s_blast.txt" %(datapath, accession, fn)
    
        blastn_obj = NcbiblastnCommandline(query=query, subject=filename, 
                                       evalue=10**-5, #num_descriptions = 100, 
                                       max_target_seqs = 5, dust = "no", # don't need lots of hits, just confirmation that it matches
                                       # see http://www.metagenomics.wiki/tools/blast/blastn-output-format-6 for outformat
                                       outfmt = 6, 
                                       task = "blastn",
                                       out = outfile)

        stdout, stderr = blastn_obj()
```

### Blast all reads against phage genomes
<a id='phage_genome_blast'></a>

Done on Niagara for Guererro 2021


## Cluster all spacers
<a id='cluster_spacers'></a>

```python
# load wild-type spacers
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"

wild_type_spacers_CR1 = load_wild_type_spacers(datapath, "CR1")
wild_type_spacers_CR2 = load_wild_type_spacers(datapath, "CR2")
```

```python
metadata = pd.read_csv("%s/../SraRunTable.txt" %datapath)

accessions = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
```

```python
#datapath = "results/2019-09-18"
#datapath = "results/2019-09-23"
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
cr = "CR2"
unique_spacers_tp = []
counts = []
time_point = []
all_spacers = []
i = 0
for accession in tqdm(accessions):
        
    spacer_subject_ids = []
    spacers = []

    fasta_file = "%s/%s_%s_spacers.txt" %(datapath, accession, cr)
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        id_str = record.id.split('_')[0]
        spacer_subject_ids.append(id_str)
        spacers.append(str(record.seq))
    
    if len(spacers) == 0:
        i += 1
        continue
    
    all_spacers += spacers
    unique_sp, spacer_counts = np.unique(spacers, return_counts = True)
    
    # remove paired-end double counts
    spacer_counts = count_paired_ends(spacers, spacer_subject_ids, unique_sp, spacer_counts)
    
    unique_spacers_tp += list(unique_sp)
    counts += list(spacer_counts)
    time_point += [i]*len(unique_sp)
    i += 1
    
```

```python
len(list(set(all_spacers)))
```

```python
if cr == "CR1":
    wild_type_spacers = wild_type_spacers_CR1
elif cr == "CR2":
    wild_type_spacers = wild_type_spacers_CR2
```

```python
# there may be overlap between wild type and all spacers at this point - will be removed later
unique_spacers = wild_type_spacers + list(set(all_spacers))
print(len(unique_spacers))
```

```python
# save list to file to associate with distance matrix
with open ('unique_spacers_%s.txt' %(cr), 'w') as f:
    for seq in unique_spacers:
        f.write(str(seq.strip()) + "\n")
```

```python
# create distance matrix using the Levenshtein similarity ratio
# this is slow for a long list of spacers, just a fact of life
 
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
np.savez_compressed("distance_matrix_%s" %(cr), distance_matrix)
```

### Agglomerative hierarchical clustering based on sequence similarity

```python
cr = "CR2"
```

```python
with np.load('results/2022-03-23/distance_matrix_%s.npz' %cr) as data:
    distance_matrix = data['arr_0']
    # data.files is how I know it's called 'arr_0'
```

```python
# load associated spacer list
with open ('results/2022-03-23/unique_spacers_%s.txt' %cr, 'r') as f:
    unique_spacers = f.readlines()
```

```python
# this takes a decently long time with lots of data as well: ~5 min
# also takes a lot of RAM
fit = AgglomerativeClustering(distance_threshold=0.15, n_clusters=None, linkage='average',
                             affinity='precomputed').fit(distance_matrix)
```

```python
# remove spacers that clustered with wild-type spacers - the wild type spacers are first in the list
wild_type_labels = fit.labels_[:len(wild_type_spacers)]
```

```python
len(np.unique(wild_type_labels))
```

### Save spacer clustering to dataframe

```python
spacer_counts = pd.DataFrame()
spacer_counts['sequence'] = unique_spacers_tp
spacer_counts['time_point'] = time_point
spacer_counts['count'] = counts
spacer_counts['sequence'] = spacer_counts['sequence'].astype(str)
spacer_counts['sequence'] = spacer_counts['sequence'].str.strip()

```

```python
unique_spacers = np.array(unique_spacers)[np.array(unique_spacers) != '\n']
```

```python
spacer_types = pd.DataFrame()
spacer_types['sequence'] = unique_spacers
spacer_types['type'] = fit.labels_
spacer_types['sequence'] = spacer_types['sequence'].astype(str)
spacer_types['sequence'] = spacer_types['sequence'].str.strip()
spacer_types = spacer_types.drop_duplicates()
```

```python
# merge to get time point, count, and type info
spacer_types_df = spacer_counts.merge(spacer_types, on = 'sequence')
```

```python
spacer_types_df.head()
```

```python
# lots of sequences are present at multiple time points
spacer_types_df.groupby('sequence')['time_point'].count().reset_index().sort_values(by = 'time_point', ascending = False).head(n = 30)
```

```python
spacer_types_df.to_csv("spacer_types_Guerrero2021_%s.csv" %(cr), index=None)
```

## Blast spacers against all read data
<a id='blast_protospacers'></a>

Blast is done on the server: `run_blast.sh`

```python
# make fasta file of unique spacer sequences

datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021"

with open("%s/unique_spacers.fasta" %(datapath), 'w') as f:

    for cr in ["CR1", "CR2"]:

        #all_unique_spacers = pd.read_csv("results/2022-02-03/spacer_types_MOI_2b_%s.csv" %cr)
        all_unique_spacers  = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_%s.csv" %cr)
        unique_spacers_df = all_unique_spacers.drop_duplicates(['sequence', 'type'])

        for i, row in tqdm(unique_spacers_df.iterrows()): # duplicates occur at different time points
            # check if this sequence is a subsequence of any other - if it appears more than once as a subsequence, skip it
            if len(unique_spacers_df[unique_spacers_df['sequence'].str.contains(row['sequence'])]) > 1:
                continue
            # remove poly-N sequences
            if row['sequence'].count('N') / len(row['sequence']) > 0.3:
                continue
            f.write(">" + str(row['type']) + "_" + str(i) + "_" + str(cr) + "\n") # include row index for uniqueness identifier
            f.write(str(row['sequence']) + "\n")
```

```python
# make fasta file of unique spacer sequences

datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"

cr = "CR2"
#all_unique_spacers = pd.read_csv("results/2022-02-03/spacer_types_MOI_2b_%s.csv" %cr)
all_unique_spacers  = pd.read_csv("results/2022-03-23/spacer_types_Guerrero2021_%s.csv" %cr)
unique_spacers_df = all_unique_spacers.drop_duplicates(['sequence', 'type'])

with open("%s/unique_spacers_%s.fasta" %(datapath,cr), 'w') as f:
    for i, row in tqdm(unique_spacers_df.iterrows()): # duplicates occur at different time points
        # check if this sequence is a subsequence of any other - if it appears more than once as a subsequence, skip it
        if len(unique_spacers_df[unique_spacers_df['sequence'].str.contains(row['sequence'])]) > 1:
            continue
        # remove poly-N sequences
        if row['sequence'].count('N') / len(row['sequence']) > 0.3:
            continue
        f.write(">" + str(row['type']) + "_" + str(i) + "\n") # include row index for uniqueness identifier
        f.write(str(row['sequence']) + "\n")
```

### Testing blast of spacers against reads

```python
accession_list = ["SRR1873837"]
datapath = "/media/madeleine/My Passport1/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"
cr = "CR3"
query = "%s/unique_spacers_%s_sample.fasta" %(datapath,cr)

for accession in accession_list:
    for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))):

        outfile = "%s/%s_%s_spacer_blast_out/%s_blast.txt" %(datapath, accession, cr, fn)
        filename = "%s/%s/%s" %(datapath, accession, fn)

        blastn_obj = NcbiblastnCommandline(query=query, subject=filename, 
                                       evalue=10**-4, #num_descriptions = 100, 
                                       max_target_seqs = 10000000, dust = "no", 
                                       # see http://www.metagenomics.wiki/tools/blast/blastn-output-format-6 for outformat
                                       outfmt = 6, 
                                       task = "blastn",
                                       out = outfile)

        stdout, stderr = blastn_obj()
```

### Run spacer BLAST on niagara (2022)
<a id='blast_protospacers_niagara'></a>

1. Make tar archive for genome files: `tar -cvzf SRR1873837_fasta.tar.gz SRR1873837`
2. Copy SRR1873837_fasta.tar.gz to niagara: `scp SRR1873837_fasta.tar.gz mbonsma@niagara.scinet.utoronto.ca:/scratch/g/goyalsid/mbonsma/2022-02-03
3. [x] Copy spacer sequences to niagara: `scp unique_spacers_CR3.fasta mbonsma@niagara.scinet.utoronto.ca:/scratch/g/goyalsid/mbonsma/2022-02-03`
`scp unique_spacers_new_CR1.fasta mbonsma@niagara.scinet.utoronto.ca:/scratch/g/goyalsid/mbonsma/2022-02-03`
4. Extract tar archive on niagara in dated folder: `tar -xvzf SRR1873837_fasta.tar.gz`
5. From dated directory on niagara, run `blast_setup_script.sh SRR1873837 unique_spacers_CR3.fasta` to set up the BLAST runs
6. From dated directory on niagara, run `sbatch blast_submit_script_niagara.sh SRR1873837`
7. Check if any had errors: run from dated folder 

```
log=$(ls slurm*.log -t | head -n 1)  # get most recent log file
cat $log | tail -n +2 > joblog.txt # trim header row from log

while read job
do 
  receive=$(echo $job | cut -d" " -f6)
  exitval=$(echo $job | cut -d" " -f7)
  i=$(echo $job | cut -d" " -f1)
  simnum=$(printf %04d $i)
  if [ $exitval -ne "0" ]; then # if non-zero exit, print simulation number
    if [ $exitval -ne "127" ]; then # if it's 127 the run never existed, not a real error
      echo $simnum
    fi
  fi
done < joblog.txt
```

8. Check if number of output files matches scripts. If not all are done, can edit the starting number higher in blast_submit_script_niagara.sh and resubmit.

```
ls doserialjob* | wc -l
ls *_blast* | wc -l
```

9. Concatenate blast output files for copying - run in top level dated folder

```
accession=SRR1873839
ls "$accession"/"$accession"* > splitfilenames.txt # in folder with split fasta data files

while read line
do
  filename=$(echo $line | cut -d'/' -f2)
  echo $filename
  cat "$accession"_blast/"$filename"* > "$accession"_blast/"$filename"_blast.txt
done < splitfilenames.txt
```

10. copy resulting combined blast output files to hard drive

```
# make tar archive of results, run in dated folder
accession=SRR1873849
ls "$accession"_blast/"$accession"??_blast.txt > "$accession"_results_transfer.txt
tar -cvzf "$accession"_results.tar.gz -T "$accession"_results_transfer.txt

# run on local machine in appropriate spacer_blast_out folder
accession=SRR1873838
date=2022-02-03 # MAKE SURE TO SET THIS TO THE RIGHT FOLDER (03 for cr3, 04 for cr1)
scp mbonsma@niagara.scinet.utoronto.ca:/scratch/g/goyalsid/mbonsma/"$date"/"$accession"_results.tar.gz .
tar -xvzf "$accession"_results.tar.gz .
```

11. Run protospacer extraction block 


## Sort and extract protospacers


### Full sequence and PAM version

```python
datapath = "/media/madeleine/My Passport1/Blue hard drive/Data/Guerrero2021/data"
metadata = pd.read_csv("%s/../SraRunTable.txt" %datapath)

accession_list = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Run'].str.rstrip().values)
dates = list(metadata[['Run', 'Collection_date']].sort_values(by = 'Collection_date')['Collection_date'].str.rstrip().values)
```

```python
datapath = "/media/madeleine/My Passport1/Blue hard drive/Data/Guerrero2021/data"
queries = "%s/unique_spacers.fasta" %(datapath) # did blast with combined list of CR1 and CR2

queries_to_remove = get_ambiguous_queries(datapath, queries)
```

Many reads match neither the phage assembly nor the bacterial genome - should I still keep protospacer hits from that? Maybe do it both ways?

```python
phage_only = True # if true, keep only spacer matches if the read also matched the phage genome
```

```python
# iterate through fasta files and detect spacers

#cr = "CR1" # or CR1 or CR2
datapath = "/media/madeleine/My Passport/Blue hard drive/Data/Guerrero2021/data"
queries = "%s/unique_spacers.fasta" %(datapath) # did blast with combined list of CR1 and CR2

queries_to_remove = get_ambiguous_queries(datapath, queries)

#spacer_types_bac_previous = pd.read_csv("results/2019-09-27/spacer_types_MOI_2b_wildtype_removed.csv") # these sequences match spacer_types_phage
#spacer_types_bac = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b_wildtype_removed.csv")
#spacer_types_bac_all = pd.read_csv("results/2021-11-05/spacer_types_MOI_2b.csv")

for accession in tqdm(accession_list):
    phage_blast_folder = "%s_phage_genome_blast" %accession
    bac_blast_folder = "%s_bac_genome_blast" %accession
    repeat_blast_folder_CR1 = "%s_CR1_repeat_blast_out" %accession
    repeat_blast_folder_CR2 = "%s_CR2_repeat_blast_out" %accession
    spacer_blast_folder = "%s_spacer_blast" %accession
        
    phage_blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, phage_blast_folder, accession), sep = '\t', header = None,
                        names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                 "num_mismatches", "num_gapopen", "query_start", "query_end",
                                 "subject_start", "subject_end", "evalue", "bitscore"])

    spacer_blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, spacer_blast_folder, accession), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])

    bac_blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, bac_blast_folder, accession), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])
        
    bac_reads = get_bac_reads(datapath, accession, bac_blast_folder, alignment_cutoff = 70)

    phage_reads = list(np.unique(phage_blast_data['query_id']))
    
    # reads in both - sort into one or the other based on match quality
    common_reads = list(set(bac_reads) & set(phage_reads))
    
    # take the best match for each bacteria read that is in the set of both
    bac_common_reads = bac_blast_data[bac_blast_data['query_id'].isin(common_reads)].sort_values(by = 
                                                                        'evalue').drop_duplicates(subset ='query_id', keep = 'first')
    phage_common_reads = phage_blast_data[phage_blast_data['query_id'].isin(common_reads)].sort_values(by = 
                                                                        'evalue').drop_duplicates(subset ='query_id', keep = 'first')
    
    merged = phage_common_reads.merge(bac_common_reads, on = 'query_id', suffixes = ('_phage', '_bac'))
    merged['log_difference_abs'] = np.abs(np.log10(merged['evalue_phage']) - np.log10(merged['evalue_bac']))
    merged['log_difference'] = np.log10(merged['evalue_phage']) - np.log10(merged['evalue_bac'])
    
    evalue_cutoff = 25
    phage_reads_new = list(merged[merged['log_difference'] < -evalue_cutoff]['query_id'])
    bac_reads_new = list(merged[merged['log_difference'] > -evalue_cutoff]['query_id'])
    
    # remove common_reads from phage_reads and bac_reads, add the divided reads back in
    # the goal is to be conservative with assigning phage reads, since we want to detect genuine protospacers
    bac_reads_final = list(set(bac_reads) - set(common_reads)) + bac_reads_new
    phage_reads_final = list(set(phage_reads) - set(common_reads)) + phage_reads_new

    ## Plot difference in evalue, show cutoff
    fig, ax = plt.subplots()
    ax.hist(merged['log_difference'], bins = 50)

    # negative value means phage match is better
    ax.set_xlabel("Difference in e-value between phage match and bacteria match")
    ax.set_ylabel("Count")

    ax.axvline(-evalue_cutoff, color = 'k', linestyle = '--')
    #ax.axvline(25, color = 'k', linestyle = '--')

    ax.set_title("%s bac reads, %s phage reads, %s overlapping, %s assigned to phage" %(len(bac_reads_final), len(phage_reads_final),
                                                                                len(common_reads), len(phage_reads_new)), fontsize = 10)
    plt.tight_layout()
    plt.savefig("Guerrero_phage_evalue_minus_bac_evalue_%s.pdf" %accession)

    # these ones are not bacteria reads
    if phage_only == False:
        protospacer_blast_data = spacer_blast_data[~spacer_blast_data['subject_id'].isin(bac_reads_final)] 
    elif phage_only == True:
        protospacer_blast_data = spacer_blast_data[spacer_blast_data['subject_id'].isin(phage_reads_final)] 
        
    if len(protospacer_blast_data) == 0:
        continue
        
    # create columns for spacer type and sequence
    protospacer_blast_data[["spacer_type", "sequence_id", "crispr"]] = protospacer_blast_data['query_id'].str.split('_', 
                                                                                            expand = True)
        
    protospacer_blast_data['sequence_id'] = protospacer_blast_data['sequence_id'].astype('int')
    protospacer_blast_data['spacer_type'] = protospacer_blast_data['spacer_type'].astype('int')
    
    # get read number for sub-selecting
    protospacer_blast_data['read_id'] = protospacer_blast_data['subject_id'].str.split('.', expand = True)[1]
    protospacer_blast_data['read_id'] = protospacer_blast_data['read_id'].astype('int')
        
    # extract protospacers
    for fn in os.listdir("%s/%s" %(datapath, accession)): # folder with split fasta files

        if phage_only == False:
            if os.path.exists("%s/protospacers/%s_protospacers.txt" %(datapath, fn)): # if this one has been done already, skip it
                continue
        elif phage_only == True:
            if os.path.exists("%s/protospacers_phage_only/%s_protospacers.txt" %(datapath, fn)): # if this one has been done already, skip it
                continue        
                
        fasta_file = "%s/%s/%s" %(datapath, accession, fn)
        
        # get start and end reads from fasta file
        start=! head -n 1 "$fasta_file"
        end=! tail -n 2 "$fasta_file" | head -n 1
        
        # sub-select protospacer_blast_data for that fasta file
        protospacer_blast_data_sub = protospacer_blast_data[(protospacer_blast_data['read_id'] >= int(start[0].split('.')[1]))
                      & (protospacer_blast_data['read_id'] <= int(end[0].split('.')[1]))]
        
        protospacer_list, PAM_list_5, PAM_list_3, bac_seqs, subject_ids = extract_protospacers_by_fasta(fasta_file, protospacer_blast_data_sub, 
                                                                queries_to_remove,  spacer_len_CR1 = 32, 
                                                                  spacer_len_CR2 = 35, similarity_threshold = 0.85)


        # save to dataframe
        protospacers_df = pd.DataFrame()
        protospacers_df['query_id'] = bac_seqs
        protospacers_df['subject_id'] = subject_ids
        protospacers_df['sequence'] = protospacer_list
        protospacers_df['PAM_region_5'] = PAM_list_5
        protospacers_df['PAM_region_3'] = PAM_list_3
        
        if phage_only == False:
            protospacers_df.to_csv("%s/protospacers/%s_protospacers.txt" %(datapath, fn), index = None)
        elif phage_only == True:
            protospacers_df.to_csv("%s/protospacers_phage_only/%s_protospacers.txt" %(datapath, fn), index = None)
```

```python

```
