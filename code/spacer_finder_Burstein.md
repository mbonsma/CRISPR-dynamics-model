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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Detect repeats and spacers in Burstein2016 data

- [ ] BLAST their detected CRISPR repeats against metagenomic data
- [ ] Detect spacers, get abundance, measure turnover
- [ ] Detect protospacers?

```python
from Bio.Blast.Applications import NcbiblastnCommandline
#help(NcbiblastnCommandline)

from Bio.Seq import Seq
from Bio import SeqIO
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

```python
def extract_spacers(fasta_file, blast_data, spacer_len = 30, similarity_threshold = 0.85, progress_bar = True):
    """
    First draft copied from spacer_finder.ipynb, designed for S. thermophilus 100 read length data
    
    Given the BLAST output from blasting the CRISPR repeat against a fasta file, take that fasta file and the 
    BLAST output and extract the neighbouring spacers.
    
    Inputs:
    fasta_file : the fasta file of reads corresponding to blast_data, i.e. SRR1873863.fasta. This is assumed to have reads
                      of length 100. 
    blast_data : a pandas dataframe of the BLAST results from blasting the repeat against fasta_file
    spacer_len : the expected spacer length - the shortest spacer will be length >= similarity_threshold * spacer_len, 
                      the longest spacer will be length spacer_len unless the distance between two repeats on the 
                      same read is longer.
                      
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
        
        data = blast_data[blast_data['subject_id'] == record.id] # select matches from blast result

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

                ids.append(str(record.id) + "_a")
                
            # get the spacer at the end of the read
            if 100 - end > min_length:
                spacer = record.seq[end : end + spacer_len]
                if reverse == True:
                    spacers.append(spacer.reverse_complement())
                else:
                    spacers.append(spacer)
                
                ids.append(str(record.id) + "_b")

        elif len(data) == 2: # multiple repeat matches, get the spacer in between

            # check if reversed       
            if int(data.iloc[0]['subject_start']) > int(data.iloc[0]['subject_end']): # match is to reverse complement
                reverse = True    

                spacer_start = min(data['subject_start'])
                spacer_end = max(data['subject_end'])

                spacer = record.seq[spacer_start : spacer_end - 1]
                spacers.append(spacer.reverse_complement())
                #print(spacer)

            else:
                reverse = False
                spacer_start = min(data['subject_end'])
                spacer_end = max(data['subject_start'])

                spacer = record.seq[spacer_start : spacer_end - 1]
                spacers.append(spacer)

                #print(spacer)

            ids.append(str(record.id))
            
        else: # can't think that it's possible to have more than 2 hits, but if it happens check it out
            print(record.id)
            print(data)
            
    return spacers, ids
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

```python
# load detected CRISPR repeats from paper
# source: https://static-content.springer.com/esm/art%3A10.1038%2Fncomms10613/MediaObjects/41467_2016_BFncomms10613_MOESM952_ESM.xlsx
Burstein_repeats = pd.read_csv("41467_2016_BFncomms10613_MOESM952_ESM.csv", skiprows = [0,1])
Burstein_repeats['query_id'] = Burstein_repeats['Genome bin'] + "_" + Burstein_repeats['Array location']
```

```python
# get reverse complement of repeat sequence
repeat_reverse_complements = []
for repeat in Burstein_repeats['Repeat sequence']:
    repeat = Seq(repeat)
    repeat_reverse_complements.append(str(repeat.reverse_complement()))
    
Burstein_repeats['Repeat reverse complement'] = repeat_reverse_complements
```

```python
# there are 197 genomes and 144 unique repeats
Burstein_repeats_unique = Burstein_repeats.drop_duplicates(subset = 'Repeat sequence')
```

```python
Burstein_repeats_unique
```

```python
Burstein_repeats_unique[Burstein_repeats_unique['Repeat sequence'].isin(Burstein_repeats_unique['Repeat reverse complement'])]
```

```python
# these are the queries not included in the blast - duplicated repeats
duplicated_queries = Burstein_repeats[~Burstein_repeats['query_id'].isin(Burstein_repeats_unique['query_id'])]['query_id']
```

```python
# make fasta file from repeat data
datapath = "/media/madeleine/My Passport/Data/Burstein2016"

with open("%s/Burstein_repeats_unique.fasta" %(datapath), 'w') as f:
    for i, row in Burstein_repeats_unique.iterrows(): # duplicates occur at different time points
        f.write(">" + str(row['Genome bin']) + "_" + str(row['Array location']) + "\n")
        f.write(str(row['Repeat sequence']) + "\n")
```

Note: I ran up to 173/346 of query suffix 1 for SRR1658462 with the repeat files in "Burstein_repeats_split", but after that I removed duplicate repeats and created new split files to run on. This should not affect the results because the query naming convention has not changed, just the total number of queries and which ones are in which query suffix split file. 

```python
query_suffixes = ["1", "2", "3", "4"]

accession_list = ["SRR1658343", "SRR1658462", "SRR1658465", "SRR1658467", "SRR1658469", "SRR1658472"]

datapath = "/media/madeleine/My Passport1/Data/Burstein2016"

for accession in accession_list:
    
    done_files = !ls -t "$datapath/$accession"_repeat_blast_out | tail -n +2 | cut -d "." -f1 # exclude the most recent file - might be partially complete
    
    for s in query_suffixes:
        query = "%s/Burstein_repeats_%s" %(datapath, s)
        for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))):
            
            if "%s_blast_%s" %(fn, s) in done_files:
                continue
            
            outfile = "%s/%s_repeat_blast_out/%s_blast_%s.txt" %(datapath, accession, fn, s)
            filename = "%s/%s/%s" %(datapath, accession, fn)
            
            blastn_obj = NcbiblastnCommandline(query=query, subject=filename, 
                               evalue=10**-4, #num_descriptions = 100, 
                               max_target_seqs = 500000, dust = "no", 
                               # see http://www.metagenomics.wiki/tools/blast/blastn-output-format-6 for outformat
                               outfmt = 6, 
                               task = "blastn",
                               out = outfile)

            stdout, stderr = blastn_obj()

```
