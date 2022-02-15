#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:17:13 2022

@author: madeleine

Process protospacer blast data on niagara
"""

from Bio.Seq import Seq
from Bio import SeqIO
import argparse
import pandas as pd

def get_ambiguous_queries(queries, N_fraction = 0.3):
    """
    Get a data frame of queries that are > N_fraction of N nucleotides. These will be removed from the results.
    
    Inputs:
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
                                   / queries_df['spacer_sequence'].str.len() > 0.3]
    
    return queries_to_remove

def extract_protospacer_from_read(data, read_seq):
    """
    Takes a dataframe corresponding to a particular read and the read sequence,
    extracts protospacer and PAM sequences.

    Parameters
    ----------
    data : blast results corresponding to a single subject_id. This should be 
            filtered so that it's one result per query, in my case the lowest
            e-value result for each matching spacer type
    read_seq : read sequence that matches subject_id

    Returns
    -------
    protospacers : a list of protospacers from blast matches
    possible_PAMs : a list of the ten nucleotides immediately after each 
                protospacer. If this runs into the end of the read, as many
                nucleotides are returned as there are.

    """
    
    protospacers = []
    possible_PAMs = []
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
            protospacer_start = max(start-11, 0) # in case this goes past the start of the read
            possible_PAMs.append(str(Seq(read_seq[protospacer_start : start - 1]).reverse_complement())) # includes 3 extra nucleotides
        else:
            protospacers.append(str(protospacer))
            possible_PAMs.append(str(read_seq[end : end + 10])) # includes 3 extra nucleotides, will take as much as available if it hits the end of the read     
            
    return protospacers, possible_PAMs

def extract_protospacers_by_fasta(fasta_file, protospacer_blast_data, queries_to_remove, 
                                  spacer_len = 30, similarity_threshold = 0.85):
    
    """
    Iterate through fasta file and find matches to the protospacer blast data
    
    Parameters
    ----------
    fasta_file :
    protospacer_blast_data :
    queries_to_remove :
    spacer_len :
    similarity_threshold :
    
    Returns
    -------
    protospacer_list
    PAM_list
    bac_seqs
    subject_ids
    
    """
    
    min_length = round(spacer_len * similarity_threshold) 

    # remove alignments that are less than the minimum length
    protospacer_blast_data = protospacer_blast_data[protospacer_blast_data['alignment_length'] >= min_length]

    # remove query sequences that have too many Ns
    protospacer_blast_data = protospacer_blast_data[~protospacer_blast_data['query_id'].isin(queries_to_remove['query_id'])]
    
    # if there are multiple hits from the same query spacer type, keep the lowest e-value one
    # sometimes there are two hits that are frame-shifted and one of them matches the PAM better - not much I can do about that
    # if there are two hits that are the same type on the same read but not in the same place, this will end up discarding one of them.
    # the above scenario is not what we want, but should be extremely rare.
    # in my tests this gets rid of ~2/3 of the matches
    sub_df = protospacer_blast_data.sort_values(by = ['subject_id','spacer_type', 'evalue']).drop_duplicates(['subject_id','spacer_type'], keep = 'first')
    
    # iterate through fasta file and get sequences from read
    
    protospacer_list = []
    PAM_list = []
    bac_seqs = []
    subject_ids = []
    
    print("Number of matched reads: %s" %len(sub_df.groupby(['subject_id'])))
    
    hit_ids = list(sub_df['subject_id'].unique())
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        
        if record.id not in hit_ids:
            continue
        
        data = sub_df[sub_df['subject_id'] == record.id] # select matches from blast result
        
        # iterate through matches to the read, get protospacer and possible PAM
        
        protospacers, possible_PAMs = extract_protospacer_from_read(data, str(record.seq))
        protospacer_list += protospacers
        PAM_list += possible_PAMs
        bac_seqs += list(data['query_id'])
        subject_ids += list(data['subject_id'])
    
    return protospacer_list, PAM_list, bac_seqs, subject_ids

def process_protospacers(accession, fn, query_file, spacer_blast_folder, cr, outfolder):
    queries_to_remove = get_ambiguous_queries(query_file)
    
    spacer_blast_data = pd.read_csv("%s/%s_blast.txt" %(spacer_blast_folder, fn), sep = '\t', header = None,
                                        names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                                 "num_mismatches", "num_gapopen", "query_start", "query_end",
                                                 "subject_start", "subject_end", "evalue", "bitscore"])

    with open("bac_reads/%s_bac_reads.txt" %(fn)) as f:
        bac_reads_unstripped = f.readlines()

    bac_reads = []
    for r in bac_reads_unstripped:
        bac_reads.append(r.rstrip())

    # these ones are not bacteria reads
    protospacer_blast_data = spacer_blast_data[~spacer_blast_data['subject_id'].isin(bac_reads)] 

    # create columns for spacer type and sequence
    protospacer_blast_data[["spacer_type", "sequence_id"]] = protospacer_blast_data['query_id'].str.split('_', 
                                                                                            expand = True)

    protospacer_blast_data['sequence_id'] = protospacer_blast_data['sequence_id'].astype('int')
    protospacer_blast_data['spacer_type'] = protospacer_blast_data['spacer_type'].astype('int')

    # extract protospacers
    fasta_file = "%s/%s" %(accession, fn)
    protospacer_list, PAM_list, bac_seqs, subject_ids = extract_protospacers_by_fasta(fasta_file, protospacer_blast_data, 
                                                                                      queries_to_remove, spacer_len = 30, similarity_threshold = 0.85)


    # save to dataframe
    protospacers_df = pd.DataFrame()
    protospacers_df['query_id'] = bac_seqs
    protospacers_df['subject_id'] = subject_ids
    protospacers_df['sequence'] = protospacer_list
    protospacers_df['PAM_region'] = PAM_list
    
    protospacers_df.to_csv("%s/%s_%s_protospacers_fast.txt" %(outfolder, fn, cr), index = None)
    

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process results of blasting spacers against all reads')
    parser.add_argument('accession', type=str, nargs='?',
                        help='accession number')
    parser.add_argument('fn', type=str, nargs='?',
                        help='filename, i.e. SRR1873837aa')
    parser.add_argument('query_file', type=str,nargs='?',
                        help='spacer query filename')
    parser.add_argument('spacer_blast_folder', type=str,nargs='?',
                        help='relative location of spacer blast results')
    parser.add_argument('cr', type=str,nargs='?',
                        help='CRISPR locus: CR1 or CR3')
    parser.add_argument('outfolder', type=str,nargs='?',
                        help='folder to save in')
    
    args = parser.parse_args()
    
    # Run simulation
    
    # define parameters
    accession = args.accession
    fn = args.fn
    query_file = args.query_file
    spacer_blast_folder = args.spacer_blast_folder
    cr = args.cr
    outfolder = args.outfolder

    process_protospacers(accession, fn, query_file, spacer_blast_folder, cr, outfolder)