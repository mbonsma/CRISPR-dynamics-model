#!/usr/bin/env python
# coding: utf-8

# get total reads matching phage and bacteria in Paez-Espino2015 data
# usage: python get_read_counts.py datapath
# example usage: python get_read_counts.py "/media/madeleine/My Passport/Blue hard drive/Data/Paez_Espino_2015/PRJNA275232"

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def get_read_counts(datapath):

    # this takes a loooooooong time, like 1 hour - can read in result a few cells below instead

    accession_list = ["SRR1873840", "SRR1873837", "SRR1873838", "SRR1873839", "SRR1873841", "SRR1873842", "SRR1873843", "SRR1873844", "SRR1873845", "SRR1873846", 
                      "SRR1873847", "SRR1873848", "SRR1873849"]
    #accession_list = ["SRR1873846", "SRR1873847", "SRR1873848", "SRR1873849"]
    #datapath = "/media/madeleine/My Passport/Data/Paez_Espino_2015/PRJNA275232"

    phage_reads_count_list = []
    bac_reads_count_list = []
    both_reads_count_list = []
    neither_reads_count_list = []
    total_reads_count_list = []

    for accession in accession_list:
        phage_blast_folder = "%s_phage_genome_blast_out" %accession
        bac_blast_folder = "%s_bac_genome_blast_out" %accession
        repeat_blast_folder = "%s_repeat_blast_out" %accession

        phage_reads_count = 0
        bac_reads_count = 0
        both_reads_count = 0
        neither_reads_count = 0
        total_reads_count = 0

        for fn in tqdm(os.listdir("%s/%s" %(datapath, accession))): # folder with split fasta files

            # count number of lines in the file of reads
            total_lines = ! wc -l "$datapath"/"$accession"/"$fn" | cut -d" " -f1 # bash
            # note that the last file in the set has an extra newline, so the extra `int` will remove the resulting 0.5
            total_reads = int(int(total_lines[0])/2) # divide by 2 to account for fasta header

            total_reads_count += total_reads

            phage_blast_data = pd.read_csv("%s/%s/%s_blast.txt" %(datapath, phage_blast_folder, fn), sep = '\t', header = None,
                                names = ["query_id", "subject_id", "percent_identity", "alignment_length", 
                                         "num_mismatches", "num_gapopen", "query_start", "query_end",
                                         "subject_start", "subject_end", "evalue", "bitscore"])

            bac_reads = get_bac_reads(datapath, fn, bac_blast_folder, repeat_blast_folder)

            phage_reads = list(np.unique(phage_blast_data['query_id']))

            phage_reads_count += len(np.unique(phage_reads))
            bac_reads_count += len(np.unique(bac_reads))
            both_reads_count += len(np.intersect1d(phage_reads, bac_reads))
            neither_reads_count += total_reads - len(np.unique(phage_reads + bac_reads))

        phage_reads_count_list.append(phage_reads_count)
        bac_reads_count_list.append(bac_reads_count)
        both_reads_count_list.append(both_reads_count)
        neither_reads_count_list.append(neither_reads_count)
        total_reads_count_list.append(total_reads_count)
        
    # make dataframe
    read_counts_df = pd.DataFrame()

    read_counts_df["accession"] = accession_list
    # there is a newline at the end of the last file in the split files, so the extra .5 should be removed.
    read_counts_df["total_reads"] = total_reads_count_list 
    read_counts_df["num_phage_reads"] = phage_reads_count_list
    read_counts_df["num_bac_reads"] = bac_reads_count_list
    read_counts_df["total_matching_both"] = both_reads_count_list
    read_counts_df["total_matching_neither"] = neither_reads_count_list
    read_counts_df["time_point"] = np.arange(0,13)

    read_counts_df.to_csv("total_reads_normalization.csv")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get phage and bacteria read counts')
    parser.add_argument('datapath', type=str,nargs='?',
                        help='datapath to fasta files')
    
    args = parser.parse_args()
    
    # define parameters
    datapath = args.datapath

    get_read_counts(datapath)



