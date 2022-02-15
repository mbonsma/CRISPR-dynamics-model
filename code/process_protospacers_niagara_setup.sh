#!/bin/bash

# run this from the dated folder in which to run processing script
# inputs: accession, query, spacer_blast_top_folder, cr
# spacer_blast_top folder is the dated folder that contains "accession_blast" with the blast results

# usage:
# bash process_protospacers_niagara_setup.sh accession query spacer_blast_top_folder cr

# example usage:
# bash process_protospacers_niagara_setup.sh SRR1873837 unique_spacers_CR3.fasta ../2022-02-03 CR3

#cp $HOME/process_protospacers_submit_script_niagara.sh . # copy submit script to current folder
cp $HOME/process_protospacers_niagara.py . # copy processing script to current folder

accession=$1
query=$2
spacer_blast_top_folder=$3
cr=$4

# create serial script for each fasta sub-file

num="$(ls "$accession"/"$accession"* | wc -l)" # get number of files to run
ls "$accession"/"$accession"* > accessions.txt


counter=1
i=$(printf %04d $counter)

# make directory for serialjobdir files
if [ ! -d "$accession"_process ]; then
  mkdir "$accession"_process 
fi

cp process_protospacers_niagara.py "$accession"_process

while read -r line || [[ -n "$line" ]];
do
  acc="$(echo $line | cut -d"/" -f2)"; # get filename
  echo $acc
  echo python process_protospacers_niagara.py $accession $acc $query "$spacer_blast_top_folder"/"$accession"_blast $cr > "$accession"_process/doserialjob$i.sh; # make run script for each accession
  ((counter+=1));
  i=$(printf %04d $counter);
done <accessions.txt




