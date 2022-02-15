# CRISPR-dynamics-model

Simulation and figure code for a model of interacting bacteria and phage with CRISPR.

## Index

### Processing data from [Paez-Espino et al. 2015](https://pubmed.ncbi.nlm.nih.gov/25900652/)

This data is publicly available in the NCBI Sequence Read Archive under the accession [PRJNA275232](https://www.ncbi.nlm.nih.gov/bioproject/275232). We used raw read data from the MOI-2B series, which has accessions SRR1873837 through SRR1873849 for the 13 time points sequenced.

* [spacer_finder.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder.md): Jupyter notebook code for detecting matches to CRISPR repeats, extracting spacers from raw reads, clustering spacers, and analyzing protospacers. Some steps of the pipeline are performed on the supercomputer cluster [Niagara](https://docs.scinet.utoronto.ca/index.php/Niagara_Quickstart), specifically blasting spacer sequences against all reads and processing protospacer hits using [process_protospacers_niagara.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara.py).
  * Note: scripts and instructions for running command-line blast on the supercomputer can be found in my [PhD-materials repository](https://github.com/mbonsma/PhD-materials).
* [spacer_sorter.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter.md): count unique spacers and protospacers, cluster all spacers and protospacers with different grouping thresholds.
* [process_protospacers_niagara.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara.py): process results of blasting all spacers against all reads. First remove any hits that match to the bacteria genome or match to the CRISPR1 repeat (the CRISPR3 repeat was not checked, but $<0.1\%$ of reads matched the CRISPR3 repeat and neither the bacterial genome or the CRISPR1 repeat). The lowest e-value hit from each spacer type to each read was kept, and 10 nt downstream were extracted to analyze the presence of PAM sequences.
* [process_protospacers_niagara_setup.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara_setup.sh): create folders and scripts for parallel processing of protospacers on the supercomputer.

### Simulation scripts

* [simulation_mutating_phage.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_mutating_phage.py): base simulation script in python.
* [simulation_mutating_phage_checkpoint_restart.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_mutating_phage_checkpoint_restart.py): simulation script to restart an in-progress simulation from a checkpoint.
