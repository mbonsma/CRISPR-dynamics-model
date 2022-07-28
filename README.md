# CRISPR-dynamics-model

Simulation and figure code for a model of interacting bacteria and phage with CRISPR. bioRxiv DOI: [10.1101/2022.07.07.498272](https://doi.org/10.1101/2022.07.07.498272).

## Index

### Figure generation scripts

All data-based figures in the main text can be generated from the following files. Figures 1 and 4 in the main text are assembled in Inkscape, and Figure 6 has text annotations added in Inkscape; all other figures are generated exactly as presented in the paper with the following scripts. Each script can be run by navigating to the `code` folder and typing `python script_name.py`; for example `python diversity_figure.py` to generate Figure 2 and associated supplementary figures. Not all supplementary figures are included in this code.

* Figure 1: [simulation_results_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_results_figure.py)
* Figure 2: [diversity_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/diversity_figure.py)
* Figure 3: [establishment_extinction_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/establishment_extinction_figure.py)
* Figure 4: [phylogeny.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/phylogeny.py)
* Figure 5: [crossreactivity_populations.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/crossreactivity_populations.py)
* Figure 6: [abundance_speed_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/abundance_speed_figure.py)
* Figure 7: [time_shift.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/time_shift.py)

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
