# CRISPR-dynamics-model

Welcome to the repository of simulation and figure code for the paper "[Dynamics of Immune Memory and Learning in Bacterial Communities](https://doi.org/10.1101/2022.07.07.498272)". This work is a model of interacting bacteria and phage with CRISPR, and a preprint can be found on bioRxiv: [10.1101/2022.07.07.498272](https://doi.org/10.1101/2022.07.07.498272).

This repository contains the code and data used to generate all main text figures and some supplementary figures. The [Index](https://github.com/mbonsma/CRISPR-dynamics-model#index) describes all the included files and data, the [Simulation pipeline](https://github.com/mbonsma/CRISPR-dynamics-model#simulation-pipeline) section describes the steps of running and analyzing the simulations, and the [Description of simulation files](https://github.com/mbonsma/CRISPR-dynamics-model#description-of-simulation-files) describes the structure of resulting simulation data. 
This repository also includes intermediate summary data for simulations and for previously published genetic data used in this study.
All raw simulation data is available on Dryad.

## Index

### Figure generation scripts

All data-based figures in the main text can be generated from the following files. Figures 1 and 4 in the main text are assembled in Inkscape, and Figure 6 has text annotations added in Inkscape; all other figures are generated exactly as presented in the paper with the following scripts. Each script can be run by cloning or downloading this repository, navigating to the `code` folder, and typing `python script_name.py` in a terminal; for example `python diversity_figure.py` to generate Figure 2 and associated supplementary figures. Not all supplementary figures are included in this code.

* Figure 1: [simulation_results_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_results_figure.py)
* Figure 2: [diversity_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/diversity_figure.py)
* Figure 3: [establishment_extinction_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/establishment_extinction_figure.py)
* Figure 4: [phylogeny.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/phylogeny.py)
* Figure 5: [crossreactivity_populations.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/crossreactivity_populations.py)
* Figure 6: [array_length_model.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/array_length_model.md)
* Figure 7: [abundance_speed_figure.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/abundance_speed_figure.py)
* Figure 8: [time_shift.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/time_shift.py)

### Processing data from [Paez-Espino et al. 2015](https://pubmed.ncbi.nlm.nih.gov/25900652/)

The source data is publicly available in the NCBI Sequence Read Archive under the accession [PRJNA275232](https://www.ncbi.nlm.nih.gov/bioproject/275232). We used raw read data from the MOI-2B series, which has accessions SRR1873837 through SRR1873849 for the 13 time points sequenced.

* [spacer_finder.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder.md): Jupyter notebook code for detecting matches to CRISPR repeats, extracting spacers from raw reads, clustering spacers, and analyzing protospacers. Some steps of the pipeline are performed on the supercomputer cluster [Niagara](https://docs.scinet.utoronto.ca/index.php/Niagara_Quickstart), specifically blasting spacer sequences against all reads and processing protospacer hits using [process_protospacers_niagara.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara.py).
  * Note: scripts and instructions for running command-line blast on the supercomputer can be found in my [PhD-materials repository](https://github.com/mbonsma/PhD-materials).
* [spacer_sorter.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter.md): count unique spacers and protospacers, cluster all spacers and protospacers with different grouping thresholds.
* [get_read_counts.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/get_read_counts.py): process BLAST data to count the total number of reads that match the phage or bacteria genome.
* [process_protospacers_niagara.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara.py): process results of blasting all spacers against all reads. First remove any hits that match to the bacteria genome or match to the CRISPR1 repeat (the CRISPR3 repeat was not checked, but $<0.1\%$ of reads matched the CRISPR3 repeat and neither the bacterial genome or the CRISPR1 repeat). The lowest e-value hit from each spacer type to each read was kept, and 10 nt downstream were extracted to analyze the presence of PAM sequences.
* [process_protospacers_niagara_setup.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/process_protospacers_niagara_setup.sh): create folders and scripts for parallel processing of protospacers on the supercomputer.
* [Banfield_spacer_correlations.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/Banfield_spacer_correlations.md): Jupyter notebook to analyze and group spacers and protospacers - produces the data in [data/PaezEspino2015](https://github.com/mbonsma/CRISPR-dynamics-model/tree/main/data/PaezEspino2015) and supplementary plots.

### Processing data from [Guerrero et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33067586/)

The source data for this section is publicly available in the NCBI Sequence Read Archive under the accession [PRJNA484416](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA484416). 

* [spacer_finder_Guerrero.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder_Guerrero.md): Jupyter notebook code based on [spacer_finder.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder.md) to detect spacers from raw reads.
* [spacer_sorter_Guerrero.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter_Guerrero.md): Jupyter notebook code based on [spacer_sorter.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter.md) to cluster spacers and protospacers with different grouping thresholds.
* [Guerrero_spacer_correlations.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/Guerrero_spacer_correlations.md): Jupyter notebook to analyze and group spacers and protospacers - produces the data in [data/Guerrero2021](https://github.com/mbonsma/CRISPR-dynamics-model/tree/main/data/Guerrero2021) and supplementary plots.

### Processing data from [Burstein et al. 2016](https://www.nature.com/articles/ncomms10613)

The source data for this section is publicly available in the NCBI Sequence Read Archive under the accession [PRJNA268031](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA268031). 

* [spacer_finder_Burstein.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder_Burstein.md): Jupyter notebook code based on [spacer_finder.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_finder.md) to detect spacers from raw reads.
* [spacer_sorter_Burstein.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter_Burstein.md): Jupyter notebook code based on [spacer_sorter.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_sorter.md) to cluster spacers and protospacers with different grouping thresholds.

### Simulation scripts

* [simulation_mutating_phage.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_mutating_phage.py): base simulation script in python.
* [simulation_mutating_phage_checkpoint_restart.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_mutating_phage_checkpoint_restart.py): simulation script to restart an in-progress simulation from a checkpoint.
* [sim_setup.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/sim_setup.sh): this script generates the individual folders for each simulation (i.e. serialjobdir0001), taking as its input a file with a list of parameters for each simulation to be run (named params_list.txt or params_list_run01.txt, etc). It generates `doserialjob0001.sh` which contains a single bash command to run `simulation_mutating_phage_niagara.py` or `simulation_mutating_phage_niagara_resume.py`.
* [niagara_submit_script_restart.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/niagara_submit_script_restart.sh): this script submits jobs to be run on the Niagara supercomputer after folders have been generated with `sim_setup.sh`. It automatically resubmits jobs 10 times, and checks if each individual simulation needs to be restarted or if it has not yet started.
* [create_params_list_scinet.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/create_params_list_scinet.py): generates a list of varying parameter combinations that can be fed to `simulation_mutating_phage.py`.
* [run_sims_parallel.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/run_sims_parallel.sh): short bash helper to use `gnu-parallel` to run multiple simulations on multiple cores.

### Simulation processing scripts

* [data_download.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/data_download.sh): bash script to download newly completed simulations from a server
* [sort-sims.md](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/sort-sims.md): process raw simulation results to create a sparse array of clone abundance over time (`pop_array_timestamp.npz`).
* [get_sims_to_analyze.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/get_sims_to_analyze.py): create a list of simulations that have not been analyzed with `simulation_stats.py` to be added to `all_data.csv`.
* [simulation_stats.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_stats.py): calculate summary quantities for simulations and store in the dataframe `all_data.csv`.
* [simulation_stats.sh](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/simulation_stats.sh): bash helper script to run `simulation_stats.py`.
* [extinction_df.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/extinction_df.py): analyze simulations that end in extinction and create `extinction_df.csv`.
* [predict_diversity.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/predict_diversity.py): numerically calculate predicted clone diversity based on simulation parameters, create summary dataframe `grouped_data.csv`.
* [sim_analysis_functions.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/sim_analysis_functions.py): functions used in simulation processing.
* [spacer_model_plotting_functions.py](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/code/spacer_model_plotting_functions.py): functions used in simulation processing and visualization.

### Data

* [all_data.csv](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/data/all_data.csv): processed simulation statistics in tabular format, created with `simulation_stats.py`
* [grouped_data.csv](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/data/grouped_data.csv): processed simulation statistics in tabular format, created with `predict_diversity.py`
* [extinction_df.csv](https://github.com/mbonsma/CRISPR-dynamics-model/blob/main/data/extinction_df.csv): processed simulation statistics relating to extinction, created with `extinction_df.py`
* [PaezEspino2015](https://github.com/mbonsma/CRISPR-dynamics-model/tree/main/data/PaezEspino2015): detected spacers and protospacers in data from Paez-Espino et al. 2015. The folders `perfect_PAM`, `partial_PAM`, and `no_PAM` contain protospacers with either a perfect CRISPR1 or CRISPR3 PAM, a partial truncated match to a PAM, or all results regardless of PAM respectively. The suffix `type_0.XX` indicates the similarity grouping threshold for assigning spacer types: `type_0.85` means that an 85% similarity threshold was used. The suffix `wt_True` or `wt_False` indicates whether matches to wild-type spacers were included in the data - `True` includes wild-type, `False` has wild-type removed. In general we used `perfect_PAM` data and included matches to wild-type spacers in our analysis.
* [Burstein2016](https://github.com/mbonsma/CRISPR-dynamics-model/tree/main/data/Burstein2016): detected spacers and protospacers in data from Burstein et al. 2016. 
* [Guerrero2021](https://github.com/mbonsma/CRISPR-dynamics-model/tree/main/data/Guerrero2021): detected spacers and protospacers in data from Guerrero et al. 2021. The folders `perfect_PAM` and `no_PAM` contain protospacers with either a perfect PAM or all results regardless of PAM respectively. The suffix `wt_True` or `wt_False` indicates whether matches to wild-type spacers were included in the data - `True` includes wild-type, `False` has wild-type removed. The suffix `phage_only_True` or `phage_only_False` indicates whether hits were required to be on reads that matched one of the draft phage genomes (DC-56 or DS-92) (`phage_only_True`) or if all matches were included regardless of read match to the phage genome (`phage_only_False`). In general we used the `perfect_PAM`, `wt_True`, and `phage_only_False` data for our analysis, thinking that a perfect PAM should indicate a genuine hit regardless of a match between a read and a draft genome. 
* `date/`: dated folders contain selected simulation data used to generate main text figures.

## Simulation pipeline - data processing & analysis

1. Generate list of parameters to simulate using `create_params_list_scinet.py`. This generates a file called `params_list.txt` with each row giving a list of parameters accepted by `simulation_mutating_phage_niagara.py`.
2. Copy each parameter combination four times for a total of five independent simulations with the same parameters: 
  `while read line; do for i in {1..5}; do echo "$line"; done; done < params_list.txt > params_list2.txt`
  `mv params_list2.txt params_list.txt`
3. Place the file `params_list.txt` in the top level folder to create a simulation folder in.
4. Run `sim_setup.sh` to create simulation run folders. We used the date as the folder name:
  `bash sim_setup.sh params_list.txt folder_name`
5. Run simulations - either using `run_sims_parallel.sh` or something like it if running directly, or using `niagara_submit_script_restart.sh` if running on supercomputer. 
  `nohup bash run_sims_parallel.sh &` # nohup detaches it from the terminal so it keeps going if it closes
6. Download completed simulations using `data_download.sh`
7. Process simulations into a single array for the entire simulation using `sort-sims.ipynb` - this script loops through each serialjobdir* directory and creates pop_array_timestamp.npz, a scipy sparse array whose structure is described above.
8. Get simulation statistics - run `get_sims_to_analyze.py` to generate a list of unanalyzed simulations, then run `simulation_stats.py` with the helper script `simulation_stats.sh`.
9. Calculate diversity using `predict_diversity.py`.

## Description of simulation files

The folder `data/` contains selected simulation results that are used to generate main text figures. The raw files `populations_timestamp.txt` and `protospacers_timestamp.txt` are not included because their information is stored in `pop_array_timestamp.npz`.

All results files from a simulation end with the suffix `_timestamp.txt`, where `timestamp` is generated with the datetime package: `timestamp = datetime.datetime.now().isoformat()`. Example timestamp: `2019-03-16T06:29:24.331296`.

`all_phages_timestamp.txt` - an ordered list of the nucleotide sequences of all protospacers present at any time in the simulation. This file is saved over every time a checkpoint is saved, meaning that it contains the list of all phages present at the last checkpoint in the simulation. Combined with `pop_array_timestamp.npz`, this file identifies the protospacer sequence for each column in `pop_array_timestamp.npz`: the index of a phage in `all_phages` corresponds to its position in `pop_array`.

`mutation_times_timestamp.txt` - an ordered list of all the times at which a particular protospacer sequence arose by mutation. The total list is the same length as `all_phages_timestamp.txt`, and the position in the outside list matches the phage sequence in `all_phages_timestamp.txt`. Each sublist is the times at which that protospacer arose by mutation. A protospacer can be mutated to more than once by the same phage population, or it can be mutated to by a different phage population that is also one mutation away.

`parameters_timestamp.txt` - a list of the parameter values used for this particular simulation. The first column is the parameter name, and the second column is its value. Parameter descriptions are above.

`parents_timestamp.txt` - an ordered list of same shape as `mutation_times` that identifies the source phage sequence for each phage mutant that appears in the simulation. For example, if `parent_list[1][0] = 0`, this means that the phage sequence identified as type 1 first arose by mutation from type 0. An example corresponding mutation time is `mutation_times[1][0] = 1.6400876`. Each sublist can have more than one entry meaning that that protospacer sequence arose by mutation more than once, and by comparison with the same sublist in `mutation_times` this identifies the times each mutation happened and which phage mutated to which other phage type.

`populations_timestamp.txt` - a list of the population sizes at each saved checkpoint. The number of lines in this file is the number of saved checkpoints. Each line can be variable length because the population array in the simulation changes size dynamically: as new phages arise by mutation the number of subpopulations grows, and periodically the simulation checks if any subpopulations have gone extinct and deletes those from the population array. This file is combined with `protospacers_timestamp.txt` to uniquely identify each population over time despite the dynamically changing fields. Each line has the following fields separated by whitespace: nB0, {nBi}, {nVi}, C, t.
nB0: the total number of bacteria without spacers (always 1 field)
{nBi}: the number of bacteria with each of the i spacers present at time t (variable number of fields: minimum 1). The number of fields is the same as the number for nVi so that types can be matched to each other. This means that there are often many zeros in the {nBi} columns.
{nVi}: the number of phages with each of the i protospacers present at time t (variable number of fields: minimum 1).
C: the nutrient concentration at time t (always 1 field)
t: the checkpoint time in minutes (always 1 field)

`protospacers_timestamp.txt` - a list of the i spacer / protospacer sequences present at each saved checkpoint. Each spacer / protospacer sequence is length L (L = 30). The number of lines in this file is the number of saved checkpoints; the file length is the same as `populations_timestamp.txt`. These sequences uniquely identify each field in `populations_timestamp.txt`. Example: the first line of `2019-05-14/run02/serialjobdir0021/populations_2019-05-14T13:13:11.268118.txt` is [210.0, 0.0, 2100.0, 90.0, 0]. This means at the start of the simulation there are 210 bacteria without spacers, no bacteria with spacers, 2100 phages of a single type, and 90 nutrients at time 0. The first line of `2019-05-14/run02/serialjobdir0021/protospacers_2019-05-14T13:13:11.268118.txt` is [[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]], which is the sequence of the single phage population's protospacer. The first four mutations happen by the 12th line of the populations file: [161.0, 0.0, 0.0, 0.0, 0.0, 7025.0, 1.0, 1.0, 1.0, 89.0, 42.11249915286325]. Now there are four phage types of sizes 7025, 1, 1, and 1 respectively, but still there are zero bacteria with spacers, so the 2nd through 5th entries of the populations line are zero. This happens at t = 42.11 minutes (the last entry). The corresponding line in the protospacers file is [[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]]. This line contains a list with four items, where each item is the sequence of the ith protospacer for the ith phage population.

`pop_array_timestamp.npz` - A scipy sparse array which is a re-organized version of the data in `populations_timestamp.txt`. Rows are time points, columns are unique populations (determined by spacer or protospacer sequence). Each unique population ever present in the simulation has its own column. `max_m` is the total number of unique populations ever present in the simulation.
    pop_array structure:

    | Columns                 | Description |
    | 0                       | $n_B^0$     |
    | 1 : max_m + 1`          | $n_B^i$     |
    | max_m + 1 : 2*max_m + 1 | $n_V^i$     |
    | 2*max_m + 2 or -2       | $C$         |
    | 2*max_m + 3 or -1       | $t$ (mins)  |



