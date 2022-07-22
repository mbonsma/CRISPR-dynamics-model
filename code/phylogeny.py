# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: spacer_phage
#     language: python
#     name: spacer_phage
# ---

# # Phylogenetic trees (Figure 4)
#
# Code for Figure 4, assembled in Inkscape.

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm
import dendropy
import toytree
import toyplot.svg
import toyplot.pdf
from cycler import cycler

# %matplotlib inline

from sim_analysis_functions import find_nearest
from sim_analysis_functions import load_simulation, find_file


def create_distance_matrix(all_phages):
    # create distance matrix for all the phages that ever lived
    all_phages = np.array(all_phages)

    distance_matrix = np.zeros((len(all_phages), len(all_phages)))

    for i in range(len(all_phages)):
        distance_matrix[i] = np.sum(np.abs(all_phages - all_phages[i]), axis = 1)
        
    return distance_matrix


def create_phages_dict(m_init, all_phages, parent_list, mutation_times, distance_matrix):
    """
    Create a dictionary of all the inheritance information in the simulation.
    
    """
    
    # create phages_dict
    # the initial phages start at position 0 in the phage list

    m_init = int(m_init)

    phages_dict = {}

    # keys: 

    for i, phage in enumerate(all_phages):
        if i < m_init:
            continue

        parent_ids = parent_list[i]

        parent_distances = []
        mutation_positions = []
        mutation_ts = mutation_times[i]

        loop_list = parent_ids
        #if len(np.unique(parent_ids)) > 1:
        #    loop_list = parent_ids

        #else:
        #    loop_list = np.unique(parent_ids)
        #    mutation_ts = [mutation_ts[0]]

        for j, pid in enumerate(loop_list):
            pid = int(pid)
            parent_distance = distance_matrix[i, pid]
            parent_distances.append(parent_distance)

            mutation_pos = np.where(np.abs(phage - np.array(all_phages[pid])) == 1)[0]
            if mutation_pos[0] < 0:
                mutation_pos += 30
            mutation_positions.append(mutation_pos)

        phages_dict[i] = {"sequence": phage, "parents": loop_list, "mutation_times": mutation_ts, "parent_distance": parent_distances,
                        "mutation_position": mutation_positions}
        
    return phages_dict


def make_newick_tree(max_m, all_phages, nvi, times, nvi_cutoff = 2):
    """Make a tree in newick format for simulation data.
    Inputs:
    max_m : the total number of protospacer types in the simulation, returned by load_simulation
    all_phages : a list of the sequence of each protospacer type, corresponding to columns in pop_array
    nvi : array of phage clone sizes over time
    times : time series that goes with nvi
    nvi_cutoff : the minimum size of a phage clone to include in the plot
    
    Returns:
    tree : the newick format tree
    extinction_times : the extinction times for each phage clone
    """
    
    names_int = np.arange(max_m) # names are needed for toytree, assign each node its index in the array as a name
    names = []
    extinction_times = [None]

    for name in names_int:
        names.append(str(name))

    taxon_namespace = dendropy.TaxonNamespace(names)
    tree = dendropy.Tree(taxon_namespace=taxon_namespace)

    node_array = [tree.seed_node] # len (all_phages)

    for i, phage in enumerate(all_phages):
        if i == 0:
            continue # skip the initial phage

        if not np.any(nvi[:,i] >= nvi_cutoff):
            node_array.append(None)
            extinction_times.append(None)
            continue

        final_extinction_time = times[nvi[:,i] > 0][-1] # the last time this clone is detected
        establishment_time = times[nvi[:,i] >= nvi_cutoff][0]

        # get the last parent
        #parent_ind = int(phages_dict[i]["parents"][-1]) 

        # force parent to have a lower index than child 
        parents = np.array(phages_dict[i]["parents"])
        parent_ind = int(parents[parents < i][-1])


        # get parent establishment time
        parent_establishment_time = times[nvi[:,parent_ind] >= nvi_cutoff][0]

        # get parent index in all_phages

        ch = dendropy.Node(edge_length=float(establishment_time - parent_establishment_time))
        ch.taxon = taxon_namespace.get_taxon(names[i])

        #if final_extinction_time - parent_extinction_time < 0:
        #    print ("negative line")

        parent_node = node_array[parent_ind]
        parent_node.add_child(ch)

        node_array.append(ch)

        extinction_times.append(final_extinction_time)
    
    return tree, extinction_times

# +
# find folder based on timestamp
all_data = pd.read_csv("../data/all_data.csv", index_col = 0)

# binary, theta = 1, theta = 2, exponential_025 with nice traveling wave
timestamps = ['2019-03-18T02:51:49.500085', '2021-06-17T13:40:03.119258','2021-06-17T15:21:36.357638', '2021-09-12T20:20:54.807443']

# alternate timestamps for SI: higher mutation rate (no exponential examples)
#timestamps = ['2019-02-25T14:24:15.599257', #binary mu = 10^-5
#             '2021-06-24T01:53:46.265313', # theta = 1
#             '2021-06-24T03:33:05.147621', # theta = 2
#             '2021-06-24T04:47:15.005351'] # theta = 3

# higher population sizes c0 = 3*10^4
#timestamps = ['2021-09-09T15:34:58.101935', # binary
#             '2021-09-20T10:03:49.723068', #exponential
#             '2021-11-06T21:22:50.020517', # theta = 1
#              '2021-11-06T21:22:50.022542'] # theta = 2
# -

all_data_subset = all_data[(all_data['timestamp'].isin(timestamps))]

# +
# load data

pop_array_list = []
mutation_times_list = []
parents_list = []
all_phages_list = []

c0_list = []
g_list = []
eta_list = []
mu_list = []
alpha_list = []
e_list = []
m_init_list = []
max_m_list = []
bac_extinct_list = []
phage_extinct_list = []
gen_max_list = []
theta_list = []
timestamps_list = []

for i, timestamp in tqdm(enumerate(timestamps)):
    
    top_folder = "../data/" + str(all_data[all_data['timestamp'] == timestamp]['folder_date'].values[0])
    
    folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, top_folder)

    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
     max_m, mutation_times, parent_list, all_phages = load_simulation(folder, timestamp, return_parents = True)


    c0_list.append(c0)
    g_list.append(g)
    eta_list.append(eta)
    mu_list.append(mu)
    m_init_list.append(m_init)
    max_m_list.append(max_m)
    alpha_list.append(alpha)
    e_list.append(e)
    gen_max_list.append(gen_max)
    theta_list.append(theta)

    #print(c0, eta, mu, m_init)
    #print(timestamp)
    #pop_array = pop_array.toarray() # convert to dense array

    pop_array_list.append(pop_array.toarray())
    mutation_times_list.append(mutation_times)
    parents_list.append(parent_list)
    all_phages_list.append(all_phages)
    timestamps_list.append(timestamp)
# -

# ### Make newick format tree

# get colours to correspond to extinction times
cmap = cm.get_cmap('viridis')

# iterate through simulations and create tree plots
for i, pop_array in enumerate(pop_array_list):
    mutation_times = mutation_times_list[i]
    all_phages = all_phages_list[i]
    parent_list = parents_list[i]
    m_init = m_init_list[i]
    eta = eta_list[i]
    mu = mu_list[i]
    max_m = max_m_list[i]
    g = g_list[i]
    c0 = c0_list[i]
    #alpha = alpha_list[i]
    timestamp = timestamps[i]
    gen_max = gen_max_list[i]
    
    distance_matrix = create_distance_matrix(all_phages)
    
    phages_dict = create_phages_dict(m_init, all_phages, parent_list, mutation_times, distance_matrix)
    
    nvi = pop_array[:, max_m+1:2*max_m+1]
    times = pop_array[:, -1]*g*c0
    
    tree, extinction_times = make_newick_tree(max_m, all_phages, nvi, times, nvi_cutoff = 2)
    
    # save in newick format
    newick = tree.as_string("newick")
    
    # make toytree object
    # check tree format here: we have all nodes named http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html
    tree0 = toytree.tree(newick, tree_format = 3)
    
    # add max population size and colours
    for node in tree0.treenode.traverse():
        node.add_feature("max_size", np.log(np.max(nvi[:, int(node.name)]) + 1))
        try:
            c = cmap(extinction_times[int(node.name)] / gen_max)
            node.add_feature("colour", matplotlib.colors.to_hex(c))
        except TypeError: # single error in 2nd simulation analyzed
            print("missing node: %s" %int(node.name))
            node.add_feature("colour", 'black')
    
    sizes = tree0.get_node_values('max_size', show_root=1, show_tips=1)
    colours = tree0.get_node_values('colour', show_root=1, show_tips=1)
    
    # define a style dictionary for toytree plotting
    mystyle = {"node_hover" : False,
               "node_labels": False,
               "tip_labels": False,
               "height":300,
               "width":600,
               "edge_style": {
                "stroke": 'black',
                "stroke-width": 0.75,
               "stroke-opacity": 0.2},
               "node_style": {"stroke": "black",
                             "stroke-width": 0.75},
                "node_sizes": sizes,
                "node_colors": colours,
               "tree_style": "n",
               "edge_type": "p"
    }

    canvas, axes, makr = tree0.draw(**mystyle)
    axes.show = True
    axes.x.ticks.show = True
    
    # save tree figure
    toyplot.svg.render(canvas, "tree-plot_%s.svg" %timestamp)

# +
fig, ax = plt.subplots(figsize = (9,0.25))

norm = matplotlib.colors.Normalize(vmin=times[0], vmax=gen_max)

cb1 = matplotlib.colorbar.ColorbarBase(ax=ax, cmap= cm.viridis,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label("Clone extinction time (bacterial generations)")


plt.savefig("phylogeny_colorbar_%s.svg" %int(gen_max))
# -

# ### Trajectory for figure insets

# +
# handpicked ranges for each simulation to highlight specific features
start_time_list = [2000, 3300, 7500, 2700]
end_time_list = [4000, 4100, 8700, 3900]

subsample = 5
colours = cm.tab20(np.linspace(0,1,20))
# -

for i, pop_array in enumerate(pop_array_list):
    mutation_times = mutation_times_list[i]
    all_phages = all_phages_list[i]
    parent_list = parents_list[i]
    m_init = m_init_list[i]
    eta = eta_list[i]
    mu = mu_list[i]
    max_m = max_m_list[i]
    g = g_list[i]
    c0 = c0_list[i]
    #alpha = alpha_list[i]
    timestamp = timestamps[i]
    gen_max = gen_max_list[i]
    
    time = start_time_list[i]
    end_time = end_time_list[i]
    
    t_ind = find_nearest(pop_array[:, -1]*g*c0, time)
    t_ind_end = find_nearest(pop_array[:, -1]*g*c0, end_time)

    fig, axs = plt.subplots(2,1, figsize = (3,1.65))

    ax = axs[0]
    ax1 = axs[1]

    custom_cycler = (cycler(color=colours))
    ax.set_prop_cycle(custom_cycler)
    ax1.set_prop_cycle(custom_cycler)

    ax.plot(pop_array[t_ind:t_ind_end:subsample, -1]*g*c0, pop_array[t_ind:t_ind_end:subsample, max_m+1 : 2*max_m+1], linewidth = 1);
    #ax.set_xlim(8000, 8800)
    ax.set_yscale('log')
    #ax.set_xlabel("Time (bacterial generations)")
    ax.set_ylabel("Phage")

    ax1.plot(pop_array[t_ind:t_ind_end:subsample, -1]*g*c0, pop_array[t_ind:t_ind_end:subsample, 1 : max_m+1], linewidth = 1);
    #ax.set_xlim(8000, 8800)
    ax1.set_yscale('log')
    #ax1.set_xlabel("Time (bacterial generations)")
    ax1.set_ylabel("Bacteria")

    ax.set_xlim(time, end_time)
    ax1.set_xlim(time, end_time)
    #ax1.set_xticks(np.arange(time, time+1000, 200))

    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig("clone_size_%s_start_%s_end_%s.pdf" %(timestamp, time, end_time))
    plt.savefig("clone_size_%s_start_%s_end_%s.svg" %(timestamp, time, end_time))
