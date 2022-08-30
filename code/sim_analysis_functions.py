#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:08:08 2018

@author: madeleine
"""

import numpy as np
import pandas as pd
from scipy import sparse
import os

def recreate_x(x_line):
    """Each line of the data file is a vector x with length 2m + 3
    x[0]: nb0
    x[1:m+1]: nbi
    x[m+1:2m+1]: nvi
    x[2m+2]: C
    x[2m+3]: t
    """
    return np.array(x_line.decode("utf-8").split('[')[1].split(']')[0].split(','), dtype = 'float32')

def recreate_phage(phage_row):
    """
    Input: the list of prophages for a particular timepoint (i.e phage[-1], where phages is read in above)
    Output: the same line formatted as a list of lists of integers
    """
    phage_list = []
    
    phages_string = phage_row.decode("utf-8").split('[')[2:]
    for phage in phages_string:
        phage = phage.split(']')[0]
        phage_list.append(list(np.array(phage.split(','),dtype=int)))
    
    return phage_list
    
def remove_zeros(x,m):
    """
    Remove empty entries in simulation data, i.e. if there are no bacteria or phage at that index
    
    Inputs:
    x: vector of length 2*m + 3 where each entry is a population abundance.
    x[0] = nb0, x[1:m+1] = nbi, x[m+1:2*m+1] = nvi, x[2*m+1] = C, x[2*m + 2] = time  
    m: number of unique phage species in the population
    
    Returns:
    x and m after removing matching zero entries. x is still length 2*m+3, but m may have decreased.
    """
    zero_inds = np.where(np.logical_and(x[m+1:2*m+1] == 0, x[1:m+1] == 0) == True)[0] # where both phage and bac are 0
    if len(zero_inds) == 0:
        return x, m
    else:
        x = np.delete(x, m+1 + zero_inds) # delete phage
        x = np.delete(x, 1 + zero_inds) # delete corresponding bacteria
        m -= len(zero_inds)
        return x, m

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_pop_array(data, all_phages, phages):
    """
    Create array that contains all the simulation data. 
    Rows are time points, columns are unique populations. 
    Each unique population ever present in the simulation has its own column.
    This is not much faster than creating an array directly, 
    but it is much faster to save and load.
    
    pop_array structure:
        
    | Columns                 | Description |
    | 0                       | $n_B^0$     |
    | 1 : max_m + 1`          | $n_B^i$     |
    | max_m + 1 : 2*max_m + 1 | $n_V^i$     |
    | 2*max_m + 2 or -2       | $C$         |
    | 2*max_m + 3 or -1       | $t$ (mins)  |
    
    Inputs:
    data: file object created from reading "populations.txt" with readlines()
    all_phages: list of all unique phages ever present in the simulation
    phages: file object created from reading "protospacers.txt" with readlines()
    
    Outputs:
    pop_array: sparse scipy array of data, structured as above
    max_m: total number of unique species (phage or bacteria) in the simulation
    """
    
    max_m = len(all_phages) # from a previous run
    nrows = len(data) 

    data_vec = []
    i_vec = []
    j_vec = []
    max_j = max_m*2 + 3
    
    for i, row in enumerate(data):
        x = recreate_x(row)
        m = int((len(x) - 3)/2)
        phage_list = recreate_phage(phages[i]) 

        # possibly do i_vec all at once:
        # i_vec += [i]*len(x)

        # nb0
        data_vec.append(x[0])
        i_vec.append(i)
        j_vec.append(0)

        # time
        data_vec.append(x[-1])
        i_vec.append(i)
        j_vec.append(max_j-1)

        # c
        data_vec.append(x[-2])
        i_vec.append(i)
        j_vec.append(max_j-2)

        # add population totals to pop_array
        for count2, phage in enumerate(phage_list):       
            ind = all_phages.index(phage)
            # nbi
            data_vec.append(x[1+count2])
            i_vec.append(i)
            j_vec.append(ind + 1)

            # nvi
            data_vec.append(x[m + 1 + count2])
            i_vec.append(i)
            j_vec.append(max_m + 1 + ind)
            
    pop_array = sparse.coo_matrix((data_vec, (i_vec, j_vec)), shape=(nrows, max_j),  dtype = 'float32')
    pop_array = sparse.csr_matrix(pop_array, dtype = 'float32')
    
    return pop_array, max_m
        
def PV(i,j, pv, e):
    if i == j:
        return pv*(1-e)
    else:
        return pv
    
def recreate_parent_list(parent_row):
    """
    Input: list of prophage parents for a particular time point (i.e. parent_list[-1], where parent_list is read
    in above)
    Output: the same line formatted as a list of tuples of integers, or 'nan' if the phage has no parent (i.e. 
    is one of the original phages)
    """
    parent_list_row = []
    parents_string = parent_row.decode("utf-8").split('[')[2:]
    for parent in parents_string:
        parent = parent.split(']')[0]
        if parent == "''": # this is one of the original phages with no back mutations
            parent_list_row.append([])
        else: # has at some point arisen by mutation
            # check if any of the list are blank
            parent = parent.split(',')
            try:
                ind = parent.index("''")
                parent[ind] = np.nan
            except:
                pass
            try:
                ind = parent.index('')
                parent[ind] = np.nan
            except:
                pass
            parent_list_row.append(list(np.array(parent,dtype='float32')))
        
    return parent_list_row

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root, name
        
def load_simulation(folder, timestamp, save_pop = True, return_parents = False):
    #print("loading parameters")
    # load parameters
    parameters = pd.read_csv(folder + "/parameters_%s.txt" %timestamp, delimiter = '\t', header=None)
    parameters.columns = ['parameter', 'value']
    parameters.set_index('parameter')

    f = float(parameters.loc[parameters['parameter'] == 'f']['value'])
    c0 = float(parameters.loc[parameters['parameter'] == 'c0']['value'])
    g = float(parameters.loc[parameters['parameter'] == 'g']['value'])
    B = float(parameters.loc[parameters['parameter'] == 'B']['value'])
    R = float(parameters.loc[parameters['parameter'] == 'R']['value'])
    eta = float(parameters.loc[parameters['parameter'] == 'eta']['value'])
    pv = float(parameters.loc[parameters['parameter'] == 'pv']['value'])
    alpha = float(parameters.loc[parameters['parameter'] == 'alpha']['value'])
    e = float(parameters.loc[parameters['parameter'] == 'e']['value'])
    L = float(parameters.loc[parameters['parameter'] == 'L']['value'])
    mu = float(parameters.loc[parameters['parameter'] == 'mu']['value'])
    m_init = float(parameters.loc[parameters['parameter'] == 'm_init']['value'])
    gen_max = float(parameters.loc[parameters['parameter'] == 'gen_max']['value'])
    max_save = float(parameters.loc[parameters['parameter'] == 'max_save']['value'])
    try:
        theta = float(parameters.loc[parameters['parameter'] == 'theta']['value'])
    except:
        theta = 0.0
        
    # load list of all phages that ever existed
    with open(folder + "/all_phages_%s.txt" %timestamp, "rb") as all_phages_file:
        all_phages = all_phages_file.readlines()
    
    #print("creating list of all phages")
    all_phages = recreate_phage(all_phages[0])
    
    #print("attempting to load pre-made pop_array...")
    try: # try loading pop_array directly
        pop_array = sparse.load_npz(folder + "/pop_array_%s.txt.npz" %timestamp) # fast <3 <3 <3
        max_m = int((pop_array.shape[1] -3)/2)
        #Sprint("loading from existing pop_array file")
            
    except: #if pop_array doesn't exist, load the necessary files and create it
        print("creating pop_array")
        with open(folder + "/populations_%s.txt" %timestamp, "rb") as popdata:
            data = popdata.readlines()
            
        with open(folder + "/protospacers_%s.txt" %timestamp, "rb") as protospacers:
            phages = protospacers.readlines()

        pop_array, max_m = create_pop_array(data, all_phages, phages)
        
        if save_pop == True: # save pop_array so it can be loaded quicker
            #print("saving pop_array")
            sparse.save_npz(folder + "/pop_array_%s.txt" %timestamp, pop_array)
    


    with open("%s/mutation_times_%s.txt" %(folder,timestamp), "rb") as mut_f:
        mutation_t = mut_f.readlines()

    mutation_times = recreate_parent_list(mutation_t[0])

    
    if return_parents == True:
        #print("loading parents and mutation times")
        with open("%s/parents_%s.txt" %(folder,timestamp), "rb") as par:
            parents = par.readlines()
        parent_list = recreate_parent_list(parents[0])
        return f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, max_m, mutation_times, parent_list, all_phages
    
    else:
        return f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, max_m, mutation_times, all_phages


if __name__ == "__main__":
    pass
    
