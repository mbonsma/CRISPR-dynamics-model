#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:08:36 2018

@author: madeleine

Usage:
python simulation_mutating_phage_niagara.py B C0 g f R eta pv alpha e mu L epsilon m_init max_save gen_max
     
Example usage: 
python simulation_mutating_phage_niagara.py 170 10000.0 2.3809523809523808e-06 0.3 0.04 0.001 0.02 2e-06 0.5 3.162277660168379e-06 30 0.03 50 20000 10000.0

"""

import numpy as np
import random, datetime, argparse
import pandas as pd
#from tqdm import tqdm

def gillespie_time(a0): 
    """
    draw a random number from an exponential distribution with rate parameter a0
    """
    return -np.log(random.uniform(0,1))/a0 

def gillespie_reaction(a, a0, numReactions):
    """
    Returns j, where j is the index of the reaction that occurs at this time step.
    """
    a_cdf = np.cumsum(a) # cumulative sum of rates for all reactions
    random_num = random.uniform(0,a0) # draw a random number between 0 and the total rate a0
    # return the first index where the sum of rates is greater than the random number drawn - this is the reaction that happens
    return np.where(a_cdf > random_num)[0][0] 

# v in the case of phage mutations - this is v up to but not including the mutation terms
def species_changes(m, B, numSpecies, numReactions):
    """
    Calculates matrix v: rows are species index, columns are reactions index.
    v(i,j) gives the change in species i if reaction j occurs.
    
    Inputs: 
    m: number of phage types
    B: phage burst size
    numSpecies: total number of species = 2*m + 2
    numReactions: total number of reactions = (9 + 2*m) * m + 3
    
    Outputs: 
    v: matrix of species changes
    """

    v = np.zeros((numSpecies, numReactions), dtype = 'int16')

    for i in range(m+1):
        # bacteria growth
        v[i, i] = 1 
        v[2*m+1, i] = -1
        # bacteria flow
        v[i, i+m+1] = -1

    #carbon in
    v[2*m+1,3*m+2]=1;
    #carbon out
    v[2*m+1,3*m+3]=-1;

    for i in range(m):
        #phage flow
        v[m+1+i,2*m+2+i]=-1; 
        
        #lose spacer
        v[i+1,3*m+4+i]=-1;
        v[0,3*m+4+i]=1;

        # v-b0, b0 wins
        v[m+1+i, 4*m+4+i]=-1

        #v-b0 gain spacer i
        v[0,5*m+4+i]=-1;
        v[m+1+i,5*m+4+i]=-1;
        v[i+1,5*m+4+i]=1;

    # don't need the phage success terms, covered by mutation case
    for j in range(m):
        # vi-b0, phage wins
        v[0,6*m+4+j]=-1;
        v[m+1+j,6*m+4+j]=B-1;

    for i in range(m):
        for j in range(m):
            #v-bi, b wins
            v[m+1+j,(7+i)*m+4+j]=-1;
    #
    for i in range(m):
        for j in range(m):
            #vi-bi, vi wins
            v[1+i,(8+i+m)*m+4+j] = -1 # bac dies
            v[m+1+j, (8+i+m)*m+4+j] = B-1
            
    return v

# total rates: rate of reaction j at current population sizes for phage mutations


def total_rates_phage(numReactions, x, m, B, g, e, eta, pv, alpha, F, r, c0, mu, L):
    """ 
    Calculates the rate of reaction j at population sizes given by the vector x
    numReactions = (9 + 2*m) * m + 3
    """
    
    a = np.zeros((numReactions))
    
    for i in range(m+1):
        # bacteria growth
        a[i] = g * x[2*m+1] * x[i] #g*C*nb
        # bacteria flow
        a[i+m+1] = F*x[i] #F*nb

    # carbon in
    a[3*m+2] = F*c0
    # carbon out
    a[3*m+3] = F*x[2*m+1]
   
    for i in range(m):
        a[2*m+2 + i] = F*x[m+1+i] # phage flow
        a[3*m+4+i] = r*x[i+1] # lose a spacer
        a[4*m+4+i] = alpha*(1-pv)*(1-eta)*x[0]*x[m+1+i] # vi vs b0, bac wins, no acquisition
        a[5*m+4+i] = alpha*(1-pv)*eta*x[0]*x[m+1+i] # vi vs b0, bac wins, spacer acquired
        a[6*m+4+i] = alpha*pv*x[0]*x[m+1+i] # vi vs b0, v wins
        
        for j in range(m):
            a[(7+i)*m+4+j] = alpha*(1-PV(i, j, pv, e))*x[1+i]*x[m+1+j] # bi vs vj, bi wins
            a[(8+m+i)*m+4+j] = alpha*PV(i,j,pv,e)*x[1+i]*x[m+1+j] # bi vs vj, vj wins
    
    return a
    
def PV(i,j, pv, e):
    """
    Probability of success for phage j against bacterium i
    """
    if i == j:
        return pv*(1-e)
    else:
        return pv
    
def tau_time_step(x, v, a, epsilon):
    """
    For every species, calculate tau', the largest timestep within tolerance
    
    Input:
    x: population sizes vector
    v: matrix of species changes (from species_changes function)
    a: vector of rates for each reaction (length numReactions, from total_rates_phage function)
    
    Out1: a vector of length numSpecies
    """       
    
    # calculate means
    mask = np.ones(a.shape,dtype=bool)
    mask[a.nonzero()[0]] = False
    mask = np.logical_not(mask)
    means = np.sum(a[mask]*v[:,mask], axis=1) # species change matrix multiplied by rates matrix, product[:,j] = v_ij a_j for all i
    
    # calculate variances
    
    variances = np.sum(a[mask]*(v[:,mask]**2), axis=1)

    # -------------------------
    # calculate tau prime for each species, then take the smallest as tau prime
    maxes = np.maximum(epsilon*x/2, np.ones(len(x)))
    means_list = maxes/np.abs(means)
    variances_list = (maxes**2)/np.abs(variances)
    
    tau_primes = np.minimum(means_list, variances_list)
    
    tau_prime = np.min(tau_primes)
    
    return tau_prime
    
def create_mutant_phages(k,x,m,v,num_mutations,phage_list):
    # draw mutations from binomial, calculate number of mutated phages
    mutations = np.random.binomial(1, mu, size = B*L) # draw from binomial B*L times
    mutations = np.reshape(mutations, (B, L))
    #M = np.count_nonzero(np.sum(mutations, axis = 1))

    if k < (7*m+4): # then b0 vs vj:
        j = k - (6*m + 4)
        i = -1

    elif k >= (8+m)*m + 4: # then bi vs vj - calculate i and j, then same as above
        j = (k-((8+m)*m + 4))%m 
        i = int((k-4-j)/m - 8 -m)

    #print(i,j,k)
    # do mutations 
    #if M > 0:
    new_phages = []
    M = np.unique(np.array(np.where(mutations > 0))[0]) # indices of mutated phages
    for l in range(len(M)): # this will have 0 entries if no mutations 
        new_phage = list(phage_list[j])    
        new_phage = list(new_phage^mutations[M[l]]) # flip the mutated bits
        num_mutations += 1
        new_phages.append(new_phage)

    # subtract 1 from nb
    try:
        x[1+i] += -1
        # phage burst - M
        x[m+1+j] += B-1-len(M)              
    except IndexError as err:
        print(i, j, k, m)
        print(x)
        print("IndexError({0}): {1}".format(err.errno, err.strerror))
        
    return x, i, j, num_mutations, new_phages
    
# changes to x and v

# first do the changes that don't involve mutations
# question though: does applying those first mean the rates change? 
#Answer: no, since the change doesn't depend on x anymore now that we've decided which reactions happen.

# create a mask that removes the virus success terms
def tau_iteration(check, a, tau, x, t, m, v, num_mutations, phage_list, parent_list, mutation_times, all_phages, numSpecies, numReactions):
    
    k_list = np.random.poisson(a*tau, size = len(a))
    mutations_mask = np.ones(k_list.shape, dtype=bool)
    mutations_mask[6*m+4:7*m+4] = False
    mutations_mask[(8+m)*m + 4:] = False

    k_indices = np.arange(numReactions) # list of k values

    for k in k_indices[mutations_mask][k_list[mutations_mask].nonzero()[0]]: # remove phage mutations
        x_new = x + v[:,k]*k_list[k]
        if np.any(x_new < 0):
            check = 1
            break
        else:
            x = x_new
            check = 0

    # now do the phage mutation terms
    all_new_phages = []
    all_new_parents = []
    for k in k_indices[np.logical_not(mutations_mask)][k_list[np.logical_not(mutations_mask)].nonzero()[0]]:
        #print(k)
        for count2 in range(k_list[k]):
            # j is the parent phage for all these new phages
            x_new, i, j, num_mutations, new_phages = create_mutant_phages(k,x,m,v,num_mutations,phage_list)
            if np.any(x_new<0):
                check = 1 # return 1 if any element of x is negative
                break
            else:
                x = x_new
                check = 0
            all_new_phages += new_phages
            for count4 in range(len(new_phages)):
                all_new_parents.append(j)
    
    # add new phages
    for count3, phage in enumerate(all_new_phages):
        # check if phage already in phage_list
        if phage not in all_phages:
            all_phages.append(phage)
            parent_list.append([all_phages.index(phage_list[all_new_parents[count3]])])
            mutation_times.append([t])
        else:
            ind2 = all_phages.index(phage)
            parent_list[ind2].append(all_phages.index(phage_list[all_new_parents[count3]]))
            mutation_times[ind2].append(t)
        if phage in phage_list: # then don't increment m
            ind = phage_list.index(phage)
            
            x[m+1+ind] += 1 # add one to the existing phage    
            
            #print('yes')
        else: # add 2*len(M) new rows to x, create new phage in phage_list
            phage_list.append(phage) 
            
            x = np.insert(x, 2*m+1, 1) # new phage
            x = np.insert(x, m+1, 0) # new bacteria for the new phage
            m += 1    
            #phage_array = np.vstack([phage_array, phage]) # append new phage to the end
            
    numSpecies = 2*m + 2  # (nb0 + nbi + nvi + c)
    numReactions = (9 + 2*m) * m + 3   # if I did the counting right
    v = species_changes(m, B, numSpecies, numReactions) # update v only if m changes
            
    return check, x, m, v, num_mutations, phage_list, parent_list, mutation_times, all_phages, numSpecies, numReactions
    
def save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime):
    with open('populations_%s.txt' %(starttime), 'a+') as file:
        x_save = list(x)
        x_save.append(t)
        file.write(str(x_save) + '\n')
    with open('protospacers_%s.txt' %(starttime), 'a+') as p:
        p.write(str(phage_list) + '\n')
        
    with open('parents_%s.txt' %(starttime), 'w') as par:
        par.write(str(parent_list) + '\n')
    with open('all_phages_%s.txt' %(starttime), 'w') as all_p:
        all_p.write(str(all_phages) + '\n')
    with open('mutation_times_%s.txt' %(starttime), 'w') as mut_t:
        mut_t.write(str(mutation_times) + '\n')

def run_simulation_tau(m_init, f, c0, g, B, R, eta, pv, alpha, e, L, mu, max_save, epsilon, gen_max):
    now = datetime.datetime.now()
    starttime = now.isoformat()
    
    # save parameters to file
    
    all_phages = []
    params_array = np.zeros((15, 2), dtype = 'object')
    params_array[:,0] = ["m_init", "f", "c0", "g", "B", "R", "eta", "pv", "alpha", "e", "L", 
                         "mu", "gen_max", "max_save", "epsilon"]
    params_array[:,1] = [m_init,    f,   c0,   g,   B,   R,   eta,   pv,   alpha,   e,   L,   
                         mu,   gen_max, max_save, epsilon]
    pd.DataFrame(params_array).to_csv("parameters_%s.txt" %starttime, sep='\t', index=False, header=False)
    
    # convert R and f to non-normalized versions
    r = R*g*c0
    F = f*g*c0
    
    # set population vector (initial)
    # initial populations
    m = m_init # number of initial protospacers
    
    numSpecies = 2*m + 2  # (nb0 + nbi + nvi + c)
    numReactions = (9 + 2*m) * m + 3   # if I did the counting right

    nbi = np.zeros((m))
    nb0 = (1-f)*c0
    c = f*c0
    nv = np.zeros((m))
    # initialize nv equally across m protospacers
    nv = int((10*nb0)/m)

    x = np.zeros((numSpecies))

    x[0] = nb0
    x[1:m+1] = nbi
    x[m+1:2*m+1] = nv
    x[2*m+1] = c

    # run simulation
    #protospacer = np.zeros((L),dtype='int')
    phage_list = []
    parent_list = []
    mutation_times = []
    
    # create intial phages: m random protospacers
    for mval in range(m):
        protospacer = np.random.choice([0,1], L)
        phage_list.append(list(protospacer))
        parent_list.append([""]) # no parent for the initial phages
        all_phages.append(list(protospacer))
        mutation_times.append([0])
    num_mutations = 0

    jump = 200
    save_interval = gen_max/max_save
    t = 0
    v = species_changes(m, B, numSpecies, numReactions) # initialize v
    n = 0 # iteration counter
    n_save = 0 # save iteration counter
    
    # initial save
    save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime)
    
    #pbar = tqdm(total=max_save, initial=n_save)
    
    while t*g*c0 <= gen_max: # stop once desired generations are reached    
        
        if np.sum(x[:m+1]) == 0:
            print("Bacteria are extinct")
            print(x)
            save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime)
            break
        if np.sum(x[m+1:2*m+1]) == 0:
            print("Phages are extinct")
            print(x)
            save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime)
            break
        
        if t*g*c0 > n_save*save_interval: # save to file 
            save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime)
            n_save += 1 # incrememt save counter
            #print(len(all_phages), len(parent_list))
            #pbar.update(1)
            
        a = total_rates_phage(numReactions, x, m, B, g, e, eta, pv, alpha, F, r, c0, mu, L)

        a0 = np.sum(a)
        dt = gillespie_time(a0)

        # calculate tau time step
        tau = tau_time_step(x, v, a, epsilon)

        # compare tau to gillespie step size
        if tau < 10/a0: # do 100 gillespie iterations
            #print("doing Gillespie")
            for counter in range(200):
                a = total_rates_phage(numReactions, x, m, B, g, e, eta, pv, alpha, F, r, c0, mu, L)
                a0 = np.sum(a)
                dt = gillespie_time(a0)
                k = gillespie_reaction(a, a0, numReactions)
                n += 1 #increment n

                # if phage mutated, update m, a, and v
                if (k >= (6*m + 4) and k < (7*m + 4)) or (k >= (8+m)*m + 4): # these are the terms where v wins
                    x_new, i, j, num_mutations, new_phages = create_mutant_phages(k,x,m,v,num_mutations,phage_list)
                    
                    # add new phages
                    for phage in new_phages:
                        if phage not in all_phages:
                            all_phages.append(phage)
                            parent_list.append([all_phages.index(phage_list[j])])
                            mutation_times.append([t])
                        else:
                            ind2 = all_phages.index(phage)
                            parent_list[ind2].append(all_phages.index(phage_list[j]))
                            mutation_times[ind2].append(t)
                        # check if phage already in phage_list
                        if phage in phage_list: # then don't increment m
                            ind = phage_list.index(phage)
                            x_new[m+1+ind] += 1 # add one to the existing phage             
                            #print('yes')            
                        else: # add 2*len(M) new rows to x, create new phage in phage_list
                            phage_list.append(phage) 
                            
                            x_new = np.insert(x_new, 2*m+1, 1) # new phage
                            x_new = np.insert(x_new, m+1, 0) # new bacteria for the new phage
                            m += 1    
                            #phage_array = np.vstack([phage_array, phage]) # append new phage to the end

                            numSpecies = 2*m + 2  # (nb0 + nbi + nvi + c)
                            numReactions = (9 + 2*m) * m + 3   # if I did the counting right
                            v = species_changes(m, B, numSpecies, numReactions) # update v only if m changes
                        
                else: # do the normal update
                    x_new = x + v[:,k]
                
                if np.any(x_new <0): # don't save this population step
                    print("negative population from Gillespie")
                    continue
                else:                
                    t += dt
                    x = x_new
            
                # check for extinct populations
                if np.sum(x[:m+1]) == 0:
                    break
                if np.sum(x[m+1:2*m+1]) == 0:
                    break
            
            if np.sum(x[:m+1]) != 0 and np.sum(x[m+1:2*m+1]) != 0:
                # remove zeros after Gillespie
                zero_inds = np.where(np.logical_and(x[m+1:2*m+1] == 0, x[1:m+1] == 0) == True)[0] # where both phage and bac are 0
                if len(zero_inds) == 0:
                    pass
    
                else:
                    #print(x)
                    x = np.delete(x, m+1 + zero_inds) # delete phage
                    x = np.delete(x, 1 + zero_inds) # delete corresponding bacteria
                    m -= len(zero_inds)
                    for o in zero_inds[::-1]:
                        del phage_list[o] # delete the entries in phage_list as well (starting from higher indices)
                    #print("deleted " + str(len(zero_inds)) + " extinct phages")
                    #print("m is now " + str(m))
                    #print(x)
                    # recalculate v
                    numSpecies = 2*m + 2  # (nb0 + nbi + nvi + c)
                    numReactions = (9 + 2*m) * m + 3    # if I did the counting right
                    v = species_changes(m, B, numSpecies, numReactions) # update v only if m changes

            # save after Gillespie
            save_files(x, t, phage_list, parent_list, all_phages, mutation_times, starttime)

        else: #do a tau leaping timestep
            check = 1
            while check ==1:
                check, x, m, v, num_mutations, phage_list, parent_list, mutation_times, all_phages, numSpecies, numReactions = tau_iteration(check, 
                                a, tau, x, t, m, v, num_mutations, phage_list, parent_list, mutation_times, all_phages, numSpecies, numReactions)
                if check == 1:
                    tau = tau/2 
                if tau < a0: # not sure what to do here if tau gets very small - best would be gillespie cycles
                    continue
                
            t += tau
            n += 1

        # if a phage population goes extinct, delete that row from x, delete entry in list, and decrement m
        # check periodically for extinct phages
        if n%jump == 0:
            zero_inds = np.where(np.logical_and(x[m+1:2*m+1] == 0, x[1:m+1] == 0) == True)[0] # where both phage and bac are 0
            if len(zero_inds) == 0:
                pass

            else:
                #print(x)
                x = np.delete(x, m+1 + zero_inds) # delete phage
                x = np.delete(x, 1 + zero_inds) # delete corresponding bacteria
                m -= len(zero_inds)
                for o in zero_inds[::-1]:
                    del phage_list[o] # delete the entries in phage_list as well (starting from higher indices)
                #print("deleted " + str(len(zero_inds)) + " extinct phages")
                #print("m is now " + str(m))
                #print(x)
                # recalculate v
                numSpecies = 2*m + 2  # (nb0 + nbi + nvi + c)
                numReactions = (9 + 2*m) * m + 3    # if I did the counting right
                v = species_changes(m, B, numSpecies, numReactions) # update v only if m changes
    
    #pbar.close()
    return num_mutations, t, x, phage_list
    
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    default_c0 = 10**4
    
    parser = argparse.ArgumentParser(description='Simulation of interacting phage and bacteria with CRISPR')
    parser.add_argument('B', type=int, default=170, nargs='?',
                        help='phage burst size')
    parser.add_argument('c0', type=float, default=default_c0, nargs='?',
                        help='inflow carbon concentration')
    parser.add_argument('g', type=float, default=1./(42*default_c0), nargs='?',
                        help='bacterial growth rate (per min)')
    parser.add_argument('f', type=float, default=0.3, nargs='?',
                        help='chemostat flow rate (F/(g*co))')
    parser.add_argument('R', type=float, default=0.04, nargs='?',
                        help='spacer loss rate (r/(g*c0))')
    parser.add_argument('eta', type=float, default=5*10**-3, nargs='?',
                        help='spacer acquisition probability')
    parser.add_argument('pv', type=float, default=0.02, nargs='?',
                        help='phage success probability (without CRISPR)')
    parser.add_argument('alpha', type=float, default=2*10**-1/default_c0, nargs='?',
                        help='phage adsorptoin rate')
    parser.add_argument('e', type=float, default=0.95, nargs='?',
                        help='reduction in phage success due to CRISPR (between 0 and 1)')
    parser.add_argument('mu', type=float, default=10**-5, nargs='?',
                        help='phage mutation probability per base per generation')
    parser.add_argument('L', type=int, default=30, nargs='?',
                        help='protospacer length (number of nucleotides)')
    parser.add_argument('epsilon', type=float, default=0.06, nargs='?',
                        help='tau leaping epsilon parameter')
    parser.add_argument('m_init', type=int, default=1, nargs='?',
                        help='number of initial protospacer clusters')
    parser.add_argument('max_save', type=int, default=10, nargs='?',
                        help='maximum number of rows of data to save')
    parser.add_argument('gen_max', type=float, default=5, nargs='?',
                        help='number of generations at which to stop simulation')
    
    args = parser.parse_args()
    
    # Run simulation
    
    # define parameters
    B = args.B
    c0 = args.c0
    g = args.g
    f = args.f
    R = args.R
    eta = args.eta
    pv = args.pv
    alpha = args.alpha
    e = args.e
    mu = args.mu # mutation probability, per base, per generation
    L = args.L
    
    p = pv*alpha/g
    
    epsilon = args.epsilon # 0.03 from paper
    m_init = args.m_init
    max_save = args.max_save
    gen_max = args.gen_max
    
    # hardcoded
    nc = 10 # from paper
    gi = 2 # highest order for every species is 2
    

    num_mutations, t, x, phage_list = run_simulation_tau(m_init, f, c0, g, B, R, eta, pv, alpha, e, L, mu, max_save, epsilon, gen_max)
