#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:12:38 2018

@author: madeleine

Generate parameters for simulations
"""

import numpy as np
import pandas as pd

from spacer_model_plotting_functions import x_fn_nu, y_fn_nu, z_fn_nu
from spacer_model_plotting_functions import cubsol3, aterm, bterm, cterm, dterm

# set parameter ranges
c0_list = np.array([300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000])
eta_list = np.logspace(-5,-2,4)
mu_list = np.logspace(-7,-4,7)
m_init_list = np.array([1, 10, 50])
e_list = np.array([0.1,0.5,0.8,0.95])
#theta_list = [1,2,3]

# generate grid of all parameter combinations
# recommend choosing order so that longest-running sims are first (i.e. high mu first)
c0vals,  muvals, etavals, mvals, evals = np.meshgrid(c0_list, mu_list, eta_list, m_init_list, e_list)

# remove combinations of mu and c0 that won't finish in time
# the 0.35 means that when mu = 10**-4, c0 won't exceed 3000, etc.
keep_inds = c0vals.flatten() * muvals.flatten() < 0.35

c0vals = c0vals.flatten()[keep_inds]
muvals = muvals.flatten()[keep_inds]
etavals = etavals.flatten()[keep_inds]
mvals = mvals.flatten()[keep_inds]
evals = evals.flatten()[keep_inds]
#thetavals = thetavals.flatten()[keep_inds]

num_runs = len(c0vals)
# set maximum running time according to c0
gen_max = np.max([np.array([10000]*len(c0vals)), c0vals], axis = 0)

# create parameter array
params_array = np.zeros((num_runs,15))

params_array[:,0] = 170 # B
params_array[:,1] = c0vals # c0
params_array[:,2] = 1./(42*c0vals) # g
params_array[:,3] = 0.3 # f
params_array[:,4] = 0.04 # R
params_array[:,5] = etavals
params_array[:,6] = 0.02 # pv
params_array[:,7] = 2.*10**-2 / c0vals #alpha
params_array[:,8] = evals # e
params_array[:,9] = muvals
params_array[:,10] = 30 #L
params_array[:,11] = 0.03 #epsilon
params_array[:,12] = mvals # m_init
params_array[:,14] = gen_max # gen_max
params_array[:,13] = gen_max*2 # max_save
#params_array[:,15] = thetavals # for PV

# check params - phages won't go extinct
for i in range(len(params_array)):
    B = params_array[i,0]
    c0 = params_array[i,1]
    g = params_array[i,2]
    f = params_array[i,3]
    R = params_array[i,4]
    alpha = params_array[i,7]
    pv = params_array[i,6]
    e = params_array[i,8]
    eta = params_array[i,5]
    p = alpha*pv/g
    
    # this is the minimum pv value below which phages go extinct
    pv_min = (1/B)*(g*f/(alpha*(1-f))+1)
    if pv < pv_min:
        print(pv, pv_min, i)

    # check mean field steady state - prediction without phage mutation,
    # it's a rough estimate but it gives an idea
    a = aterm(f,p,pv,e,B,R,eta).astype(complex)
    b = bterm(f,p,pv,e,B,R,eta).astype(complex)
    c = cterm(f,p,pv,e,B,R,eta).astype(complex)
    d = dterm(f,p,pv,e,B,R,eta).astype(complex)
    
    nustar = cubsol3(a, b, c, d)
    nustar = np.real(nustar)
    xstar = x_fn_nu(nustar, f, p, pv, e, B, R, eta)
    ystar = y_fn_nu(nustar, f, p, pv, e, B, R, eta)
    zstar = z_fn_nu(nustar, f, p, pv, e, B, R, eta)

    nb0 = (1 - nustar)*xstar*c0
    Nv = ystar*c0
    
    print("Nb, Nv, C, nu at steady-state: %s, %s, %s, %s" %(xstar*c0, Nv, zstar*c0, nustar))

# save parameters
df = pd.DataFrame(params_array)

df[0] = df[0].astype(int) # set B column to int
df[10] = df[10].astype(int) # set L column to int 
df[12] = df[12].astype(int) # set m_init column to int
df[13] = df[13].astype(int) # set max_save column to int

df.to_csv("params_list.txt", header = None, index = None, sep = ' ')