# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
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

# # Predicting $m$, number of clones, using parameters and recursive matching of rates

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import quad

# %matplotlib inline

from spacer_model_plotting_functions import (nbi_steady_state, nvi_steady_state, analytic_steady_state,
                                             predict_m, recursive_bac_m)
from spacer_model_plotting_functions import (bacteria_clone_backwards_extinction, bac_extinction_time_numeric)


def get_numeric_mean(P0_vals, time_vals, n_samples = 5000):
    """
    Estimate the mean of a cumulative distribution by sampling randomly from the distribution.
    
    time_vals should be in units of bacterial generations.
    """
    inds = []
    for n in np.random.rand(n_samples):
        inds.append(np.argmin(np.abs((1 - P0_vals) - n)))

    tvals = time_vals[inds]
    
    return np.mean(tvals) # in generations


def recursive_bac_m_plot(m_vals_test, f, g, c0, alpha, B, pv, e, mu, R, eta, xlim=np.nan, ylim=np.nan, 
                         target_m = np.nan, target_m_std = np.nan, close_fig = True):
        
    """
    Predict bacteria m recursively using a range of input m and finding the 
    intersection between the input and output m.
    Plot the input and output m
    
    target_m : the measured value of m from simulation (can also be nan if plotting theoretically only)
    target_m_std : standard deviation of target m 
    m_vals_test : the range of m to iterate over
    """
    
    pred_m_list = []
    m_test = []
    
    for m in m_vals_test:
        
        if e/m > 1: # this is a non-sensical case to test
            continue
            
        m_test.append(m)        
        pred_m = predict_m(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
        pred_m_list.append(pred_m)

    m_test = np.array(m_test)
    
    fig, ax = plt.subplots(figsize = (6,5))
    ax.plot(m_test, pred_m_list, marker = 'o', label = "predicted m")
    ax.errorbar([target_m], [target_m], 
                xerr = [target_m_std], yerr = [target_m_std], 
                markersize = 7, markeredgecolor = 'k', marker = 's',
                color = 'r', label = "mean m from simulation")
    
    ax.plot(np.arange(0,50), np.arange(0, 50), 'k--', label = r"$y =x$")
    ax.legend()
    ax.set_xlabel("Input m")
    ax.set_ylabel("Predicted m")
    try:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    except ValueError:
        ax.set_ylim(0, np.nanmax(pred_m_list)*1.8)
        ax.set_xlim(0, np.nanmax(pred_m_list)*1.8)
    
    plt.savefig("m_prediction_c0_%s_eta_%s_e_%s_mu_%s_B_%s_pv_%s.png" %(c0, eta, e, mu, B, pv), dpi = 150)
    if close_fig == True:
        plt.close()

def integrand_P0_long(t, s, B, delta, n0):
    """
    This is the integral of the long-time approximation for P0(t),
    to find the mean time to extinction for large clones
    
    n0: initial clone size (n0 = nvi_ss for large clones)
    """
    return 1 - (((-1 + np.exp(s*t))*(-2*s + B*(delta + s)))/(2*s + B*(-1 + np.exp(s*t))*(delta + s)))**n0
    


def mean_time(t_max, s, B, delta, n0):
    """
    First output is integral
    Second output is upper bound on the error
    """
    return quad(integrand_P0_long, 0, t_max, args = (s, B, delta, n0))


# +
all_data = pd.read_csv("all_data.csv", index_col = 0)
all_data = all_data.drop_duplicates()

data_non_extinct = all_data 
# -

data_non_extinct.shape

# +
# predicted establishment using either beta and delta or P0(t)
R = 0.04
c0 = data_non_extinct['C0']
alpha = 2*10**-2 / c0
g = (1./(42*c0))
L = 30
eta = data_non_extinct['eta']
mu = data_non_extinct['mu']
P0 = np.exp(-data_non_extinct['mu']*L)
e = data_non_extinct['e']
B = data_non_extinct['B']
f = data_non_extinct['f']
pv = data_non_extinct['pv']
nv = data_non_extinct['mean_nv']
nb = data_non_extinct['mean_nb']
C = data_non_extinct['mean_C']
nb0 = nb*(1-data_non_extinct['mean_nu'])
F = f*g*c0
beta = nb*alpha*pv
delta = F + alpha*nb*(1-pv)

mutation_rate_pred = (alpha*B*(1-P0)*pv*nv
                    *nb*(1-e*data_non_extinct['mean_nu']/data_non_extinct['mean_m']))

#predicted_establishment_rate = (1 - delta/(delta+beta)) * mutation_rate_pred / (g*c0 )
predicted_establishment_rate = data_non_extinct['predicted_establishment_fraction'] * mutation_rate_pred / (g*c0 )

# calculate time to extinction for small clones

freq = 1/nv
mean_T_backwards_small = 2*nv*freq*(1 -np.log(freq))*g*c0/((B-1)**2 * beta + delta)

# time to extinction for large clones using nbi_ss

nvi_ss = nvi_steady_state(nb, nv, C, nb0, f, g, c0, e, alpha, B, mu, pv, R, eta)
nbi_ss = nbi_steady_state(nb, f, g, c0, e, alpha, B, mu, pv)

beta = nb*alpha*pv - alpha*pv*e*nbi_ss
delta = F + alpha*nb*(1-pv) + alpha*pv*e*nbi_ss

freq = nvi_ss / nv
mean_T_backwards_nvi_ss_nbi_ss = 2*nv*freq*(1-np.log(freq))*g*c0/((B-1)**2 * beta + delta)
# -

data_non_extinct['mean_T_backwards_small'] = mean_T_backwards_small
data_non_extinct['mean_T_backwards_nvi_ss_nbi_ss'] = mean_T_backwards_nvi_ss_nbi_ss
data_non_extinct['predicted_establishment_rate'] = predicted_establishment_rate
data_non_extinct['mutation_rate_pred'] = mutation_rate_pred  / (g*c0 )

grouped_data = data_non_extinct.groupby(['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv', 'm_init', 'pv_type', 'theta', 'gen_max'])[['mean_m', 'mean_phage_m', 
                            'e_effective', 'mean_nu', 'mean_nb', 'mean_nv', 'rescaled_phage_m',
                            'mean_large_phage_m', 'mean_large_phage_size',
                            'mean_C', 'fitness_discrepancy', 
                            'mean_size_at_acquisition', 'fitness_at_90percent_acquisition',
                            'fitness_at_mean_acquisition',
                            'num_bac_acquisitions', 'mean_bac_acquisition_time', 
                            'mean_establishment_time', 
                            'median_bac_acquisition_time', 'first_bac_acquisition_time',
                            'mean_bac_extinction_time', 'mean_bac_extinction_time_phage_present',                                                          
                            'mean_trajectory_length','mean_large_trajectory_length_nvi_ss',
                            'mean_T_backwards_nvi_ss',
                            'establishment_rate_nvi_ss',
                            'mean_T_backwards_small', 'mean_T_backwards_nvi_ss_nbi_ss', 'predicted_establishment_rate',
                            'turnover_speed',
                            'mutation_rate_pred', 'measured_mutation_rate',
                            'establishment_rate_bac', 'mean_bac_establishment_time',
                            'bac_speed_mean',
                           'bac_speed_std', 'phage_speed_mean', 'phage_speed_std',
                           'bac_spread_mean', 'bac_spread_std', 'phage_spread_mean',
                           'phage_spread_std', 'net_phage_displacement', 'net_bac_displacement',
                           'max_phage_displacement', 'max_bac_displacement',
                           'bac_phage_distance_mean', 'bac_phage_distance_std',
                           'time_to_reach_bac_0.25', 'time_to_reach_phage_0.25',
                           'time_to_reach_bac_0.5', 'time_to_reach_phage_0.5',
                           'time_to_reach_bac_1', 'time_to_reach_phage_1', 'time_to_reach_bac_2',
                           'time_to_reach_phage_2', 'time_to_reach_bac_5', 'time_to_reach_phage_5',
                           'time_to_reach_bac_10', 'time_to_reach_phage_10',
                           'time_to_reach_bac_15', 'time_to_reach_phage_15', 'pred_num_establishments',
                           'measured_num_establishments', 'measured_num_establishments_8000',
                           'time_to_full_turnover_phage', 'time_to_full_turnover_bac',
                           'phage_abundance_speed_mean', 'phage_abundance_speed_std',
                           'bac_abundance_speed_mean', 'bac_abundance_speed_std',
                           'bac_clan_number', 'phage_clan_number', 'bac_clan_number_std',
                           'phage_clan_number_std', 'bac_clan_size', 'phage_clan_size',
                           'bac_clan_size_std', 'phage_clan_size_std', 'slope', 'peak_immunity']].agg([np.nanmean, np.nanstd,
                                                                                                'count']).reset_index()

grouped_data.shape

# ## Predict m

# +
# get bacteria m prediction - 1 value per group since it only depends on parameters

grouped_data['pred_bac_m_recursive'] = np.nan
grouped_data['pred_bac_m_recursive_uncertainty'] = np.nan
grouped_data['mean_T_backwards_nvi_ss_nbi_ss_recursive'] = np.nan
grouped_data['predicted_establishment_fraction_recursive'] = np.nan
grouped_data['mutation_rate_pred_recursive'] = np.nan
final_pred_m = []

#B = 170
#pv = 0.02
#f = 0.3
R = 0.04

for i, row in tqdm(grouped_data.iterrows()):
    c0 = float(row['C0'])
    eta = float(row['eta'])
    mu = float(row['mu'])
    e = float(row['e'])
    B = float(row['B'])
    f = float(row['f'])
    pv = float(row['pv'])
    alpha = 2*10**-2 / c0
    g = 1/(42*c0)
    mean_m = float(row['mean_m']['nanmean'])
    
    #if mean_m < 3:
    #    continue

    # iterate over data and calculate predicted bac m
    try:
        m_vals_test = np.arange(mean_m*0.2, mean_m*4, float(mean_m * 0.05))
        pred_m_uncertainty = m_vals_test[1] - m_vals_test[0]
        (recursive_m, establishment_fraction_pred_recursive, mutation_rate_pred_recursive, 
         extinction_time_pred_recursive) = recursive_bac_m(m_vals_test, f, g, c0, alpha, B, pv, e, mu, R, eta)
    except ValueError: # no mean m from the simulation
        #m_vals_test = np.arange(0.5, 50, 0.1)
        recursive_m = np.nan
        pred_m_uncertainty = np.nan
        establishment_fraction_pred_recursive = np.nan
        mutation_rate_pred_recursive = np.nan
        extinction_time_pred_recursive = np.nan

    grouped_data.loc[i, 'pred_bac_m_recursive'] = recursive_m
    grouped_data.loc[i, 'pred_bac_m_recursive_uncertainty'] = pred_m_uncertainty
    grouped_data.loc[i, 'mean_T_backwards_nvi_ss_nbi_ss_recursive'] = extinction_time_pred_recursive
    grouped_data.loc[i, 'predicted_establishment_fraction_recursive'] = establishment_fraction_pred_recursive
    grouped_data.loc[i, 'mutation_rate_pred_recursive'] = mutation_rate_pred_recursive
# -

# ### Calculate total population size using predicted diversity

# +
# get predicted total population size
grouped_data['nb_pred_recursive'] = np.nan
grouped_data['nv_pred_recursive'] = np.nan
grouped_data['nu_pred_recursive'] = np.nan
grouped_data['C_pred_recursive'] = np.nan

for i, row in tqdm(grouped_data.iterrows()):
    c0 = float(row['C0'])
    eta = float(row['eta'])
    mu = float(row['mu'])
    e = float(row['e'])
    B = float(row['B'])
    f = float(row['f'])
    pv = float(row['pv'])
    alpha = 2*10**-2 / c0
    g = 1/(42*c0)
    mean_m = float(row['mean_m']['nanmean'])
    
    m = float(row['pred_bac_m_recursive'])
    
    e_effective = e/m

    # get predicted mean field quantities
    nb, nv, C, nu = analytic_steady_state(pv, e_effective, B, R, eta, f, c0, g, alpha)

    nb = float(nb)
    nv = float(nv)
    C = float(C)
    nu = float(nu)
    
    grouped_data.loc[i,'nb_pred_recursive'] = nb
    grouped_data.loc[i,'nv_pred_recursive'] = nv
    grouped_data.loc[i,'nu_pred_recursive'] = nu
    grouped_data.loc[i,'C_pred_recursive'] = C
# -

# ## Bacteria clone extinction

# +
# get predicted total population size
grouped_data['pred_bac_extinction_time_nodrift'] = np.nan
grouped_data['pred_bac_extinction_time_numeric'] = np.nan

for i, row in tqdm(grouped_data.iterrows()):
    c0 = float(row['C0'])
    eta = float(row['eta'])
    mu = float(row['mu'])
    e = float(row['e'])
    B = float(row['B'])
    f = float(row['f'])
    pv = float(row['pv'])
    alpha = 2*10**-2 / c0
    g = 1/(42*c0)
    mean_m = float(row['mean_m']['nanmean'])
    
    m = float(row['pred_bac_m_recursive'])
    
    if np.any(np.isnan(m)): # if no predicted m solution, skip analytic part
        grouped_data.loc[i, 'pred_bac_extinction_time_nodrift'] = np.nan
        grouped_data.loc[i, 'pred_bac_extinction_time_numeric'] = np.nan
        continue
    
    nb_pred = float(row['nb_pred_recursive'])
    nv_pred = float(row['nv_pred_recursive'])
    C_pred = float(row['C_pred_recursive'])
    nu_pred = float(row['nu_pred_recursive'])
    nb0_pred = (1-nu_pred)*nb_pred
    nbs_pred = nu_pred*nb_pred
    
    nvi_ss_pred = nvi_steady_state(nb_pred, nv_pred, C_pred, nb0_pred, f, g, c0, e, alpha, B, mu, pv, R, eta)
    nbi_ss_pred = nbi_steady_state(nb_pred, f, g, c0, e, alpha, B, mu, pv)
    
    # analytic solution without drift term
    b = g*C_pred
    d = f*g*c0 + R*g*c0 + alpha*pv*(nv_pred - e*nvi_ss_pred)
    D = alpha*eta*nb0_pred*(nvi_ss_pred)*(1-pv)
    mean_bac_extinction_time = bacteria_clone_backwards_extinction(b,d, D, nbi_ss_pred, nbs_pred)*g*c0
    
    # numerical solution
    mean_bac_extinction_time_changing_nvi = bac_extinction_time_numeric(nb_pred, nv_pred, 
                                                C_pred, nb0_pred, nbs_pred, nvi_ss_pred, nbi_ss_pred, 
                                                        mean_bac_extinction_time, 
                                                        f, g, c0, e, alpha, B, mu, pv, R, eta)
    
    grouped_data.loc[i, 'pred_bac_extinction_time_nodrift'] = mean_bac_extinction_time
    grouped_data.loc[i, 'pred_bac_extinction_time_numeric'] = mean_bac_extinction_time_changing_nvi

# -

grouped_data_multisample = grouped_data[grouped_data['mean_m']['count'] > 2]

# save predicted m data to csv
grouped_data.to_csv("grouped_data.csv")


