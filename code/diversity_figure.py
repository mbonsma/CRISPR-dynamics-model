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

# # Diversity and Average Immunity (Figure 2)
#
# Code to generate Figure 2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib import gridspec
import matplotlib

# %matplotlib inline

from sim_analysis_functions import find_nearest, load_simulation, find_file
from spacer_model_plotting_functions import (analytic_steady_state, x_fn_nu, y_fn_nu, effective_e)


# from https://stackoverflow.com/a/53191379
################### Function to truncate color map ###################
def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


# +
def a_coefficient(g,c0,f,B,eta,e,alpha,pv,R,mu,L):
    """
    Note: approximating e^(-mu L) as mu L
    """
    
    nu = nuapprox_small_e(f, g, c0, alpha, pv, B, R, eta, 0, 1) #set e = 0
    nv = nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta)
    A = Aterm(f, g, alpha, pv, B)
    
    return (2*e*nu/(B-1))*(alpha*B*mu*L*pv*nv*(1-f)/(g*A))*(2*nv*(1- 1/(B*pv))/(f*(B-1)))

def a_coefficient_approx(g,c0,f,B,eta,e,alpha,pv,R,mu,L):
    """
    Most aggressive approximation for a, equation 162 in SI
    """
    r = R*g*c0
    return 4*e*mu*L*eta*(1-pv)*(g*c0*(1-f))**3 / ((B*pv-1)**2 * alpha**2 * pv*r)

def a_coefficient_approx_B(g,c0,f,B,eta,e,alpha,pv,R,mu,L):
    r = R*g*c0
    return 4*e*mu*L*eta*(1-pv)*(g*c0*(1-f))**3 / ((B**2*pv**3 * alpha**2 * r))

def m_approx(a):
    """
    Approximation from Sid (May 27)
    """
    z = 1 + (1/3)*np.log(a)
    return (a*z*(1+np.log(z)/(3*z-1)))**(1/3)

def m_approx_aggressive(a):
    return a**(1/3)

### Compact approximations for T and nu

def nuapprox_small_e(f,g,c0,alpha,pv,B,R,eta,e,m):
    """
    Assume nu = -d/c
    """
    r = R*g*c0
    A = Aterm(f,g,alpha,pv,B)
    nv = nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta)
    
    return 1 / (1 + r/(eta*(1-pv)*alpha*nv) - e*pv/(m*eta*(1-pv)) + (A*B*pv*e/m) /((A-1)*(B*pv - 1)))

def nv_no_CRISPR(f,g,c0,alpha,pv,B,R,eta):
    
    """nv without CRISPR is setting nu and e to zero"""
    nu = 0
    e = 0
    
    return c0*y_fn_nu(nu, f, pv*alpha/g, pv, e, B, R, eta)

def nb_no_CRISPR(f,g,c0,alpha,pv,B,R,eta):
    
    """nb without CRISPR is setting nu and e to zero"""
    nu = 0
    e = 0
    
    return c0*x_fn_nu(nu, f, pv*alpha/g, pv, e, B, R, eta)

def Aterm(f,g,alpha,pv,B):
    """
    A > 1 is phage existence criterion (without CRISPR)
    """
    
    return (1-f)*(B*pv-1)*alpha/(f*g)


# -

analytic_steady_state_vec = np.vectorize(analytic_steady_state)

# +
grouped_data = pd.read_csv("grouped_data_predicted_m_Fokker_Planck.csv", 
                           index_col = 0, header=[0,1])

# remove unnamed levels
new_columns = []

for label in grouped_data.columns:
    if 'Unnamed' in label[1]:
        new_columns.append((label[0], ''))
    else:
        new_columns.append(label)

grouped_data.columns = pd.MultiIndex.from_tuples(new_columns)

grouped_data_multisample = grouped_data[grouped_data['mean_m']['count'] > 2]
# -

# load data
all_data = pd.read_csv("all_data.csv", index_col = 0)

all_data.shape

# +
c0_select = 10**5
e_select = 0.95
mu_select = 3*10**-7

all_data_subset = all_data[(all_data['C0'] == c0_select)
                      & (all_data['e'] == e_select)
                      & (all_data['mu'] > mu_select*0.9)
                      & (all_data['mu'] < mu_select*1.1)
                      & (all_data['pv_type'] == 'binary')
                      & (all_data['pv'] == 0.02)
                      & (all_data['B'] == 170)
                      & (all_data['f'] == 0.3)
                      & (all_data['m_init'] == 1)]

all_data_subset = all_data_subset.groupby('eta').head(4) # keep only 4 unique simulations, same number for each set of parameters
# -

timestamps = list(all_data_subset.sort_values(by='eta')['timestamp'])
top_folders = list(all_data_subset.sort_values(by='eta')['folder_date'])

# ### Load simulation data

# +
# load simulations

B = 170
pv = 0.02
f = 0.3
R = 0.04
L = 30

pop_array_list = []
mutation_times_list = []
parents_list = []
all_phages_list = []
timestamps_list = []

c0_list = []
g_list = []
eta_list = []
mu_list = []
m_init_list = []
max_m_list = []
alpha_list = []
e_list = []
gen_max_list = []


for i, timestamp in tqdm(enumerate(timestamps)):
    top_folder = "/media/madeleine/My Passport/Data/results/" + str(top_folders[i])
    folder, fn = find_file("pop_array_%s.txt.npz" %timestamp, top_folder)

    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
     max_m, mutation_times, all_phages = load_simulation(folder, timestamp);
        
    c0_list.append(c0)
    g_list.append(g)
    eta_list.append(eta)
    mu_list.append(mu)
    m_init_list.append(m_init)
    max_m_list.append(max_m)
    alpha_list.append(alpha)
    e_list.append(e)
    gen_max_list.append(gen_max)

    pop_array_list.append(pop_array)
    mutation_times_list.append(mutation_times)
    #parents_list.append(parent_list)
    #all_phages_list.append(all_phages)
    timestamps_list.append(timestamp)
# -

data_subset = grouped_data[(grouped_data['C0'] == float(np.unique(c0_list)))
             & (grouped_data['mu'] == float(np.unique(mu_list)))
             & (grouped_data['e'] == float(np.unique(e_list)))  
             & (grouped_data['m_init'] == 1)
            & (grouped_data['B'] == 170)
            #& (grouped_data['pv'] == 0.02)
            # & (grouped_data['f'] == 0.3)
            & (grouped_data['pv_type'] == 'binary')]

# ## Figure 2

# +
# generate clone size histograms

num_samples = 1000

x_vals_bac = []
y_vals_bac = []
x_vals_phage = []
y_vals_phage = []

for j, eta in enumerate(np.unique(eta_list)): # loop through eta values
        
    bac_clones_main_list = []
    phage_clones_main_list = []

    for i in np.where(np.array(eta_list) == eta)[0]: # loop through simulations for each eta value
        pop_array = pop_array_list[i]
        g = g_list[i]
        c0 = c0_list[i]
        eta = eta_list[i]
        mu = mu_list[i]
        gen_max = gen_max_list[i]
        max_m = max_m_list[i]
        alpha = alpha_list[i]
        data = data_subset[data_subset['eta'] == eta]
    
        t_ss = gen_max / 5
        t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

        skip = int(pop_array[t_ss_ind:, -1].shape[0] / num_samples)

        bac_clones_list = pop_array[t_ss_ind::skip, 1:max_m+1].toarray().flatten()
        phage_clones_list = pop_array[t_ss_ind::skip, max_m+1:2*max_m+1].toarray().flatten()

        bac_clones_list = bac_clones_list[np.nonzero(bac_clones_list)] # use only nonzero clones
        phage_clones_list = phage_clones_list[np.nonzero(phage_clones_list)]
        
        bac_clones_main_list += list(bac_clones_list)
        phage_clones_main_list += list(phage_clones_list)
    
    # get phage clone size histogram
    phage_bin_width = 6000
    phage_vals, bin_edges = np.histogram(phage_clones_main_list, 
                                         bins= np.arange(0, np.max(phage_clones_main_list)+phage_bin_width, phage_bin_width))
    phage_bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # get bac clone size histogram
    bac_bin_width = 30
    bac_vals, bin_edges = np.histogram(bac_clones_main_list, 
                                       bins= np.arange(0, np.max(bac_clones_main_list)+bac_bin_width, bac_bin_width))
    bac_bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # predicted clone size
    
    row = grouped_data[(grouped_data['C0'] == c0)
            & (grouped_data['eta'] == eta)
            & (grouped_data['e'] == e)
            & (grouped_data['mu'] < mu*1.1)
            & (grouped_data['mu'] > mu*0.9)
            & (grouped_data['B'] == 170)
            & (grouped_data['m_init'] == 1)
            & (grouped_data['pv_type'] == 'binary')]
    
    mean_bac_clone_size = float(row['mean_nu']['nanmean']*row['mean_nb']['nanmean']/row['mean_m']['nanmean'])

    mean_phage_clone_size = float(row['mean_nv']['nanmean']/row['rescaled_phage_m']['nanmean'])
    
    if mean_phage_clone_size < 0: # use bacteria m instead of rescaled phage m
        print("Negative phage clone size: %s" %eta)
        mean_phage_clone_size = float(row['mean_nv']['nanmean']/row['mean_m']['nanmean'])
    
    x_vals_bac.append(bac_bin_centres / mean_bac_clone_size)
    y_vals_bac.append(bac_vals / (np.sum(bac_vals)*bac_bin_width/mean_bac_clone_size))
    x_vals_phage.append(phage_bin_centres / mean_phage_clone_size)
    y_vals_phage.append(phage_vals / (np.sum(phage_vals)*phage_bin_width/mean_phage_clone_size))


# +
mu_vals = list(np.unique(grouped_data['mu']))
c0_vals = list(np.unique(grouped_data['C0']))
eta_vals = list(np.unique(grouped_data['eta']))
e_vals = list(np.unique(grouped_data['e']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd', 'X', 'h']
colours_m = sns.color_palette("hls", len(c0_vals))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

colours = cm.viridis(np.linspace(0,0.9, 4)) # for eta

# +
# mean m plot setup
e_mu_eta_c0_g_alpha_vals = np.logspace(-9, 2, 50)

c0_select = 10**4 # this is a dummy variable that doesn't actually impact the value of the trendline in this formulation
g = 1/(42*c0_select)
B = 170
alpha = 2*10**-2 / c0_select
r = R*g*c0_select

# approximating m ~ a**(1/3)
m_trendline = (e_mu_eta_c0_g_alpha_vals*(4*(1-pv)*L*((1-f))**3)/((B*pv-1)**2*pv*r))**(1/3)

m_init_select = 1
data_subset = grouped_data_multisample[(grouped_data_multisample['m_init'] == m_init_select)
                                       & (grouped_data_multisample['pv_type'] == 'binary')
                                      #& (grouped_data_multisample['eta'] >= 10**-4)
                                     & (grouped_data_multisample['mu'] > 3*10**-8)
                                      & (grouped_data_multisample['B'] == 170) ]

colour = 'C0'
shape = 'eta'
#line = 'eta'

colour_label = 'C_0'
shape_label = '\eta'
#line_label = '\eta'

legend_elements = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor='grey',markersize = 10, linestyle = "None"))
    
legend_elements_eta = []
for i in range(len(np.sort(data_subset[shape].unique()))):
    legend_elements_eta.append(Line2D([0], [0], marker=markerstyles[i],  
                                  label='$%s = %s$' %(shape_label, round(np.sort(data_subset[shape].unique())[i], 8)),
                          markerfacecolor=colours[i],markersize = 10, linestyle = "None"))

#for i in range(len(np.sort(data_subset[colour].unique()))):
#    legend_elements.append(Line2D([0], [0], marker='o', 
#                                  label='$%s = %s$' %(colour_label, 
#                                int(np.sort(data_subset[colour].unique())[i])),
#                          markerfacecolor=colours_m[i], markeredgecolor = 'k', markersize = 10, linestyle = "None"))


# +
fig = plt.figure(figsize = (10,7))

title_fontsize = 18
absolute_left = 0.09
absolute_right = 0.94
absolute_bottom = 0.11
absolute_top = 0.95

title_fontsize = 18
legend_fontsize = 9
label_fontsize = 10

# clone size distributions
gs_clones = gridspec.GridSpec(2,1)
gs_clones.update(left=absolute_left, right=0.35, bottom = absolute_bottom, top = absolute_top, hspace = 0.22)
ax1 = plt.subplot(gs_clones[0])
ax2 = plt.subplot(gs_clones[1])

left_fitness = 0.42
right_fitness = 0.62

gs_m = gridspec.GridSpec(1,1)
gs_m.update(left=0.43, right=0.91, bottom = absolute_bottom, top = absolute_top)
ax3 = plt.subplot(gs_m[0])

x0 = 0.09
y0 = 0.68
width = 0.4
height = 0.3
ax4 = ax3.inset_axes([x0, y0, width, height])

gs_cbar = gridspec.GridSpec(1,1)
gs_cbar.update(left=0.92, right=absolute_right, bottom = absolute_bottom, top = absolute_top)
ax_cbar = plt.subplot(gs_cbar[0])
#cmap = sns.color_palette("hls", len(c0_vals))
#cmap = ListedColormap(sns.color_palette('hls', len(c0_vals)).as_hex())
cmap = sns.color_palette("hls", as_cmap=True)
# figure out how much to truncate colormap to end at 10^6
extent = np.log10(3*10**6) - np.log10(300) # log-range of c0 values for original color mapping
new_extent = np.log10(10**6) - np.log10(300) # want to end at 10^6 instead
fraction = (extent - new_extent)/extent
new_cmap =  truncate_colormap(cmap, 0, 1-fraction, n=100)

cbar = fig.colorbar(cm.ScalarMappable(norm= matplotlib.colors.LogNorm(vmin=min(c0_vals), vmax=10**6), cmap=new_cmap),
             cax=ax_cbar, orientation='vertical', label='Nutrient concentration $C_0$')
cbar.ax.fontsize=16

## Clone size distributions



for j, eta in enumerate(np.unique(eta_list)): # loop through eta values
        
    ax1.plot(x_vals_bac[j], y_vals_bac[j], color = colours[j],
                linewidth = 3, label = r"$\eta = %s$" %eta)
    ax2.plot(x_vals_phage[j], y_vals_phage[j],  color = colours[j], 
                linewidth = 3, label = r"$\eta = %s$" %eta)

ax1.axvline(1, linestyle = '--', color = 'k', linewidth = 1, label = "Mean clone\nsize")
ax2.axvline(1, linestyle = '--', color = 'k', linewidth = 1) 
ax1.set_yscale('log')
ax2.set_yscale('log')
#axs[0].set_xscale('log')
#axs[1].set_xscale('log')
ax1.set_xlim(0, 6.5)
ax1.set_ylim(10**-3, 1.2*10**0)
ax2.set_xlim(0, 3)
ax2.set_ylim(10**-3, 2*10**1)

ax1.set_ylabel("Bacteria clone\nprobability density", fontsize = label_fontsize)
ax2.set_ylabel("Phage clone\nprobability density", fontsize = label_fontsize)
ax1.set_xlabel("Normalized bacteria clone size", fontsize = label_fontsize)
ax2.set_xlabel("Normalized phage clone size", fontsize = label_fontsize)

ax1.legend(loc = 'upper right', fontsize = legend_fontsize)

## mean m plots

for group in data_subset.groupby([colour, shape]):
    data = group[1].sort_values(by = 'mu')
    c0 = data['C0']
    eta = data['eta']
    e = data['e']   
    g = 1/(42*c0)
    alpha = 2*10**-2/c0
    mu = data['mu']
    
    eta = group[0][1]
    
    if eta < 10**-4:
        alphaval = 0.3
        #col = colours[0]
    else:
        alphaval = 0.6
        #col = colours[2]
    
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    #a = a_coefficient(g,c0,f,B,eta,e,alpha,pv,R,mu,L)
    #a = a_coefficient_approx(g,c0,f,B,eta,e,alpha,pv,R,mu,L)
    a = a_coefficient_approx_B(g,c0,f,B,eta,e,alpha,pv,R,mu,L)

    ax4.errorbar(data['pred_bac_m_recursive'], data['mean_m']['nanmean'],
            yerr = data['mean_m']['nanstd'],
            c = colours[shape_ind], 
           alpha = alphaval, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")

    # only plot highest eta vals with "a" version

    if eta < 10**-3:
        alphaval = 0.3
    else:
        alphaval = 0.6
    
    ax3.errorbar(a, data['mean_m']['nanmean'],
            yerr = data['mean_m']['nanstd'],
            c = colours_m[colour_ind], 
       alpha = alphaval, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")

                     
#t, = ax.plot(a_vals, m_solutions, color = 'k', label = r"Small $e_{eff}$" + "\napproximation")
#t1, = ax3.plot(a_vals, m_approx(a_vals), color = 'k', linestyle = '-', linewidth = 2,
#              label = "Theory")

a_vals = np.logspace(-5, 7, 100)
t1, = ax3.plot(a_vals, m_approx_aggressive(a_vals), color = 'k', linestyle = '-', linewidth = 2,
              label = "Theory")

## 1/3 slope line
x = np.array([9*10**1, 10**5])
t, = ax3.plot(x, (x/200)**(1/3), linestyle = '--', color = 'k', label = r"$\frac{1}{3}$ slope")
ax3.annotate(r'$\frac{1}{3}$', xy = (2*10**4, 3), xycoords = 'data', fontsize = 16)

#axs[0].annotate('Decreasing spacer\nacquisition probability',
#            xy=(1, 10**-2), xycoords='data',
#            xytext=(2, 3*10**-1), textcoords='data',
#            arrowprops=dict(facecolor='black', arrowstyle="->"))

ax3.set_yscale('log')
ax3.set_xscale('log')
#ax.set_xlim(0.3, 45)
ax3.set_ylim(0.09, 75)
ax3.set_xlim(5*10**-5, 2*10**7)

ax4.plot([10**-2, 10**2], [10**-2, 10**2], 'k')
ax4.set_xlim(4*10**-1, 400)
ax4.set_ylim(4*10**-2, 88)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Predicted number of clones', fontsize = label_fontsize)

#legend_elements.append(t)
#legend_elements.append(t1)
#ax.legend(handles=legend_elements, loc='lower right', ncol = 2, fontsize = 9)
ax3.set_xlabel(r"$\frac{4 e \mu \eta L (1-p_V)(g C_0(1-f))^3}{B^2 \alpha^2 p_V^3 r}$", fontsize = 16)

ax3.set_ylabel("Mean number of clones $m$", fontsize = label_fontsize)
#ax4.set_ylabel("Mean number of clones $m$")

#legend_elements.append(t)

ax3.legend(handles=legend_elements, loc='lower right', ncol = 1, fontsize = legend_fontsize)
ax4.legend(handles=legend_elements_eta, loc='lower right', ncol = 1, fontsize = legend_fontsize)

ax1.set_title("A", loc = 'left', fontsize = title_fontsize)
#ax2.set_title("B", loc = 'left', fontsize = title_fontsize)
ax3.set_title("B", loc = 'left', fontsize = title_fontsize)

#plt.tight_layout()
plt.savefig("diversity_figure.svg")
plt.savefig("diversity_figure.pdf")

# -

# ## Supplementary and presentation figures

# +
fig = plt.figure(figsize = (5.5,3.5))

title_fontsize = 18
absolute_left = 0.14
absolute_right = 0.9
absolute_bottom = 0.14
absolute_top = 0.95

title_fontsize = 18
legend_fontsize = 9
label_fontsize = 10

gs_m = gridspec.GridSpec(1,1)
gs_m.update(left=absolute_left, right=0.87, bottom = absolute_bottom, top = absolute_top)
ax = plt.subplot(gs_m[0])

gs_cbar = gridspec.GridSpec(1,1)
gs_cbar.update(left=0.88, right=absolute_right, bottom = absolute_bottom, top = absolute_top)
ax_cbar = plt.subplot(gs_cbar[0])
#cmap = sns.color_palette("hls", len(c0_vals))
#cmap = ListedColormap(sns.color_palette('hls', len(c0_vals)).as_hex())
cmap = sns.color_palette("hls", as_cmap=True)
# figure out how much to truncate colormap to end at 10^6
extent = np.log10(3*10**6) - np.log10(300) # log-range of c0 values for original color mapping
new_extent = np.log10(10**6) - np.log10(300) # want to end at 10^6 instead
fraction = (extent - new_extent)/extent
new_cmap =  truncate_colormap(cmap, 0, 1-fraction, n=100)

cbar = fig.colorbar(cm.ScalarMappable(norm= matplotlib.colors.LogNorm(vmin=min(c0_vals), vmax=10**6), cmap=new_cmap),
             cax=ax_cbar, orientation='vertical', label='Nutrient concentration $C_0$')
cbar.ax.fontsize=16


c0_select = 10**4 # this is a dummy variable that doesn't actually impact the value of the trendline in this formulation
g = 1/(42*c0_select)
B = 170
alpha = 2*10**-2 / c0_select
r = R*g*c0_select

m_init_select = 1
data_subset = grouped_data_multisample[(grouped_data_multisample['m_init'] == m_init_select)
                                       & (grouped_data_multisample['pv_type'] == 'binary')
                                      #& (grouped_data_multisample['eta'] >= 10**-4)
                                     & (grouped_data_multisample['mu'] > 3*10**-8)
                                      & (grouped_data_multisample['B'] == 170) ]

colour = 'C0'
shape = 'eta'
#line = 'eta'

colour_label = 'C_0'
shape_label = '\eta'
#line_label = '\eta'

legend_elements = []
legend_elements.append(Line2D([0], [0], marker=markerstyles[0],  
                                  label='Simulation data',
                          markerfacecolor=colours_m[0], markeredgecolor = 'k',markersize = 10, linestyle = "None"))


for group in data_subset.groupby([colour, shape]):
    data = group[1].sort_values(by = 'mu')
    c0 = data['C0']
    eta = data['eta']
    e = data['e']   
    g = 1/(42*c0)
    alpha = 2*10**-2/c0
    mu = data['mu']
    
    eta = group[0][1]
    
    if eta < 10**-4:
        alphaval = 0.3
    else:
        alphaval = 0.6
    
    colour_variable = group[0][0]
    shape_variable = group[0][1]

    colour_ind = list(np.sort(data_subset[colour].unique())).index(colour_variable)
    shape_ind = list(np.sort(data_subset[shape].unique())).index(shape_variable)
    
    a = a_coefficient_approx_B(g,c0,f,B,eta,e,alpha,pv,R,mu,L)


    # only plot highest eta vals with "a" version

    if eta < 10**-3:
        alphaval = 0.3
    else:
        alphaval = 0.6
    
    ax.errorbar(a, data['mean_m']['nanmean'],
            yerr = data['mean_m']['nanstd'],
            c = colours_m[colour_ind], 
       alpha = alphaval, marker = markerstyles[shape_ind], mec ='k', markersize = 8, linestyle = "None")

                     
a_vals = np.logspace(-5, 7, 100)
t1, = ax.plot(a_vals, m_approx_aggressive(a_vals), color = 'k', linestyle = '-', linewidth = 2,
              label = "Theory")

## 1/3 slope line
x = np.array([9*10**1, 10**5])
t, = ax.plot(x, (x/200)**(1/3), linestyle = '--', color = 'k', label = r"$\frac{1}{3}$ slope")
ax.annotate(r'$\frac{1}{3}$', xy = (1*10**4, 1.5), xycoords = 'data', fontsize = 16)

ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_xlim(0.3, 45)
ax.set_ylim(0.05, 85)
ax.set_xlim(4*10**-5, 6*10**6)

#legend_elements.append(t)
legend_elements.append(t1)
ax.legend(handles=legend_elements, loc='lower right', ncol = 1, fontsize = 9)
ax.set_xticks([])
ax.set_xlabel(r"Combined parameter ($\propto$" + "mutation rate,\nspacer effectiveness, spacer acquisition)", fontsize = 12)

ax.set_ylabel("Diversity", fontsize = 12)

legend_elements.append(t)

plt.tight_layout()
plt.savefig("mean_m_vs_params.pdf")
plt.savefig("mean_m_vs_params.svg")
plt.savefig("mean_m_vs_params.png", dpi = 300)
# -

# ### Theta pv simulation

timestamp = '2021-06-17T13:40:03.119258'
folder = "/media/madeleine/My Passport/Data/results/2021-06-11/serialjobdir0530/"
#timestamp = '2021-10-14T03:19:46.217650'
#folder = "/media/madeleine/My Passport1/Data/results/2021-09-13/serialjobdir0396"

f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
     max_m, mutation_times, parent_list, all_phages = load_simulation(folder, timestamp, return_parents = True);

# +
t_ss_ind = t_ind = find_nearest(pop_array[:, -1].toarray()*g*c0, gen_max / 5)

skip = 100
nbi = pop_array[t_ss_ind::skip, 1:max_m +1].toarray()
nvi = pop_array[t_ss_ind::skip, max_m+1:2*max_m +1].toarray()
nbs = np.sum

e_effective_list = effective_e(nbi, nvi, all_phages, 'theta_function', e, theta)
# -

analytic_steady_state_vec = np.vectorize(analytic_steady_state)

nb_pred, nv_pred, C_pred, nu_pred = analytic_steady_state_vec(pv, e_effective_list, B, R, eta, f, c0, g, alpha)

# +
fig, axs = plt.subplots(2,1, figsize = (5,3.5))

colours = cm.viridis(np.linspace(0,0.9, 2))

time =  pop_array[t_ss_ind::skip,-1].toarray()*g*c0
nv = np.sum(nvi, axis = 1)
nb = np.sum(pop_array[t_ss_ind::skip, :max_m +1].toarray(), axis = 1).flatten()
C = pop_array[t_ss_ind::skip, -2].toarray()
nu = np.sum(nbi, axis = 1) / nb

axs[0].plot(time, nv, linewidth = 2, color = 'rebeccapurple', alpha = 0.9, label = "Simulation data")
axs[0].plot(time, nv_pred, color = 'k', linestyle = '--', label = "Theoretical prediction\n using measured average immunity")

axs[1].plot(time, nb, linewidth = 2, color = 'lightseagreen', alpha = 0.9)
axs[1].plot(time, nb_pred, color = 'k',linestyle = '--')

for i, ax in enumerate(axs):
    ax.set_xlim(2500, 7500)
    if i != 1:
        ax.set_xticks([])
    
axs[0].set_ylabel("Total phage")
axs[1].set_ylabel("Total bacteria")
axs[1].set_xlabel("Simulation time (bacterial generations)")

axs[1].tick_params(axis='y', colors='lightseagreen')
axs[1].yaxis.label.set_color('lightseagreen')
axs[0].tick_params(axis='y', colors='rebeccapurple')
axs[0].yaxis.label.set_color('rebeccapurple')

axs[0].legend(loc = 'upper left')

plt.tight_layout()
plt.savefig("pop_sizes_simulation_and_theory_small_%s.pdf" %timestamp)

# -






