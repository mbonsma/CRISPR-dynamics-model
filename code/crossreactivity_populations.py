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

# # Figure 5: cross-reactivity and average immunity

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ColorConverter
import colorsys


# from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


from spacer_model_plotting_functions import analytic_steady_state

# %matplotlib inline

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

# ## Total population size predictions with crossreactivity

import warnings
warnings.simplefilter('ignore', UserWarning)

# +
mu_vals = list(np.unique(grouped_data['mu']))
c0_vals = list(np.unique(grouped_data['C0']))
eta_vals = list(np.unique(grouped_data['eta']))
e_vals = list(np.unique(grouped_data['e']))

markerstyles = ['D', 'o', 's', 'P', '*', 'v', '>', 'd', 'X', 'h']
colours = sns.color_palette("hls", len(c0_vals))
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)),  (0, (3, 5, 1, 5, 1, 5)), (0, (3, 5, 1, 5, 3, 5)) ] 

colours = cm.viridis(np.linspace(0,0.9, 4)) # for eta
# -

d_dict = {"binary": 0, "theta_function": 0, "exponential": 1, "exponential_025": 4}
grouped_data['d'] = grouped_data['pv_type'].map(d_dict)

# +
markersize = 7
colour = 'C0'
shape = 'eta'
line = 'e'

e_select = 0.95
eta_select = 10**-4
mu_select = 10**-6
m_init_select = 1
B=170
pv=0.02
f=0.3
R = 0.04 # this is never varied
data_subset = grouped_data[(grouped_data['m_init'] == m_init_select)
                                    & (grouped_data['eta'] == eta_select)
                                    & (grouped_data['mu'] == mu_select)
                                    & (grouped_data['mean_m']['nanmean'] >= 1)
                                    & (grouped_data['B'] == B)
                                    & (grouped_data['pv'] == pv)
                                    & (grouped_data['f'] == f)]
                                    #& (grouped_data['theta'] < 2)]
                                    #& (grouped_data['pv_type'] == 'exponential')]

        
phage_colour = ColorConverter.to_rgb('rebeccapurple')
bac_colour = ColorConverter.to_rgb('lightseagreen')

theta_rgbs_phage = [scale_lightness(phage_colour, scale) for scale in list(np.linspace(0, 1.9, 4))]
d_rgbs_phage = [scale_lightness(phage_colour, scale) for scale in [0, 0.8, 1.5]]

theta_rgbs_bac = [scale_lightness(bac_colour, scale) for scale in list(np.linspace(0, 1.9, 4))]
d_rgbs_bac = [scale_lightness(bac_colour, scale) for scale in [0, 0.8, 1.5]]

d_color = ColorConverter.to_rgb("red")
theta_color = ColorConverter.to_rgb("blue")
theta_rgbs = [scale_lightness(theta_color, scale) for scale in list(np.linspace(0, 1.9, 4))]
d_rgbs = [scale_lightness(d_color, scale) for scale in [0, 0.8, 1.5]]

legend_elements = []

# add binary case
legend_elements.append(Line2D([0], [0], marker='*',  
                                  label=r'$\theta =0$',
                          markerfacecolor='k',markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

for i in range(len(data_subset['d'].unique())-1):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[1],  
                                  label=r'$\theta = %s$ (exp)' %np.sort(data_subset['d'].unique())[i+1],
                          markerfacecolor=d_rgbs[i+1],markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

for i in range(len(data_subset['theta'].unique())-1):
    legend_elements.append(Line2D([0], [0], marker=markerstyles[0],  
                                  label=r'$\theta = %s$' %int(np.sort(data_subset['theta'].unique())[i+1]),
                          markerfacecolor=theta_rgbs[i+1],markeredgecolor = 'k', markersize = markersize, linestyle = "None"))


legend_elements.append(Line2D([0], [0], label = 'Theory', linestyle = '-', color = 'k'))
#legend_elements.append(Line2D([0], [0], label = 'Simulation', linestyle = '-', alpha = 0.5, color = 'grey'))
    
# leged for phage and bacteria plot
legend_elements2 = []
# add binary case
legend_elements2.append(Line2D([0], [0], marker='*',  
                                  label=r'$\theta =0$',
                          markerfacecolor='k',markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

for i in range(len(data_subset['d'].unique())-1):
    legend_elements2.append(Line2D([0], [0], marker=markerstyles[1],  
                                  label=r'$\theta = %s$ (exp)' %np.sort(data_subset['d'].unique())[i+1],
                          markerfacecolor=d_rgbs_phage[i+1],markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

for i in range(len(data_subset['theta'].unique())-1):
    legend_elements2.append(Line2D([0], [0], marker=markerstyles[0],  
                                  label=r'$\theta = %s$' %int(np.sort(data_subset['theta'].unique())[i+1]),
                          markerfacecolor=theta_rgbs_phage[i+1],markeredgecolor = 'k', markersize = markersize, linestyle = "None"))



legend_elements2.append(Line2D([0], [0], label = 'Theory', linestyle = '-', color = 'k'))
#legend_elements.append(Line2D([0], [0], label = 'Simulation', linestyle = '-', alpha = 0.5, color = 'grey'))

# leged for phage and bacteria plot
legend_elements3 = []

legend_elements3.append(Line2D([0], [0], marker='D',  
                                  label='Phage',
                          markerfacecolor='rebeccapurple',markeredgecolor = 'k', markersize = markersize, linestyle = "None"))

legend_elements3.append(Line2D([0], [0], marker='D',  
                                  label='Bacteria',
                          markerfacecolor='lightseagreen',markeredgecolor = 'k', markersize = markersize, linestyle = "None"))
#legend_elements.append(Line2D([0], [0], label = 'Simulation', linestyle = '-', alpha = 0.5, color = 'grey'))

# +
fig = plt.figure(figsize = (10,4))

title_fontsize = 18
absolute_left = 0.04
absolute_right = 0.91
absolute_bottom = 0.13
absolute_top = 0.92

title_fontsize = 18
legend_fontsize = 9
label_fontsize = 10

# clone size distributions
gs_pop = gridspec.GridSpec(1,2)
gs_pop.update(left=0.42, right=absolute_right, bottom = absolute_bottom, top = absolute_top, wspace=0.08)
ax0 = plt.subplot(gs_pop[0])
ax1 = plt.subplot(gs_pop[1])

gs_est = gridspec.GridSpec(1,1)
gs_est.update(left=absolute_left, right=0.39, bottom = absolute_bottom, top = absolute_top)
ax2 = plt.subplot(gs_est[0])


# population size plots

for group in data_subset.groupby([colour, shape, line, 'theta', 'd']):
    data = group[1].sort_values(by = 'm_init')
    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]
    theta = group[0][3]
    d = group[0][4]

    theta_ind = list(np.sort(data_subset['theta'].unique())).index(theta)
    d_ind = list(np.sort(data_subset['d'].unique())).index(d)
        
    if d == 0 and theta == 0:
        marker_ind = 4
        colour_ind = 0
        phage_colours = theta_rgbs_phage
        bac_colours = theta_rgbs_bac
    elif d==0:
        marker_ind = 0
        colour_ind = theta_ind
        phage_colours = theta_rgbs_phage
        bac_colours = theta_rgbs_bac
    elif theta == 0:
        marker_ind = 1
        colour_ind = d_ind
        phage_colours = d_rgbs_phage
        bac_colours = d_rgbs_bac

    ax0.errorbar(  data['mean_m']['nanmean'], 
                     data['mean_nv']['nanmean'] /data['C0'],
                        yerr = data['mean_nv']['nanstd']/data['C0'],
                        xerr = data['mean_m']['nanstd'], 
                    c = phage_colours[colour_ind], 
               alpha = 0.5, marker = markerstyles[marker_ind], mec ='k', markersize = markersize, linestyle = "None")
        
    ax0.errorbar(  data['mean_m']['nanmean'], 
                     data['mean_nb']['nanmean'] /data['C0'],
                        yerr = data['mean_nb']['nanstd']/data['C0'],
                        xerr = data['mean_m']['nanstd'], 
                    c = bac_colours[colour_ind], 
               alpha = 0.5, marker = markerstyles[marker_ind], mec ='k', markersize = markersize, linestyle = "None")
    
        
    ax1.errorbar(  data['e_effective']['nanmean'], 
                     data['mean_nv']['nanmean'] /data['C0'],
                        yerr = data['mean_nv']['nanstd']/data['C0'],
                        xerr = data['e_effective']['nanstd'], 
                    c = phage_colours[colour_ind],  
               alpha = 0.5, marker = markerstyles[marker_ind], mec ='k', markersize = markersize, linestyle = "None")
        
    ax1.errorbar(  data['e_effective']['nanmean'], 
                     data['mean_nb']['nanmean'] /data['C0'],
                        yerr = data['mean_nb']['nanstd']/data['C0'],
                        xerr = data['e_effective']['nanstd'], 
                     c = bac_colours[colour_ind],  
               alpha = 0.5, marker = markerstyles[marker_ind], mec ='k', markersize = markersize, linestyle = "None")
    
ax0.set_xscale('log')
ax0.set_yscale('log')
ax1.set_yscale('log')
ax0.set_ylim(0,30)
ax1.set_ylim(0,30)
#ax0b.set_ylim(0, 25)
#ax1b.set_ylim(0, 25)
ax0.set_xlim(0.7,30)
#ax0b.set_xlim(0.7,30)

#ax1b.tick_params(axis='y', colors='lightseagreen')
#ax1b.yaxis.label.set_color('lightseagreen')
#ax1.tick_params(axis='y', colors='rebeccapurple')
#ax1.yaxis.label.set_color('rebeccapurple')
#ax1.set_ylabel(r"Mean phage population size $/C_0$")
#ax1b.set_ylabel(r"Mean bacteria population size $/C_0$")

ax0.set_xlabel("Mean number of clones " +r"$m$")
ax1.set_xlabel("Average bacterial immunity")

#ax0b.tick_params(axis='y', colors='lightseagreen')
#ax0b.yaxis.label.set_color('lightseagreen')
#ax0.tick_params(axis='y', colors='rebeccapurple')
#ax0.yaxis.label.set_color('rebeccapurple')
ax0.set_yticks([])
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")
ax1.set_ylabel(r"Mean population size $/C_0$")
#ax0b.set_ylabel(r"Mean bacteria population size $/C_0$")

# add theoretical line
e_effective_list = np.arange(0.0001, 0.9999, 0.01)
c0_select = 10**4
g = 1/(42*c0_select)
alpha = 2*10**-2/c0_select
analytic_steady_state_vec = np.vectorize(analytic_steady_state)
nb, nv, C, nu = analytic_steady_state_vec(pv, e_effective_list, B, R, eta_select, f, c0_select, g, alpha)

ax1.plot(e_effective_list, nb/c0_select, color = 'k', linewidth = 2)
ax1.plot(e_effective_list, nv/c0_select, color = 'k', linewidth = 2)

ax0.plot(1/e_effective_list, nb/c0_select, color = 'k', linewidth = 2)
ax0.plot(1/e_effective_list, nv/c0_select, color = 'k', linewidth = 2)

ax0.legend(handles = legend_elements2, loc = 'lower right', bbox_to_anchor = (1,0.28))
ax1.legend(handles = legend_elements3, loc = 'lower left', bbox_to_anchor = (0,0.4))

### Establishment with crossreactivity

data_subset["pred_e_eff"] = data_subset['e'] / data_subset['pred_bac_m_recursive']

for group in data_subset.groupby([colour, shape, line, 'theta', 'd']):
    data = group[1].sort_values(by = 'm_init')

    colour_variable = group[0][0]
    shape_variable = group[0][1]
    line_variable = group[0][2]
    theta = group[0][3]
    d = group[0][4]

    theta_ind = list(np.sort(data_subset['theta'].unique())).index(theta)
    d_ind = list(np.sort(data_subset['d'].unique())).index(d)
        
    if d == 0 and theta == 0:
        marker_ind = 4
        colour_ind = 0
        colours = theta_rgbs
    elif d==0:
        marker_ind = 0
        colour_ind = theta_ind
        colours = theta_rgbs
    elif theta == 0:
        marker_ind = 1
        colour_ind = d_ind
        colours = d_rgbs
        
    P_est_std = ((data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean']) 
            * np.sqrt((data['establishment_rate_nvi_ss']['nanstd'] / data['establishment_rate_nvi_ss']['nanmean'])**2 
            + (data['measured_mutation_rate']['nanstd'] / data['measured_mutation_rate']['nanmean'])**2 ))

    yerr = np.stack([np.zeros(P_est_std.shape), P_est_std])

    #if np.any(data['pv_type'] == 'exponential'):
    ax2.errorbar(data['e_effective']['nanmean'], 
                data['establishment_rate_nvi_ss']['nanmean'] / data['measured_mutation_rate']['nanmean'],
                    xerr = data['e_effective']['nanstd'], 
                  yerr = yerr,
                c = colours[colour_ind],
                 marker = markerstyles[marker_ind], mec ='k', markersize = markersize, linestyle = "None")

# plot theoretical line
data = data_subset[data_subset['pv_type'] == 'binary'].sort_values(by = "pred_e_eff")
e = data['e']

t, = ax2.plot(e/data['pred_bac_m_recursive'], 
    data['predicted_establishment_fraction_recursive'],
    linestyle = '-', color = 'k', linewidth = 2, zorder = 2, label = 'Theory')

#ax2.annotate("Increasing\ncross-reactivity",
#            xy=(3*10**-1, 6*10**-6), xycoords='data',
#            xytext=(1.5*10**-2, 2*10**-4), textcoords='data',
#            arrowprops=dict(facecolor='grey', arrowstyle="->"))

ax2.set_xscale('log')
ax2.set_yscale('log')


ax2.set_yticks([])
#ax3b.set_ylabel("Establishment probability")
ax2.set_xlabel(r"Average bacterial immunity")
ax2.set_ylabel("Phage establishment probability")

ax2.legend(handles = legend_elements)

ax2.set_title("A", loc = 'left', fontsize = title_fontsize)
ax0.set_title("B", loc = 'left', fontsize = title_fontsize)
ax1.set_title("C", loc = 'left', fontsize = title_fontsize)

#plt.savefig("populations_vs_mean_m_e_eff_crossreactivity.pdf")
#plt.savefig("populations_vs_mean_m_e_eff_crossreactivity.svg")
plt.savefig("fig5.pdf")
plt.savefig("fig5.svg")
# -


