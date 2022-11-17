---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: spacer_phage
    language: python
    name: spacer_phage
---

## Simulating the effect of longer arrays on average immunity


The average overlap calculation depends on knowing the number of spacers and protospacers per organism: basically, by considering all spacers and protospacers as belonging to separate organisms, we underestimate average immunity. For example, if phages have 3 protospacers each, a bacterium needs only one of the three to be immune to it. 

Table S2: "Average length of the CRISPR loci by time point. CRISPR locus expansion was calculated by dividing the difference between the total and the wild-type locus length by the length of the corresponding spacer-repeat segment. Locus length was calculated using the repeat-containing reads divided by the host coverage by time point and CRISPR locus."

For CRISPR1 (the data I use), the average lengths are around 4 spacers. This wouldn't give a straight factor of 4 boost to average immunity though, since some new spacers would target the same phage as others in the locus, not boosting average immunity. 

My simple theory is that the initial overlap assuming each spacer is a single organism (regardless of the number of protospacers) gives a "base fraction" that determines what the effect is of adding more spacers. Let's call the intial single-spacer overlap $1-C$: $C = 1-a_1$. Then the average immunity $a$ if all bacteria have $n$ spacers is:

$$a_{n+1} = 1- (1 - a_n)C$$

Plugging in $a_1 = C$, we get

$$a_n = 1 - C^n$$

Now we'll explore this theory for a few different assumptions of how the spacers and arrays are distributed


```python
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import string
import matplotlib.cm as cm
from matplotlib.lines import Line2D
```

```python
%matplotlib inline
```

```python
def count_matches(phages, spacers):
    """
    count the total number of phage matches from any of the spacers in spacers
    Just have to be the same string to match
    """
    presence_counter = 0
    for p in phages:
        present = 0
        for sp in spacers:
            if sp in p:
                present = 1
        presence_counter += present
        
    return presence_counter
```

```python
def generate_spacers(dist, spacer_seqs, num_spacers = 420, beta = 6):
    """
    Generate a set of bacteria spacers drawn from the `spacer_seqs` list with distribution `dist`.
    
    Inputs:
    dist: either 'exponential' or 'uniform'
    spacer_seqs: a list of possible sequences (letters of the alphabet here)
    num_spacers: total number of spacers to sample
    beta: parameter for the exponential distribution. Beta is the distribution mean.
    
    Returns:
    spacers_total: list of all the spacer sequences
    """
    
    # generate set of bacteria spacers
    spacers_total = []
    np.random.shuffle(spacer_seqs) # shuffle list so order doesn't matter
    for n in range(num_spacers):
        # draw indices from exponential distribution
        if dist == 'exponential':
            ind = int(np.random.exponential(beta))
            while ind >= len(spacer_seqs):
                ind = int(np.random.exponential(beta))
            spacers_total.append(spacer_seqs[ind])
        elif dist == 'uniform':
        # sample uniformly from the list
            spacers_total.append(np.random.choice(spacer_seqs))
            
    return spacers_total
```

```python
def create_arrays(array_size, spacers_total, length_dist, sigma = 2):
    """
    Randomly split the set of `spacers_total` into arrays of mean length `array_size` sampled from `length_dist` distribution.
    
    Inputs:
    array_size : characteristic array length
    spacers_total : set of total spacers
    length_dist : distribution from which to draw array lengths: either 'constant', 'gaussian', or 'exponential'
    sigma : scale parameter for the `gaussian` length_dist
    
    Returns:
    arrays: list of the returned arrays    
    """
    
    inds = np.arange(0,len(spacers_total),1)
    
    arrays = []
    for i in range(int(len(spacers_total) / array_size)):
        if length_dist == 'constant':
            size = array_size
        elif length_dist == 'gaussian':
            size = 0
            while size < 1:
                size = round(np.random.normal(loc = array_size, scale = sigma))
        elif length_dist == 'exponential':
            size = 0
            while size < 1:
                size = round(np.random.exponential(array_size))
        else:
            print("Invalid length distribution")
            
        if size > len(inds):
            choice = inds
        else:
            choice = np.random.choice(inds, size=size, replace=False)
        spacers = np.array(spacers_total)[choice]
    
        arrays.append(list(spacers))

        # delete the sampled spacers
        for j in choice:
            to_delete = np.where(inds == j)[0]
            inds = np.delete(inds, to_delete)
        
        if len(inds) == 0: # if all the spacers have been sampled
            break
            
    return arrays
```

## Effect of array length on average immunity, all else being equal

```python
spacer_seqs = list(string.ascii_uppercase)
```

```python
# generate phage sequences
# series of 5 unique letters randomly drawn from the alphabet
phages = []
num_phages = 50
for n in range(num_phages):
    phages.append(list(np.random.choice(spacer_seqs, 5, replace = False)))
```

```python
# distribution of 
dists = ['uniform', 'exponential']
array_sizes = np.arange(1, 8, 1)
n_iter = 50
length_dists = ['constant', 'gaussian', 'exponential']
num_spacers = 420
    
fig, axs = plt.subplots(2,3, figsize = (8,5))

# iterate over spacer distributions
for i, dist in enumerate(dists):
    spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)

    # iterate over array length distributions
    for j, length_dist in enumerate(length_dists):
        mean_avg_immunity = []
        std_avg_immunity = []
        
        for array_size in tqdm(array_sizes):
            avg_immunity_vals = []

            # simulate array splitting n_iter times
            for n in range(n_iter): 
                avg_immunity = 0
                arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
                for spacers in arrays:
                    overlap = count_matches(phages, spacers)
                    avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

                avg_immunity_vals.append(avg_immunity)
                
            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))
            
        # initial average immunity:
        C = 1-mean_avg_immunity[0]
        axs[i,j].errorbar(array_sizes, np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), label = "Simulated spacer\ndistributions")

        axs[i,j].plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
        
        axs[i,j].set_title("%s spacers,\n%s array length" %(dist, length_dist))
        
        axs[i,j].set_ylim(0.1, 0.8)

axs[0,0].set_ylabel("Average immunity")
axs[1,0].set_ylabel("Average immunity")
axs[1,1].set_xlabel("Mean bacterial array length")

axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[0,2].set_xticklabels([])

#ax.set_xlabel("Mean bacteria array length")
axs[0,0].legend()
plt.tight_layout()

plt.savefig("simulated_avg_immunity_vs_array_length_full_nphage_%s_nbac_%s.pdf" %(num_phages, num_spacers))
```

```python
spacer_seqs = list(string.ascii_uppercase)[:6]
```

```python
# generate phage sequences
# series of 5 unique letters randomly drawn from the alphabet
phages = []
num_phages = 100
for n in range(num_phages):
    phages.append(list(np.random.choice(spacer_seqs, 5, replace = False)))
```

```python
# sort protospacers to calculate phage diversity
phages_list = []

for phage in phages:
    phage.sort()
    phages_list.append("".join(phage))
```

```python
len(set(phages_list))
```

```python
# Single plot
dist = 'exponential'
length_dist = 'gaussian'

array_sizes = np.arange(1, 8, 1)
n_iter = 50
num_spacers = 1260

mean_avg_immunity = []
std_avg_immunity = []
mean_diversity = []
std_diversity = []
total_arrays = [] 

spacers_total = generate_spacers(dist, spacer_seqs, num_spacers = num_spacers, beta = 6)

for array_size in tqdm(array_sizes):
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    # simulate array splitting n_iter times
    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(spacers_total) / array_size)

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))
        
        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
    
    mean_diversity.append(np.mean(diversity))
    std_diversity.append(np.std(diversity))
    mean_avg_immunity.append(np.mean(avg_immunity_vals))
    std_avg_immunity.append(np.std(avg_immunity_vals))
    total_arrays.append(np.mean(num_arrays))  

```

```python
# plot of diversity vs average immunity for fixed total number of spacers but variable array size


# initial average immunity:
C = 1-mean_avg_immunity[0]

fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar(np.array(mean_avg_immunity), mean_diversity,  marker = 'o', 
           xerr = np.array(std_avg_immunity), label = "Simulated spacer distributions")
ax.plot(np.array(mean_avg_immunity), total_arrays, marker = 'o', label = "Total population size")

#ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_xlabel("Average immunity")
ax.set_ylabel("Bacterial diversity")
ax.legend()
plt.tight_layout()
```

```python

# initial average immunity:
C = 1-mean_avg_immunity[0]

fig, ax = plt.subplots(figsize = (5,4))
ax.errorbar(array_sizes, np.array(mean_avg_immunity), marker = 'o', 
           yerr = np.array(std_avg_immunity), label = "Simulated spacer distributions")

ax.plot(array_sizes, 1 - C**array_sizes, linestyle = '--', color = 'k', label = "Theory")
#plt.yscale('log')

ax.set_ylabel("Average immunity")
ax.set_xlabel("Mean bacteria array length")
ax.legend()
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_array_length_nphage_%s_nbac_%s_%s_array_size_%s_spacers.pdf" %(num_phages, num_spacers, length_dist, dist))
```

## Common2020 experiment


Experimental setup: combine equal proportions of bacterial strains with a different CRISPR spacer with 1 phage strain that is either targeted by all spacers or escaped from one of the spacers. There is then 1 susceptible bacterial strain per mixture when the escape phage is used. All experiments also contain 1 surface mutant bacterial strain that is always immune to the phage. 

We can directly calculate the expected initial average immunity based on this setup. For $m$ bacterial CRISPR strains, 1 surface mutant strain, and 1 escape phage, total bacteria $N_b$, total phage $N_v$, each bacterial strain has abundance $N_b/m$ and $v_j = N_v$. 

Average immunity = $\frac{\sum_{i,j} b_i v_j \delta_{ij}}{N_i N_j}$

We can include the surface mutant or not, it slightly shifts the total diversity but the trend is unaffected. For instance, for $m=3$ CRISPR clones, the phage is able to infect 1 of them, so average immunity is $\frac{N_v N_b/m + N_v N_b/m}{N_b N_v} = \frac{2}{m} = \frac{2}{3}$. If we included the surface mutant in equal proportions, we would get $3/4$ instead. 

```python
# based on Common2020
diversity_levels = np.array([1,3,6,12,24])
e_eff_levels = (diversity_levels -1)/ diversity_levels # 1 susceptible bacterial strain per mixture, equal proportions
```

In their setup, average immunity increases as diversity increases.

```python
fig, ax = plt.subplots(figsize = (4,3))
#ax.plot(total_diversity, e_eff_levels, marker = 'o')
ax.plot(diversity_levels, e_eff_levels, marker = 'o')
ax.set_xlabel("Diversity (number of bacterial strains)")
ax.set_ylabel("Initial average immunity")
plt.tight_layout()
plt.savefig("Common2020_average_immunity_vs_diversity.png", dpi = 150)
```

## Model 1 - randomly arranged spacers and protospacers

In the previous section, I fix the total set of spacers and protospacers, then shuffle them around differently. 

Here, I want to increase the number of possible spacers (which would increase diversity) for a fixed array size, then try different phage array sizes. 

At some point, even though phages have multiple protospacers, bacteria will be dividing their eggs among too many baskets and average immunity will go down.

```python
def simulate_arrays(bac_array_length, phage_array_length, diversity, spacer_seqs_total, n_iter, 
                    method='random', length_dist = 'constant', dist = 'exponential'):
    """
    Generate synthetic protospacer and spacer arrays. 
    
    Input:
    bac_array_length : generate bacterial arrays omax_diversitylength. If length_dist == 'constant', all arrays 
        will be the same length.
    phage_array_length : generate phage arrays of this length.
    diversity : total size of the spacer/protospacer pool to draw from; the total number of unique sequences in the population.
    spacer_seqs_total : a list of unique spacer sequences (must be longer than diversity)
    n_iter : number of times to simulate dividing spacers into bacterial arrays
    method : "sequential", "parallel", or "random" - how to divide protospacers into phage arrays.
        If "sequential", generate a number of phages equal to the diversity where each phage is 
        one protospacer different from the one before it in the list. This simulates successive mutational sweeps.
        If "parallel", generate a number of phages equal to the diversity plus phage_array_length where
        each phage has a different final protospacer but all others are shared. This simulates clonal interference.
        If "random", divide protospacers randomly among num_phages so that each phage has 
        phage_array_length unique protospacers. Sequences may be resampled in different phages. 
    length_dist : "constant", "gaussian", or "exponential" - the distribution for bacterial array lengths.
    dist : "exponential" or "uniform" - the distribution to sample spacer sequences from to create bacterial arrays.
    
    Returns: 
    diversity : number of unique bacterial genotypes generated for each iteration
    phage_diversity : number of unique phage genotypes generated
    avg_immunity_vals : average immunity for each iteration
    num_arrays : number of bacteria for each iteration
    actual_diversity : number of unique protospacer and spacer sequences used 
        (different from the input diversity only if method == "parallel")
    
    """
    # make a set of spacers to arrange into arrays
    if method == "parallel":
        spacer_seqs = spacer_seqs_total[:d+phage_array_length]
    else:
        spacer_seqs = spacer_seqs_total[:d]

    actual_diversity = len(spacer_seqs)   
        
    if method == "sequential":
        # generate sequentially mutated phages
        phages = []
        for i in range(len(spacer_seqs) - phage_array_length + 1):
            phages.append(spacer_seqs[i:i+phage_array_length])
    elif method == "parallel":
        # generate parallel mutated phages
        phages = []
        for i in range(len(spacer_seqs) - phage_array_length + 1):
            base = spacer_seqs[:phage_array_length-1]
            base.append(spacer_seqs[phage_array_length + i-1])
            phages.append(base)
    else:
        # generate phage sequences
        # series of `phage_array_length` unique letter combinations
        # use the same spacer sequences that bacteria get
        phages = []
        for n in range(num_phages):
            phages.append(list(np.random.choice(spacer_seqs, phage_array_length, replace = False)))

    protospacers = list(np.array(phages).flatten()) # all the unique protospacers
    # sample in proportion to the protospacer list
    spacers_total = generate_spacers(dist, protospacers, num_spacers = num_spacers, beta = 6)
            
    # calculate phage diversity
    phages_list = []

    for phage in phages:
        phage.sort()
        phages_list.append("".join(phage))

    phage_diversity.append(len(set(phages_list)))
    
    # simulate array splitting n_iter times
    avg_immunity_vals = []
    diversity = []
    num_arrays = []

    for n in range(n_iter): 
        avg_immunity = 0
        arrays_concat = []
        arrays = create_arrays(array_size, spacers_total, length_dist, sigma = 2)
        for spacers in arrays:
            overlap = count_matches(phages, spacers)
            avg_immunity += overlap / (len(phages) * len(arrays))

            # calculate bacterial diversity: number of unique genotypes
            spacers.sort()
            arrays_concat.append("".join(spacers))

        diversity.append(len(set(arrays_concat)))
        avg_immunity_vals.append(avg_immunity)
        num_arrays.append(len(arrays))
    
    return diversity, phage_diversity, avg_immunity_vals, num_arrays, actual_diversity
```

```python
#generate two-letter spacer codes for higher total diversity
spacer_seqs_total = []
for letter1 in list(string.ascii_uppercase):
    for letter2 in list(string.ascii_uppercase):
        spacer_seqs_total.append(letter1+letter2)
```

If the phage diversity is also increasing as bacterial diversity increases, average immunity will go down as diversity increases, provided the diversity is larger than the bacterial array size.

```python
# set parameters
array_sizes = [1,2,4] # bacteria array size - want this to be low to be in the low-bacterial array size limit, but high enough that can get >26 values of diversity
#array_sizes = [1, 4]
dist = 'uniform' #  exponential or uniform
#length_dist = 'gaussian'
length_dist = 'gaussian' # or constant or exponential
n_iter = 20
phage_array_lengths = [1,4,7,10]
#phage_array_lengths = [1,10]
#num_spacers = array_size * 400
num_spacers = 1000 # number of spacers to draw from the distribution
num_phages = 100
max_diversity = 100
diversity_step = 10
```

```python
bac_array_sizes_m1 = []
phage_array_sizes_m1 = []
diversity_vals_m1 = []
mean_avg_immunity_list_m1 = []
std_avg_immunity_list_m1 = []
mean_diversity_list = []
std_diversity_list = []
total_arrays_list = [] 
phage_diversity_list = []

for i, array_size in enumerate(array_sizes):
    for j, phage_array_length in tqdm(enumerate(phage_array_lengths)):
        diversity_vals_list = np.arange(phage_array_length, max_diversity, diversity_step)
        
        mean_avg_immunity = []
        std_avg_immunity = []
        mean_diversity = []
        std_diversity = []
        total_arrays = [] 
        phage_diversity = []

        for k, d in enumerate(diversity_vals_list):
            diversity, phage_diversity, avg_immunity_vals, num_arrays, actual_diversity = simulate_arrays(array_size, phage_array_length,
                                                                                d, spacer_seqs_total, n_iter,
                                                                                method = "random", length_dist = length_dist,
                                                                                                         dist = dist)
            mean_diversity.append(np.mean(diversity))
            std_diversity.append(np.std(diversity))
            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))
            total_arrays.append(np.mean(num_arrays))  
            
        bac_array_sizes_m1.append(array_size)
        phage_array_sizes_m1.append(phage_array_length)
        diversity_vals_m1.append(diversity_vals_list)
        mean_avg_immunity_list_m1.append(mean_avg_immunity)
        std_avg_immunity_list_m1.append(std_avg_immunity)
        mean_diversity_list.append(mean_diversity) # number of unique bacterial genotypes
        std_diversity_list.append(std_diversity)
        total_arrays_list.append(total_arrays)
        phage_diversity_list.append(phage_diversity)
        
```

### Plot

```python
markers = ['o', 'v', '*', 's']
linestyles = ['-', '--', '-.', ':']
colours = cm.viridis(np.linspace(0,1, len(phage_array_lengths)))
ms = 10
```

```python
legend_elements = []
for i in range(len(array_sizes)):
    if i == 0:
        label = "1 spacer"
    else:
        label='%s spacers' %(array_sizes[i])
    legend_elements.append(Line2D([0], [0], marker=markers[i],  
                                  label = label,
                          markerfacecolor='grey', markeredgecolor = 'None', markersize = ms, linestyle = linestyles[i], 
                          color = 'grey'))


for i in range(len(phage_array_lengths)):
    if i == 0:
        label = "1 protospacer"
    else:
        label='%s protospacers' %(phage_array_lengths[i])
    legend_elements.append(Line2D([0], [0], marker=markers[0],  
                                  label = label,
                          markerfacecolor=colours[i], markeredgecolor = 'None', markersize = ms, linestyle = "None"))
```

```python
fig, ax = plt.subplots(figsize = (7,4))

for i in range(len(bac_array_sizes_m1)):
    bac_ind = array_sizes.index(bac_array_sizes_m1[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m1[i])
    ax.errorbar(diversity_vals_m1[i],  np.array(mean_avg_immunity_list_m1[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 7,
                linestyle = linestyles[bac_ind],
           yerr = np.array(std_avg_immunity_list_m1[i]), color = colours[phage_ind],
            label = "Simulated spacer distributions")
    

ax.set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
ax.set_ylabel("Average immunity")
ax.legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))
ax.set_xlim(1,50)
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_spacer_diversity_%s_lengthdist_%s_dist_%s.pdf" %("random", length_dist, dist))
```

## Model 2 - Sequential phage mutations, successive sweeps regime


Now we simulate the difference between a set of more closely related phages with escape mutations. First, we assume that mutations are sequentially building on top of each other, i.e. that the phage clone that mutates a protospacer first is also the clone that mutates the second protospacer, etc. 

We assume with increasing diversity that phage clones are present in equal proportion. This is not necessary to assume but any other choice is harder to justify and should really be done in a future project with a proper population dynamics model.

Bacteria sample spacers either exponentially from the distribution of protospacers (mimicking data) or uniformly (reasonable if we ignore selection). 

```python
bac_array_sizes_m2 = []
phage_array_sizes_m2 = []
diversity_vals_m2 = []
mean_avg_immunity_list_m2 = []
std_avg_immunity_list_m2 = []

for i, array_size in enumerate(array_sizes): 
    for j, phage_array_length in tqdm(enumerate(phage_array_lengths)):
        diversity_vals_list = np.arange(phage_array_length, max_diversity, diversity_step)

        mean_avg_immunity = []
        std_avg_immunity = []
        
        for k, d in enumerate(diversity_vals_list):
            diversity, phage_diversity, avg_immunity_vals, num_arrays, actual_diversity = simulate_arrays(array_size, 
                                                                                                          phage_array_length,
                                                                                        d, spacer_seqs_total, n_iter,
                                                                                       method = "sequential", length_dist = length_dist,
                                                                                                         dist = dist)

            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))

        bac_array_sizes_m2.append(array_size)
        phage_array_sizes_m2.append(phage_array_length)
        diversity_vals_m2.append(diversity_vals_list)
        mean_avg_immunity_list_m2.append(mean_avg_immunity)
        std_avg_immunity_list_m2.append(std_avg_immunity)
        
```

```python
fig, ax = plt.subplots(figsize = (7,4))

for i in range(len(bac_array_sizes_m2)):
    bac_ind = array_sizes.index(bac_array_sizes_m2[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m2[i])
    ax.errorbar(diversity_vals_m2[i],  np.array(mean_avg_immunity_list_m2[i]),
                yerr = np.array(std_avg_immunity_list_m2[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 7,
                linestyle = linestyles[bac_ind], color = colours[phage_ind],
            label = "Simulated spacer distributions")
    

ax.set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
ax.set_ylabel("Average immunity")
ax.legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))
#ax.set_xlim(1,26)
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_spacer_diversity_%s_lengthdist_%s_dist_%s.pdf" %("sequential", length_dist, dist))
```

## Model 3 - parallel phage mutations, clonal interference regime


What about phages where just one protospacer is different and the rest are the same?

```python
bac_array_sizes_m3 = []
phage_array_sizes_m3 = []
diversity_vals_m3 = []
mean_avg_immunity_list_m3 = []
std_avg_immunity_list_m3 = []
actual_diversity_list = []

for i, array_size in enumerate(array_sizes): 
    for j, phage_array_length in tqdm(enumerate(phage_array_lengths)):
        diversity_vals_list = np.arange(phage_array_length, max_diversity, diversity_step)

        mean_avg_immunity = []
        std_avg_immunity = []
        actual_diversity_mean = []
        
        for k, d in enumerate(diversity_vals_list):
            diversity, phage_diversity, avg_immunity_vals, num_arrays, actual_diversity = simulate_arrays(array_size, 
                                                                                                          phage_array_length,
                                                                                        d, spacer_seqs_total, n_iter,
                                                                                       method = "parallel", 
                                                                                            length_dist = length_dist,
                                                                                            dist = dist)

            mean_avg_immunity.append(np.mean(avg_immunity_vals))
            std_avg_immunity.append(np.std(avg_immunity_vals))
            actual_diversity_mean.append(np.mean(actual_diversity))

        bac_array_sizes_m3.append(array_size)
        phage_array_sizes_m3.append(phage_array_length)
        diversity_vals_m3.append(diversity_vals_list)
        mean_avg_immunity_list_m3.append(mean_avg_immunity)
        std_avg_immunity_list_m3.append(std_avg_immunity)
        actual_diversity_list.append(actual_diversity_mean)
        
```

```python
fig, ax = plt.subplots(figsize = (7,4))

for i in range(len(bac_array_sizes_m3)):
    bac_ind = array_sizes.index(bac_array_sizes_m3[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m3[i])
    ax.errorbar(actual_diversity_list[i],  np.array(mean_avg_immunity_list_m3[i]),
                yerr = np.array(std_avg_immunity_list_m3[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 7,
                linestyle = linestyles[bac_ind], color = colours[phage_ind],
            label = "Simulated spacer distributions")
    

ax.set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
ax.set_ylabel("Average immunity")
ax.legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))
#ax.set_xlim(1,26)
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("simulated_avg_immunity_vs_spacer_diversity_%s_lengthdist_%s_dist_%s.pdf" %("parallel", length_dist, dist))
```

## Combined figure

```python
fig, axs = plt.subplots(1,4, figsize = (13,3.5))

# model 1
for i in range(len(bac_array_sizes_m1)):
    bac_ind = array_sizes.index(bac_array_sizes_m1[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m1[i])
    axs[1].errorbar(diversity_vals_m1[i],  np.array(mean_avg_immunity_list_m1[i]), 
                    yerr = np.array(std_avg_immunity_list_m1[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")

# model 2
for i in range(len(bac_array_sizes_m2)):
    bac_ind = array_sizes.index(bac_array_sizes_m2[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m2[i])
    axs[2].errorbar(diversity_vals_m2[i],  np.array(mean_avg_immunity_list_m2[i]), 
                    yerr = np.array(std_avg_immunity_list_m2[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")

# model 3
for i in range(len(bac_array_sizes_m3)):
    bac_ind = array_sizes.index(bac_array_sizes_m3[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m3[i])
    axs[3].errorbar(actual_diversity_list[i],  np.array(mean_avg_immunity_list_m3[i]), 
                    yerr = np.array(std_avg_immunity_list_m3[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")
    

for ax in axs:
    ax.set_ylim(0,1)
    
for ax in axs[1:]:
    ax.set_yticks([])
    ax.set_xlim(0, 90)
    
#axs[1].set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
axs[3].legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))
axs[1].set_title("B", loc = 'left', fontsize = 16)

axs[2].set_xlabel("Diversity (number of unique spacer and protospacer types)")
#axs[3].set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
#axs[2].legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))
axs[2].set_title("C", loc = 'left', fontsize = 16)
axs[3].set_title("D", loc = 'left', fontsize = 16)

axs[0].plot(diversity_levels, e_eff_levels, marker = 'o', color = 'k')
axs[0].set_xlabel("Diversity (number of spacer types)")
axs[0].set_ylabel("Average immunity")
#axs[0].set_title("Experiment", fontsize = 16)
#axs[2].set_title("Simulation", fontsize = 16)
axs[0].set_title("A", loc = 'left', fontsize = 16)

plt.tight_layout()
plt.savefig("Average_immunity_diversity_comparison_lengthdist_%s_dist_%s.pdf" %(length_dist, dist))
plt.savefig("Average_immunity_diversity_comparison_lengthdist_%s_dist_%s.svg" %(length_dist, dist))
```

```python
# figure for supplementary

fig, axs = plt.subplots(1,3, figsize = (10,3.5))

# model 1
for i in range(len(bac_array_sizes_m1)):
    bac_ind = array_sizes.index(bac_array_sizes_m1[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m1[i])
    axs[0].errorbar(diversity_vals_m1[i],  np.array(mean_avg_immunity_list_m1[i]), 
                    yerr = np.array(std_avg_immunity_list_m1[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")

# model 2
for i in range(len(bac_array_sizes_m2)):
    bac_ind = array_sizes.index(bac_array_sizes_m2[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m2[i])
    axs[1].errorbar(diversity_vals_m2[i],  np.array(mean_avg_immunity_list_m2[i]), 
                    yerr = np.array(std_avg_immunity_list_m2[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")

# model 3
for i in range(len(bac_array_sizes_m3)):
    bac_ind = array_sizes.index(bac_array_sizes_m3[i])
    phage_ind = phage_array_lengths.index(phage_array_sizes_m3[i])
    axs[2].errorbar(actual_diversity_list[i],  np.array(mean_avg_immunity_list_m3[i]), 
                    yerr = np.array(std_avg_immunity_list_m3[i]), markeredgecolor = 'None', 
                marker = markers[bac_ind], markersize = 6,
                linestyle = linestyles[bac_ind],
            color = colours[phage_ind],
            label = "Simulated spacer distributions")
    

for ax in axs:
    ax.set_ylim(0,1)
    ax.set_xlim(0, 90)
    
for ax in axs[1:]:
    ax.set_yticks([])
    
#axs[1].set_xlabel("Diversity (number of unique spacer\nand protospacer types)")
axs[2].legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1,0.5))

axs[1].set_xlabel("Diversity (number of unique spacer and protospacer types)")
axs[1].set_title("B", loc = 'left', fontsize = 16)
axs[2].set_title("C", loc = 'left', fontsize = 16)

axs[0].set_ylabel("Average immunity")
axs[0].set_title("A", loc = 'left', fontsize = 16)

plt.tight_layout()
plt.savefig("Average_immunity_diversity_comparison_lengthdist_%s_dist_%s_SI.pdf" %(length_dist, dist))
plt.savefig("Average_immunity_diversity_comparison_lengthdist_%s_dist_%s_SI.svg" %(length_dist, dist))
```
