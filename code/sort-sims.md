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

# Process simulations and create sparse population array

```python
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
```

```python
from sim_analysis_functions import recreate_x, load_simulation
```

```python
from sim_analysis_functions import find_nearest
```

```python
top_folder = "/media/madeleine/My Passport/Data/results/"
data_folder = "2019-05-14"
fn_list = []
folder_list = []

# get list of files to analyze
if data_folder == "2019-05-14" or data_folder == "2019-03-14": # these have a different structure with sub-folders
    for folder in glob("%s/run*" %(top_folder + data_folder)):
        for fn in os.listdir(folder):
            if fn[:-4] == 'serialjobdir':
                fn_list.append(fn)
                folder_list.append(folder)        
else:
    for fn in os.listdir(top_folder + data_folder):
        if fn[:-4] == 'serialjobdir':
            fn_list.append(fn)
            folder_list.append(top_folder + data_folder)
            
for i, fn in enumerate(fn_list):
    folder = folder_list[i]
    
    if data_folder == "2019-05-14":

        if int(folder[-2:]) < 2:
            continue

    print(folder, fn)

    for fn2 in os.listdir("%s/%s" %(folder, fn)):          
        if fn2[:10] == 'parameters':
            timestamp = fn2[11:].split('.txt')[0]
            try:
                f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
                 max_m, mutation_times, all_phages = load_simulation("%s/%s" %(folder, fn), timestamp)
            except:
                print("failed to load %s" %timestamp)
                raise
                continue


```
```python

```
