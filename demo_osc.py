
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import matplotlib.pyplot as plt
from osc_decomp import *
from plot_decomp import *
from plot_phase import *

with open('Lynx_new.csv', 'r', encoding='utf-8-sig') as csv_file:
    Data = csv.reader(csv_file)
    y = []
    for line in Data:
        y.append(np.log(10**float(line[0]))) #For Lynx_new data
        #y.append(np.log(float(line[0])+1)) #For sunspot data


y = np.array([y])
fs = 1
MAX_COMPONENT = 2
decomp_phase, decomp_mu, decomp_cov, AIC_osc, osc_param = osc_decomp(y, fs, MAX_COMPONENT)
num_component = np.argmin(AIC_osc)
K = num_component + 1 
tmp = np.amin(AIC_osc)
plot_decomp(y, fs, decomp_mu, decomp_cov, osc_param, K)
plot_phase(y, fs, decomp_phase, decomp_mu, decomp_cov, K)

