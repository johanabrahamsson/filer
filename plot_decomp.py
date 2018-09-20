
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def plot_decomp(y,fs,decomp_mu,decomp_cov,osc_param,num_component):
    T = len(y.T)
    T1 = 0
    T2 = T
    Tstart = 0
    fig, ax = plt.subplots(num_component+2,1, figsize=(20, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .4, wspace=.0001)
    plt.figure(1)
    ax[0].plot(np.array([Tstart+np.arange(T1,T2)/fs]).T, y[T1:T2].T)
    ax[0].set_xlim([T1, T2])
    vars1 = np.zeros((num_component,T2))
    comp_sum = np.zeros((1,T2))
    c = np.sqrt(osc_param[num_component-1, 2*num_component+0])/np.sqrt(1-osc_param[num_component-1,0]**2)
    for i in np.arange(num_component):
        c = np.sqrt(osc_param[num_component-1, 2*num_component+i])/np.sqrt(1-osc_param[num_component-1,i]**2)
        std_bar = (np.sqrt(decomp_cov[2*i,2*i,T1:T2,num_component-1])).reshape(1,T2)
        vars1[i,:] = decomp_cov[2*i,2*i,T1:T2,num_component-1]
        xx = np.array([np.concatenate(((Tstart+np.arange(T1,T2)/fs).conjugate().transpose(), (Tstart+np.arange(T2-1,T1-1,-1)/fs).conjugate().transpose()), axis=0)]).conjugate().transpose()
        yy = np.array(np.concatenate((c*(np.array([decomp_mu[2*i,T1:T2,num_component-1]])-chi2.ppf(0.9, 1)*std_bar), c*(np.array([decomp_mu[2*i,T2::-1,num_component-1]])+chi2.ppf(0.9, 1)*std_bar[:,T2::-1])), axis=1)).conjugate().transpose()
        ax[i+1].fill(xx, yy, 'gray')
        ax[i+1].plot(np.array([Tstart+np.arange(T1,T2)/fs]).conjugate().transpose(), (c*decomp_mu[2*i,T1:T2,num_component-1]).conjugate().transpose())
        ax[i+1].set_xlim([T1, T2])
        comp_sum = comp_sum+c*decomp_mu[2*i,T1:T2,num_component-1]
    ax[num_component+1].plot(np.array([Tstart+np.arange(T1,T2)/fs]).conjugate().transpose(), (y[:,T1:T2]-comp_sum).conjugate().transpose())
    ax[num_component+1].set_xlim([T1, T2])
    plt.show()

