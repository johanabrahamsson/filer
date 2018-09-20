
# coding: utf-8

# In[ ]:

from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from angle_conf_MC import *
from plot_phase_area import *
from plot_phase_nocross import *

def plot_phase(y,fs,phi_prop,decomp_mu,decomp_cov,num_component):
    nmc = 10**4
    T = len(y.T)
    T1 = 0
    T2 = T
    Tstart = 0
    phase1 = np.zeros((1,T2))
    phase2 = np.zeros((1,T2-T1))
    seeds = np.random.standard_normal((2,nmc))
    fig, ax = plt.subplots(num_component,1, figsize=(20, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .4, wspace=.0001)
    plt.figure(1)
    for k in np.arange(num_component):
        for t in np.arange(T1,T2):
            tmp = angle_conf_MC(decomp_mu[2*k:2*k+2,t,num_component-1], decomp_cov[2*k:2*k+2,2*k:2*k+2,t,num_component-1], 2*norm.cdf(1)-1, seeds)
            phase1[:,t] = tmp[:,0]
            phase2[:,t] = tmp[:,1]
        plot_phase_area(np.array([Tstart+np.arange(T1,T2)/fs]).conjugate().transpose(),
                    phi_prop[k,T1:T2,num_component-1], phase1, phase2, ax, k)
        plot_phase_nocross(np.array([Tstart+np.arange(T1,T2)/fs]).conjugate().transpose(),
                    phi_prop[k,T1:T2,num_component-1], ax, k, T1)
    plt.show()

