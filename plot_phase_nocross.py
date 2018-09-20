
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

def plot_phase_nocross(t, phase, ax, k, T1):
    t = np.reshape(t, len(t))
    for i in np.arange(len(t)-1):
        if phase[i]-phase[i+1] > np.pi:
            turn = t[i]+(np.pi-phase[i])/(phase[i+1]+2*np.pi-phase[i])*(t[i+1]-t[i])
            ax[k].plot(np.concatenate((np.array([[t[i]]]), np.array([[turn]])), axis=1).conj().transpose(),
                       np.concatenate((np.array([[phase[i]]]), np.array([[np.pi]])), axis=1).conj().transpose(), color='b', linewidth=2.0)
            
            ax[k].plot(np.concatenate((np.array([[turn]]), np.array([[t[i+1]]])), axis=1).conj().transpose(), 
                       np.concatenate((np.array([[-np.pi]]), np.array([[phase[i+1]]])), axis=1).conj().transpose(), color='b', linewidth=2.0)
            ax[k].set_xlim([T1, len(t)])
            ax[k].set_ylim([-np.pi, np.pi])
            ax[k].set_yticks([-3.14, 0, 3.14])
        else:
            if phase[i]-phase[i+1] < -np.pi:
                turn = t[i]+(phase[i]+np.pi)/(phase[i]-phase[i+1]+2*np.pi)*(t[i+1]-t[i])
                ax[k].plot(np.concatenate((np.array([[t[i]]]), np.array([[turn]])), axis=1).conj().transpose(),
                       np.concatenate((np.array([[phase[i]]]), np.array([[-np.pi]])), axis=1).conj().transpose(), color='b', linewidth=2.0)
                
                ax[k].plot(np.concatenate((np.array([[turn]]), np.array([[t[i+1]]])), axis=1).conj().transpose(), 
                       np.concatenate((np.array([[np.pi]]), np.array([[phase[i+1]]])), axis=1).conj().transpose(), color='b', linewidth=2.0)
                ax[k].set_xlim([T1, len(t)])
                ax[k].set_ylim([-np.pi, np.pi])
                ax[k].set_yticks([-3.14, 0, 3.14])
            else:
                ax[k].plot(np.concatenate((np.array([[t[i]]]), np.array([[t[i+1]]])), axis=1).conj().transpose(), 
                       np.concatenate((np.array([[phase[i]]]), np.array([[phase[i+1]]])), axis=1).conj().transpose(), color='b', linewidth=2.0)
                ax[k].set_xlim([T1, len(t)])
                ax[k].set_ylim([-np.pi, np.pi])
                ax[k].set_yticks([-3.14, 0, 3.14])
    return ax

