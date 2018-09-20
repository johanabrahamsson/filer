
# coding: utf-8

# In[ ]:

from numpy import linalg as LA
import numpy as np
from scipy.linalg import sqrtm

def angle_conf_MC(mu,Sigma,prob,seeds):
    nmc = np.size(seeds,1)
    phi = np.arctan2(mu[1],mu[0])
    P = np.concatenate((np.concatenate((np.array([[np.cos(phi)]]), np.array([[np.sin(phi)]])),axis=1), np.concatenate((np.array([[-np.sin(phi)]]), np.array([[np.cos(phi)]])),axis=1)), axis=0)
    Sigmasqrt = sqrtm(Sigma)
    seeds = LA.norm(mu)*np.concatenate((np.ones((1,nmc)), np.zeros((1,nmc))), axis=0)+np.matmul(np.matmul(P, Sigmasqrt), seeds)
    phases = np.arctan2(seeds[1,:], seeds[0,:])
    tmp = np.sort(np.abs(phases))
    I = np.argsort(np.abs(phases))
    phases = np.sort(phases[I[0:int(np.ceil(prob*nmc))]])
    conf = np.concatenate((np.array([[phi+phases[0]]]), np.array([[phi+phases[len(phases)-1]]])), axis=1)
    if conf[:,0] < -np.pi:
        conf[:,0] = conf[:,0]+2*np.pi
    if conf[:,1] > np.pi:
        conf[:,1] = conf[:,1]-2*np.pi
    return conf

