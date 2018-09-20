
# coding: utf-8

# In[ ]:

import numpy as np
import scipy.optimize
from scipy.linalg import toeplitz

def ARwithnoise_ll2(y,param):
    T = len(y.T)
    ARdeg = len(param)-1
    A = np.concatenate((np.array([[1]]), param[:,0:ARdeg].T), axis=1)
    E = np.exp(param[ARdeg])
    R = 1
    x_pred1 = np.zeros((ARdeg,T))
    x_filt = np.zeros((ARdeg,T))
    V_pred1 = np.zeros((ARdeg,ARdeg,T))
    V_filt = np.zeros((ARdeg,ARdeg,T))
    F = np.concatenate((-A[:,1:ARdeg+1], np.concatenate((np.identity(ARdeg-1), np.zeros((ARdeg-1,1))), axis=1)), axis=0)
    Q = np.concatenate((np.concatenate(([E],np.zeros((1,ARdeg-1))),axis=1), np.zeros((ARdeg-1,ARdeg))),axis=0)
    H = np.concatenate((np.array([[1]]), np.zeros((1,ARdeg-1))), axis=1)
    K = np.zeros((ARdeg+1,ARdeg+1))
    for i in np.arange(ARdeg+1):
        K[i,i:0:-1] = K[i,i:0:-1] + A[:,0:i]
        K[i,0:ARdeg-i+1] = K[i,0:ARdeg-i+1] + A[:,i:ARdeg+1]
    c = np.linalg.solve(K,np.concatenate(([E],np.zeros((ARdeg,1))),axis=0))
    V_pred1[:,:,0] = toeplitz(c[0:ARdeg])
    for t in np.arange(0,T-1):
        x_filt[:,[t]] = x_pred1[:,[t]] + np.matmul(V_pred1[:,:,t], 
                        H.T*np.linalg.solve(np.matmul(np.matmul(H, V_pred1[:,:,t]), H.T)+R, 
                        y[:,t]-np.matmul(H, x_pred1[:,t])))
        V_filt[:,:,t] = V_pred1[:,:,t] - np.matmul(np.matmul(V_pred1[:,:,t], 
                        H.T*np.linalg.solve(np.matmul(np.matmul(H, 
                        V_pred1[:,:,t]), H.T)+R, H)), V_pred1[:,:,t])
        x_pred1[:,[t+1]] = np.matmul(F, x_filt[:,[t]])
        V_pred1[:,:,t+1] = np.matmul(F, np.matmul(V_filt[:,:,t], F.T))+Q
    Rhat = 0
    for t in np.arange(1,T+1):
        Rhat = Rhat*(t-1)/t + (y[:,t-1] - 
                np.matmul(H, x_pred1[:,[t-1]]))**2/(np.matmul(np.matmul(H, V_pred1[:,:,t-1]), H.T)+R)/t
    ll = -T*np.log(Rhat)/2-T/2
    for t in np.arange(0,T):
        ll = ll-np.log(np.matmul(H, np.matmul(V_pred1[:,:,t], H.T))+R)/2
    mll = -ll
    return Rhat
    

