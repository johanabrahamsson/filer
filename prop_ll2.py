
# coding: utf-8

# In[ ]:

import numpy as np

def prop_ll2(y, param):
    param = param.reshape(1,len(param.T))
    T = len(y.T)
    num_osc = int(len(param.T)/3)
    F = np.zeros((2*num_osc,2*num_osc))
    Q = np.zeros((2*num_osc,2*num_osc))
    for i in np.arange(num_osc):
        F[2*i:2*i+2,2*i:2*i+2] = (np.tanh(param[:,i])+1)/2*np.concatenate((np.concatenate(([np.cos(param[:,num_osc+i])], 
                                    [-np.sin(param[:,num_osc+i])]), axis=1), np.concatenate(([np.sin(param[:,num_osc+i])], 
                                    [np.cos(param[:,num_osc+i])]), axis=1)), axis=0)
        Q[2*i:2*i+2,2*i:2*i+2] = np.exp(param[:,2*num_osc+i])*np.identity(2)
    H = np.zeros((1,2*num_osc))
    H[:,0:2*num_osc:2] = 1
    R = 1

    x_pred1 = np.zeros((2*num_osc,T))
    x_filt = np.zeros((2*num_osc,T))
    V_pred1 = np.zeros((2*num_osc,2*num_osc,T))
    V_filt = np.zeros((2*num_osc,2*num_osc,T))
    x_pred1[:,[0]] = np.zeros((2*num_osc,1))
    for i in np.arange(num_osc):
        V_pred1[2*i:2*i+2,2*i:2*i+2,0] = Q[2*i:2*i+2,2*i:2*i+2]/(1-F[2*i:2*i+2,2*i:2*i+2]**2-F[2*i,2*i+1]**2)
    for t in np.arange(T-1):
        x_filt[:,[t]] = x_pred1[:,[t]] + np.matmul(V_pred1[:,:,t], 
                        H.conj().transpose()*np.linalg.solve(np.matmul(np.matmul(H, V_pred1[:,:,t]), H.conj().transpose())+R, 
                        y[:,t]-np.matmul(H, x_pred1[:,t])))
        V_filt[:,:,t] = V_pred1[:,:,t] - np.matmul(np.matmul(V_pred1[:,:,t], 
                        H.conj().transpose()*np.linalg.solve(np.matmul(np.matmul(H, 
                        V_pred1[:,:,t]), H.conj().transpose())+R, H)), V_pred1[:,:,t])
        x_pred1[:,[t+1]] = np.matmul(F, x_filt[:,[t]])
        V_pred1[:,:,t+1] = np.matmul(F, np.matmul(V_filt[:,:,t], F.conj().transpose()))+Q
    Rhat = 0
    for t in np.arange(1,T+1):
        Rhat = Rhat*(t-1)/t + (y[:,t-1] - 
                np.matmul(H, x_pred1[:,[t-1]]))**2/(np.matmul(np.matmul(H, V_pred1[:,:,t-1]), H.conj().transpose())+R)/t
    ll = -T*np.log(Rhat)/2-T/2
    for t in np.arange(0,T):
        ll = ll-np.log(np.matmul(H, np.matmul(V_pred1[:,:,t], H.T))+R)/2
    mll = -ll
    return Rhat

