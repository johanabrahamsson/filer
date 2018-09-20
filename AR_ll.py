
# coding: utf-8

# In[ ]:
import numpy as np


def AR_ll(y,p):
    T = len(y.T)
    ARdeg = len(p.T)
    p = p.reshape(ARdeg,1)
    c = (np.exp(p)-1)/(np.exp(p)+1)
    a = np.zeros((ARdeg,ARdeg))
    for m in np.arange(ARdeg):
        a[m,m] = c[m,:]
    for m in np.arange(1,ARdeg):
        for i in np.arange(m):
            a[i,m] = a[i,m-1]-c[m]*a[m-i-1,m-1]
    K = np.zeros((ARdeg+1,ARdeg+1))
    A = np.concatenate((np.array([[1]]), -a[:,[ARdeg-1]].T), axis=1)
    for i in np.arange(ARdeg+1):
        K[i,i:0:-1] = K[i,i:0:-1] + A[:,0:i]
        K[i,0:ARdeg-i+1] = K[i,0:ARdeg-i+1] + A[:,i:ARdeg+1]
    C = np.linalg.solve(K, np.concatenate((np.array([[1]]), np.zeros((ARdeg,1))),axis=0))
    F = np.zeros((ARdeg,ARdeg))
    F[:,[0]] = a[:,[ARdeg-1]]
    F[0:ARdeg-1,1:ARdeg+1] = np.identity(ARdeg-1)
    G = np.concatenate((np.array([[1]]), np.zeros((ARdeg-1,1))),axis=0)
    Q = G*G.T
    H = np.concatenate((np.array([[1]]), np.zeros((1,ARdeg-1))),axis=1)
    R = 0
    x_pred1 = np.zeros((ARdeg,T))
    x_filt = np.zeros((ARdeg,T))
    V_pred1 = np.zeros((ARdeg,ARdeg,T))
    V_filt = np.zeros((ARdeg,ARdeg,T))
    x_pred1[:,[0]] = np.zeros((ARdeg,1))
    V_pred1[0,0,0] = C[0]
    for i in np.arange(1,ARdeg):
        V_pred1[i,0,0] = np.matmul(C[1:ARdeg-i+1,:].T, a[i:ARdeg,ARdeg-1])
        V_pred1[0,i,0] = V_pred1[i,0,0]
    for i in np.arange(1,ARdeg):
        for j in np.arange(i, ARdeg):
            for p in np.arange(i, ARdeg):
                for q in np.arange(j, ARdeg):
                    V_pred1[i,j,0] = V_pred1[i,j,0]+a[p,ARdeg-1]*a[q,ARdeg-1]*C[np.abs(q-j-p+i)]
            V_pred1[j,i,0] = V_pred1[i,j,0]
    for t in np.arange(T-1):
        x_filt[:,[t]] = x_pred1[:,[t]] + np.matmul(V_pred1[:,:,t], 
                        H.T*np.linalg.solve(np.matmul(np.matmul(H, V_pred1[:,:,t]), H.T)+R, 
                        y[:,t]-np.matmul(H, x_pred1[:,t])))
        V_filt[:,:,t] = np.matmul((np.identity(ARdeg) - 
                                np.matmul(V_pred1[:,:,t], 
                                np.matmul(H.T, H))/(np.matmul(np.matmul(H, V_pred1[:,:,t]), H.T)+R)), 
                                V_pred1[:,:,t])
        x_pred1[:,[t+1]] = np.matmul(F, x_filt[:,[t]])
        V_pred1[:,:,t+1] = np.matmul(F, np.matmul(V_filt[:,:,t], F.T))+Q
    Ehat = 0
    for t in np.arange(1,T+1):
        Ehat = Ehat*(t-1)/t + (y[:,t-1] - 
                np.matmul(H, x_pred1[:,[t-1]]))**2/(np.matmul(np.matmul(H, V_pred1[:,:,t-1]), H.T))/t
    ll = -T*np.log(Ehat)/2-T/2
    for t in np.arange(0,T):
        ll = float(ll-np.log(np.matmul(H, np.matmul(V_pred1[:,:,t], H.T)))/2)
    mll = -ll
    return mll
    

