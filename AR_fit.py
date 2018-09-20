
# coding: utf-8

# In[ ]:

import numpy as np
from AR_MLE import *
from ARwithnoise_ll import *
from ARwithnoise_ll2 import *

def AR_fit(y,MAX_AR):
    AIC_ARraw = np.zeros((1,MAX_AR))
    ARraw_param = np.zeros((MAX_AR,MAX_AR+1))
    AIC_ARwithnoise = np.zeros((1,MAX_AR))
    ARwithnoise_param = np.zeros((MAX_AR,MAX_AR+2))
    for ARdeg in np.arange(1,MAX_AR+1):
        A,E = AR_MLE(y,ARdeg)
        a = np.zeros((ARdeg,ARdeg))
        a[:,[ARdeg-1]] = -A[:,1:ARdeg+1].T
        for m in np.arange(ARdeg-1,0,-1):
            for i in np.arange(m):
                a[i,m-1] = (a[i,m]+a[m,m]*a[m-i-1,m])/(1-a[m,m]**2)
        c = np.zeros((ARdeg,1))
        for m in np.arange(ARdeg):
            c[m,:] = a[m,m]
        AIC_ARraw[:,ARdeg-1] = 2*AR_ll(y,(np.log(1+c)-np.log(1-c)+2*(ARdeg+1)).conjugate().transpose())
        ARraw_param[ARdeg-1,0:ARdeg+1] = np.concatenate((A[:,1:ARdeg+1], E), axis=1)
        z = np.roots(A.reshape(len(A.T),))
        k = 0
        while np.amax(np.abs(z)) >= 1:
            A,E = AR_MLE(y,ARdeg-k)
            z = np.roots(A.reshape(len(A.T),))
            A = np.concatenate(A, np.zeros((1,k)))
            k = k+1
        A,E,R = armyule(y,ARdeg)
        param = np.concatenate((A[:,1:ARdeg+1], np.log(E)-np.log(R)), axis=1)
        mll = ARwithnoise_ll(y,param.T)
        R = ARwithnoise_ll2(y,param.T)
        AIC_ARwithnoise[:,ARdeg-1] = 2*mll+2*(ARdeg+2)
        ARwithnoise_param[ARdeg-1,0:ARdeg+2] = np.concatenate((np.concatenate((param[:,0:ARdeg], 
                                                np.exp(param[:,ARdeg])*R), axis=1), R), axis=1)
    return ARwithnoise_param, AIC_ARwithnoise

