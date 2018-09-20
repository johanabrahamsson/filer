
# coding: utf-8

# In[ ]:

import numpy as np
from armyule import *
from AR_ll import *
from AR_ll2 import *

def AR_MLE(y,ARdeg):
    A,E,R = armyule(y,ARdeg)
    a = np.zeros((ARdeg,ARdeg))
    a[:,[ARdeg-1]] = -A[:,1:ARdeg+1].T
    for m in np.arange(ARdeg-1,0,-1):
        for i in np.arange(m):
            a[i,m-1] = (a[i,m]+a[m,m]*a[m-i-1,m])/(1-a[m,m]**2)
    c = np.zeros((ARdeg,1))
    for m in np.arange(ARdeg):
        c[m,:] = a[m,m]
    init = np.log(1+c)-np.log(1-c)
    init[init>20] = 20
    init[init<-20] = -20
    init[c>=1] = 20
    init[c<=-1] = -20
    tmp = scipy.optimize.minimize(lambda p: AR_ll(y,p), np.reshape(init, len(init)), options={'gtol':1e-6})
    tmp = tmp.x.reshape(len(tmp.x),1)
    c = (np.exp(tmp)-1)/(np.exp(tmp)+1)
    E = AR_ll2(y, tmp.T)
    tmp = AR_ll(y, tmp.T)
    for m in np.arange(ARdeg):
        a[m,m] = c[m,:]
    for m in np.arange(1,ARdeg):
        for i in np.arange(m):
            a[i,m] = a[i,m-1]-c[m]*a[m-i-1,m-1]
    A = np.concatenate((np.array([[1]]), -a[:,[ARdeg-1]].T), axis=1)
    return A,E

