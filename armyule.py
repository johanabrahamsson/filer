
# coding: utf-8

# In[ ]:

from numpy import linalg as LA
import numpy as np
from ARwithnoise_ll import *

def armyule(y,ARdeg):
    T = len(y.T)
    acov = np.zeros((ARdeg+1, 1))
    for k in np.arange(ARdeg+1):
        acov[k] = float(y[:,0:T-k].dot(y.T[k:T,:])/T)
    C = np.zeros((ARdeg,ARdeg))
    for k in np.arange(ARdeg):
        C[k,0:k+1] = acov[k::-1].T
        C[k,k+1:ARdeg] = acov[1:ARdeg-k].T
    c = acov[1:ARdeg+1]
    eigs = LA.eigvals(np.concatenate((np.concatenate((C, np.flip(c,0)), axis=1),
                                      np.concatenate((np.fliplr(c.T), [acov[0]]), axis=1)), axis=0))
    R = scipy.optimize.fminbound(lambda R: ARwithnoise_ll(y, np.concatenate((np.linalg.solve(-(C-R*np.identity(ARdeg)), c), 
                    np.log(acov[0]-R+(np.linalg.solve(-(C-R*np.identity(ARdeg)), c)).T.dot(acov[1:ARdeg+1,:]))-
                    np.log(R)), axis=0)), 0, np.amin(eigs), xtol=1e-04)
    A = np.linalg.solve(C-R*np.identity(ARdeg), c)
    E = acov[0]-R-np.matmul(A.T, acov[1:ARdeg+1])
    A = np.concatenate((np.array([[1]]), -A.T), axis=1)
    return A,E,R

