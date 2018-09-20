
# coding: utf-8

# In[ ]:

import numpy as np

def expGLM_ll(X, y, beta):
    mll = 0
    g = np.zeros((np.size(X,1),1))
    for i in np.arange(len(y)):
        mll = float(mll) + float(np.log(np.dot(X[[i],:], beta))) + float(y[i,:]/np.dot(X[[i],:], beta))
        g = g+X[[i],:].conj().transpose()/np.dot(X[[i],:], beta) - y[i,:]*X[[i],:].conj().transpose()/np.dot(X[[i],:], beta)**2
    return mll

