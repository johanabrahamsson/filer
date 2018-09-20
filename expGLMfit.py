
# coding: utf-8

# In[ ]:

import numpy as np
from expGLM_ll import *

def expGLMfit(X,y):
    beta = scipy.optimize.minimize(lambda b: expGLM_ll(X,y,np.exp(b)), np.reshape(np.zeros((np.size(X,1),1)), np.size(X,1)), options={'gtol': 1e-6})
    return np.exp(beta.x)

