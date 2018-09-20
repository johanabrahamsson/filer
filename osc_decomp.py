
# coding: utf-8

# In[ ]:

import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from AR_fit import *
from prop_ll import *
from prop_ll2 import *
from expGLMfit import *

def osc_decomp(y,fs,MAX_COMPONENT):
    T = len(y.T)
    phi_prop = np.zeros((MAX_COMPONENT,T,MAX_COMPONENT))
    decomp_mu = np.zeros((2*MAX_COMPONENT,T,MAX_COMPONENT))
    decomp_cov = np.zeros((2*MAX_COMPONENT,2*MAX_COMPONENT,T,MAX_COMPONENT))
    AIC_osc = np.zeros((1,MAX_COMPONENT))
    osc_param = np.zeros((MAX_COMPONENT,3*MAX_COMPONENT+1))
    MAX_AR = 2*MAX_COMPONENT
    ARwithnoise_param,AIC_ARwithnoise = AR_fit(y,MAX_AR)
    for num_component in np.arange(1,MAX_COMPONENT+1):
        minAIC = np.inf
        minAIC2 = np.inf
        for ARdeg in np.arange(num_component, 2*num_component):
            rootvector = np.concatenate((np.array([[1]]), [ARwithnoise_param[ARdeg-1,0:ARdeg]]),axis=1)
            tmp = np.roots(rootvector.reshape(len(rootvector.T),))
            if ARdeg - np.count_nonzero(tmp.imag)/2 == num_component and AIC_ARwithnoise[:,ARdeg-1] < minAIC:
                z0 = tmp
                E0 = ARwithnoise_param[ARdeg-1,ARdeg]
                R0 = ARwithnoise_param[ARdeg-1,ARdeg+1]
                minAIC = AIC_ARwithnoise[:,ARdeg-1]
                optARdeg = ARdeg
            if ARdeg - np.count_nonzero(tmp.imag)/2 >= num_component and AIC_ARwithnoise[:,ARdeg-1] < minAIC2:
                z1 = tmp
                E1 = ARwithnoise_param[ARdeg-1,ARdeg]
                R1 = ARwithnoise_param[ARdeg,ARdeg+1]
                minAIC2 = AIC_ARwithnoise[:,ARdeg-1]
                optARdeg2 = ARdeg
        if minAIC == np.inf:
            if minAIC2 == np.inf:
                print('no AR model with ',num_component,' oscillators')
            z0 = z1
            E0 = E1
            R0 = R1
            optARdeg = optArdeg2
            B = np.sort(np.abs(z0))[::-1]
            z0 = z0[:,0:len(B.T)+1]
        ARdeg = np.argmin(AIC_ARwithnoise) + 1
        rootvector = np.concatenate((np.array([[1]]), [ARwithnoise_param[ARdeg-1,0:ARdeg]]),axis=1)
        tmp = np.roots(rootvector.reshape(len(rootvector.T),))
        if sum(tmp.imag>=0) >= num_component:
            z0 = tmp
            E0 = ARwithnoise_param[ARdeg-1,ARdeg]
            R0 = ARwithnoise_param[ARdeg-1,ARdeg+1]
            optARdeg = ARdeg
        VV = np.zeros((optARdeg,optARdeg), dtype=np.complex_)
        for j in np.arange(1,optARdeg+1):
            for i in np.arange(1,optARdeg+1):
                VV[i-1,j-1] = z0[j-1]**(1-i)
        QQ = np.matmul(inv(VV), 
                        np.matmul(np.concatenate((np.concatenate((np.array([[E0]]), 
                        np.zeros((1,optARdeg-1))), axis=1), np.zeros((optARdeg-1,optARdeg))), axis=0), 
                        inv(VV).conj().transpose()))
        
        B = (np.sort((np.diag(QQ.real)/(1-np.abs(z0)**2))))[::-1]
        z0 = np.sort(z0[0:len(B.T)+1])[::-1]
        init_a = np.zeros((1,num_component))
        init_f = np.zeros((1,num_component))
        kk = 1
        for k in np.arange(num_component):
            init_a[:,k] = np.abs(z0[kk-1])
            init_f[:,k] = np.abs(np.angle(z0[kk-1]))
            if z0[kk-1].imag == 0:
                kk = kk+1
            else:
                kk = kk+2
        B = np.sort(init_f)
        init_a = init_a[:,0:len(B.T)+1]
        init_f = np.sort(init_f)[:,0:len(B.T)+1]
        num_osc = np.count_nonzero(init_f)
        #nf0 = num_component-num_osc
        freq = init_f
        P = np.zeros((len(freq.T),num_component))
        """
        if nf0 > 0:
            freq[:,1:nf0] = np.pi*np.random.rand(1,nf0-1)
            for k in np.arange(nf0):
                a = init_a[:,k]
                A = (a**2-1)/a;
                b = (A-2*a+np.sqrt((A-2*a)**2-4))/2
                for j in np.arange(len(freq.T)):
                    P[j,k] = -(a/b)*np.abs(1+b*np.exp(-1j*freq[:,j]))**2/np.abs(1-2*a*np.exp(-1j*freq[:,j])+a**2*np.exp(-2*1j*freq[:,j]))**2
        """
        for k in np.arange(num_component):
            a = init_a[:,k]
            theta = init_f[:,k]
            A = (1-2*a**2*np.cos(theta)**2+a**4*np.cos(2*theta))/a/(a**2-1)/np.cos(theta)
            b = (A-2*a*np.cos(theta)+np.sign(np.cos(theta))*np.sqrt((A-2*a*np.cos(theta))**2-4))/2
            for j in np.arange(len(freq.T)):
                P[j,k] = -(a*np.cos(theta)/b)*np.abs(1+b*np.exp(-1j*freq[:,j]))**2/np.abs(1-2*a*np.cos(theta)*np.exp(-1j*freq[:,j])+a**2*np.exp(-2*1j*freq[:,j]))**2
        p = np.zeros((len(freq.T),1))
        for j in np.arange(len(freq.T)):
            p[j,:] = np.abs(np.dot(y, np.exp(-1j*freq[:,j]*(np.arange(T)).T)))**2/T
        if np.linalg.cond(np.matmul(P.conj().transpose(), P)) < 10**6:
            init_sigma = np.linalg.solve(np.matmul(P.conj().transpose(), P), np.matmul(P.conj().transpose(), p))
        else:
            init_sigma = expGLMfit(P,p)
            init_sigma = np.reshape(init_sigma, (len(init_sigma), 1))
        init_sigma[init_sigma<0] = R0
        initvec = np.reshape(np.concatenate((np.concatenate((np.arctanh(2*init_a-1), init_f), axis=1), 
                np.log(init_sigma.T/R0)), axis=1), len(np.concatenate((np.concatenate((np.arctanh(2*init_a-1), init_f), axis=1), 
                np.log(init_sigma.T/R0)), axis=1).conjugate().transpose()))
        param = scipy.optimize.minimize(lambda param: prop_ll(y, param), 
               np.array(list(initvec)), options={'gtol':1e-6, 'eps': 1.490116119384766e-08})
        mll = param.fun
        param = np.array([param.x])
        mll = prop_ll(y,param)
        tmp = prop_ll2(y,param)
        AIC_osc[:,num_component-1] = 2*mll+2*(3*num_component+1)
        concat1 = np.concatenate(((np.tanh(param[:,0:num_component])+1)/2, param[:,num_component:2*num_component]*fs/2/np.pi), axis=1)
        concat2 = np.concatenate((np.exp(param[:,2*num_component:3*num_component])*tmp, tmp),axis=1)
        osc_param[num_component-1,0:3*num_component+1] = np.concatenate((concat1, concat2),axis=1)
        a = osc_param[num_component-1,0:num_component]
        theta = osc_param[num_component-1,num_component:2*num_component]*2*np.pi/fs
        sigma = osc_param[num_component-1,2*num_component:3*num_component]
        sigman = osc_param[num_component-1,3*num_component-1]
        m = 2*num_component
        x_pred1 = np.zeros((m,T))
        x_filt = np.zeros((m,T))
        x_smooth = np.zeros((m,T))
        V_pred1 = np.zeros((m,m,T))
        V_filt = np.zeros((m,m,T))
        V_smooth = np.zeros((m,m,T))
        F = np.zeros((m,m))
        Q = np.zeros((m,m))
        H = np.zeros((1,m))
        for k in np.arange(num_component):
            F[2*k:2*k+2,2*k:2*k+2] = a[k]*np.concatenate((np.concatenate((np.array([[np.cos(theta[k])]]), np.array([[-np.sin(theta[k])]])), axis=1), np.concatenate((np.array([[np.sin(theta[k])]]), np.array([[np.cos(theta[k])]])), axis=1)), axis=0)
            Q[2*k,2*k] = 1-a[k]**2
            Q[2*k+1,2*k+1] = 1-a[k]**2
        H[:,0:m:2] = np.sqrt(sigma)/np.sqrt(1-a**2)
        R = sigman
        x_pred1[:,[0]] = np.zeros((m,1))
        for k in np.arange(num_component):
            V_pred1[2*k:2*k+2,2*k:2*k+2,0] = np.identity(2)
        for t in np.arange(T-1):
            Kg = np.matmul(V_pred1[:,:,t], np.matmul(H.conj().transpose(), inv(np.matmul(np.matmul(H, V_pred1[:,:,t]), H.conj().transpose())+R)))
            x_filt[:,[t]] = x_pred1[:,[t]] + np.matmul(Kg, (y[:,t]-np.matmul(H, x_pred1[:,[t]])))
            V_filt[:,:,t] = np.matmul((np.identity(m)-np.matmul(Kg, H)), V_pred1[:,:,t])
            x_pred1[:,[t+1]] = np.matmul(F, x_filt[:,[t]])
            V_pred1[:,:,t+1] = np.matmul(F, np.matmul(V_filt[:,:,t], F.conj().transpose()))+Q
        Kg = np.matmul(V_pred1[:,:,T-1], np.matmul(H.conj().transpose(), inv(np.matmul(np.matmul(H, V_pred1[:,:,T-1]), H.conj().transpose())+R)))
        x_filt[:,[T-1]] = x_pred1[:,[T-1]] + np.matmul(Kg, (y[:,T-1]-np.matmul(H, x_pred1[:,[T-1]])))
        V_filt[:,:,T-1] = np.matmul((np.identity(m)-np.matmul(Kg, H)), V_pred1[:,:,T-1])
        x_smooth[:,[T-1]] = x_filt[:,[T-1]]
        V_smooth[:,:,T-1] = V_filt[:,:,T-1]
        for t in np.arange(T-2,-1,-1):
            x_smooth[:,[t]] = x_filt[:,[t]] + np.matmul(V_filt[:,:,t], np.matmul(F.conj().transpose(), np.linalg.solve(V_pred1[:,:,t+1], x_smooth[:,[t+1]]-x_pred1[:,[t+1]])))
            CC = np.matmul(V_filt[:,:,t], np.matmul(F.conj().transpose(), inv(V_pred1[:,:,t+1])))
            V_smooth[:,:,t] = V_filt[:,:,t] + np.matmul(CC, np.matmul(V_smooth[:,:,t+1]-V_pred1[:,:,t+1], CC.conj().transpose()))
        for k in np.arange(num_component):
            phi_prop[k,:,num_component-1] = np.arctan2(x_smooth[[2*k+1],:], x_smooth[[2*k],:])
        decomp_mu[0:m,:,num_component-1] = x_smooth
        decomp_cov[0:m,0:m,:,num_component-1] = V_smooth
    return phi_prop, decomp_mu, decomp_cov, AIC_osc, osc_param

