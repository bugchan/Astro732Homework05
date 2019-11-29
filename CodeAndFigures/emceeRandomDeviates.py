#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:43:54 2019

@author: sbustamanteg
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt

#%% Definitions
def probx(x):
  px=np.zeros(len(x))
  px[(x>0)*(x<np.pi)]=np.sin(x[(x>0)*(x<np.pi)])/2
  return px

def logPi(x,xmean,xsigma):
  if x<=0:
    p=-1000000
  elif x>=np.pi:
    p=-1000000
  else:
    p=-((xmean-probx(x))**2)/(2*xsigma**2)
  return p

#def logPi(params,y,x,yerr,model):
#    chi=-((y-model(x,*params))**2)/(2*yerr**2)
#    return chi.sum(axis=0)

#%% stats of my distribution
x=np.linspace(0,np.pi,100)
px=probx(x)

meanx=np.pi/2
sigmax=1
#%% emcee
ndim, nwalkers = 1, 10
x0 = np.random.randn(nwalkers, ndim)*meanx+sigmax

sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                logPi,
                                args=[meanx,sigmax])

pos, prob, state = sampler.run_mcmc(x0, 10000)

#%%


samples = sampler.flatchain
plt.hist(samples[:, 0], 100,density=True)
plt.plot(x,probx(x))
#plt.xlabel(r"$\theta_1$")
#plt.ylabel(r"$p(\theta_1)$")
#plt.gca().set_yticks([]);