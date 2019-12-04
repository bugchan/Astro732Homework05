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
  return np.sin(x)/2

def logPi(x):
  if 0<=x<=np.pi:
    p=np.log(np.sin(x))
  else:
    p=-np.inf
  return p

#%% stats of my distribution
x=np.linspace(0,np.pi,100)
px=probx(x)

meanx=np.pi/2
sigmax=1
N=10000
#%% emcee
ndim, nwalkers = 1, 100
x0 = np.random.rand(nwalkers, ndim)*np.pi

sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                logPi,a=10)

pos, prob, state = sampler.run_mcmc(x0,100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos,N)

#%%

samples = sampler.flatchain
hist=plt.hist(samples[:,0],100,density=True)
plt.plot(x,probx(x),label='sinx/2')
#plt.text(0,0.55,'AcceptanceRate:%1.2f'%sampler.acceptance_fraction.mean())
#plt.xlim(-.5,np.pi+.5)
plt.legend()