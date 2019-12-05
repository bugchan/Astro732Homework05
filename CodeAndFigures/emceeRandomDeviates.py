#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:43:54 2019

@author: sbustamanteg
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
import SetupPlots as SP

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
N=100000
#%% emcee
ndim, nwalkers = 1, 100
x0 = np.random.rand(nwalkers, ndim)*np.pi

sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                logPi,a=10)

pos, prob, state = sampler.run_mcmc(x0,100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos,N)

#extract the samples of the distribution
samples = sampler.flatchain

#%% Plot 
width,height=SP.setupPlot(singleColumn=False)

fig,axs = plt.subplots(1,1,figsize=(width,height))
hist=axs.hist(samples[:,0],100,density=True)
axs.plot(x,probx(x),label='sinx/2')
axs.set_xlabel('x')
axs.set_ylabel('P(x)')
axs.legend()
axs.grid()

fig.tight_layout()
fig.savefig('emceeRandomDeviatesPlot.pdf')
