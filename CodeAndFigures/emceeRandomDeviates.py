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
def logprior(x):
    xmin=0
    xmax=np.pi
    #prob of values outside of range is 0. 
    lp=np.log(1) if xmin<x<xmax else -np.inf
    return lp

def loglikelihood(x):
    return -np.inf

def logposterior(x):
    return -np.inf
    
def probx(x):
  return np.sin(x)/2

def logPi(x,xmean,xsigma):
  if x<=0 or x>=np.pi:
    p=-np.inf
#  elif x>=np.pi:
#    p=-1000000
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
N=10000
#%% emcee
ndim, nwalkers = 1, 10
x0 = np.random.randn(nwalkers, ndim)*meanx+sigmax

sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                logPi,
                                args=[meanx,sigmax])

pos, prob, state = sampler.run_mcmc(x0,N)

#%%

samples = sampler.flatchain
plt.hist(samples[:, 0], 100,density=True)
plt.plot(x,probx(x),label='sinx/2')
plt.text(0,0.6,'AcceptanceRate:%1.2f'%sampler.acceptance_fraction.mean())
plt.legend()

