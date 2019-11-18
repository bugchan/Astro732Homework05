# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:36:39 2019

@author: sandr
"""

import numpy as np
import corner
import emcee
import lmfit
from scipy.linalg import pinv
import matplotlib.pyplot as plt
#%% Definitions
def linearmodel(coeff,x):
  f=a[0]+a[1]*x
  return f

def pi(a,y,x,sigma):
  chi=np.exp(-((y-linearmodel(x,a))**2)/(2*sigma**2))
  p=chi.prod(axis=0)
  print(p)
  return p

def logProb(x,mu,cov):
  #important first parameter to be the walker
  #mu is N-dimensional vector position of the mean of the density
  #
  diff=x-mu
  p=-0.5 * np.dot(diff, np.linalg.solve(cov, diff))
  return p

def log_likelihood(coeff, x, y, sigma):
  sigma2 = sigma**2
  llres=-0.5*np.sum((y - linearmodel(coeff,x))**2 /sigma2 + np.log(sigma2))
  return llres
  


#%% import data

data=np.load('linfit_data.npz')
x=data['x']
y=data['y']
sigma=data['sigma']

#%% Find initial values of parameters
npts=len(x)
xSVD=x.reshape(npts,1)
#using weighted
ySVD=y.reshape(npts,1)
A=np.concatenate((xSVD**0,xSVD**1),axis=1)
Apinv=pinv(A)
#determine coefficients
Coeff=Apinv.dot(ySVD)

#Verifying it makes sense
yfitSVD=A.dot(Coeff)
plt.plot(ySVD)
plt.plot(yfitSVD)

#%% 

a=Coeff.copy()#initial coefficientes
ndim=2
nwalkers=100
sampler=emcee.EnsembleSampler(nwalkers,ndim,linearmodel,args=[a])
