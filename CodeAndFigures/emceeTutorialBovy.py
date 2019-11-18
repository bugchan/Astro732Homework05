# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:08:15 2019

@author: sandr
"""
import numpy as np
import corner
import emcee
#import scipy.optimize as opt

#%%
# Basic linear fit used as initialization
def linModel(x,a1,a0):
  model=a1*x+a0
  return model

def linfit(y,x,yerr):
    # Generate Y and X arrays
    A= np.vander(x,N=2)
    Cinv= np.diag(1./yerr**2.)
    Y= np.dot(A.T,np.dot(Cinv,y))
    A= np.dot(A.T,np.dot(Cinv,A))
    return (np.linalg.solve(A,Y),np.linalg.inv(A))

def lnprob(x,data):
    # Parameters: m,b,Pb,Yb,lnVb
    xdata=data[:,1]
    ydata=data[:,2]
    sigmay=data[:,3]

    if x[2] < 0: return -1000000000000.
    if x[2] > 1: return -1000000000000.
    if np.exp(x[4]/2.) > 10000.:  return -1000000000000.
    if np.exp(x[4]/2.) < 1.:  return -1000000000000.
    return np.sum(\
        np.log((1.-x[2])/sigmay*\
               np.exp(-0.5*\
                      (ydata-x[0]*xdata-x[1])**2./sigmay**2.)
                 + x[2]/np.sqrt(sigmay**2.+np.exp(x[4]))\
                         *np.exp(-0.5*(ydata-x[3])**2./(sigmay**2.+np.exp(x[4])))))

#%%
data= np.array([[1,201,592,61,9,-0.84],
                   [2,244,401,25,4,0.31],
                   [3,47,583,38,11,0.64],
                   [4,287,402,15,7,-0.27],
                   [5,203,495,21,5,-0.33],
                   [6,58,173,15,9,0.67],
                   [7,210,479,27,4,-0.02],
                   [8,202,504,14,4,-0.05],
                   [9,198,510,30,11,-0.84],
                   [10,158,416,16,7,-0.69],
                   [11,165,393,14,5,0.30],
                   [12,201,442,25,5,-0.46],
                   [13,157,317,52,5,-0.03],
                   [14,131,311,16,6,0.50],
                   [15,166,400,34,6,0.73],
                   [16,160,337,31,5,-0.52],
                   [17,186,423,42,9,0.90],
                   [18,125,334,26,8,0.40],
                   [19,218,533,16,6,-0.78],
                   [20,146,344,22,5,-0.56]])

#%%
nwalkers= 20
# Start from best-fit line above

X, Xcov= linfit(data[4:,2],data[4:,1],data[4:,3])
p0= [[X[0]+np.random.normal()*np.sqrt(Xcov[0,0]),
      X[1]+np.random.normal()*np.sqrt(Xcov[1,1]),
      0.3+(np.random.uniform()/5.-0.1),
      np.mean(data[:,2])+np.random.normal()\
      *np.std(data[:,2])/4.,
     (2.*np.log(np.std(data[:,2]))\
      *(1.+0.1*np.random.normal()))]
    for w in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers,len(p0[0]),
                                lnprob,args=[data])
# Burn=in
pos, prob, state = sampler.run_mcmc(p0,1000)
sampler.reset()
pos, prob, state= sampler.run_mcmc(pos,10000)

#%%
fig = corner.corner(sampler.flatchain[::10],
                    labels=["$m$", "$b$", "$P_b$",
                            "$Y_b$","$\ln\,V_b$"],
                    range=[(1.8,2.8),(-50.,150.),(0.,1.),
                           (0.,800.),(0.,14.)],
                    show_titles=True)