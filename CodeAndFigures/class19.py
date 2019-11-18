# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:10:25 2019

@author: Sandra Bustamante


"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy
#from scipy import interpolate
#from scipy.linalg import pinv
import scipy.optimize as opt

def model(x,a0,a1):
  f=a0+a1*x
  return f

def candidate(a,sigma):
  c=np.random.normal(a,sigma)
  return c

def pi(a0,a1,y,x,sigma):
  chi=np.exp(-((y-model(x,a0,a1))**2)/(2*sigma**2))
  p=chi.prod(axis=0)
  #print('pi:', p)
  return p

def logPi(a0,a1,y,x,sigma):
  chi=-((y-model(x,a0,a1))**2)/(2*sigma**2)
  p=chi.sum(axis=0)
  #print('log pi:', p)
  return p

def alpha(pia_prev,pia_cand):
  alpha=pia_cand/pia_prev
  if alpha>1:
    alpha=1
  return alpha
  

#%% import data
data=np.load('linfit_data.npz')
x=data['x']
y=data['y']
sigma=data['sigma']

##%% doing fit to determine a, since a is linear it can be done using svd
#npts=len(x)
#xSVD=x.reshape(npts,1)
##using weighted
#ySVD=y.reshape(npts,1)
#A=np.concatenate((xSVD**0,xSVD**1),axis=1)
#Apinv=pinv(A)
##determine coefficients
#coeff=Apinv.dot(ySVD)
##calculate
#
##Verifying it makes sense
#yfitSVD=A.dot(P)
#plt.plot(ySVD)
#plt.plot(yfitSVD)
#%% Initial guess of parameters

#using curve fit to find my initial guess

coeff,pcov=opt.curve_fit(model,x,y,sigma=sigma)
#calculate covariance
cov=np.sqrt(np.diag(pcov))


#%% MCMC
accept=0
#start with cov as width and then optimize according to acceptance rate
coeff=np.array([-50,1])
width=cov*.25
N=1000
aArray=np.zeros((N,len(coeff)))
candArray=np.zeros((N,len(coeff)))
aArray[0]=coeff
accept=np.zeros(N)
#determine candidate position
for i in range(1,N):
  cand=candidate(coeff,width)
  candArray[i]=cand
  pia_prev=logPi(aArray[i-1,0],aArray[i-1,1],y,x,sigma)
  pia_cand=logPi(cand[0],cand[1],y,x,sigma)
  acceptanceprob=alpha(np.exp(pia_prev),np.exp(pia_cand))
  u=np.random.uniform()
  if u<acceptanceprob:
    #accept candidate step
    aArray[i]=cand
    accept[i]=1
    
acceptRate=(accept.sum()/N)*100
print('acceptance rate: %1.2f percentage'%acceptRate)
index=np.argwhere(accept==1)

plt.scatter(aArray[index,0],aArray[index,1])
plt.scatter(candArray[:,0],candArray[:,1],r'o')

