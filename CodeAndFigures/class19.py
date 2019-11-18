# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:10:25 2019

@author: Sandra Bustamante


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import corner
import time

def model(x,a0,a1):
  f=a0+a1*x
  return f

def logPi(a0,a1,y,x,sigma):
  chi=-((y-model(x,a0,a1))**2)/(2*sigma**2)
  p=chi.sum(axis=0)
  #print('log pi:', p)
  return p


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
#cov=np.sqrt(np.diag(pcov))
#cov=np.diag(pcov)


#%% MCMC

start=time.time()
accept=0
#start with cov as width and then optimize according to acceptance rate
#coeff=np.array([-30,0])
N=10000
aArray=np.zeros((N,len(coeff)))
candArray=np.zeros((N,len(coeff)))
width=4.9*pcov

#set initial guess as first value of arrays
aArray[0]=coeff

#Start Metropolis Hasting Algorithm
for i in range(1,N):
  #determine candidate values from normal distribution
  cand=np.random.multivariate_normal(aArray[i-1],width)
  candArray[i]=cand
  #calculate likelihood of candidates and previous value
  logpi_prev=logPi(aArray[i-1,0],aArray[i-1,1],y,x,sigma)
  logpi_cand=logPi(cand[0],cand[1],y,x,sigma)
  #calculate the acceptance probability
  acceptanceprob=min(1,np.exp(logpi_cand-logpi_prev))
  #print(acceptanceprob)
  #generate a random number from uniform distribution
  u=np.random.uniform()
  if u<acceptanceprob:
    #accept candidate step
    aArray[i]=cand
    accept+=1.
  else:
    #reject candidate and new value becomes the prev value
    aArray[i]=aArray[i-1]

#calculates the acceptance rate and print it
acceptRate=(accept/N)*100.
print('acceptance rate: %1.2f percentage'%acceptRate)

end = time.time()
print('Time to run: %0.2f'%(end - start))

#make plot of parameters

fig1=corner.corner(aArray[500:],
                   labels=['$a_0$','$a_1$'],
                   show_titles=True,
                   levels=(0.683,0.95),
                   color='C0')
plt.savefig('LinearFitMetropolisHastings.pdf')


