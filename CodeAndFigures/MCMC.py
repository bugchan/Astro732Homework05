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

def linearModel(x,a0,a1):
  return a0+a1*x

def gaussModel(x,a0,a1,a2,a3):
    return a0+a1*np.exp(-.5*((x-a2)/a3)**2)

def logPi(params,y,x,yerr,model):
    chi=-((y-model(x,*params))**2)/(2*yerr**2)
    return chi.sum(axis=0)

def MCMCHastings(N,width,params,model,x,y,sigma):
    start=time.time()
    #Initialize arrays and variables
    accept=0
    aArray=np.zeros((N,len(params)))
    candArray=np.zeros((N,len(params)))

    #set initial guess as first value of arrays
    aArray[0]=params

    #Start Metropolis Hasting Algorithm
    for i in range(1,N):
      #determine candidate values from normal distribution
      cand=np.random.multivariate_normal(aArray[i-1],width)
      candArray[i]=cand
      #calculate likelihood of candidates and previous value

      logpi_prev=logPi(aArray[i-1],y,x,sigma,model)
      logpi_cand=logPi(cand,y,x,sigma,model)
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
    return aArray


#%% Linear FIT

# import data
data1=np.load('linfit_data.npz')
#show key of data
#data.files
x1=data1['x']
y1=data1['y']
sigma1=data1['sigma']

#Visual inspection of data
#plt.plot(x,y)

#using curve fit to find my initial guess
coeff1,pcov1=opt.curve_fit(linearModel,x1,y1,sigma=sigma1)

#start with pcov as width and then optimize according to acceptance rate
N=10000
width=4.9*pcov1
accept=0

aArray1=MCMCHastings(N,width,coeff1,linearModel,x1,y1,sigma1)

# Make plot of parameters
fig1=corner.corner(aArray1[500:],
                   labels=['$a_0$','$a_1$'],
                   show_titles=True,
                   levels=(0.683,0.95),
                   color='C0',
                   title="Linear Model,Metropolis-Hastings",
                   #truths=coeff,
                   )
plt.savefig('LinearFitMetropolisHastings.pdf')

#%% Gaussian Fit

data2=np.load('gaussfit_data.npz')
#show key of data
#data.files

x2=data2['x']
y2=data2['y']
sigma2=data2['sigma'][:30]

#Visual inspection of data
#plt.plot(x2,y2)


#using curve fit to find my initial guess
coeff2,pcov2=opt.curve_fit(gaussModel,x2,y2,sigma=sigma2)

#plt.plot(x2,gaussModel(x2,coeff2[0],coeff2[1],
#                       coeff2[2],coeff2[3]))

#start with pcov as width and then optimize according to acceptance rate
N=10000
width=1.65*pcov2
#width=.1*np.diag(coeff2)
accept=0

aArray2=MCMCHastings(N,width,coeff2,gaussModel,x2,y2,sigma2)
# Make plot of parameters
fig2=corner.corner(aArray2[0:],
                   labels=['$a_0$','$a_1$','$a_2$','$a_3$'],
                   show_titles=True,
                   levels=(0.683,0.95),
                   color='C0',
                   title="Gaussian Model Metropolis-Hastings",
                   #truths=coeff,
                   )
plt.savefig('GaussFitMetropolisHastings.pdf')

fig3=plt.figure()
grid=plt.GridSpec(4,4)
ax1=add


