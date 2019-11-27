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
import emcee
import SetupPlots as SP

#%% Definitions

def linearModel(x,a0,a1):
  return a0+a1*x

def gaussModel(x,a0,a1,a2,a3):
    return a0+a1*np.exp(-.5*((x-a2)/a3)**2)

def logPi(params,y,x,yerr,model):
    chi=-((y-model(x,*params))**2)/(2*yerr**2)
    return chi.sum(axis=0)

def MCMCHastings(N,width,params,model,x,y,sigma,debug=False):
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
      ratio=np.exp(logpi_cand-logpi_prev)
      if debug:
        print('logpi prev',logpi_prev)
        print('logpicand',logpi_cand)
        print('ratio',ratio)
      acceptanceprob=min(1,ratio)
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


#%% Linear Model Import Data

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

#%% Linear Model Metropolis Hastings
#start with pcov as width and then optimize according to acceptance rate = 23.4
N=10000
nwalkers1=20
#width=4.9*np.sqrt(np.diag(pcov1))
width=4.9*pcov1
accept=0

aArray1=MCMCHastings(nwalkers1*N,width,
                     coeff1,linearModel,x1,y1,sigma1)

#%% Linear Model Plot Metropolis Hastings
width,height=SP.setupPlot(singleColumn=False)
fig1=corner.corner(aArray1[::10],
                   labels=['$a_0$','$a_1$'],
                   show_titles=True,
                   levels=(0.683,0.95),
                   color='C0',
                   title="Linear Model Metropolis-Hastings",
                   #truths=coeff,
                   )
fig1.set_size_inches((width,width))
fig1.savefig('LinearModelMetropolisHastings.pdf')

#%% Linear Model Emcee Algorithm

p01= [[coeff1[0]+np.random.normal()*np.sqrt(pcov1[0,0]),
      coeff1[1]+np.random.normal()*np.sqrt(pcov1[1,1])]
      for w in range(nwalkers1)]
#p0= [np.random.multivariate_normal(coeff1,pcov1)
#      for w in range(nwalkers)]

start=time.time()

sampler1=emcee.EnsembleSampler(nwalkers1,len(p01[0]),
                      logPi,args=[y1,x1,sigma1,linearModel])
pos1, prob1, state1 = sampler1.run_mcmc(p01,1000)
sampler1.reset()
pos1, prob1, state1= sampler1.run_mcmc(pos1,N)

end = time.time()
print('Time to run: %0.2f'%(end - start))


#%% Linear Model Plot Emcee
fig3 = corner.corner(sampler1.flatchain[::10],
                    labels=["$a_0$", "$a_1$"],
                    show_titles=True,
                    color='C0',
                    levels=(0.683,0.95),
                    title="Linear Model Emcee",
                    )
fig3.set_size_inches((width,width))
fig3.savefig('LinearModelEmcee.pdf')

#%% Gaussian Model Import Data

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

#%% Gaussian Model Metropolis Hasting
#start with pcov as width and then optimize according to acceptance rate
N=10000
nwalkers2=20
width=1.65*pcov2
accept=0

aArray2=MCMCHastings(N*nwalkers2,width,
                     coeff2,gaussModel,x2,y2,sigma2)

#%% Gaussian Model Plot
width,height=SP.setupPlot(singleColumn=True)
fig2=corner.corner(aArray2[0:],
                   labels=['$a_0$','$a_1$','$a_2$','$a_3$'],
                   show_titles=True,
                   levels=(0.683,0.95),
                   color='C0',
                   title="Gaussian Model Metropolis-Hastings",
                   )
fig2.set_size_inches((width,width))
fig2.savefig('GaussianModelMetropolisHastings.pdf')

#%% Gaussian Model Emcee Algorithm


p02=[[coeff2[0]+np.random.normal()*np.sqrt(pcov2[0,0]),
      coeff2[1]+np.random.normal()*np.sqrt(pcov2[1,1]),
      coeff2[2]+np.random.normal()*np.sqrt(pcov2[2,2]),
      coeff2[3]+np.random.normal()*np.sqrt(pcov2[3,3])]
      for w in range(nwalkers2)]
#p0= [np.random.multivariate_normal(coeff1,pcov1)
#      for w in range(nwalkers)]

start=time.time()
sampler2=emcee.EnsembleSampler(nwalkers2,len(p02[0]),
                      logPi,args=[y2,x2,sigma2,gaussModel])

pos2, prob2, state2 = sampler2.run_mcmc(p02,1000)
sampler2.reset()
pos2, prob2, state2= sampler2.run_mcmc(pos2,N)

end = time.time()
print('Time to run: %0.2f'%(end - start))

#%% Gaussian Model Plot Emcee Algorithm
fig4 = corner.corner(sampler2.flatchain[::10],
                    labels=['$a_0$','$a_1$','$a_2$','$a_3$'],
                    show_titles=True,
                    color='C0',
                    levels=(0.683,0.95),
                    title="Linear Model Emcee",
                    )
fig4.set_size_inches((width,width))
fig4.savefig('gaussianModelEmcee.pdf')





