#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:26:10 2019

@author: Sandra Bustamante

Focussing the LMT

weight: (1/sigma2)
Putting all this together, use the lmfit Levenberg-Marquardt
fitting package to fit each image to a 2-d gaussian.

"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import lmfit
import SetupPlots as SP
from scipy import linalg
#import emcee
import corner

#%% Definitions

def quadModel(z,a0,a1,a2):
  return a0+a1*z+a2*z**2

def gauss2d(x,y,x0,y0,sigmax,sigmay,theta,A):
  a=(np.cos(theta)**2)/(2*sigmax**2)\
  + (np.sin(theta)**2)/(2*sigmay**2)
  b=-(np.sin(2*theta))/(2*sigmax**2)\
  +(np.sin(2*theta))/(2*sigmay**2)
  c=(np.sin(theta)**2)/(2*sigmax**2)\
  + (np.cos(theta)**2)/(2*sigmay**2)
  return A*np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2)

#def gauss2d(x,y,x0,y0,sigmax,sigmay,A):
#  return A*np.exp((-(x-x0)**2)/(2*sigmax**2)-((y-y0)**2)/(2*sigmay**2))

def logPi(params,x,y,signal,weight):
    chi=-((signal-gauss2d(x,y,*params))**2)*weight*0.5
    return chi.sum(axis=0)

#%% Import Data

#hdu=fits.open('aztec_2015-02-07_035114_00_0001_maps_signal_unfilt.fits')

s4=fits.getdata('aztec_2015-02-07_035114_00_0001_maps_signal_unfilt.fits')
s4=s4[:,3:315]
w4=fits.getdata('aztec_2015-02-07_035114_00_0001_maps_weight_unfilt.fits')
w4=w4[:,3:315]
s5=fits.getdata('aztec_2015-02-07_035115_00_0001_maps_signal_unfilt.fits')
w5=fits.getdata('aztec_2015-02-07_035115_00_0001_maps_weight_unfilt.fits')
s6=fits.getdata('aztec_2015-02-07_035116_00_0001_maps_signal_unfilt.fits')
w6=fits.getdata('aztec_2015-02-07_035116_00_0001_maps_weight_unfilt.fits')
s7=fits.getdata('aztec_2015-02-07_035117_00_0001_maps_signal_unfilt.fits')
w7=fits.getdata('aztec_2015-02-07_035117_00_0001_maps_weight_unfilt.fits')
s8=fits.getdata('aztec_2015-02-07_035118_00_0001_maps_signal_unfilt.fits')
w8=fits.getdata('aztec_2015-02-07_035118_00_0001_maps_weight_unfilt.fits')

#weights are 1/sigma^2
signals=[s4,s5,s6,s7,s8]
weights=[w4,w5,w6,w7,w8]
paramslabels=['x0','y0','sigmax','sigmay','theta','A']

#initialize x and y coordinates
ylen,xlen=s4.shape
x=np.arange(xlen)
y=np.arange(ylen)
x,y=np.meshgrid(x,y)

#%% Gaussian Fit to pointing images
#Initialize the gaussian model
start=time.time()

gaussModel=lmfit.Model(gauss2d,
                       independent_vars=('x','y'),
                       param_names=paramslabels,
                       nan_policy='omit')

#initializes arrays
ndim=len(paramslabels)
nmaps=len(signals)
valuesArray=np.zeros((nmaps,ndim))
fits=np.zeros(np.shape(signals))
covarArray=np.zeros((nmaps,ndim,ndim))

for i in np.arange(nmaps):
  result=gaussModel.fit(data=signals[i],
                        weights=weights[i],
                        x=x,y=y,
                        x0=156,y0=177,
                        sigmax=1,sigmay=1,
                        theta=0,A=1)
  covarArray[i]=result.covar
  #get the values out of the dictionary
  valuesArray[i]=np.array([*result.best_values.values()])
  #creates an array of fits
  fits[i]=gauss2d(x,y,*valuesArray[i])

residuals=signals-fits

end=time.time()
print('Gauss Fit took ',(end-start))

#%% Quadratic Fit to Amplitudes

amps=valuesArray[:,-1]
z=np.array([-3.0,-2.0,-1.0,0.0,1.0])

#%% quadfit using SVD
start=time.time()
#Reminder of what we have Az*coeff=Amp
#create A matrix
Az=np.vander(z,N=3,increasing=True)
#calculate pseudo inverse
Apinv=linalg.pinv(Az)
#find coeff and cov matrix
quadCoeff=Apinv.dot(amps)
quadcov=np.sqrt(1/np.diag(np.dot(Az.T,Az)))
#generate fit data using coeff
zfit=np.linspace(-3,1,1000)
quadfitSVD=quadModel(zfit,*quadCoeff)
#determine max of fit
index=np.argmax(quadfitSVD)

end=time.time()
print('Quad fit w/ SVD took ',(end-start))

#%% quadFit using lmfit
#start=time.time()
#quadM=lmfit.Model(quadModel,
#                  independent_vars=('z'),
#                  param_names=['a0','a1','a2'])
#quadfit=quadM.fit(data=amps,z=z,a0=1,a1=1,a2=1)
#quadBestValues=np.array([*quadfit.best_values.values()])
#zfit=np.linspace(-3,1,1000)
#quadfitResult=quadModel(zfit,*quadBestValues)
##find the index of z position of the maximum value
#index=np.argmax(quadfitResult)
#zPosition=zfit[index]
#end=time.time()
#print('Quad fit with lmfit took ',(end-start))

#%% posterior probability for fit1
M=2
niter=10000

posteriorParams=np.zeros((niter,ndim))
posteriorChi=np.zeros(niter)

start=time.time()

for i in range(niter):
  #paramslabels=['x0','y0','sigmax','sigmay','theta','A']
  p0=valuesArray[M]+np.random.normal(size=ndim)
  fakeData=gauss2d(x,y,*p0)
  result=gaussModel.fit(data=fakeData,x=x,y=y,
                        x0=156,y0=177,sigmax=1,sigmay=1,theta=0,A=1)
  #get the values out of the dictionary
  posteriorParams[i]=np.array([*result.best_values.values()])
  #calculate the chi sq.
  fit=gauss2d(x,y,*posteriorParams[i])
  posteriorChi[i]=(((signals[M]-fit)**2)*weights[M]*0.5).mean()

end=time.time()
print('Posterior analysis took ',(end-start))

#%% Using emcee
#walkers=20
#N=10000

#Determine new parameters

#p0= [[valuesArray[2,0]+np.random.normal()*np.sqrt(covarArray[2,0,0]),
#      valuesArray[2,1]+np.random.normal()*np.sqrt(covarArray[2,1,1]),
#      valuesArray[2,2]+np.random.normal()*np.sqrt(covarArray[2,2,2]),
#      valuesArray[2,3]+np.random.normal()*np.sqrt(covarArray[2,3,3]),
#      valuesArray[2,4]+np.random.normal()*np.sqrt(covarArray[2,4,4]),
#      valuesArray[2,5]+np.random.normal()*np.sqrt(covarArray[2,5,5])]
#      for w in range(walkers)]
#start=time.time()
#
#sampler=emcee.EnsembleSampler(walkers,len(p0[0]),
#                      logPi,args=[fits[2].flatten(),
#                                  x.flatten(),y.flatten(),
#                                  weights[2].flatten()])
#pos, prob, state = sampler.run_mcmc(p0,100)
#sampler.reset()
#pos, prob, state= sampler.run_mcmc(pos,N)
#
#end = time.time()
#print('Posterior Analysis: %0.2f'%(end - start))

#%%
width,height=SP.setupPlot(singleColumn=False)
grid = plt.GridSpec(1,3)
mapcolor='magma'

for i in range(5):
  fig,axs = plt.subplots(1,3,figsize=(width,.6*height))
  cmap1=axs[0].imshow(signals[i],cmap=mapcolor,
           interpolation='none',vmin=-.4,vmax=1.1)
  axs[0].set_title('Raw')
  axs[0].set_aspect('equal')
  axs[0].set_xticks([])
  axs[0].set_yticks([])

  cmap2=axs[1].imshow(fits[i],cmap=mapcolor,
           interpolation='none',vmin=0,vmax=0.9)
  axs[1].set_title('Fit')
  axs[1].set_aspect('equal')
  axs[1].set_xticks([])
  axs[1].set_yticks([])

  cmap3=axs[2].imshow(residuals[i],cmap=mapcolor,
           interpolation='none',vmin=-.4,vmax=0.6)
  axs[2].set_title('Residuals')
  axs[2].set_aspect('equal')
  axs[2].set_xticks([])
  axs[2].set_yticks([])


  fig.tight_layout()
  fig.colorbar(cmap1,ax=axs[0])
  fig.colorbar(cmap2,ax=axs[1])
  fig.colorbar(cmap3,ax=axs[2])
  fig.savefig('DataFits%i.pdf'%(i+4))

#%%
width,height=SP.setupPlot(singleColumn=False)
fig,axs = plt.subplots(1,1,figsize=(width,height))
axs.errorbar(z,amps,yerr=2*np.sqrt(covarArray[:,5,5]),fmt='.',label='Data Points')
#axs.plot(zfit,quadfitResult,label='LM fit')
axs.plot(zfit,quadfitSVD,label='SVD fit')
axs.plot(zfit[index],quadfitSVD[index],'*',label='z: %1.2f mm'%zfit[index])
axs.legend()
axs.set_xlabel('mm')
axs.set_ylabel('Amplitude')
axs.grid()

fig.savefig('QuadFitPlot.pdf')

#%%
width,height=SP.setupPlot(singleColumn=True)

fig1 = corner.corner(posteriorParams,
                    labels=paramslabels,
                    show_titles=True,
                    color='C0',
                    #levels=(0.683,0.95),
                    )
fig1.set_size_inches((width,width))
fig1.savefig('FocusingLMTPosteriorCorner.pdf')