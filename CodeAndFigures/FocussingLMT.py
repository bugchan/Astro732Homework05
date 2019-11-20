#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:26:10 2019

@author: Sandra Bustamante

Focussing the LMT

weight: (1/σ2)
Putting all this together, use the lmfit Levenberg-Marquardt
fitting package to fit each image to a 2-d gaussian.

"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import lmfit

#%% Definitions

def model(z,a0,a1,a2):
  return a0+a1*z+a2*z**2

def gauss2d(x,y,x0,y0,sigmax,sigmay,theta,A):
  a=(np.cos(theta)**2)/(2*sigmax**2)\
  + (np.sin(theta)**2)/(2*sigmay**2)
  b=(np.sin(2*theta))/(2*sigmax**2)\
  +(np.sin(2*theta))/(2*sigmay**2)
  c=(np.sin(theta)**2)/(2*sigmax**2)\
  + (np.cos(theta)**2)/(2*sigmay**2)
  return A*np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2)



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

z=np.array([-3.0,-2.0,-1.0,0.0,1.0])
ylen,xlen=s4.shape
x=np.arange(xlen)
y=np.arange(ylen)
x,y=np.meshgrid(x,y)

#%%
gaussModel=lmfit.Model(gauss2d,
                       independent_vars=('x','y'),
      param_names=('x0','y0','sigmax','sigmay','theta','A'),
      nan_policy='omit')

result=gaussModel.fit(data=s4,weights=w4,
                      x=x,y=y,
                      x0=156,y0=177,
                      sigmax=1,sigmay=1,
                      theta=0,A=1)
covar=result.covar
bestValues=result.best_values
j=0
valuesArray=np.zeros(len(bestValues))
valuesNames=['']*len(bestValues)
for i in bestValues:
  valuesArray[j]=bestValues[i]
  valuesNames[j]=i
  j+=1

plt.imshow(gauss2d(x,y,*valuesArray))

#plt.imshow(x, y, s4)
#plt.imshow(x,y,result.best_fit)
#
#plt.plot(x, result.init_fit, 'k--', label='initial fit')
#plt.plot(x, result.best_fit, 'r-', label='best fit')
#plt.legend(loc='best')




