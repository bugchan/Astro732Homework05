# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:11:21 2019

@author: Sandra Bustamante

Gauss-Laguerre
Evaluate this integral using Gauss-Laguerre 
quadrature for \tau = 1, \sigma*t = 0.5 and at 100 points
for t \in (-2, 6). Use N=5, 10, 15, 20, and 30 knots 
and compute both the relative and absolute
error using the analytic solution.
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as integrate
import scipy.special as sp
from numpy.polynomial import laguerre
import SetupPlots as SP

#%% Definitions

def conv(tprime,t,tau,sigmat):
  A=np.exp(-(t-tprime)**2/(2*sigmat**2))\
  *np.exp(-tprime/tau)
  B=tau*np.sqrt(2*np.pi*sigmat**2)
  return A/B

def analytic(t,tau,sigmat):
  A=np.exp(((sigmat**2)/(2*tau**2))-(t/tau))
  B=sp.erfc((sigmat/(np.sqrt(2)*tau))-\
            (t/(np.sqrt(2)*sigmat)))
  C=1/(2*tau)
  return A*B*C

#when trying to solve integral(dx w(x)f(x)) the 
#integrand is f(x) after making change of variables
def integrand(x,t,tau,sigmat):
  I=np.exp(-(t-x*tau)**2/(2*sigmat**2))
  return I

#%% 
tau=1
sigmat=0.5
tArray=np.linspace(-2,6,100)
N=[5,10,15,20,30]
integralArray=np.zeros((len(N),len(tArray)))

#%% Integrate using Gauss-Laguerre

for n in range(len(N)):
  for t in range(len(tArray)):
    #obtain the weight for gaus'laguerre
    roots,weights=laguerre.laggauss(N[n])
    cons=1/(np.sqrt(2*np.pi*sigmat**2))
    integral=cons*np.sum(weights*integrand(roots,
                                           tArray[t],
                                           tau,
                                           sigmat))
    integralArray[n,t]=integral
    
#%% Relative error
real=analytic(tArray,tau,sigmat)
relError=np.abs(real-integralArray)/real

#%% Absolute error
absError=np.abs(real-integralArray)

#%% Plot
width,height=SP.setupPlot(singleColumn=False)

fig,axs = plt.subplots(3,1,
                       figsize=(width,2*height),
                       sharex=True,
                       gridspec_kw = {'hspace':0})

for n in range(len(N)):
  axs[0].plot(tArray,integralArray[n],
           label='N: %i'%N[n])
  axs[1].semilogy(tArray,relError[n],
           label='N: %i'%N[n])
  axs[2].semilogy(tArray,absError[n],
           label='N: %i'%N[n])

axs[0].plot(tArray,real,'o',label='Analytic')
axs[0].legend()
axs[0].set_ylabel('Integral Solution')
axs[0].grid()

axs[1].legend()
axs[1].set_ylabel('Relative Error')
axs[1].grid()

axs[2].legend()
axs[2].set_ylabel('Absolute Error')
axs[2].set_xlabel('Time')
axs[2].grid()

fig.savefig('GaussLaguerreQuadrature.pdf')
