# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:55:46 2019

@author: Sandra Bustamante

Acceptance/Rejection Method

"""

import numpy as np
import matplotlib.pyplot as plt
import SetupPlots as SP
import scipy.interpolate as interpolate
import pandas as pd
#%% Definitions

def probx(x):
  return np.sin(x)/2

def func1(x):
  x=np.array(x)
  return .6*np.ones(x.size)

def Area1(x):
  return .6*x

def func2(x):
  return -.2*x**2+.2*np.pi*x+.05

def Area2(x):
  return -(.2/3)*x**3 + .1*np.pi*x**2 + .05*x

#%%

x=np.linspace(0,np.pi,100)
#desired probability distribution
px=np.sin(x)/2
#analytic and integrable function f(x):constant
fx1=func1(x)
#analytic and integrable function f(x):quadratic
fx2=func2(x)

#%% Accept-Reject Algorithm Constant Function
N=1000
x0Array=np.zeros(N)
f1Array=np.zeros(N)

#determine the total Area under the curve
A=Area1(np.pi)
#inverse function Fx=int(f(x))
Fxinv=interpolate.interp1d(Area1(x),x)

for i in range(N):
  # Draw a uniform deviate from [0,A]
  y=A*np.random.uniform()
  #Evaluate the inverse function
  x0=Fxinv(y)
  #2nd deviate from [0,f(x0)]
  f1=func1(x0)*np.random.uniform()

  if f1<=probx(x0):
    #accept f1
    x0Array[i]=x0
    f1Array[i]=f1

infactor1=N/len(x0Array[x0Array!=0])
print('Inefficiency Factor Constant: %1.2f'\
      %infactor1)
print('Area Constant: %1.2f'%A)

#%% Accept-Reject Algorithm Constant Function

#Initialize Arrays
x0Array2=np.zeros(N)
f1Array2=np.zeros(N)

#determine the total Area under the curve
A2=Area2(np.pi)
#inverse function Fx=int(f(x))
Fxinv2=interpolate.interp1d(Area2(x),x)

for i in range(N):
  # Draw a uniform deviate from [0,A]
  y=A2*np.random.uniform()
  #Evaluate the inverse function
  x0=Fxinv2(y)
  #2nd deviate from [0,f(x0)]
  f2=func2(x0)*np.random.uniform()

  if f2<=probx(x0):
    #accept f1
    x0Array2[i]=x0
    f1Array2[i]=f2

infactor2=N/len(x0Array2[x0Array2!=0])
print('Inefficiency Factor Quadratic:%1.2f'\
      %infactor2)
print('Area Quadratic %1.2f'%A2)

#%% Plot constant function
width,height=SP.setupPlot(singleColumn=False)

fig,axs = plt.subplots(1,1,
                       figsize=(width,height),
                       sharex=True,)

#density=True, normalizes the histogram
hist=axs.hist(x0Array[x0Array!=0],bins=50,
              density=True,alpha=0.8,label='Hist')
axs.plot(x,px,label='px')
axs.plot(x,fx1,label='f1')
axs.scatter(x0Array[x0Array!=0],
   f1Array[f1Array!=0],
   label='probx',color='C3',alpha=0.3)
axs.legend()
axs.grid()
axs.set_xlim([-.1,np.pi+.1])
axs.set_ylim([-.05,.65])

fig.savefig('AcceptanceRejectionPlotf1N%i.pdf'%N)

#%% Plot quadratic function
width,height=SP.setupPlot(singleColumn=False)

fig,axs = plt.subplots(1,1,
                       figsize=(width,height),
                       sharex=True,)

#density=True, normalizes the histogram
hist=axs.hist(x0Array2[x0Array2!=0],bins=50,
              density=True,alpha=0.8,label='Hist')
axs.plot(x,px,label='px')
axs.plot(x,fx2,label='f2')
axs.scatter(x0Array2[x0Array2!=0],
   f1Array2[f1Array2!=0],
   label='probx2',color='C3',alpha=0.3)
axs.legend()
axs.grid()
axs.set_xlim([-.1,np.pi+.1])
axs.set_ylim([-.05,.65])

fig.savefig('AcceptanceRejectionPlotf2N%i.pdf'%N)

#%% Save Data to csv file

indexNames=['Constant','Quadratic']
colNames=np.array(['$Ineficiency Factor$',
                   '$TotalArea$'])
row1=np.array([infactor1,A])
row2=np.array([infactor2,A2])
rows=[row1,row2]

df = pd.DataFrame(rows,columns=colNames,
                  index=indexNames)

with open('AcceptanceRejectionTableN%i.tex'%N,
          'w') as tf:
    tf.write(df.to_latex(float_format='%2.4f',
                         index=True,
                         escape=False))
