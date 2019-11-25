# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:55:46 2019

@author: Sandra Bustamante

Acceptance/Rejection Method

"""

import numpy as np
import matplotlib.pyplot as plt

#%% Definitions

def probx(x):
  return np.sin(x)/2

def func1(x):
  return .6*np.ones(len(x))

def func2(x):
  return (-x**2+np.pi*x)/3



x=np.linspace(0,np.pi,100)

px=np.sin(x)/2

plt.plot(x,px,label='px')
plt.plot(x,func1(x),label='f1')
plt.plot(x,func2(x),label='f2')
plt.legend()

