#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:08:16 2019

@author: sbustamanteg
"""

import numpy as np

def leapfrog2D(dvdt,a,b,h,IV,dim=2):
  #h = (b-a)/float(N)
  t=np.arange(a,b+h,h)
  x=np.zeros((len(t),dim))
  v=np.zeros((len(t),dim))
  x[0], v[0] = IV
  for n in np.arange(1,len(t)):
    k1=x[n-1]+0.5*h*v[n-1]
    v[n]=v[n-1]+h*dvdt(t,k1,v)
    x[n]=k1+0.5*h*v[n]
  return t, x, v

def RK4(dvdt, a, b, h, IV,dim=2):
    t = np.arange(a,b+h,h)  # create time
    #N=int((b-a)/h)
    x = np.zeros((len(t),dim))       # initialize x
    v = np.zeros((len(t),dim))       # initialize x
    x[0], v[0] = IV       # set initial values
    # apply Fourth Order Runge-Kutta Method
    for i in np.arange(1,len(t)):
        k1= h*dvdt(t[i-1], x[i-1], v[i-1])
        j1= h*v[i-1]
        k2= h*dvdt(t[i-1]+h/2.0, x[i-1]+j1/2.0,
                   v[i-1]+k1/2.0)
        j2= h*(v[i-1]+k1/2.0)
        k3= h*dvdt(t[i-1]+h/2.0, x[i-1]+j2/2.0,
                   v[i-1]+k2/2.0)
        j3= h*(v[i-1]+k2/2.0)
        k4= h*dvdt( t[i], x[i-1] + j3, v[i-1] + k3)
        j4= h*(v[i-1] + k3)
        v[i] =v[i-1] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        x[i] =x[i-1] + (j1 + 2.0*j2 + 2.0*j3 + j4)/6.0
    return t, x, v
