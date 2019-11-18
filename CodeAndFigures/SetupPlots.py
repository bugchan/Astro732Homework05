#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:24:55 2019

@author: sbustamanteg
"""
import numpy as np
import matplotlib.pyplot as plt

def setupPlot(singleColumn):

  if singleColumn:
    width=6.9    
  else:
    width=3.39

  fontsize=8  
  linewidth=0.4
  markersize=1

  height=width*(np.sqrt(5.)-1.)/2.
  params = {'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'font.size': fontsize,
            'legend.fontsize': fontsize-2,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'lines.linewidth': linewidth,
            'grid.linewidth' : linewidth*.7,
            'axes.axisbelow' : True,
            'pgf.rcfonts' : False,
            'lines.markersize' : markersize,
            }
  plt.rcParams.update(params)
  return width,height

#def 3x1Plot():
