#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:56:22 2017

@author: baillard

"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


plt.close('all')
file='/home/baillard/Dropbox/_Moi/Presentations/Presentation_AGU_2017/Data/citations.txt'
year_range=[1960,2019]

year,citation=np.loadtxt(file,skiprows=1,unpack=True)

year_bins=np.arange(year_range[0],year_range[1],1)

citation_bins=np.interp(year_bins,year,citation)

plt.figure()
plt.bar(year_bins,citation_bins,align='edge',color=(0.984, 0.682, 0.156))

plt.xlim([1960,2018])
plt.xlabel('Year')
plt.ylabel('Num. Publications')

diffy=np.diff(citation_bins)

perc=(diffy/citation_bins[0:-1])*100

xx=np.arange(ini_x,year_range[1],1)

ini_x=1998
ini_y=337

#####

alpha=0.14

yy=[]

y1=ini_y
yy.append(ini_y)
for kk in range(len(xx)-1):
    y2=(1+alpha)*y1
    
    yy.append(y2)
    y1=y2
    
plt.plot(xx,yy,'k',lw=2)

######

alpha=0.03

yy=[]

y1=ini_y
yy.append(ini_y)
for kk in range(len(xx)-1):
    y2=(1+alpha)*y1
    
    yy.append(y2)
    y1=y2
    
plt.plot(xx,yy,':k',lw=2)

#####

plt.ylim([0,2700])

plt.figure()
new_x=year_bins[1:]
plt.bar(new_x,perc,align='edge')


np.mean(perc[(new_x>=2000) & (new_x<2010)])