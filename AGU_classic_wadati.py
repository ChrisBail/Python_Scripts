#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:20:12 2017

@author: baillard
"""

from lotos import LOTOS_class
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from general import projection
import numpy as np
from itertools import combinations
from general import wadati
import matplotlib
from general.util import convert_values
import sys
import matplotlib.backends.backend_pdf
from general import util as gutil



plt.close('all')
input_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/inidata/rays0_3082.dat'
frameon=False

A=LOTOS_class.Catalog()
A.read(input_file,'bin')
tp,ts,_=A.get_t_wadati()

wadati.classic_wadati(tp,ts,x_bins=None,force=True,color=(0.490, 0.666, 0.890),markersize=2,alpha=0.5,marker='o')

plt.xlim([0,1.6])
plt.savefig('classic_wadati.pdf')


### Select eastern events

center=[7.5,2]
angle_deg=110
len_prof=[0,6]
width_prof=[1.5,1.5]
station_code=[1,2,3]
#

#East
#center=[10.5,2]
#angle_deg=110
#len_prof=[0,7]
#width_prof=[1.5,1.5]
#station_code=[4,5,6,7]

Ray=A.in_box(center,angle_deg,len_prof,width_prof,flag_plot=True)

#### Classic

x_bins=np.linspace(0.4,1,3)

FRay=Ray.select_ray(station_code)
tp,ts,st=FRay.get_t_wadati()


wadati.classic_wadati(tp,ts,x_bins=None,force=True,markersize=3,color='r',alpha=0.5,marker='o')
plt.savefig('classic_wadati_east.pdf')


rgb=gutil.get_colors(len(station_code),colormap='jet',flag_plot=False)

fig,ax=plt.subplots(1)
for kk in range(len(station_code)):
    NewRay=Ray.select_ray(station_code[kk])

    tp,ts,st=NewRay.get_t_wadati()

    wadati.classic_wadati(tp,ts,x_bins=None,force=True,markersize=3,ax=ax,color=rgb[kk],label=str(station_code[kk]),alpha=0.5,marker='o')

plt.xlim([0,1.2])

plt.legend()    
plt.savefig('classic_wadati_west.pdf')

