#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:45:50 2017

@author: baillard

Script made to translate rays0 binary files into text files 
Program made to read rays.dat files binaries into text file
the input in the LOTOS rays binary file with station source coordiantes and travel times specfication
The nomenclature follows what you have in the LOTOS guide for rays.dat file except that the 4th column
of obserbvation is actually the theoretical travel time
Created on Wed May 31 15:08:21 2017
"""

import numpy as np
import sys
from util import xy2ll


### Parameters

input_binary=sys.argv[1]

#### Check option
flag_option=False

if len(sys.argv)==3:
    flag_option=True
    try:
        geo_ref=[float(x) for x in sys.argv[2].split(',')]
        lon0,lat0=geo_ref[0],geo_ref[1]
    except ValueError:
        print('reference should be given like lon0,lat0')
        sys.exit()
        
###Start reading

data = np.fromfile(input_binary, dtype=np.float32, count=-1) # Array of floats 
data_int = np.fromfile(input_binary, dtype=np.int32, count=-1) # Array of ints

### Start storing and printing

k=0
while k<len(data):
    xo = data[k]
    k=k+1
    yo = data[k]
    k=k+1
    zo = data[k]
    k=k+1
    n = data_int[k]
    k=k+1
    
    if flag_option: # Convert to lon/lat
        lon,lat=xy2ll(xo,yo,lon0,lat0)
        print('{0:10.4f} {1:10.4f} {2:10.4f} {3:3d}'.format(lon,lat,zo,n))
    else:
        print('{0:10.4f} {1:10.4f} {2:10.4f} {3:3d}'.format(xo,yo,zo,n))
    n_line=1 
    
    while n_line<=n:
        type_pha = data_int[k]
        k=k+1
        sta_code= data_int[k]
        k=k+1
        t_obs=data[k]
        k=k+1
        t_ref=data[k]
        k=k+1
        n_line=n_line+1
        print('{0:1d} {1:3d} {2:10.4f} {3:10.4f}'.format(type_pha,sta_code,t_obs,t_ref))
