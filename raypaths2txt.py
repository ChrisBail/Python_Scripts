#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:39:05 2017

@author: baillard
"""


import numpy as np
import sys

### Parameters

#input_binary='/home/baillard/PROGRAMS/LOTOS13_unix/TMP_files/tmp/ray_paths_1.dat'
input_binary=sys.argv[1]

data = np.fromfile(input_binary, dtype=np.float32, count=-1) # Array of floats 
data_int = np.fromfile(input_binary, dtype=np.int32, count=-1) # Array of ints

### Start storing and printing

k=0

while k<len(data):
    num_node = data_int[k]
    
    #### Check if end of number of events reached
    
    if num_node>=100000:
        break
    
    print('{0:10d}'.format(num_node))

    k=k+1
    n=1 
    
    while n<=num_node:
        x_r = data[k]
        k=k+1
        y_r = data[k]
        k=k+1
        z_r = data[k]
        k=k+1

        n=n+1
 
        print('{0:10.4f} {1:10.4f} {2:10.4f} '.format(x_r,y_r,z_r))
