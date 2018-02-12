#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:18:41 2017

@author: baillard
"""

### Parameters

from nlloc import util as nllocutil
from lotos.model_3d.vgrid import VGrid,read,get_grid_range
import matplotlib.pyplot as plt
from general import cube
import copy
import matplotlib.pyplot as plt

import numpy as np


plt.close('all')

### Parameters

file_in='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/3D/AXIAL_VELOCITY.P.mod.buf'
file_header='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/3D/AXIAL_VELOCITY.P.mod.hdr'
file_1d='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/AXIAL_1D_PS_model.zv'


### Load one d file

z,vp,vs=np.loadtxt(file_1d,unpack=True)

vpvs=vp/vs

### Load 3D file

dic=nllocutil.read_nlloc_header(file_header)

A=nllocutil.read_nlloc_model(file_in,dic)




vp_cube=A.data2matrix()
    
### interpolate


ratio_cube=A.extend_1D_z(z,vpvs)


vs_cube=vp_cube/ratio_cube

A.matrix2data(vs_cube)

A.plot_slice('x=5')


A.write('AXIAL_VELOCITY',output_type='nlloc',phase='S',unit='VELOCITY')

A.data[:,3]=(1/A.data[:,3])*0.1


A.write('AXIAL_SLOW_LEN',output_type='nlloc',phase='S',unit='SLOW_LEN')