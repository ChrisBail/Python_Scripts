#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:19:55 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from copy import copy
from lotos.model_3d import vgrid
from general import cube
import matplotlib.colors as colors
from scipy.interpolate import interp2d
import scipy.ndimage
from lotos import lotos_util


plt.close('all')


### Parameters 

file_ray='/home/baillard/PROGRAMS/LOTOS13_unix/TMP_files/tmp/ray_paths_3.dat'
file_cube='/home/baillard/Dropbox/_Moi/Projects/Axial/PROG/3D_RESULTS/vs_cube_09.dat'
dvs_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/data/dv_v21.dat'

dvs_cube=vgrid.read(dvs_file,'lotos')
v_cube=vgrid.read(file_cube,'lotos')
density_cube=vgrid.read_raypaths(file_ray,v_cube.grid_spec)

density_cube.plot_slice('y=5.5',color_range=[0,100])
dvs_cube.plot_slice('y=4')
density_cube.plot_slice('z=1.4',color_range=[0,100])
fig=plt.gcf()
ax=fig.axes[0]
density_cube.plot_contour('y=4',levels=[100],ax=ax,colors='k')

plt.ion()
z_slices=np.arange(0,2,0.2)

v_cube=dvs_cube
for kk in range(len(z_slices)):
    plt.close('all')
    z_slice=z_slices[kk]
    z_str='z='+str(z_slice)


    v_cube.plot_slice(z_str)
    fig=plt.gcf()
    v_cube.plot_lines(fig=fig)
    v_cube.read_stations()
    v_cube.plot_stations()
    
    fig=plt.gcf()
    ax=fig.axes[0]
    density_cube.plot_contour(z_str,levels=[100],ax=ax,colors='k')
    fig_name='tomo_slice_dv'+str(int(z_slice*10))+'.pdf'
    plt.savefig(fig_name)

#density_cube.plot_slice('y=6',color_range=[0,300],cmap=plt.cm.get_cmap('jet'))
#density_cube.plot_slice('z=0.2',cmap=plt.cm.get_cmap('jet'))

#v_cube.mask=density_cube.data
#
#X_slice=X[:,:,5]
#Y_slice=Y[:,:,5]
#V_slice=V[:,:,5]
#
#V_slice= scipy.ndimage.zoom(V_slice, 3)
#palette = copy(plt.cm.jet)
#
#gray_pal=copy(plt.cm.gray)
#gray_pal.set_under('w',alpha=0)
#gray_pal.set_over('w',alpha=0.5)
#
#Mask=np.ma.masked_where(V_slice<=50, V_slice)
##Mask_dat=np.ma.masked_where(V_slice<=100, V_slice)
#
##plt.pcolormesh(X_slice,Y_slice,V_slice,cmap=palette,linewidth=0,vmin=0, vmax=500)
#
##plt.pcolormesh(X_slice,Y_slice,Mask.mask,cmap=gray_pal,vmin=0.5, vmax=0.5,linewidth=0,shading='gouraud')
##plt.colorbar()
##plt.axis('equal')
#extent=[0,15,0,15]
#origin='lower'
#
#
#plt.figure()
#plt.imshow(np.transpose(V_slice), interpolation='bilinear',cmap=palette,origin='lower',extent=[0,15,0,15])
#plt.colorbar()
#plt.contour(np.transpose(V_slice),levels=[200], antialiased=True,colors='w', origin=origin, extent=extent,linewidths=0.5)
#
##plt.imshow(np.transpose(Mask.mask), interpolation='bilinear',cmap=gray_pal,origin='lower',vmin=0.49, vmax=0.5,extent=[0,15,0,15])
##plt.figure()
##plt.imshow(V_slice, interpolation='bilinear',cmap=palette,extent=[0,15,0,15])
#
#plt.plot(5,5,'ow')