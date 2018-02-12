#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:14:24 2018

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import shutil 
import datetime
import os,sys
import copy

import nlloc.util as nllocutil
from lotos.model_3d.vgrid import VGrid,read,read,read_raypaths
from lotos.LOTOS_class import Catalog


plt.close('all')


### Paramters

name_suffix='check_hor_1_ver_1_real'
model_dir='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/BOARD_01/'
data_dir=model_dir+'data/'
dv_file=model_dir+'data/dv_v24.dat'
ini_file=model_dir+'ref_3D_mod2.dat'
check_file=model_dir+'anomaly_2.dat'
anomaly_param=model_dir+'anomaly.dat'
output_root='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/CHECKER/'
ray_file=data_dir+'rays1.dat'
ini_1d=model_dir+'ref_syn.dat'
raypath_file='/home/baillard/PROGRAMS/LOTOS13_unix/TMP_files/tmp/ray_paths_4.dat'
#dv_file=output_root+'154607/dv_v21.dat'
copy_flag=True

flag_1D=0

slice_vals=['y=4','y=6','x=8','x=9','z=0.1','z=0.4','z=0.8','z=1','z=1.2']
ax_map=[3.6,12,0.6,8.6]
ax_profile=[4,10,4,0]
ax_dic={'z':ax_map,'x':[2,10,4,0],'y':[5,11,4,0]}
#slice_vals=['z=0.1']

### Load 1D model

z,vp,vs=np.loadtxt(ini_1d,skiprows=1,unpack=True)


#### Generate random name based on clock 

if copy_flag:
    if name_suffix==None:
        name_suffix=datetime.datetime.now().strftime('%H%M%S')
    output_dir=output_root+name_suffix+'/' 
    os.makedirs(output_dir,exist_ok=True)

    ### Copy files to directorys
    
    shutil.copy(dv_file,output_dir)
    shutil.copy(model_dir+'anomaly_2.dat',output_dir)
    shutil.copy(anomaly_param,output_dir)
    shutil.copy(model_dir+'MAJOR_PARAM.DAT',output_dir)

### Process the data

INI=read(ini_file,'lotos')


if flag_1D==1:
        
    INI_1D=copy.deepcopy(INI)
    matrx=INI_1D.extend_1D_z(z,vp)
    INI_1D.matrix2data(matrx)
    INI=copy.copy(INI_1D)

DV=read(dv_file,'lotos')
CHECK=read(check_file,'lotos')
RAY=read_raypaths(raypath_file,DV.grid_spec)
SC_1=DV.apply_operator(INI,'div')
SC_2=SC_1.apply_operator(100,'mul')

#plt.close('all')
#
#INI.plot_slice('y=6',cmap=plt.cm.get_cmap('jet'))
#DV.plot_slice('y=6',c_center=0,cmap=plt.cm.get_cmap('bwr'))
#SC_2.plot_slice('z=1.23',c_center=0,cmap=plt.cm.get_cmap('bwr'))
#CHECK.plot_slice('z=1.23',c_center=0,cmap=plt.cm.get_cmap('bwr'))
##
#CHECK.read_stations()
#CHECK.plot_stations()
#ax=plt.gcf().axes[0]
#ax.axis([4,11,2,8])
#if copy_flag:
#        plt.savefig(output_dir+'checker.png')
#        
#CHECK.plot_slice('y=4',c_center=0,cmap=plt.cm.get_cmap('bwr'))
#if copy_flag:
#        plt.savefig(output_dir+'checker_cross.png')
##sys.exit()
    
### Plot it

SC_2.read_stations()
CHECK.read_stations()

plt.close('all')
kk=0
for slice_val in slice_vals:
    if slice_val[0] in ['x','y']:
        fig, axarr = plt.subplots(2, 1,sharex=True,sharey=True)  
    else:
        fig, axarr = plt.subplots(1, 2,sharex=True,sharey=True) 
        
    iter_val=-1
    kk=kk+1
    for CUBE in [CHECK,SC_2]:
        iter_val=iter_val+1
        
        print(kk)
        ax=axarr[iter_val]

        (ax,_)=CUBE.plot_contour(slice_val,c_center=0,cmap=plt.cm.get_cmap('bwr'),filled=True,ax=ax,vmin=-8,vmax=8)
        plt.colorbar()
        if slice_val[0]=='z':
            CUBE.plot_stations(ax=ax)
            CUBE.plot_lines(ax=ax)
        
        if CUBE==SC_2:
            (ax,_)=CUBE.plot_contour(slice_val,c_center=0,colors='black',ax=ax,vmin=-8,vmax=8)
            (ax,_)=CHECK.plot_contour(slice_val,c_center=0,colors='black',ax=ax,levels=[0])

        RAY.plot_contour(slice_val,colors='black',filled=True,alpha=0.7,levels=[-10,10],ax=ax)
        ax.axis('equal')
        ax.axis(ax_dic[slice_val[0]])

        ### Save fig
        
    name_file='cross_'+str(kk)+'.png'
    
    if copy_flag:
        plt.savefig(output_dir+name_file)




