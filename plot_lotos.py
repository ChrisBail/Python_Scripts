#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:07:29 2018

@author: baillard
"""

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
model_dir='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/'
data_dir=model_dir+'data/'
dv_s_file=model_dir+'data/dv_v24.dat'
ini_s_file=model_dir+'ref_3D_mod2.dat'
ini_p_file=model_dir+'ref_3D_mod1.dat'
check_file=model_dir+'anomaly_2.dat'
anomaly_param=model_dir+'anomaly.dat'
output_root='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/CHECKER/'
ray_file=data_dir+'rays1.dat'
ini_1d=model_dir+'ref_syn.dat'
raypath_file='/home/baillard/PROGRAMS/LOTOS13_unix/TMP_files/tmp/ray_paths_5.dat'
#dv_file=output_root+'154607/dv_v21.dat'
copy_flag=False

flag_1D=0

slice_vals=['y=4','y=6','x=8','x=9','z=0.1','z=0.4','z=0.8','z=1','z=1.2','z=1.5']
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

### Read all files

INI_S=read(ini_s_file,'lotos')
INI_P=read(ini_p_file,'lotos')

INI_P.plot_slice('z=1.2')

DV_S=read(dv_s_file,'lotos')

FIN_P=INI_P
FIN_S=INI_S.apply_operator(DV_S,'add')

INI_VPVS=INI_P.apply_operator(INI_S,'div')
FIN_VPVS=FIN_P.apply_operator(FIN_S,'div')

RAY=read_raypaths(raypath_file,DV_S.grid_spec)
SC_1=DV_S.apply_operator(INI_S,'div')
SC_2=SC_1.apply_operator(100,'mul')

RAY_MASK=copy.deepcopy(RAY)
RAY_MASK.data[RAY.data[:,3]>=10,3]=1
RAY_MASK.data[RAY.data[:,3]<10,3]=np.nan

FIN_S_MASKED=FIN_S.apply_operator(RAY_MASK,'mul')

############  Plot Cubes ###########


FIN_S_MEAN=FIN_S_MASKED.get_mean(axis='z')
FIN_S_VAR=FIN_S.apply_operator(FIN_S_MEAN,'sub')

slice_val='z=1.2'

cmap_dic={'INI_S':plt.cm.get_cmap('jet_r'),
          'FIN_S':plt.cm.get_cmap('jet_r'),
          'SC_2':plt.cm.get_cmap('bwr_r'),
          'DV_S':plt.cm.get_cmap('bwr_r'),
          'FIN_S_VAR':plt.cm.get_cmap('bwr_r')}

color_val=[2,4]
color_dic={'INI_S':color_val,
           'FIN_S':color_val}

vm_dic={'SC_2':[-50,50],
        'DV_S':[None,None],
        'FIN_S_VAR':[-1,1]}

plot_style={'INI_S':'slice',
            'FIN_S':'slice',
            'SC_2':'contour',
            'DV_S':'contour',
            'FIN_S_VAR':'contour'}

plt.close('all')

names=['INI_S','FIN_S','DV_S','FIN_S_VAR']


for name in names:
    CUBE=eval(name)
    
    if plot_style[name]=='slice':
        (ax,_)=CUBE.plot_slice(slice_val,cmap=cmap_dic.get(name,None),color_range=color_dic[name])
    else:
        (ax,_)=CUBE.plot_contour(slice_val,cmap=cmap_dic.get(name,None),c_center=0,
         vmin=vm_dic[name][0],vmax=vm_dic[name][1],filled=True)
        plt.colorbar()
        CUBE.plot_contour(slice_val,c_center=0,colors='black',
         vmin=vm_dic[name][0],vmax=vm_dic[name][1],ax=ax)

    if slice_val[0]=='z':
        CUBE.read_stations()
        CUBE.plot_stations(ax=ax)
        CUBE.plot_lines(ax=ax)
    RAY.plot_contour(slice_val,colors='black',filled=True,alpha=0.7,levels=[-10,1],ax=ax)
    
    ax.axis(ax_dic[slice_val[0]])
    
    
sys.exit()
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


plt.close('all')
kk=0
for slice_val in slice_vals:
    fig, axarr = plt.subplots(1, 1,sharex=True,sharey=True)  


 
    kk=kk+1
    for CUBE in [SC_2]:
        
        
        print(kk)
        ax=axarr

        (ax,_)=CUBE.plot_contour(slice_val,c_center=0,cmap=plt.cm.get_cmap('bwr'),filled=True,ax=ax)
        plt.colorbar()
        if slice_val[0]=='z':
            CUBE.plot_stations(ax=ax)
            CUBE.plot_lines(ax=ax)
        
        if CUBE==SC_2:
            (ax,_)=CUBE.plot_contour(slice_val,c_center=0,colors='black',ax=ax)


        RAY.plot_contour(slice_val,colors='black',filled=True,alpha=0.7,levels=[-10,10],ax=ax)
        ax.axis('equal')
        ax.axis(ax_dic[slice_val[0]])

        ### Save fig
        
    name_file='cross_'+str(kk)+'.png'
    
    if copy_flag:
        plt.savefig(output_dir+name_file)




