#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:24:31 2018

@author: baillard

Code made to remove the magma chamber anomaly from Adrien's model and extend it using 
Endaovour vp gradient.
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from general import projection,math
from scipy.interpolate import griddata,interp2d
from scipy.ndimage import gaussian_filter

import nlloc.util as nllocutil
from lotos.model_3d.vgrid import VGrid,read
from lotos.LOTOS_class import Catalog


plt.close('all')
file_1d='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/AXIAL_1D_PS_model.zv'
file_3d='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/3D/VGRID_x15_y15_z4_100m_1525_2kms.lotos'
file_mcm='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/top_amc_1525.llz'
file_caldera='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/caldera_smooth.ll'

#### Paramters

ini_lon=-130.1
ini_lat=45.9
z_last=1.05
slice_val='x=5'
color_range=[2,7]

#### Laod and convert mcm

lon,lat,z=np.loadtxt(file_mcm,unpack=True)

#plt.scatter(lon,lat,z,z)

x_mcm,y_mcm=projection.ll2xy(lon,lat,ini_lon,ini_lat)
plt.scatter(x_mcm,y_mcm,z,z)
plt.axis('equal')
#####

data_mcm=np.column_stack((x_mcm,y_mcm,z))
data_mcm_xy=np.column_stack((x_mcm,y_mcm))
z_mcm=copy.copy(z)
if slice_val[0]=='x':
    center=[float(slice_val[2]),0]
    angle_deg=90
elif slice_val[0]=='y':
    center=[0,float(slice_val[2])]
    angle_deg=0
    
len_prof=[0,15]
width_prof=[0.5,0.5]

fig,ax_mcm=plt.subplots()
proj_data,select_boolean=projection.project(data_mcm,center,angle_deg,len_prof,width_prof)
#plt.plot(proj_data[:,0],proj_data[:,2],'or')


x_mcm,y_mcm=math.average_data(proj_data[:,0],proj_data[:,2],num_points=100,mode='min',flag_plot=False)
plt.plot(x_mcm,y_mcm,'ok',markerfacecolor='k',markersize=1)
plt.axis([0,15,4,0])



#### Load 3D

cube_ini=read(file_3d,'lotos')
cube_ini.plot_slice(slice_val,color_range=color_range,ax=ax_mcm)

Xn,Yn,Zn,Vn=cube_ini.data2matrix(unpack=True)

### Interpolate

grid_z0 = griddata(data_mcm_xy, z_mcm, (Xn,Yn), method='nearest')
grid_z0 = grid_z0-0.15

#### Mask

MASK=np.ones(Zn.shape)

MASK[Zn>=grid_z0]=np.nan
V_MASKED=copy.deepcopy(Vn)
V_MASKED=Vn*MASK


###### Add Layer Cake

plt.close('all')
X_test=Xn[:,:,0]
Y_test=Yn[:,:,0]
x_val=X_test.reshape(-1,)
y_val=Y_test.reshape(-1,)
    
V_NEW=np.ones(V_MASKED.shape)*np.nan

for kk in range(V_MASKED.shape[2]):


    
    V_test=V_MASKED[:,:,kk]
    
    
    if np.isnan(V_test).all():
        break
        
    if np.isnan(V_test).any():
        V_test=gaussian_filter(V_test, 1)
    
    v_val=V_test.reshape(-1,)
    
    ### Interpolate
    V_inter=griddata((x_val[~np.isnan(v_val)],y_val[~np.isnan(v_val)]),
                 v_val[~np.isnan(v_val)],(X_test,Y_test), method='linear')
    
    v_val=V_inter.reshape(-1,)
    
    V_inter_2=griddata((x_val[~np.isnan(v_val)],y_val[~np.isnan(v_val)]),
                 v_val[~np.isnan(v_val)],(X_test,Y_test), method='nearest')
    

    ##### Add nearest neighbord hood after linear !!!
    
    #### Smooth
    V_smooth=gaussian_filter(V_inter_2, 1)
    
    #### Refeed
    
    V_NEW[:,:,kk]=V_smooth
 
    
#    plt.figure()
#    plt.imshow(V_test)

#    
#    DIFF=abs(V_test-V_new)
#    plt.figure()
#    plt.imshow(DIFF)
#    plt.colorbar()


print(kk)
#V_CLEAN=gaussian_filter(V_NEW, 1)

plt.close('all')
plt.figure()  
plt.imshow(Vn[70,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_MASKED[70,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_NEW[70,:,:],vmin=2,vmax=7.5)


### Extend

#
##### Load 1D

z_1d,v_1d=np.loadtxt(file_1d,usecols=[0,1],unpack=True)

#### Change gradient in oder to avoid moho
grad_end=0.063

for kk in range(v_1d.shape[0]-4,v_1d.shape[0]):
    v_1d[kk]=v_1d[kk-1]+grad_end

### define gradient

z_1d_interp=np.arange(z_1d[0],z_1d[-1],0.1)
v_1d_interp=np.interp(z_1d_interp,z_1d,v_1d)
#v_1d_interp=v_1d

#### Extend cube in depth using gradient

V_NEW=gaussian_filter(V_NEW, 0.5)   
V_EXTEND=copy.deepcopy(V_NEW)

for i in range(Xn.shape[0]):
    for j in range(Xn.shape[1]):
        z_prof=Zn[i,j,:]
        v_prof=copy.copy(V_NEW[i,j,:])
        ind_nan=np.sort(np.argwhere(np.isnan(v_prof)))
        ind_nan=ind_nan.reshape(-1,)
        if not ind_nan.size:
            continue
        

        ind_nan_first=ind_nan[0]
        v_min=v_prof[ind_nan_first-1]
        
        v_prof[ind_nan_first:]=v_min
        v_grad=(v_1d_interp-v_min)/v_min
    
        ind_min=np.argmin(abs(v_grad))
        v_grad[0:ind_min]=0
        v_grad=v_grad+1
        
        v_extend=np.ones(100,)*v_grad[-1]
        
        v_grad=np.append(v_grad,v_extend)
        
        v_new=np.ones(v_prof.shape)
        v_new=np.append(v_new[0:ind_nan_first],v_grad[ind_min:])
        v_new=v_new[0:v_prof.shape[0]]
        
        V_EXTEND[i,j,:]=v_prof*v_new


#####
        
V_CLEAN=gaussian_filter(V_EXTEND, 0.5)   
V_DIFF=V_MASKED-V_CLEAN
        
plt.close('all')
slice_i=80
plt.figure()  
plt.imshow(V_MASKED[slice_i,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_NEW[slice_i,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_EXTEND[slice_i,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_CLEAN[slice_i,:,:],vmin=2,vmax=7.5)
plt.figure()  
plt.imshow(V_DIFF[:,:,20])
plt.colorbar()
  
#### Write to lotos file
        
cube_final=copy.deepcopy(cube_ini)
cube_final.matrix2data(V_CLEAN)

cube_diff=cube_final.apply_operator(cube_ini,'sub')
plt.close('all')
slice_dir='x=7.5'
cube_final.plot_slice(slice_dir,color_range=[3,7])
cube_ini.plot_slice(slice_dir,color_range=[3,7])
cube_diff.plot_slice(slice_dir)


cube_final.write('VGRID_x15_y15_z4_100m_1525_2kms_NOMCM.lotos','lotos')

#### Build the S cube

cube_final_s=copy.deepcopy(cube_final)
z_1d,v_p,v_s=np.loadtxt(file_1d,usecols=[0,1,2],unpack=True)

cube_final_s=cube_final.apply_operator(np.column_stack((z_1d,v_p/v_s)),'div')

cube_final_s.write('VGRID_x15_y15_z4_100m_1525_2kms_NOMCM_S.lotos','lotos')

cube_vpvs=cube_final.apply_operator(cube_final_s,'div')
plt.close('all')
cube_final.plot_slice(slice_dir)
cube_final_s.plot_slice(slice_dir)
cube_vpvs.plot_slice(slice_dir)


################## ADD GRADIENT ############"
#
###### Load 1D
#
#z_1d,v_1d=np.loadtxt(file_1d,usecols=[0,1],unpack=True)
#
##### Change gradient in oder to avoid moho
#grad_end=0.063
#
#for kk in range(v_1d.shape[0]-4,v_1d.shape[0]):
#    v_1d[kk]=v_1d[kk-1]+grad_end
#
#### define gradient
#
#z_1d_interp=np.arange(z_1d[0],z_1d[-1],0.1)
#v_1d_interp=np.interp(z_1d_interp,z_1d,v_1d)
##v_1d_interp=v_1d
#
#
##### Extend cube in depth using gradient
#
#V_EXTEND=copy.deepcopy(V_MASKED)
#
#for i in range(Xn.shape[0]):
#    for j in range(Xn.shape[1]):
#        z_prof=Zn[i,j,:]
#        v_prof=copy.copy(V_MASKED[i,j,:])
#        ind_nan=np.sort(np.argwhere(np.isnan(v_prof)))
#        ind_nan=ind_nan.reshape(-1,)
#        if not ind_nan.size:
#            continue
#        
#        
#        ind_nan_first=ind_nan[0]
#        v_min=v_prof[ind_nan_first-1]
#        
#        v_prof[ind_nan_first:]=v_min
#        v_grad=(v_1d_interp-v_min)/v_min
#    
#        ind_min=np.argmin(abs(v_grad))
#        v_grad[0:ind_min]=0
#        v_grad=v_grad+1
#        
#        v_extend=np.ones(100,)*v_grad[-1]
#        
#        v_grad=np.append(v_grad,v_extend)
#        
#        v_new=np.ones(v_prof.shape)
#        v_new=np.append(v_new[0:ind_nan_first],v_grad[ind_min:])
#        v_new=v_new[0:v_prof.shape[0]]
#        
#        V_EXTEND[i,j,:]=v_prof*v_new
#
#
#plt.figure()
#plt.imshow(V_MASKED[:,:,15])
#plt.figure()
#plt.imshow(V_EXTEND[:,:,15])
#
#plt.figure()
#plt.imshow(V_MASKED[:,90,:])
#plt.figure()
#plt.imshow(V_EXTEND[:,90,:])
#plt.colorbar()
#
#


#### Get proper slice
#
#X,Y,Z,V=cube_ini.data2matrix(unpack=True)
#
#z_cube=Z[0,0,:]
#ind_min=np.argmin(abs(z_cube-z_last))
#v_slice_min=V[:,:,ind_min]
#
#### Extend cube
#
#V_ext=copy.copy(V)
#
#for kk in range(ind_min,V_ext.shape[2]):
#    V_ext[:,:,kk]=v_slice_min
#    
#### Plot
#    
#cube_ext=copy.copy(cube_ini)
#cube_ext.matrix2data(V_ext)
#cube_ext.plot_slice(slice_val,color_range=color_range)
#
##### Apply operator
#
#cube_fin=cube_ext.apply_operator(grad_mat,'mul')
#cube_fin.plot_slice(slice_val,color_range=color_range)
