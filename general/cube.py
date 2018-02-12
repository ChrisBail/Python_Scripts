#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:25:20 2017

@author: baillard
"""

import logging 
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.ndimage
import copy
from scipy.ndimage.filters import gaussian_filter
import general

def xyz2matrix(nx,ny,nz,data):
    """
    Function made to reorganize data (a 1D numpy array) into 3D numpy array
    First place the element along z, then y and finally x
    
    Parameters
    ----------
    
    nx,ny,nz : float
    data: np.ndarray
    """
    ### Check that the data is of type array

    if not isinstance(data, np.ndarray):
        print('Not an array')
        return
    
    if nx*ny*nz != len(data):
        print('Number of elemenst dont match length of array')
        return
    
    ### Create 3D matrix
    
    matrix_v=np.reshape(data,(nx,ny,nz),order='C')
#    
#    matrix_v=np.empty((nx,ny,nz))
#    k=0
#    for ix in range(0,nx):
#        for iy in range(0,ny):
#            for iz in range(0,nz):
#                matrix_v[ix,iy,iz]=data[k]
#                k=k+1
#                
    return matrix_v

def matrix2xyz(matrix_v):
    """
    Function made to reshape the matrix
    equivalent to z loop inside y loop inside x loop
    """
    
    data=np.reshape(matrix_v,(-1),order='C')
    return data

def get_xyz(nx,ny,nz,data):
    """ 
    function made to output the space range into 1D arrays x, y and z
    """
    
    x=xyz2matrix(nx,ny,nz,data[:,0])
    y=xyz2matrix(nx,ny,nz,data[:,1])
    z=xyz2matrix(nx,ny,nz,data[:,2])

    xi=x[:,0,0]
    yi=y[0,:,0]
    zi=z[0,0,:]
    
    return xi,yi,zi


def get_slice(gridi,nx,ny,nz,slice_param):
    
    """
    Function made to plot cross sections from a given Nx4 2D arrays, the first 3 columns are the coordinates
    and the last column is the data itself (velocity)
    Loop is first on x, then y, finally z
    slice_param is given in km of the form 'x=4'
    """
    
    ### Check type and size
    
    if not isinstance(gridi,np.ndarray):
        print('Input data not an array')
        return
    
    if gridi.shape[0]!=(nx*ny*nz):
        print('mismatch between given grid and given increments')
        return
    
    ### start process
    
    ### Transfrom 2D array to 3D matrix
    
    t=time.time()
    v=xyz2matrix(nx,ny,nz,gridi[:,3])
    elapsed=time.time()-t
    logging.info('Conversion in sec = %f'%elapsed)
    x=xyz2matrix(nx,ny,nz,gridi[:,0])
    y=xyz2matrix(nx,ny,nz,gridi[:,1])
    z=xyz2matrix(nx,ny,nz,gridi[:,2])

    xi=x[:,0,0]
    yi=y[0,:,0]
    zi=z[0,0,:]
    ### Choose which section to get
    
    ### Get slice
    
    scr=slice_param.split("=");
    slice_dir=scr[0]
    slice_val=float(scr[1])
    slice_diri=slice_dir+'i'
    
    slice_ind=np.argmin(abs(eval(slice_diri)-slice_val)) # find index 
    
    if slice_dir=='x':
        X,Y,Z=y[slice_ind,:,:],z[slice_ind,:,:], v[slice_ind,:,:]
    elif slice_dir=='y':
        X,Y,Z=x[:,slice_ind,:],z[:,slice_ind,:], v[:,slice_ind,:]
    elif slice_dir=='z':
        X,Y,Z=x[:,:,slice_ind],y[:,:,slice_ind], v[:,:,slice_ind]
    
    return X,Y,Z

def plot_slice(gridi,nx,ny,nz,slice_param,ax=None,coef_std=None,c_center=None,color_range=None,cmap=plt.cm.get_cmap('jet_r')):
    
    """
    Function made to plot cross sections from a given Nx4 2D arrays, the first 3 columns are the coordinates
    and the last column is the data itself (velocity)
    Loop is first on x, then y, finally z
    slice_param is given in km of the form 'x=4'
    c_center: center of colormap, for symetric plots
    """
    
    #### Make current axes
    
    if ax is None:
        f1, ax = plt.subplots()
    
    plt.sca(ax)
    
    ### Get requested slize X,Y,Z are 2D matrix, x,y, being coordinates and z the values
    
    X,Y,Z=get_slice(gridi,nx,ny,nz,slice_param)
    
    ### Define colormap range (useful for variations)
    
    clip_max=None
    clip_min=None
    
    if c_center!=None:
        z_array=Z.reshape((-1,))
        
        if c_center=='mean':
            c_center=np.mean(z_array)
        std_value=np.std(z_array)
        
        max_abs=np.max(np.abs(z_array))
        
        if coef_std==None:
             clip_max=c_center+max_abs
             clip_min=c_center-max_abs
        else:
            clip_max=c_center+coef_std*std_value
            clip_min=c_center-coef_std*std_value
        
    print(clip_min,clip_max)
        
    ### Choose which section to plot
    
    ### Get slice
    
    scr=slice_param.split("=");
    slice_dir=scr[0]
    
    extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)]
    print(extent)
    
    if color_range!=None:
        #h1=plt.pcolormesh(X,Y,Z,vmin=color_range[0],vmax=color_range[1])
        h1=plt.imshow(np.transpose(Z),vmin=color_range[0],vmax=color_range[1], interpolation='bilinear',cmap=cmap,origin='lower',extent=extent)
    else:
        #h1=plt.pcolormesh(X,Y,Z,cmap=cmap,shading='gouraud')
        #Z= scipy.ndimage.zoom(Z, 3)
        h1=plt.imshow(np.transpose(Z), interpolation='bilinear',cmap=cmap,vmin=clip_min,vmax=clip_max,origin='lower',extent=extent)
    
    if slice_dir=='x':
        plt.xlabel('Y [km]')
        plt.ylabel('Z [km]')
        if ax.axis()[2]<ax.axis()[3]:
            plt.gca().invert_yaxis() # Only invet if axis is not already inverted in previous plots
    elif slice_dir=='y':
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        if ax.axis()[2]<ax.axis()[3]:
            plt.gca().invert_yaxis()
    elif slice_dir=='z':
        plt.xlabel('X [km]')
        plt.ylabel('Y [km]')

    
    plt.title('Slice '+slice_param)
    plt.axis('equal')
    plt.axis('tight')
    plt.colorbar(h1)
    
    ### Save slice if asked
    
        
    return (ax,h1)

def plot_contour(gridi,nx,ny,nz,slice_param,ax=None,
                 coef_std=None,c_center=None,
                 filled=False,linewidths=0.5,levels=None,vmin=None,vmax=None,**contourargs):
    
    """
    Function made to plot cross sections from a given Nx4 2D arrays, the first 3 columns are the coordinates
    and the last column is the data itself (velocity)
    Loop is first on x, then y, finally z
    slice_param is given in km of the form 'x=4'
    main contour args are linewidths and levels and colors
    
    Examples:
    plt.contourf(Zu,cmap=plt.cm.get_cmap('jet'),levels=np.linspace(0,420,10))
    plt.colorbar()
    plt.contour(Zu,levels=np.linspace(0,420,10),colors='black',linewidths=0.5)
    
    plt.contourf(Zu,colors='gray',alpha=0.7,levels=[-10,60])
        
    Parameters:
    --------
    
    filled: use contourf instead of contours
    
    ax: axes object
        Axe on which to plot the figure
    
    """
    
    ax_ini=copy.copy(ax)
    ### Manage figure
    
    if ax is None:
        fig, ax = plt.subplots()
    
    plt.sca(ax)
    
    ### Get requested slize X,Y,Z are 2D matrix, x,y, being coordinates and z the values
    
    X,Y,Z=get_slice(gridi,nx,ny,nz,slice_param)
    
    ### Choose which section to plot
    
    ### Define colormap range (useful for variations)
    
    vmina=vmin
    vmaxa=vmax
    
    z_array=Z.reshape((-1,))
    if c_center!=None:
     
        if c_center=='mean':
            c_center=np.mean(z_array)
        std_value=np.std(z_array)
        
        max_abs=np.max(np.abs(z_array))
        
        if coef_std==None:
             clip_max=c_center+max_abs
             clip_min=c_center-max_abs
        else:
            clip_max=c_center+coef_std*std_value
            clip_min=c_center-coef_std*std_value
            
    else:
        clip_max=np.max(z_array)
        clip_min=np.min(z_array)
        
    if vmina !=None:
        clip_min=vmina
        clip_max=vmaxa
        

    ### Choose levels
    
    if levels==None:
        levels=general.math.perfect_linspace(clip_min,clip_max,15)
    
    print(levels)
    ### Choose which section to plot
    
    print(vmina,vmaxa)
    print(clip_min,clip_max)
    
    ### Get slice
    
    scr=slice_param.split("=");
    slice_dir=scr[0]
    
    extent=[np.min(X),np.max(X),np.min(Y),np.max(Y)]
    origin='lower'
    
    ### Plot contour
    
    Z = gaussian_filter(Z,0.5)
#    min_val=np.min(Z)
#    Z= scipy.ndimage.zoom(Z, 5)
#    Z[Z<=min_val]=min_val
    
    #### levels
    

    if filled==False:
        h1=plt.contour(np.transpose(Z), antialiased=True,vmin=clip_min,vmax=clip_max, origin=origin, extent=extent,linewidths=linewidths,
                   levels=levels,**contourargs)
    else:
        
        h1=plt.contourf(np.transpose(Z), antialiased=True,vmin=clip_min,vmax=clip_max,  origin=origin, extent=extent,
                   levels=levels,**contourargs)


    ### Add label
    
    if slice_dir=='x':
        plt.xlabel('Y [km]')
        plt.ylabel('Z [km]')
        if ax_ini is None:
            plt.gca().invert_yaxis()
    elif slice_dir=='y':
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        if ax_ini is None:
            plt.gca().invert_yaxis()
    elif slice_dir=='z':
        plt.xlabel('X [km]')
        plt.ylabel('Y [km]')

    
    plt.title('Slice '+slice_param)
    plt.axis('equal')
    plt.axis('tight')


    ### Save slice if asked
    
        
    return (ax,h1)
    
def write_slice(gridi,nx,ny,nz,slice_param,output_file):
    
    """
    Function made to plot cross sections from a given Nx4 2D arrays, the first 3 columns are the coordinates
    and the last column is the data itself (velocity)
    Loop is first on x, then y, finally z
    slice_param is given in km of the form 'x=4'
    """
    
    ### Get requested slize X,Y,Z are 2D matrix, x,y, being coordinates and z the values
    
    X,Y,Z=get_slice(gridi,nx,ny,nz,slice_param)
    
    xi=X.reshape(-1,1)
    yi=Y.reshape(-1,1)
    zi=Z.reshape(-1,1)
    
    data=np.column_stack((xi,yi,zi))

    
    ### Write into file

    np.savetxt(output_file,data,fmt='%.3f %.3f %.3f')
    
    
