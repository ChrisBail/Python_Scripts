#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:13:17 2017

@author: baillard
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import utm
import pyproj

def rotate_z(data,angle_deg,flag_plot=None):
    """
    Function made to change basis along the vertical direction, angle deg is 
    counter clockiwse from x
    """
    
    #### Check data
    
    if not isinstance(data,np.ndarray):
        logging.error('Data is not an array')
        return
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
    
    if num_col not in [2,3]:
        logging.error('Data has wrong shape, num col is %i'%num_col)
        return
    
    angle_rad=angle_deg*np.pi/180
    
    ### Rotation (Change of basis matrix)
    
    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    
    ##### Check Matrix
    
    if num_col==2:
        Rz=np.array([[cosa , sina],
                 [-sina, cosa]])
    else:
        Rz=np.array([[cosa , sina, 0],
                 [-sina, cosa,0],
                 [0 ,0 ,1]])
        
    data=data.transpose()
    
    ### Apply product
    
    new_data=np.dot(Rz,data).transpose()
    
    ### Return and plot
    
    if flag_plot:
        
        plt.figure()
        
        plt.subplot(121)
        plt.title('Un-rotated')
        plt.plot(data[0,:],data[1,:],'or')
        plt.axis('equal')
        plt.grid('on')
        
        plt.subplot(122)
        plt.title('Rotated with angle %.2f (CCW from x)'%(angle_deg))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
    
    
    return new_data

def translate(data,xt=0,yt=0,zt=0,flag_plot=None):
    """
    Function made to translate to +x,+y,+z
    """
    if not isinstance(data,np.ndarray):
        raise TypeError('Provide array as first argument')
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
    
    if num_col==2:
        shift_array=np.array([xt,yt])
    else:
        shift_array=np.array([xt,yt,zt])
    
    new_data=data+shift_array
    
    ### Plot if asked
    
    if flag_plot:
    
        plt.figure()
        
        plt.subplot(121)
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.axis('equal')
        plt.grid('on')
        
        plt.subplot(122)
        plt.title('translated with shift dx=%.2f and dy=%.2f '%(xt,yt))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
    
    return new_data

def recenter(data,xo=0,yo=0,zo=0,flag_plot=None):
    
    new_data=translate(data,-xo,-yo,-zo)
    
     ### Plot if asked
    
    if flag_plot:
    
        plt.figure()
        
        ax1=plt.subplot(121)
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.axis('equal')
        plt.grid('on')
        
        ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
        plt.title('recentered xo=%.2f and yo=%.2f '%(xo,yo))
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')
        
    return new_data

def inbox(data,**kwargs):
    
    try:
        num_col=data.shape[1]
    except:
        num_col=data.size
        
    x_border=kwargs.get('x_border')
    y_border=kwargs.get('y_border')
    z_border=kwargs.get('z_border')
    flag_plot=kwargs.get('flag_plot')
 
    
    if x_border is None:
        x_border=[np.min(data[:,0]),np.max(data[:,0])]
    if y_border is None:
        y_border=[np.min(data[:,1]),np.max(data[:,1])]
    if z_border is None:
        z_border=[np.min(data[:,2]),np.max(data[:,2])]
        
    logging.info('x_border: %s'%(str(x_border)))
    logging.info('y_border: %s'%(str(y_border)))
    logging.info('z_border: %s'%(str(z_border)))
        
    bool_x=np.logical_and(data[:,0]>=x_border[0],data[:,0]<=x_border[1])
    bool_y=np.logical_and(data[:,1]>=y_border[0],data[:,1]<=y_border[1])
    bool_z=np.logical_and(data[:,2]>=z_border[0],data[:,2]<=z_border[1])

    bool_array=bool_x & bool_y & bool_z
    
    new_data=data[bool_array,:]
    
    if flag_plot is True:
        plt.figure()
        
        plt.title('Un-translated')
        plt.plot(data[:,0],data[:,1],'or')
        plt.plot(new_data[:,0],new_data[:,1],'ob')
        plt.axis('equal')
        plt.grid('on')

    
    return new_data,bool_array

def project(data,center,angle_deg,len_prof,width_prof):
    """
    Function made to project x,y,z data onto cross section p,z
    """
    
    if not isinstance(data,np.ndarray):
        raise TypeError('Provide array as first argument')
        
    if data.ndim!=2:
        raise TypeError('Data array must be 2D')
    
    
    num_initial=data.shape[0]
    logging.info('Number of initial events %i'%num_initial)
    
    #### Do the projection
    
    proj_data=recenter(data,center[0],center[1],0)
    proj_data=rotate_z(proj_data,angle_deg)
    proj_data,bool_array=inbox(proj_data,
                               x_border=[-len_prof[0],len_prof[1]],
                               y_border=[-width_prof[0],width_prof[1]])
                               
            
    ### Return

    num_final=proj_data.shape[0]
    logging.info('Number of initial events %i'%num_final)
    
    return proj_data,bool_array
    
def xy2ll(x,y,ini_lon,ini_lat):

    ### get UTM
    
    zone_utm='%i'%utm.from_latlon(ini_lat, ini_lon)[2]+\
    utm.from_latlon(ini_lat, ini_lon)[3]
    
    ### Convert
    
    P1=pyproj.Proj(proj='utm', zone=zone_utm, ellps='WGS84')
    x_o,y_o=P1(ini_lon ,ini_lat)
    
    new_x=x*1000+x_o
    new_y=y*1000+y_o

    lon,lat=P1(new_x,new_y,inverse=True)
    
    return lon,lat
    
def ll2xy(lon,lat,ini_lon,ini_lat):
    
    if isinstance(lon,list):
        lon=np.array(lon)
        
    if isinstance(lat,list):
        lat=np.array(lat)
    
    
    ### get UTM
    
    zone_utm='%i'%utm.from_latlon(ini_lat, ini_lon)[2]+\
    utm.from_latlon(ini_lat, ini_lon)[3]

    ### Convert
    
    P1=pyproj.Proj(proj='utm', zone=zone_utm, ellps='WGS84')
    x_o,y_o=P1(ini_lon ,ini_lat)
    
    x,y=P1(lon,lat)
    
    new_x=(x-x_o)/1000
    new_y=(y-y_o)/1000

    return new_x,new_y

def is_in_circle(x,y,xo,yo,radius,flag_plot=None):
    """
    Function made to return x,y and boolean of elements that are isinde the 
    circle of center xo and yo with radius.
    x and y are numpy array, xo and yo and radius are floats
    if flag_plot=true, we will plot elements that in the circle
    """
    
    ### Compute val
    
    val=(x-xo)*(x-xo)+(y-yo)*(y-yo)

    ### Get boolean
    
    boolean=val<=radius**2
    x_sel=x[boolean]
    y_sel=y[boolean]
    
    ### Print
    
    logging.info('Number of elemnents selected is %i'%(len(x_sel)))
    
    if flag_plot:
        
    ### Figure
    
        plt.figure()
        plt.plot(x,y,'.',color='g')
        plt.plot(x_sel,y_sel,'.',color='r')

        plt.axis('equal')
    
        
    return x_sel,y_sel,boolean


def get_circle(xo,yo,radius,flag_plot=False):
    """
    Function made to get the coordinates of a circle
    
    Parameter
    ---------
    
    xo,yo,radius: float
        values of center and radius
    """
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2*np.pi, 100)
    
    # the radius of the circle
    
    
    # compute x1 and x2
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    
    x=x+xo
    y=y+yo
    
    if flag_plot:
             
        plt.plot(x,y,'r')
        plt.axis('equal')
        
    return x,y
        
        
    
    

    


   
    