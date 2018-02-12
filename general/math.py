#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:31:31 2017

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


def interp_step(x_interp,x_data,y_data,flag_plot=False):
    """
    Function made to interpolate data, but it is not linear interp
    y_interp=y_data(1) if x_data(1)<=x_interp<x_data(2) 
    """
    
    y_interp=np.zeros(x_interp.shape)
    

    for kk in range(len(y_data)-1):
        y_interp[ (x_interp>=x_data[kk]) & (x_interp<x_data[kk+1])]=y_data[kk]
        
    y_interp[ x_interp>=x_data[-1]]=y_data[-1]
    
    
    if flag_plot:
        plt.figure()
        plt.step(x_data,y_data,where='post',color='k')
        plt.plot(x_interp,y_interp,'+r')
        
    return y_interp

def xy2mesh(x_data,y_data,x_lin,y_lin):
        

    #plt.plot(x_interp,y_interp,'+b')
    
    #plt.plot(x_data,y_data,'+r')
    
    x_interp=np.copy(x_lin)
    y_interp=np.interp(x_interp,x_data,y_data)
    
    X,Y=np.meshgrid(x_lin,y_lin)
    
    Z=np.zeros(X.shape)
    
    
    for ii in range(X.shape[0]-1):
        for kk in range(X.shape[1]-1):
            boolx=(x_interp>= X[ii][kk]) & (x_interp< X[ii+1][kk+1])
            booly=(y_interp>= Y[ii][kk]) & (y_interp< Y[ii+1][kk+1])
            bool_all=boolx & booly
            if np.any(bool_all):
                Z[ii][kk]=1
    
    return X,Y,Z
#    f=interp2d(X, Y, Z, kind='cubic')
#    xnew = x_lin
#    ynew = y_lin
#    Z1 = f(xnew,ynew)
#    Xn, Yn = np.meshgrid(xnew, ynew)
# 
#    return Xn,Yn,Z1
    
def smoothmesh(X,Y,Z,xp,yp):
    
    y_data=Y[:,0]
    x_data=X[0,:]
    f = interp2d(x_data, y_data, Z, kind='cubic')
    Z2 = f(xp, yp)
    X2, Y2 = np.meshgrid(xp, yp)
   
    return X2,Y2,Z2

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def average_data(x,y,num_points=1000,mode='mean',flag_plot=False):
    """
    Function made average the data, or take the upper or lower envelope of the data
    
    Parameters:
    
        x,y: arrays containig the data
    
    """
    
    if not isinstance(x,np.ndarray):
        raise ValueError('x should be np array')
        
    if not isinstance(y,np.ndarray):
        raise ValueError('y should be np array')
        

    if mode not in ['mean','max','min']:
         raise ValueError('mode should be "mean","max","min" ')
        
    ### Sort according to x
    
    x_new=x[np.argsort(x)]
    y_new=y[np.argsort(x)]
    
    x=np.copy(x_new)
    y=np.copy(y_new)
    
    ### Get value
    x_interp=np.linspace(x[0],x[-1],num_points)
    y_interp=[]
    x_new=[]
    for kk in range(len(x_interp)-1):
        y_in_bin=y[(x>=x_interp[kk]) & (x<x_interp[kk+1])]
        if not y_in_bin.size:
            continue

        y_interp.append(eval('np.'+mode+'(y_in_bin)'))

        x_val=(x_interp[kk] + x_interp[kk+1])/2
        x_new.append(x_val)
    
    y_interp=np.array(y_interp)
    x_new=np.array(x_new)
    
    yy=smooth(np.array(y_interp), window_len=10, window='hanning')
    
    if flag_plot:
        plt.figure()
        plt.plot(x,y,'+b')

        
        plt.plot(x_new,yy,'g')
        
    return x_new,yy

def cluster_1d(x,dis,flag_plot=False):
    """
    Function made to cluster a 1D data based on a distance criterion 
    and return the mean of each group
    """
    
    
    diffx_left=np.diff(x)
    bool_l=diffx_left<=dis
    bool_left=np.hstack((bool_l,0))
    
    bool_right=np.hstack((0,bool_l))
    
    
    diffx_left=np.diff(x)
    bool_left=diffx_left<=dis
    bool_left=np.hstack((bool_left,0))
    
    
    group=bool_left-bool_right
    index_start=np.where(group==1)[0]
    index_end=np.where(group==-1)[0]
    
    
    x_mean=[]
    
    index_pair=np.column_stack((index_start,index_end))
    
    for kk in range(index_pair.shape[0]):
        index_1=index_pair[kk,0]
        index_2=index_pair[kk,1]
        
        val_med=np.median(x[index_1:index_2+1])
        
        
        x_mean.append(val_med)
        
    
    single_bool=(bool_left==False) & (bool_right==False)
    x_single=list(x[single_bool])
    
    x_mean.extend(x_single)
         
    x_mean=np.array(x_mean)
    
    if flag_plot:
        plt.figure()
    
        plt.plot(x_mean,np.ones(x_mean.shape),'ob')
        
        plt.plot(x,np.ones(x.shape),'+r')
    
    return x_mean


def perfect_linspace(vmin,vmax,num_step):
    """
    function made to give a linspace with nice step interval
    the first and last elements of the output list are included into the 
    vmin < list < vmax (i.e. vmin and vmax are not included).
    This is particularly useful for defining proper levels contours in plt.contourf
    
    Parameters
    ----------
    
    vmin: float
        lower value
    vmax: float
        upper value
    num_step: float,int
        number of step
        
    Returns
    -------
    
    x_range : list
        output list containing elements
    """
        
    a=(vmax-vmin)/num_step
    
    ### Change step
    
    str_value='%e'%a
    decimal=abs(int(str_value.split('e')[1]))
    new_step=float(np.around(a,decimals=decimal))
    
    ### Define new lower/upper limits
    
    up_val=(divmod(vmax,new_step)[0]+1)*new_step
    low_val=(divmod(vmin,new_step)[0]+0)*new_step
    print(low_val,up_val,new_step)
    
    x_range=np.arange(low_val,up_val,new_step)
    
    if x_range[-1]<up_val:
        x_range=np.append(x_range,up_val)
    
    return list(x_range)
    
                    
        
    