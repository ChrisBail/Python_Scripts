#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:53:09 2017

@author: baillard
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scstat
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from general import util as gutil


def classic_wadati(tp,ts,station=None,x_bins=None,force=True,fig=None,ax=None,color='0.5',label=None,**kwargs):
    
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        plt.figure(fig.number)
        fig, ax= plt.subplots(1,1)
        
    
    ### Get values
    
    delta_t=ts-tp

    ### Start plotting

    
    if station is not None:
        station_list=np.unique(station)
        
        rgb=gutil.get_colors(len(station_list),'jet')
        for kk in range(len(station_list)):
            
            new_tp=tp[station==station_list[kk]]
            new_ts=ts[station==station_list[kk]]
            
            new_delta=new_ts-new_tp
            plt.plot(new_tp,new_delta,'+',color=rgb[kk], markeredgewidth=0.3)
            
    else:
        plt.plot(tp,delta_t,'+',color=color, markeredgewidth=0.3,label=label,**kwargs)
    
    if x_bins is None:
        x_bins=np.array([np.min(tp),np.max(tp)])
        
    
    
    ### Compute regression
    
    vpvs_ratio=[]
    
    for kk in range(len(x_bins)-1):
        
        new_tp=tp[(tp>=x_bins[kk]) & (tp<=x_bins[kk+1])]
        new_delta_t=delta_t[(tp>=x_bins[kk]) & (tp<=x_bins[kk+1])]
        
        if np.size(new_tp)==0:
            continue
        
        if force:
            new_tp_col=new_tp[:,np.newaxis]
    
            slope, _, _, _ = np.linalg.lstsq(new_tp_col,new_delta_t)
            intercept=0
        else:
            slope,intercept, r_value, p_value, std_err =scstat.linregress(new_tp,new_delta_t)

        ratio=1+slope

        vpvs_ratio.append(ratio)
        
        y=slope*x_bins[kk:kk+2]+intercept
          
        plt.plot(x_bins[kk:kk+2],y,color='k',lw=1)
        
        plt.text(np.sum(x_bins[kk:kk+2])/2,np.sum(y)/2+0.15,'$V_P/V_S = %.2f$'%ratio,color='k',weight='bold',horizontalalignment= 'center',
                 fontsize=9)
        
    plt.xlabel('$t_P$ [s]')
    plt.ylabel(r'$t_S - t_P$ [s]')
    plt.axis([0,plt.axis()[1],0,plt.axis()[3]])
    ax=plt.gca()
    ax.set_title('Classic Wadati Diagram')
    
    return vpvs_ratio


def delta_wadati(dtp,dts,min_delta='0.8',force=True,title_suffix='',size=5):
    
    
    #plt.figure(figsize=(12, 8))
    plt.figure()
    x_bins=np.array([np.min(dtp),np.max(dtp)])

    #plt.plot(dtp,dts,'+')
    
    
    plt.scatter(dtp,dts,edgecolors='none',s=size,c=min_delta,zorder=3)
    
  
    
    ### Draw rectangle
    
    ax=plt.gca()
    limit_ax=ax.axis()
    ax.add_patch(patches.Rectangle(
        (limit_ax[0], 0.0),
        np.abs(limit_ax[0]),
        np.abs(limit_ax[3]),
        facecolor='0.9',zorder=1      # remove background
    ))
    
    ax.add_patch(patches.Rectangle(
        ( 0.0,limit_ax[2]),
        np.abs(limit_ax[1]),
        np.abs(limit_ax[2]),
        facecolor='0.9',zorder=2      # remove background
    ))
 
    if isinstance(min_delta,np.ndarray):
        plt.jet()
        cbar=plt.colorbar()
        cbar.ax.set_ylabel(r'$t_S - t_P [s]$')

    dtp_col=dtp[:,np.newaxis]
    
    if force:
        slope, _, _, _ = np.linalg.lstsq(dtp_col,dts)
        intercept=0
        slope=slope[0]
    else:
        slope,intercept, r_value, p_value, std_err =scstat.linregress(dtp,dts)
    
    y=slope*x_bins+intercept
        
    plt.plot(x_bins,y,'k',zorder=4)
    plt.text(np.max(dtp)-0.1*( np.max(dtp) -np.min(dtp)  ),np.max(dts),'Vp/Vs = %.2f'%slope,color='r',horizontalalignment= 'right',
                 fontsize=9,zorder=5)
    
    plt.xlabel(r'$\Delta t_P$')
    plt.ylabel(r'$\Delta t_S$')
    ax=plt.gca()
    ax.axis(limit_ax)
    ax.set_title('Delta Wadati Diagram'+'\n'+title_suffix)
    
    return slope
    
    