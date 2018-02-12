#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:36:58 2017

@author: baillard
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import glob
from matplotlib.backends.backend_pdf import PdfPages

from lotos import param
from general.util import full_path_list,merge_pdfs
from lotos import lotos_util
from lotos import LOTOS_class

#from lotos.model_1d.util import read_1D_lotos, plot_1D_multi

def read_1D_lotos(model_file):
    """
    Reads LOTOS 1D model file
    
    model files can ref_start.dat, refmod.dat or even ref1.dat, obtained during
    1D optimization
    
    Parameters
    ----------
    
    model_file : str
        path to 1D model file
    """

    ### read the file
    
    data=np.array([])
    
    data=np.loadtxt(model_file,dtype='float',skiprows=1,usecols=(0,1,2))
    
    ## return
    
    return data

def write_1D_lotos(data,output_file=None):
    """
    Function made to generate a lotos model file ref_start.dat,refmod.dat
    
    Parameter
    --------
    data: np.ndarray
        2D numpy array with z,vp,vs in each column
    output_file: str,opt
        output filename
    """
    
    ### Check type
    
    if np.ndim(data)!=2:
        raise ValueError('Data should have dimension 2')
        
    if data.shape[1]!=3:
        raise ValueError('Data needs 3 columns')
    
    ### Open
    
    fic=None
    if output_file:
        fic=open(output_file,'wt')
  
    format_str='{:.2f} {:.3f} {:.3f}'
    
    #### Logging
    
    logging.info('Print 1D model file from np.array')
    
    print('0.0		Ratio vp/vs',file=fic)
    
    for kk in range(data.shape[0]):
        format_str.format(data[kk,0],data[kk,1],data[kk,2])
        print(format_str.format(data[kk,0],data[kk,1],data[kk,2]),file=fic)
        
    if output_file:
        fic.close()
        
def plot_1D_lotos(model_file,model_name=None,output_fig=None):
    """
    Function made to plot a 'refmod' or 'refstart.dat' file

    """
    
    data=read_1D_lotos(model_file)
    
    depth=data[:,0]
    vp=data[:,1]
    vs=data[:,2]
    
    vpvs=vp/vs
    
    #### Plot
    
    fig = plt.figure()
    if model_name:
        fig.suptitle(model_name)
    
    ax= fig.add_subplot(131)
    velocities_arr=vp
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vp [km/s]')
    
    ax= fig.add_subplot(132)
    velocities_arr=vs
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vs [km/s]')
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])
    
    ax= fig.add_subplot(133)
    velocities_arr=vpvs
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vp/Vs')
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])
    
    #### save figure
    
    if output_fig:
        plt.savefig(output_fig)
    
    


def plot_1D_multi(ax,depth,velocities_arr):
    """
    made to plot velocities (size 100,5 for example) versus depth (size 100,1)
    must be arrays, ax is the plt ax object
    """
    
    if np.ndim(velocities_arr)==1:
        velocities_arr=np.reshape(velocities_arr, (-1,1))
    cmap = plt.cm.get_cmap('winter', velocities_arr.shape[1])

    min_vel=np.min(velocities_arr)
    max_vel=np.max(velocities_arr)
    diff_vel=max_vel-min_vel
    ylim_inf=min_vel-diff_vel*0.2
    ylim_sup=max_vel+diff_vel*0.2
    
    for k in range(velocities_arr.shape[-1]):
        rgb = cmap(k)[:3]
        ax.step(velocities_arr[:,k],depth,linewidth=1.5,c=rgb)
        
 
    ax.set_ylabel('Depth [km]')
    ax.set_xlabel('V [km/s]')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlim([ylim_inf,ylim_sup])
    ax.set_ylim([np.min(depth),np.max(depth)])
    ax.invert_yaxis()
    ax.grid()
    
    
def plot_velo_1DOPT(model_directory,max_iter,z_min,z_max,dz,output_fig=None):
    """
    Function made to plot velocity models outputted by LOTOS during
    the 1D process optimization
    """


    
    ### Parameters
    
    #model_directory='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01'
    #max_iter=4  # will read all models from initial to ref4.dat
    #z_min=0
    #z_max=4
    #dz=0.2
    #output_figure='test.pdf'
    
    ### Process
    
    model_name=model_directory.split('/')[-1]
    
    data_directory=model_directory+'/data'
    
    ### List all files in directory
    
    max_iter=int(max_iter)
    start_with=[('ref%d'%x) for x in range(1,max_iter+1) ]
    model_list=full_path_list(data_directory,'ref_start','.dat')
    model_list=model_list+full_path_list(data_directory,start_with,'.dat')
    
    ### Read the file and interpolate
    
    z=np.arange(z_min,z_max+dz,dz)
    vp=np.zeros((z.size,len(model_list)))
    vs=np.zeros((z.size,len(model_list)))
    vpvs=np.zeros(vp.shape)

    for i in range(len(model_list)):
        data=read_1D_lotos(model_list[i])
        vp_interp=np.interp(z,data[:,0],data[:,1])
        vs_interp=np.interp(z,data[:,0],data[:,2])
        
        vp[:,i]=vp_interp
        vs[:,i]=vs_interp
        
    ### Get vpvs
        
    vpvs=vp/vs
    
    #### Plot
    
    fig = plt.figure(figsize=(11.69,8.27))
    fig.suptitle(model_name)
    
    ax= fig.add_subplot(131)
    velocities_arr=vp
    depth=z
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vp [km/s]')
    
    ax= fig.add_subplot(132)
    velocities_arr=vs
    depth=z
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vs [km/s]')
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])
    
    ax= fig.add_subplot(133)
    velocities_arr=vpvs
    depth=z
    plot_1D_multi(ax,depth,velocities_arr)
    ax.set_xlabel('Vp/Vs')
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([])
    
    #### save figure
    
    if output_fig:
        plt.savefig(output_fig)
        

def plot_diff_1DOPT(model_directory,last_iter,axis_option='z',output_fig=None):
    """
    Function made to plot diff file
    """
    
    ### Process
    
    model_name,data_directory=lotos_util.get_name(model_directory)
    
    ### Get first and last ray file for comparaison 
    
    list_cmd='%srays_it[1-%1d].dat'%(data_directory,last_iter)
    ray_list=glob.glob(list_cmd)
    ray_list.sort()
    
    logging.info('rays processed for comparaison will be:\n%s\n%s'%(ray_list[0],ray_list[-1]))
    
    lotos_util.plot_diff(ray_list[0],ray_list[-1],axis_option=axis_option,output_fig=output_fig)
    

def plot_cross_1DOPT(model_directory,max_iter,center,angle_deg,
                     len_prof,width_prof,output_fig=None,
                     map_xlim=None,map_ylim=None,z_lim=None):

    """
    Function made to plot cross sections 
    """
                     
    ### Process
    
    model_name,data_directory=lotos_util.get_name(model_directory)
    
    ### Ray file list to process
    
    list_cmd='%srays_it[1-%1d].dat'%(data_directory,max_iter)
    ray_list=glob.glob(list_cmd)
    ray_list.sort()
    
    ### Open pdf if asked 
    
    if output_fig is not None:
        pdf=PdfPages(output_fig)
    
    ### Start processing through ray files
    
    
    for file in ray_list:
        A=LOTOS_class.Catalog()
        A.read(file,'bin')
        A.plot_cross(center,angle_deg,len_prof,width_prof,
                     map_ylim=map_ylim,map_xlim=map_xlim,z_lim=z_lim)
        if output_fig is not None:
            pdf.savefig()
    #util.plot_velo_1DOPT(model_directory,5,0,4,0.2)
        
    if output_fig is not None:
        pdf.close()

def plot_cross(ray_list,center,angle_deg,
                     len_prof,width_prof,output_fig=None,
                     map_xlim=None,map_ylim=None,z_lim=None):

    """
    Function made to plot cross sections 
    """
                     
    ### Open pdf if asked 
    
    if output_fig is not None:
        pdf=PdfPages(output_fig)
    
    ### Start processing through ray files
    
    
    for file in ray_list:
        A=LOTOS_class.Catalog()
        A.read(file,'bin')
        A.plot_cross(center,angle_deg,len_prof,width_prof,
                     map_ylim=map_ylim,map_xlim=map_xlim,z_lim=z_lim)
        if output_fig is not None:
            pdf.savefig()
    #util.plot_velo_1DOPT(model_directory,5,0,4,0.2)
        
    if output_fig is not None:
        pdf.close()
        
def plot_statinfo(ray_list,output_fig=None):
    
    ### Open pdf if asked 
    
    if output_fig is not None:
        pdf=PdfPages(output_fig)
    
    ### Start processing through ray files
    
    
    for file in ray_list:
        A=LOTOS_class.Catalog()
        A.read(file,'bin')
        A.plot_statinfo()
        if output_fig is not None:
            pdf.savefig()
    #util.plot_velo_1DOPT(model_directory,5,0,4,0.2)
        
    if output_fig is not None:
        pdf.close()

def get_residual_1DOPT(model_directory,max_iter,output_file=None):

    """
    Function made get the residuals 
    """
                     
    
    ### Process
    
    model_name,data_directory=lotos_util.get_name(model_directory)
    
    ### Ray file list to process
    
    list_cmd='%srays_it[1-%1d].dat'%(data_directory,max_iter)
    ray_list=glob.glob(list_cmd)
    ray_list.sort()
    
    ### Start processing through ray files
    
    mean_p=[]
    mean_s=[]
    std_p=[]
    std_s=[]
    rms_p=[]
    rms_s=[]
    tsigma_p=[]
    tsigma_s=[]
    num_obs_all=[]
    
    
    for file in ray_list:
        A=LOTOS_class.Catalog()
        A.read(file,'bin')
        vals=A.get_stat()[0]
        ss=A.get_stat()[2]
        num_obs=len(A.get_stat()[1])+len(A.get_stat()[2])
        
        num_obs_all.append(num_obs)
        
        mean_p.append(vals['mean_P'])
        mean_s.append(vals['mean_S'])
        std_p.append(vals['std_P'])
        std_s.append(vals['std_S'])
        rms_p.append(vals['rms_P'])
        rms_s.append(vals['rms_S'])
        tsigma_p.append(vals['two_sigma_P'])
        tsigma_s.append(vals['two_sigma_S'])
        
    
    #### Print to file
    
    print('Ray_file    iter  mean_P RMS_P redu       mean_S RMS_S redu      num_obs')
    if output_file is not None:
        fic=open(output_file,'wt')
        fic.write('Ray_file    iter  mean_P RMS_P redu       mean_S RMS_S redu      num_obs\n')
        
    line_format='%s %d   %6.3f %6.3f %6.3f     %6.3f %6.3f %6.3f    %8i\n'
    
    for kk,value in enumerate(ray_list):
        
        if kk==0:
            redu_p=0
            redu_s=0
        else:
            tsigma_p_old=rms_p[kk-1]
            tsigma_p_new=rms_p[kk]
            tsigma_s_old=rms_s[kk-1]
            tsigma_s_new=rms_s[kk]
            
            redu_p=((tsigma_p_old-tsigma_p_new)/tsigma_p_old) * 100
            redu_s=((tsigma_s_old-tsigma_s_new)/tsigma_s_old) * 100
            
        print(line_format %(
                ray_list[kk].split('/')[-1],kk,
                mean_p[kk],rms_p[kk],redu_p,
                mean_s[kk],rms_s[kk],redu_s,num_obs_all[kk]))
        
        if output_file is not None:
            fic.write(line_format %(
                    ray_list[kk].split('/')[-1],kk,
                mean_p[kk],rms_p[kk],redu_p,
                mean_s[kk],rms_s[kk],redu_s,num_obs_all[kk]))
            
    if output_file is not None:
        fic.close()
        
def get_residual(ray_list,output_file=None):

    """
    Function made get the residuals 
    """
                     
    ### Start processing through ray files
    
    mean_p=[]
    mean_s=[]
    std_p=[]
    std_s=[]
    rms_p=[]
    rms_s=[]
    tsigma_p=[]
    tsigma_s=[]
    num_obs_all=[]
    
    
    for file in ray_list:
        A=LOTOS_class.Catalog()
        A.read(file,'bin')
        vals=A.get_stat()[0]
        num_obs=len(A.get_stat()[1])+len(A.get_stat()[2])
        
        num_obs_all.append(num_obs)
        
        mean_p.append(vals['mean_P'])
        mean_s.append(vals['mean_S'])
        std_p.append(vals['std_P'])
        std_s.append(vals['std_S'])
        rms_p.append(vals['rms_P'])
        rms_s.append(vals['rms_S'])
        tsigma_p.append(vals['two_sigma_P'])
        tsigma_s.append(vals['two_sigma_S'])
        
    
    #### Print to file
    
    print('Ray_file    iter  mean_P RMS_P redu       mean_S RMS_S redu      num_obs')
    if output_file is not None:
        fic=open(output_file,'wt')
        fic.write('Ray_file    iter  mean_P RMS_P redu       mean_S RMS_S redu      num_obs\n')
        
    line_format='%s %d   %6.3f %6.3f %6.3f     %6.3f %6.3f %6.3f    %8i\n'
    
    for kk,value in enumerate(ray_list):
        
        if kk==0:
            redu_p=0
            redu_s=0
        else:
            tsigma_p_old=rms_p[kk-1]
            tsigma_p_new=rms_p[kk]
            tsigma_s_old=rms_s[kk-1]
            tsigma_s_new=rms_s[kk]
            
            redu_p=((tsigma_p_old-tsigma_p_new)/tsigma_p_old) * 100
            redu_s=((tsigma_s_old-tsigma_s_new)/tsigma_s_old) * 100
            
        print(line_format %(
                ray_list[kk].split('/')[-1],kk,
                mean_p[kk],rms_p[kk],redu_p,
                mean_s[kk],rms_s[kk],redu_s,num_obs_all[kk]))
        
        if output_file is not None:
            fic.write(line_format %(
                    ray_list[kk].split('/')[-1],kk,
                mean_p[kk],rms_p[kk],redu_p,
                mean_s[kk],rms_s[kk],redu_s,num_obs_all[kk]))
            
    if output_file is not None:
        fic.close()

def read_resi_1DOPT(resi_file):
    """
    Function made to read the residual file obatained by get_residual_1DOPT
    """
            
    #log_file='/home/baillard/Dropbox/_Moi/Projects/Axial/PROG/tmp/resi.log'
    
    fic=open(resi_file,'rt')
    
    lines=fic.readlines()
    
    fic.close()

    
    #### Generate dic keys
    
    param_dic={key:[] for key in lines[0].split() }
    
    ### Feed dic
    
    for k in range(1,len(lines)):
        rms_p=float(lines[k].split()[3])
        rms_s=float(lines[k].split()[6])
        num_obs=int(lines[k].split()[8])
        
        param_dic['RMS_P'].append(rms_p)
        param_dic['RMS_S'].append(rms_s)
        param_dic['num_obs'].append(num_obs)
        

    
    return param_dic

def plot_1DOPT(model_directory,output_dir,id_key='',max_iter=None,
               center=[8,5],angle_deg=20,len_prof=[4,3],
               width_prof=[2,4]):
    """
    Function made to plot everything from a model directory and compute residuals
    
    Parameters
    ----------
    model_directory : str
        Path to the LOTOS MODEL directory
    output_dir : str
        Path to store all the plots and the residual file
    id_key : str, optional
        ID key to be added to the filenames
        
    """
    
    plt.close('all')
    plt.ioff()
    
    output_dir=output_dir+'/'
#    ### Parameters
#    
#    model_directory='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01'
#    output_dir='./tmp/'
#    id_key=''
#    center=[8,5]
#    angle_deg=20
#    len_prof=[4,3]
#    width_prof=[2,4]
    
    ### Get iteration
    
    dic_param=param.read_param(model_directory+'/MAJOR_PARAM.DAT')
    if max_iter is None:
        max_iter=dic_param['model_1d']['num_iter']
    #   max_iter=4
    
    ###
    
    ### Plot Velocity profiles
    
    
    plot_velo_1DOPT(model_directory,max_iter,0,4,0.2,output_fig='velo.pdf')
    
    ### Plot Cross-sections
    
    plot_cross_1DOPT(model_directory,max_iter,center,angle_deg,
                         len_prof,width_prof,output_fig='cross.pdf',
                         map_xlim=[3,13],map_ylim=[0,10],z_lim=4)
    
    ### Plot Locations differences
    
    plot_diff_1DOPT(model_directory,max_iter,axis_option='x',output_fig='diff.pdf')
    
    ### Cpmpute residuals
    
    get_residual_1DOPT(model_directory,max_iter,output_file=output_dir+'resi_'+id_key+'.log')
    
    
    #### Merge all pdfs
    
    pdfs = ['velo.pdf','cross.pdf','diff.pdf']
    
    merge_pdfs(pdfs,output_dir+'plot_'+id_key+'.pdf')
    

        


    
               