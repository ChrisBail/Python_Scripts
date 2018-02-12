#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:02:51 2017

@author: baillard

Basic Modules and functions specifically designed to work for LOTOS
"""

import glob
import os
import subprocess
import logging
import sys
from obspy.core.inventory import Station

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import logging
from matplotlib.backends.backend_pdf import PdfPages


import importlib
import lotos
importlib.reload(lotos)
from lotos import lotos_util
from lotos import LOTOS_class




def get_name(model_directory):
    """
    Made to get data direcoty and model name from a LOTOS model directory
    i.e. model_directory='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01'
    """
    
    #model_directory='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01'
    
    model_directory=model_directory.rstrip('/')
    
    ### Check that directory exist
    
    data_directory=model_directory+'/data/'
    model_name=model_directory.split('/')[-1]
    area_name=model_directory.split('/')[-2]
    
    logging.info('Model name is %s'%model_name)
    logging.info('Data directory is %s'%data_directory)
    
    return (area_name,model_name,data_directory)

def plot_ray_GMT(GMT_plot,GMT_config,ray_file):
    """
    Function made to plot a lotos ray binary file
    
    GMT_plot and GMT_config are the path to the GMT plot script
    and the GMT config file which load the ray_file
    """
    ### Parameters
    
    #GMT_plot='/home/baillard/Dropbox/_Moi/Scripts/plot_GMT.sh'
    #GMT_config='/home/baillard/Dropbox/_Moi/Projects/Axial/GMT/GMT_config_rayjkm.sh'
    #ray_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/data/rays_it1.dat'
    
    ### Warning
    
    logging.warning('Please check the GMT_config file to be sure that the'
                 'geographical conversion uses the correct lon0/lat0')
    
    ### Clean to be sure its file not directory
    
    GMT_plot=GMT_plot.rstrip('/')  
    GMT_config=GMT_config.rstrip('/')
    ray_file=ray_file.rstrip('/')
    
    ### Process
    
    if not (os.path.exists(GMT_plot) and 
            os.path.exists(GMT_config) and
            os.path.exists(ray_file)):
        logging.error('GMT_plot, GMT_config or ray file does not exist')
        sys.exit()
    
    ### Call plotting script
    
    cmd_plot=('%s %s %s'%(GMT_plot,GMT_config,ray_file))
    
    proc = subprocess.Popen([cmd_plot], stdout=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    logging.debug("%s"%out.decode('ascii'))
    
    #### Output Info
    
    logging.info("Figures copied to ./figures/")
    
def read_station(station_file):
    """
    Read station files and put them in Obpsy list of staton class
    """
    if not os.path.isfile(station_file):
        raise IOError('No such file')
    
    list_dic=[]
    fic = open(station_file, 'r') 
    for line in fic:
        elements=line.split()
        lon,lat,depth=float(elements[0]),float(elements[1]),float(elements[2])
        sta_name=elements[3]
        
        sta_temp=Station(sta_name,lat,lon,depth*1000)
        
        list_dic.append(sta_temp)
        
    return list_dic

def plot_diff(ray_file_1,ray_file_2,axis_option='z',output_fig=None):
    """
    Function made to plot the location shifts between two sets of locations
    in two ray_files
    
    Parameters
    ----------
    ray_file_1,ray_file_2: str
        Full path to ray files
    axis_option : {'z', 'x', 'y','events'}
        List of possible axis options
    output_fig: str
        Name the output pdf file
    """
    
    possible_options=['x','y','z','events']
    
    ### Check option
    
    if axis_option.lower() not in possible_options:
        logging.info('%s is not a possible option, please choose'\
                     ' ammongs x,y,z or events, default=z'%axis_option)
        axis_option='z'
    
    A=LOTOS_class.Catalog()
    A.read(ray_file_1,'bin')
    xa,ya,za=A.get_xyz()
    
    B=LOTOS_class.Catalog()
    B.read(ray_file_2,'bin')
    xb,yb,zb=B.get_xyz()
    
    #### Check that number of events in the file are the same
    
    if len(xa)!=len(xb):
        logging.error('Number of events does not match in the two files')
        raise
    
    #### Compute difference
    
    #### Define x axis
    
    if axis_option=='events':
        x_axis=np.arange(1,len(xa)+1)
        xlabel='# '+axis_option
    else:
        cmd=axis_option.lower()+'a'
        x_axis=np.array(eval(cmd))
        xlabel=axis_option.upper()+' [km]'
    
    dx=np.array(xb)-np.array(xa)
    dy=np.array(yb)-np.array(ya)
    dz=np.array(zb)-np.array(za)
    
    data=[dx,dy,dz]
    data_label=['dX','dY','dZ']
    
    ### Open pdf file if asked
    
    if output_fig is not None:
        pdf=PdfPages(output_fig)
    
    #### Start plotting
    
    #### Get histograms
    
    nbins=20
    color_m='0.5'
    
    plt.figure(figsize=(11.69,8.27))
    
    gs = gridspec.GridSpec(3, 2, width_ratios=[5, 1]) 
    
    kk=0
    
    for index,value in enumerate(data):
    
        ax1=plt.subplot(gs[kk])
        
        kk=kk+1
        
        if index==0:
            plt.title('Location shifts (final - initial)')
        
        ax1.plot(x_axis,value,'+',color=color_m)
        ax1.set_xlim([0,np.max(x_axis)])
        ax1.set_ylabel(data_label[index])
        
        ax2=plt.subplot(gs[kk])
        kk=kk+1
        x_bins=np.linspace(ax1.get_ylim()[0],ax1.get_ylim()[1],nbins)
        
        plt.hist(value,x_bins,orientation='horizontal',color=color_m)
        ax2.get_yaxis().set_ticks([])
        
        if index==len(data)-1:
            ax1.set_xlabel(xlabel)
            
        if index!=len(data)-1:
            ax1.get_xaxis().set_ticklabels([])
        
        
    plt.tight_layout()
    
    if output_fig is not None:
        pdf.savefig()
    
    ############### Start 3D plot
    
    fig_3d= plt.figure(figsize=(11.69,8.27))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    ax_3d.quiver(np.array(xa),np.array(ya), np.array(za), dx, dy, dz,linewidth=0.5,color='k',
                 arrow_length_ratio=0.2)
    ax_3d.scatter(xa,ya,za,c='0.5',label='Initial')
    ax_3d.scatter(xb,yb,zb,c='r',label='Final')
    
    ax_3d.legend(loc='upper right', shadow=True)
    
    ax_3d.set_xlabel('X [km]')
    ax_3d.set_ylabel('Y [km]')
    ax_3d.set_zlabel('Z [km]')
    
    plt.gca().invert_zaxis()
    
    ax_3d.view_init(azim=-90, elev=90)
    
    if output_fig is not None:
        pdf.savefig()
        pdf.close()
        logging.info('Figure saved in %s'%output_fig)
    
def run_exe(file_exe,flag_out=False):
    """
    Function made to run executable file by going into that directory and 
    back to the current directory
    
    file_exe: str
        Full path to exe file
    """


    #### Get full path and function name
    
    fortran_path='/'.join(file_exe.split('/')[:-1])+'/'
    fortran_code=file_exe.split('/')[-1]
    
    ### Get current path
    
    curr_path=os.getcwd()
    
    ### Cd to fotran path
    
    os.chdir(fortran_path)
    
    #### Run fortran code
    
    batcmd='./'+fortran_code
    
    #os.system(batcmd)
    
    p = subprocess.Popen(batcmd, stdout=subprocess.PIPE)
    
    
    while p.poll() is None:
        l = p.stdout.readline().decode("utf-8").rstrip('\n')  # This blocks until it receives a newline.
        if 'ERROR REFRAYS' in l:
            os.chdir(curr_path)
            raise ValueError(l)
        if flag_out:
            print(l)
    
    print(p.stdout.read())

    ### go back to current directory
    
    os.chdir(curr_path)
    
    ### Logging
    
    logging.info('Running fortran code %s'%(fortran_code))
        
def write_allareas(ar,md,niter,file_out='all_areas.dat'):
    """
    Function made to write a lotos allareas.dat, that is useful to read
    the number of max iterations, the model and the area of consideration.
    Its mainly used in automatation process such as RUN_3D_LOTOS
    
    Parameters
    ----------
    
    ar: str
        name of the area
    md: str
        name of the model
    niter: int
        number of max iterations to be considered
    file_out: str
        name of the output file
    
    """
    
    #file_out='test.areas'
    
    ### Check lenght of arguments
    
    if len(ar)>8:
        raise ValueError('AREA %s has more than 8 characters'%(ar))
    if len(md)>8:
        raise ValueError('MODEL %s has more than 8 characters'%(md))
        
    ### Write into file
    
    fic=open(file_out,'wt')
    
    fic.write(' 1: name of the area (any 8 characters)\n\
 2: name of the model (any 8 characters)\n\
 3: number of iterations\n\
 *******************************************\n\
')
    
    fic.write('%8s %8s %i'%(ar,md,niter))
    fic.close()
    
    ### Logging
    logging.info('Writing elements in %s'%(file_out))
    
    
def read_raypaths(input_binary):
    """
    function made to read the lotos binary file raypath which has the ray coordinates
    into a (x,y,z,num_ray) tuple
    
    """
    
    data = np.fromfile(input_binary, dtype=np.float32, count=-1) # Array of floats 
    data_int = np.fromfile(input_binary, dtype=np.int32, count=-1) # Array of ints
    
    ### Start storing and printing
    
    k=0
    
    x=[]
    y=[]
    z=[]
    num_ray=0
    
    while k<len(data):
        num_node = data_int[k]
        num_ray=num_ray+1
        
        #### Check if end of number of events reached
        
        if num_node>=100000:
            break
        
        k=k+1
        n=1 
        
        while n<=num_node:
            x_r = data[k]
            k=k+1
            y_r = data[k]
            k=k+1
            z_r = data[k]
            k=k+1
    
            n=n+1
            
            x.append(x_r)
            y.append(y_r)
            z.append(z_r)
            
            
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
     
    return (x,y,z,num_ray)
        