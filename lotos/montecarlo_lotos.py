#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:48:51 2017

@author: baillard

Script made to run the montecarlo process testing multiple velocities
and lotos

"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
import datetime
from shutil import copyfile
import sys
from shutil import copyfile
import copy


from general import math
from lotos.model_1d import util as util1d
from lotos import lotos_util
from general.util import full_path_list,merge_pdfs
from lotos import LOTOS_class


#### Parameters ######

ini_vel='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/AXIAL_1D_PS_model.zv'
#ini_vel='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/AXAS2_VP.zv'
model_dir='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/'
ini_ray='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/inidata/rays0_3082.dat'
location_code='/home/baillard/PROGRAMS/LOTOS13_unix/PROGRAMS/2_INVERS_3D/1_locate/locate.exe'
tmp_dir='MC_11/'
result_file=tmp_dir+'/'+'results.log'

plt.close('all')
plt.ioff()
center=[8,5]
angle_deg=20
len_prof=[4,3]
width_prof=[1.5,1.5]

### PREPARE
  
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    
fic=open(result_file,'wt')
fic.write('%15s %7s %7s %8s\n'%(
        'MODEL_KEY','RMS_P','RMS_S','NUM_OBS'
        ))
fic.close()

#### Get data directory

model_name,data_dir=lotos_util.get_name(model_dir)
work_cwd = os.getcwd()

#### SELECT EVENTS

### WEST ####
#center_sel=[7.5,2]
#angle_deg_sel=110
#len_prof_sel=[0,6]
#width_prof_sel=[1.5,1.5]
#station_code=[1,2,3]

### EAST ####
center_sel=[10.5,2]
angle_deg_sel=110
len_prof_sel=[0,7]
width_prof_sel=[1.5,1.5]
station_code=[3,4,5,6,7]

Ray=LOTOS_class.Catalog()
Ray.read(ini_ray,'bin')
New_Ray=Ray.in_box(center_sel,angle_deg_sel,len_prof_sel,width_prof_sel,flag_plot=True)
New_Ray=New_Ray.select_ray(station_code=station_code)

#New_Ray.write(data_dir+'rays0.dat',format_file='bin')
Ray.write(data_dir+'rays0.dat',format_file='bin')


#### Load Initial Velocity #####

data=np.loadtxt(ini_vel)
depth=data[:,0]
vp=data[:,1]

#### Generate VpVs combinations ######

depth_vpvs=[0,0.7]

values_vpvs=[ list(np.arange(1.8,2.8,0.1)),
             list(np.arange(1.4,1.9,0.1)),
             ]

combi_vpvs=list(itertools.product(*values_vpvs))


####  Generate list of velocities

list_vel=[]

for kk in range(len(combi_vpvs)):

    vpvs_interp=math.interp_step(depth,depth_vpvs,combi_vpvs[kk],flag_plot=False) 
    vs=vp/vpvs_interp
    data_vel=np.column_stack((depth,vp,vs))
    
    list_vel.append(data_vel)
    

    
#### RUN THE PROCESS #####
    
for kk in range(len(list_vel)):
    
    ##### OPEN RESULTS FILE
    
    plt.close('all')
    
    ##### EDIT ########
    
    id_key=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data=list_vel[kk]
    
    
    #### Write  velocity file
    
    velocity_file='refmod.dat'
    util1d.write_1D_lotos(data,output_file=velocity_file)
    copyfile(velocity_file, data_dir+velocity_file)
    copyfile(velocity_file, tmp_dir+id_key+'_'+velocity_file)


    
    ###### RUN #########
    
    try:
        lotos_util.run_exe(location_code,flag_out=True)
    except ValueError as vrr:
        print(vrr)
        #max_iter=float(str(vrr).split('=')[1])-1
        continue
        

    ##### PLOT ########
    
   
    list_rays=[data_dir+'rays1.dat'] 
    

    util1d.plot_1D_lotos(data_dir+velocity_file,output_fig='velo.pdf')
    util1d.plot_cross(list_rays,center,angle_deg,
                         len_prof,width_prof,output_fig='cross.pdf',
                         map_xlim=[3,13],map_ylim=[0,10],z_lim=4)
    util1d.plot_statinfo(list_rays,output_fig='statinfo.pdf')
    
    pdfs = ['velo.pdf','cross.pdf','statinfo.pdf']
    merge_pdfs(pdfs,tmp_dir+'plot_'+id_key+'.pdf')
    
    ### Compute residuals
    
    util1d.get_residual(list_rays,output_file=tmp_dir+'resi_'+id_key+'.log')
    
    ### WRITE RESULTS #######
    fic=open(result_file,'at')
    resi_dic=util1d.read_resi_1DOPT(tmp_dir+'/resi_'+id_key+'.log')
    fic.write('%15s %7.3f %7.3f %8i\n'%(
                id_key,resi_dic['RMS_P'][-1],resi_dic['RMS_S'][-1],resi_dic['num_obs'][-1]
                ))
    
    ### Copy rays to file
    
    copyfile(data_dir+'rays1.dat',tmp_dir+'/rays_'+id_key+'.dat' )
        
    #### END LOOP ####
    
    fic.close()
    

    

    

