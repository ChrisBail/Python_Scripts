#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:55:58 2017

@author: baillard
"""

from lotos.model_3d import vgrid
import matplotlib.pyplot as plt
from general import util,projection
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stat_xy.dat'
plt.close('all')
station_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/data/stat_xy.dat'
caldera_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/caldera_smooth.ll'
ini_vel='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/AXIAL_1D_PS_model.zv'
fig_title='East Caldera'
#
#grid_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/ref_3D_mod1.dat'
##grid_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/3D/VGRID_x15_y15_z4_100m_1525.lotos'
#
#
#A=vgrid.read(grid_file,'lotos')
#
#A.get_1D_velocity(8,8,0.1,flag_plot=True)
#A.plot_slice('z=0')


vel_data=np.loadtxt(ini_vel)
grid_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/VELOCITY/3D/VGRID_x15_y15_z4_100m_1525_2kms.lotos'

A=vgrid.read(grid_file,'lotos')

data=np.loadtxt(station_file)

x_center=data[:,0]
y_center=data[:,1]
labels=['AXAS1','AXAS2','AXCC1','AXEC1','AXEC2','AXEC3','AXID1']


radius=0.5
A.read_stations(station_file)
plt.cm.get_cmap('jet')
A.plot_slice('z=0.5')
fig=plt.gcf()
A.plot_stations(fig=fig)
A.plot_lines(caldera_file,fig=fig)
plt.plot(x_center,y_center,'+r')
#x_c,y_c=projection.get_circle(x_center,y_center,radius)

#plt.plot(x_c,y_c,'w',lw=1)
plt.axis('equal')

#x_a=x_center[0:3]
#y_a=y_center[0:3]
#labels_a=labels[0:3]

x_a=x_center[1]
y_a=y_center[1]
labels_a=labels[1]


depth,vel,_=A.get_1D_velocity(x_a,y_a,radius,flag_plot=True,labels=labels_a)
plt.step(vel_data[:,1],vel_data[:,0],'r',lw=1)
ax=plt.gca()
fig=plt.gcf()
fig.set_size_inches([5,6])
ax.set_xlabel(r'$V_P$ [km/s]')
ax.set_xlim([1.5,6.5])
A.write_1D_velocity(x_a,y_a,radius,'AXAS2_VP.zv')
plt.grid(color='0.7', linestyle=':')
fig.suptitle(fig_title)

util.figs2pdf('test.pdf')
