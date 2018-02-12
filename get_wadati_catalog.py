#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:55:26 2017

@author: baillard
"""

from lotos import LOTOS_class
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from general import projection
import numpy as np
from itertools import combinations
from general import wadati
import matplotlib
from general.util import convert_values
import sys
import matplotlib.backends.backend_pdf


plt.close('all')
input_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/inidata/rays0_3082.dat'

A=LOTOS_class.Catalog()
A.read(input_file,'bin')



x=[]
y=[]
for event in A.events:
    x.append(event.x)
    y.append(event.y)
    
x=np.array(x)
y=np.array(y)
z=np.ones(x.shape)


data=np.column_stack((x,y,z))

#### Get station permutation

index_list=range(len(A.stations_realname))
index_combi=list(combinations(index_list, 2))
station_val=list(range(1,len(A.stations_realname)+1))
x_sta,y_sta=A.conv_station()

######################
#### START LOOP ######

prof_shift=0.5
width_prof=[0.5,0.5]

vpvs_ratio=[]
num_obs=[]

plt.ioff()

pdf = matplotlib.backends.backend_pdf.PdfPages("new.pdf")


for kk,index_pair in enumerate(index_combi):
   
    #### Indexes
    
    station_1=station_val[index_pair[0]]
    station_2=station_val[index_pair[1]]
    

    ### Select Events in box define by pair
    
    x1=x_sta[index_pair[0]]
    y1=y_sta[index_pair[0]]
    
    x2=x_sta[index_pair[1]]
    y2=y_sta[index_pair[1]]
    
    dist=np.sqrt((x2-x1)**2+(y2-y1)**2)
    angle_deg=np.arctan2((y2-y1), (x2-x1)) * 180 / np.pi

    len_prof=[prof_shift,dist+prof_shift]
    
    SelEv=A.in_box([x1,y1],angle_deg,len_prof,width_prof,map_xlim=[4,12],map_ylim=[1.5,8.5],flag_plot=True)
    plt.axis([4,12,1.5,8.5])

    plt.plot([x1,x2],[y1,y2],'^',markersize=10,color=(0.5,1,0),markeredgecolor='k')
    plt.text(x1,y1+0.15,'1',va='bottom',ha='center')
    plt.text(x2,y2+0.15,'2',va='bottom',ha='center')
   
    
    pdf.savefig(plt.gcf())
    ### Select only obs associated to pair
    
    SelObs=SelEv.select_ray(station_code=[station_1,station_2])
    
    
    dtp,dts,min_delta=SelObs.get_dt_wadati()
    
    
   
    slope=wadati.delta_wadati(dtp,dts,min_delta,force=True,title_suffix='',size=30)
    num_obs.append(dtp.size)

    pdf.savefig(plt.gcf())
    
    vpvs_ratio.append(slope)
    
    plt.close('all')
    


######
    
pdf.close()

plt.close('all')
    
vpvs=vpvs_ratio

num_obs=np.array(num_obs)

new=convert_values(num_obs,[100,500,800,1000],[1,2,3,4,5]) 
    
cm = plt.get_cmap('jet')
plt.ion()
norm = clrs.Normalize(vmin=np.min(vpvs), vmax=np.max(vpvs))


A.plot_map()
for kk,index_pair in enumerate(index_combi):
    
    station_1=station_val[index_pair[0]]
    station_2=station_val[index_pair[1]]
    
    ### Select Events in box define by pair
    
    x1=x_sta[index_pair[0]]
    y1=y_sta[index_pair[0]]
    
    x2=x_sta[index_pair[1]]
    y2=y_sta[index_pair[1]]
    
    plt.plot([x1,x2],[y1,y2],color=cm(norm(vpvs[kk])),lw=new[kk])
    
plt.axis([4,12,1.5,8.5])

    
sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
cbar=plt.colorbar(sm)
cbar.ax.set_ylabel('Vp/Vs')

plt.savefig('vpvs_map.pdf')
