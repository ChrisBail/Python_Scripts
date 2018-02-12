#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:24:29 2018

@author: baillard
"""

import nlloc.util as nllocutil
from obspy.core.utcdatetime import UTCDateTime
import numpy as np

import matplotlib.pyplot as plt


file_nlloc='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/CATALOG/AXIAL_3D_ALL_ID.xyz'
file_dd='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/CATALOG/Axial_hypoDDPhaseInput.dat'

  
### Read dd

fic=open(file_dd,'rt')
lines=fic.readlines() 
fic.close()  

id_list=[x.split()[-1] for x in lines if len(x)>30]

ot=[]
for line in lines:
    if len(line)<30:
        continue
    
    A=line.split()
    
    year=int(A[1])
    month=int(A[2])
    day=int(A[3])
    hour=int(A[4])
    minute=int(A[5])
    second=float(A[6])
    if second>=60.0:
        second=59.99
    
    ot.append(UTCDateTime(year,month,day,hour,minute,second))
    
## Read nlloc
    
#Cat=nllocutil.read_nlloc_sum(file_nlloc)
#ot_nlloc=[event.ot for event in Cat.events]
#id_nlloc=[event.id for event in Cat.events]

fic=open(file_nlloc,'rt')
lines=fic.readlines() 
lines=lines[1:]
fic.close()  

ot_nlloc=[UTCDateTime(x.split()[0]) for x in lines]
id_nlloc=[x.split()[-1] for x in lines]


id_delta=[]
ot_delta=[]

for kk in range(len(id_nlloc)):
    
    id_ref=id_nlloc[kk]
    ot_ref=ot_nlloc[kk]
    
    if id_ref in id_list:
        ot_comp=ot[id_list.index(id_ref)]
    else:
        continue
    
    id_delta.append(float(id_ref))
    ot_delta.append(ot_comp-ot_ref)
    
plt.plot(id_delta,ot_delta)
plt.suptitle('Origin time diff (William-NLLOC)')
plt.xlabel('Event IDs')
plt.ylabel('delta t [s]')
    
dd=np.diff(id_delta)
    
    
    
    