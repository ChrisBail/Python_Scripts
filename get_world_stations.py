#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:21:02 2017

@author: baillard
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


### Parameters

inventory_file='inventory.pickle'
output_dir='Stations_GMT/'
lon_range=[-130,-65]
lat_range=[25,50]
year_range=[1960,2019]

### Check

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### Read or Load

if os.path.isfile(inventory_file):
    inventory = pickle.load(open( "inventory.pickle", "rb" ))
else:
    starttime = UTCDateTime("%4i-01-01"%(year_range[0]))
    endtime = UTCDateTime("%4i-01-01"%(year_range[1]))
    
    
    client = Client("IRIS")
    inventory = client.get_stations(network="*", station="*",
                                    starttime=starttime,
                                    endtime=endtime,
                                    minlongitude=lon_range[0],
                                    maxlongitude=lon_range[1],
                                    minlatitude=lat_range[0],
                                    maxlatitude=lat_range[1]
                                    )
    with open(inventory_file,"wb") as f:
        pickle.dump(inventory, f)
        
        
### Feed
        
lon,lat,code,start_year,end_year=[],[],[],[],[]

for network in inventory:
    if network.code=='SY':
        continue
    for station in network.stations:
        lon.append(station.longitude)
        lat.append(station.latitude)
        code.append(station.code)
        start_year.append(station.start_date.year)
        end_year.append(station.end_date.year)
        
##### Sort depending on year
        
lon=np.array(lon)
lat=np.array(lat)
code=np.array(code)
start_year=np.array(start_year)
end_year=np.array(end_year)

tot_num=[]

year_bins=np.arange(year_range[0],year_range[1],1)
year=[]
num_sta=[]


for kk in range(len(year_bins)):
    file_name=output_dir+'file_%4i.dat'%(int(year_bins[kk]))
    bool_sel=((start_year<=year_bins[kk]) & (end_year>=year_bins[kk]))
    a=start_year[bool_sel]
    a[a<year_bins[kk]]=0
    a[a==year_bins[kk]]=1
    np.savetxt(file_name,np.c_[lon[bool_sel],lat[bool_sel],
                          a
                          ],
        fmt='%f %f %i')
    
    num_file_name=output_dir+'num_%4i.dat'%(int(year_bins[kk]))
    
    num_sta.append(len(lat[bool_sel]))
    year.append(year_bins[kk])
    
    x=np.array(year)
    y=np.array(num_sta)
    np.savetxt(num_file_name,np.c_[x,y            
                          ],
        fmt='%i %f')
    

    

    

