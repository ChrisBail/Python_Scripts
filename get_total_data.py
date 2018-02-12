#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:19:26 2017

@author: baillard
"""

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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



### Parameters

inventory_file='inventory.pickle'

lon_range=[-180,180]
lat_range=[-90,90]
year_range=[1960,2019]

num_chan=3
daily_chan_size=30 # in Mb

coef=num_chan*daily_chan_size


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

    bool_sel=((start_year<=year_bins[kk]) & (end_year>=year_bins[kk]))

    
    num_sta.append(len(lat[bool_sel]))
    year.append(year_bins[kk])
    
    
x=np.array(year)
y=np.array(num_sta)
plt.bar(x,y,align='edge',color=(0.207, 0.603, 0.870),width=0.8)
    
plt.xlim([1960,2018])
plt.xlabel('Year')
plt.ylabel('Num. Stations')

    

