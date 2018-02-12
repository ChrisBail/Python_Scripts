#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:45:37 2017

@author: baillard
"""


import numpy as np
import matplotlib.pyplot as plt

from obspy.core.event import Event
from obspy.clients.filesystem.sds import Client as sds_client
from util import getPickForArrival,getPicks



def plot_event(single_event,sds_root,time_delay,time_before,_network_code='*',
               _station_code='*',
               _location_code='*',
               _channel_code='*Z'):
    
    ### Define pick colors
    
    pick_colormap={'P':'k','S':'r'}
    
    if type(single_event) is not type(Event()):
        raise Exception('Event given is not and obspy Event')

    
    ### Read SDS client
    
    client=sds_client(sds_root,sds_type='D',format='MSEED')
    
    ### Read starttime from preferred origin
    
    origin_pref=[x for x in single_event.origins if x.resource_id==single_event.preferred_origin_id][0]
    _starttime=origin_pref.time-time_before
    _endtime=_starttime+time_delay
    
    st= client.get_waveforms(_network_code,_station_code,_location_code,_channel_code,_starttime,_endtime)
    
    ### Pre-process streams
    
    st.sort(keys=['network','station','channel'])
    st.taper(type="cosine",max_percentage=0.1)
    st.filter("bandpass",freqmin=3,freqmax=30,corners=4)
    st.taper(type="cosine",max_percentage=0.1)
    
    #st.plot(starttime=origin_pref.time, endtime=origin_pref.time+3, fig=fig2)
    
    ### Start plotting waveforms 
    
    
    fig=plt.figure(figsize=(20, 12))
    
    st_num=len(st)
    axs=[]
    
    for i,tr in enumerate(st):
        net, sta, loc, cha = tr.id.split(".")
        if loc=='':
                loc=None
    
        starttime_relative=tr.stats.starttime-_starttime
        sampletimes = np.arange(starttime_relative,
                starttime_relative + (tr.stats.delta * tr.stats.npts),
                tr.stats.delta)
#    
        if len(sampletimes) == tr.stats.npts + 1:
            sampletimes = sampletimes[:-1]
    
        if i == 0:
            ax = fig.add_subplot(st_num, 1, i+1)
        else:
            ax = fig.add_subplot(st_num, 1, i+1, sharex=axs[0])
    
        axs.append(ax)
        ax.plot(sampletimes,tr.data)
        ### Add label
        ax.text(0.1, 0.90, tr.id, transform=ax.transAxes)
    
    fig.subplots_adjust(hspace=0)
    axs[0].set_xlim(0,sampletimes[-1])
    
    ### Plot Picks on top of the waveforms
    
    picks=single_event.picks
    arrivals=single_event.origins[0].arrivals
    
    # Select picks that are in arrivals
    
    picks_sel=[getPickForArrival(picks, arrival) for arrival in arrivals]
        
    for i,tr in enumerate(st):
        ax=axs[i]
        net,sta,loc,chan=tr.id.split('.')
        if net=='':
            net=None
        if sta=='':
            continue
        if loc=='':
            loc=None
        print(sta)
        picks_tr=getPicks(picks_sel,net,sta,loc)
        for pick_tr in picks_tr:
            phase_pick=pick_tr.phase_hint
            rel_time_pick=pick_tr.time-_starttime
            ax.axvline(x=rel_time_pick, ymin=0, ymax=1,color=pick_colormap[phase_pick],lw=2)