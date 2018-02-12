#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:14:46 2017

@author: baillard
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:22:28 2017

@author: baillard

Function made to read a phase file formatted for hypoDD, the file looks like

# 2015  1 22  0  8 58.891   45.94934 -129.99501  0.000  0.0  0.210  0.156  0.093  20831
 AXCC1  0.489 0.75 P
 AXEC1  0.569 1.00 P

"""

from obspy import UTCDateTime
from obspy.core.event import Event, Origin, Magnitude, Catalog
from obspy.core.event import QuantityError,OriginQuality
from obspy.core.event import Pick, WaveformStreamID, Arrival
from util import weight2error

import numpy as np
import copy




def read_DD(event_file,network_code):
        
    """
    Read hypoDD
    """
    ### Parameters
    
    
    #event_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/test.dat'
    
    ### Start process
    
    f=open(event_file, 'r') 
    catalog = Catalog()
    
    k=0
    for line in f:
        k=k+1
        if line[0]=='#':
            if k>1:
                catalog.events.append(new_event)
            new_event=read_header_line(line)
        else:
            read_pick_line(line,new_event,network_code)
    
    ### Append last event when eof reacehd
    
    catalog.events.append(new_event)
    f.close()
    
    return catalog
    
    
def read_header_line(string_line):
    
    new_event=Event()
    line=string_line
    
    param_event=line.split()[1:]
    
    ### check if line as required number of arguments
    
    if len(param_event)!=14:
        return new_event
        
    ### Get parameters
    
    year,month,day=[int(x) for x in param_event[0:3]]
    hour,minu=[int(x) for x in param_event[3:5]]
    sec=float(param_event[5])
    if sec>=60:
        sec=59.999
    lat,lon,z=[float(x) for x in param_event[6:9]]
    mag=float(param_event[9])
    errh,errz,rms=[float(x) for x in param_event[10:13]]
    
    _time=UTCDateTime(year,month,day, hour,minu,sec)
    _origin_quality=OriginQuality(standard_error=rms)
    
            
    # change what's next to handle origin with no errors estimates
    
    origin=Origin(time=_time,
                  longitude=lon,latitude=lat,depth=z,
                  longitude_errors=QuantityError(uncertainty=errh),
                  latitude_errors=QuantityError(uncertainty=errh),
                  depth_errors=QuantityError(uncertainty=errz),
                  quality=_origin_quality)
    
    magnitude=Magnitude(mag=mag,origin_id=origin.resource_id)
    
    ### Return
    
    new_event.origins.append(origin)
    new_event.magnitudes.append(magnitude)
    new_event.preferred_origin_id=origin.resource_id
    new_event.preferred_magnitude_id=magnitude.resource_id

    return new_event
    

def read_pick_line(string_line,new_event,_network_code):

    time_origin=new_event.origins[0].time
    _method_id='K'
    _evaluation_mode='automatic'
    time_error_ref=[0.5,0.25,0.1,0.01] # translate weight into time uncertainty
    
    ### Start script
    
    line=string_line
    
    ### Parse line

    _station_code=line[1:6].strip()
    tt=float(line[7:14])
    weight=float(line[14:18])
    _phase_hint=line[19:21].strip()
    
    abs_time=time_origin+tt
    
    _waveform_id = WaveformStreamID(network_code=_network_code,station_code=_station_code)
    
    ### Put into Pick object
    
    _time_errors=weight2error(weight,time_error_ref)
    pick=Pick(waveform_id=_waveform_id, phase_hint=_phase_hint,
                   time=abs_time, method_id=_method_id, evaluation_mode=_evaluation_mode,time_errors=_time_errors)
    
    
    ### Put into Arrival object
    
    arrival=Arrival(pick_id=pick.resource_id, phase=pick.phase_hint)
    arrival.time_weight=weight
    
    ### Append to event
    
    new_event.picks.append(pick)
    new_event.origins[0].arrivals.append(arrival)
        
    return new_event

def dd2nllocobs(dd_file,nlloc_file):
    
    ### Read/write
    
    Catalog=read_DD(dd_file,'OO')
    
    foc_global=open(nlloc_file, 'wb') 
    for ev in Catalog:
    
        ev.write(foc_global,format='NLLOC_OBS')
        foc_global.write('\n'.encode())
    
    ### Close file
        
    foc_global.close()
    
def read_nlloc_header(file_in):
    fic=open(file_in,'rt')
    
    line=fic.readline().rstrip('\n')
    
    
    values=line.split()
    
    nx,ny,nz=[int(x) for x in values[0:3]]
    xo,yo,zo,dx,dy,dz=[float(x) for x in values[3:9]]
    
    dic={}
    
    dic.update({
            'xo': xo, 'dx': dx, 'nx': nx,
            'yo': yo, 'dy': dy, 'ny': ny,
            'zo': zo, 'dz': dz, 'nz': nz,
            })
    
    fic.close()
    
    return dic

def write_nlloc_header(dic,lon0,lat0,prefix_file,phase,unit='VELOCITY'):
    """
    Write model header for nlloc
    dic :  dict
        dictionnary contaning grid specifications
    lon0,lat0: float
        Origin of the grid
    prefix_file: str
    phase: {'P','S'}
    
    """
    
    ### Check
    
    unit_choice=['VELOCITY','VELOCITY_METERS','SLOWNESS','SLOW_LEN']
    phase_choice=['P','S']
    
    if unit not in unit_choice:
        raise ValueError("Select proper unit")
        
    if phase not in phase_choice:
        raise ValueError("Select proper phase")
    
    ### Get file_name
    
    file_out='{}.{}.mod.hdr'.format(prefix_file,phase)

    ### Write
    
    fic=open(file_out,'wt')
    
    fic.write('{:d} {:d} {:d} {:f} {:f} {:f} {:f} {:f} {:f} {} FLOAT\n'.format(
            dic['nx'],dic['ny'],dic['nz'],
            dic['xo'],dic['yo'],dic['zo'],
            dic['dx'],dic['dy'],dic['dz'],
            unit)
    )
    
    fic.write('TRANSFORM  SIMPLE LatOrig {:f}  LongOrig {:f}  RotCW 0.000000\n\n'.format(
            lat0,lon0)
              )
    
    fic.close()
    
def read_nlloc_model(file_in,dic):
    
    from lotos.model_3d.vgrid import VGrid # Avoid Circular depedency
 
    #### Read file
    
    val_array = np.fromfile(file_in, dtype=np.float32)
    
    xo,dx,nx=dic['xo'],dic['dx'],dic['nx']
    yo,dy,ny=dic['yo'],dic['dy'],dic['ny']
    zo,dz,nz=dic['zo'],dic['dz'],dic['nz']
    
    x_array=np.arange(xo,xo+nx*dx, dx)
    y_array=np.arange(yo,yo+ny*dy, dy)
    z_array=np.arange(zo,zo+nz*dz, dz)
    
    k=0
    data= np.empty((nx*ny*nz, 4))
    for x in x_array:
        for y in y_array:
            for z in z_array:
                dat=np.array([x, y, z, val_array[k]])
                data[k]=dat
                k=k+1
                
    Grid=VGrid()
    Grid.grid_spec.update({'xo':xo,'yo':yo,'zo':zo,
                           'dx':dx,'dy':dy,'dz':dz,
                           'nx':nx,'ny':ny,'nz':nz})
    Grid.data=data
    
    return Grid
    
def read_nlloc_sum(file_in):
    """
    Function made to read a nlloc hypocenter-file and store it into a simple LOTOS_class Catalog
    The ID is read from the event.comments part
    """    
    from obspy.io.nlloc.core import read_nlloc_hyp
    from lotos.LOTOS_class import Catalog,Event,Phase
    from general import util as gutil

    
    #file_in='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc3/AXIAL.20170130.005908.grid0.loc.hyp'
    #file_in='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc3/sum.nlloc'
    
    
    Ray=Catalog()
    cat =read_nlloc_hyp(file_in) 
    
    stations_dic=Ray.stations_realname
    
    for event in cat:
        
        id_event=event.comments[0].text
        origin=event.preferred_origin()
        OT=origin.time
        
        #### Initialize Event
        
        Event_p=Event()
        
        Event_p.x=origin.longitude
        Event_p.y=origin.latitude
        Event_p.z=origin.depth/1000
        Event_p.id=id_event
        Event_p.ot=OT
        Event_p.num_phase=origin.quality.used_phase_count
        Picks_p=event.picks
      
        
        for arrival in origin.arrivals:
            Phase_p=Phase()
            
            if arrival.phase in ['P','Pn']:
                Phase_p.type=1
            else:
                Phase_p.type=2
    
            Pick_p=gutil.getPickForArrival(Picks_p, arrival)
            Phase_p.station=stations_dic[Pick_p.waveform_id.station_code]
            Phase_p.t_obs=Pick_p.time-OT
            Phase_p.t_tho=Phase_p.t_obs-arrival.time_residual
            
            Event_p.phases.append(Phase_p)
            
    
        Ray.events.append(Event_p)
    
    return Ray
     
def combine_id(id_list,nlloc_file,output_file):
    """
    Function made to intergrate the id list of str into the nlloc file.
    The ID is a simple str that identifies the event.
    Its put in the COMMENt line
    """
    
    #file_nlloc='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc_3D_3082/sum.nlloc'
    #file_output='test_combine.nlloc'

    ### Read nlloc lines
    
    fic=open(nlloc_file,'rt')
    lines=fic.readlines()     
    fic.close()
    
    ### Integrate IDS
    
    foc=open(output_file,'wt')
    new_lines=copy.deepcopy(lines)
    ind=0
    
    for kk in range(len(new_lines)):
        if 'COMMENT' in lines[kk]:
            new_line='COMMENT "%s"\n'%(id_list[ind])
            new_lines[kk]=new_line
            ind=ind+1
        
        foc.write("%s"%(new_lines[kk]))
        
    foc.close()

def combine_id_fromdd(dd_file,nlloc_file,output_file):
    
    fic=open(dd_file,'rt')
    lines=fic.readlines()   
    id_list=[x.split()[-1] for x in lines if len(x)>30]
    fic.close()
    
    combine_id(id_list,nlloc_file,output_file)

