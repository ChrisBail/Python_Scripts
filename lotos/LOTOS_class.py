#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:10:12 2017

@author: baillard

Program made to define the classes to read rays.txt or binaries files
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import logging
from lotos import lotos_util
import itertools
from general import projection,math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42




### Parameters


class Catalog(object):
        
    def __init__(self):
        """
        """
        self.events=[]
        self.file=None
        self.stations_realname={'AXAS1':1,'AXAS2':2,
                                'AXCC1':3,'AXEC1':4,'AXEC2':5,
                                'AXEC3':6,'AXID1':7}
        self.station_file='/home/baillard/PROGRAMS/LOTOS13_unix'\
        '/DATA/AXIALSEA/inidata/stat_ft.dat'
        self.line_files=['/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/caldera_smooth.ll']
        self.surf_files=['/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/top_amc_1525.llz']
        self.ini_lon=-130.1
        self.ini_lat=45.9
        self.surf=[]
        self.lines=[]
        
    def read(self,input_file,format_file=None):
        """
        """
        
        ### Get filename
        
        file_short=input_file.split('/')[-1]
        self.file=file_short
        
        ### Transform binary to text if necessary
        
        if format_file=='bin':
            cmd='rays2txt.py '+input_file+' > tmp_ray.txt'
            print(cmd)
            os.system(cmd)
            input_file='tmp_ray.txt'
            
        
        ### Read file and put everything into an array

        fic=open(input_file,'rt')
        line=1
        
        while line:
            line=fic.readline()
            if not line:
                break
            event_tmp=Event()
            self.events.append(event_tmp)
            event_tmp.read_header(line)
            phase_counter=1
     
            while phase_counter<=event_tmp.num_phase:
                line=fic.readline()
                event_tmp.phases.append(Phase(line))
                phase_counter=phase_counter+1

            
        fic.close()
        
    def write(self,output_file,format_file='txt'):
        
        fic=open(output_file,'wt')
        
        for event in self.events:
            event.write(fic)
        
        fic.close()
        
        if format_file=='bin':
            cmd='txt2rays.py '+ output_file +' tmp_ray.bin'
            os.system(cmd)
            cmd='mv tmp_ray.bin '+ output_file
            os.system(cmd)
            
    def read_surf(self,filein=None):
        """
        Read lon lat z surface file
        """
        if filein is None:
            filein=self.surf_files
            
        for file in filein:
            surf=np.loadtxt(file)
            
            surf=surf[:,0:3]
            
            #### Convert to x,y
            
            if self.ini_lon is not None:
                x,y=projection.ll2xy(surf[:,0],surf[:,1],self.ini_lon,self.ini_lat)
                surf[:,0]=x
                surf[:,1]=y
            
            self.surf.append(surf)     
            
    def read_line(self,filein=None):
        """
        Read lon lat line file
        """
        if filein is None:
            filein=self.line_files
            
        for file in filein:
            lines=np.loadtxt(file)
            
            lines=lines[:,0:2]
            
            #### Convert to x,y
            
            if self.ini_lon is not None:
                x,y=projection.ll2xy(lines[:,0],lines[:,1],self.ini_lon,self.ini_lat)
                lines[:,0]=x
                lines[:,1]=y
            
            self.lines.append(lines)   
            

    def ll2xy(self,ini_lon,ini_lat):
        
        new_events=self.events
        for kk,event in enumerate(self.events):
            lon=event.x
            lat=event.y
            
            x,y=projection.ll2xy(lon,lat,ini_lon,ini_lat)
        
            
            new_events[kk].x=x
            new_events[kk].y=y
            
        
        self.events=new_events
        
    def xy2ll(self,ini_lon,ini_lat):
        
        new_events=self.events
        for kk,event in enumerate(self.events):
            x=event.x
            y=event.y
            
            lon,lat=projection.xy2ll(x,y,ini_lon,ini_lat)
        
            
            new_events[kk].x=lon
            new_events[kk].y=lat
            
        
        self.events=new_events
        
    def get_xyz(self):
        """
        Function made to return xyz catalog from the catalog
        """
    
        x=[event.x for event in self.events]
        y=[event.y for event in self.events]
        z=[event.z for event in self.events]
        
        return x,y,z


    def get_stat(self,type_stat=None):
        
        events=self.events[:]
        resid_P_all=[]
        resid_S_all=[]
        num_P_all=0
        num_S_all=0
        
        outdict=dict()
        
        for event in events:
            resid_P,num_P,resid_S,num_S,resid_all=event.get_stat()
            num_P_all=num_P_all+num_P
            num_S_all=num_S_all+num_S
            resid_P_all.extend(resid_P)
            resid_S_all.extend(resid_S)
        
        ### Compute stat
        
        mean_P,std_P,rms_P,two_sigma_P=statistics(resid_P_all,type_stat)
        mean_S,std_S,rms_S,two_sigma_S=statistics(resid_S_all,type_stat)
        
        outdict.update({
                'mean_P':mean_P,'std_P':std_P,'rms_P':rms_P,'two_sigma_P':two_sigma_P,
                'mean_S':mean_S,'std_S':std_S,'rms_S':rms_S,'two_sigma_S':two_sigma_S,
                'num_P_all':num_P_all,'num_S_all':num_S_all
                        })
 
        ### Return 
        
        return outdict,resid_P_all,resid_S_all
        
    def plot_histo(self,station_code=None,phase_code=None):
        
        Ray=self.select_ray(station_code=None,phase_code=None)
        
        
        
        ### Get stat
        
        indict,resid_p,resid_s=Ray.get_stat()
        
        ### Figure

        #fig_title=r'Time residuals ($t_{obs}$ - $t_{theo}$) for station(s): %s'%(','.join(station_code))
        plt.figure(figsize=[8.2,6.55])
        xbins=np.linspace(-0.3,0.3,30)
        
        ax = plt.subplot(2, 1,1)
        ax.hist(resid_p,xbins,color='0.75',edgecolor='white')
        ax.text(0.1, 0.8, 'mean = {:.3f}\n$\sigma$ = {:.3f}\n$2\sigma$ = {:.3f}'.format(
                indict['mean_P'], indict['std_P'],indict['two_sigma_P']),transform=ax.transAxes,horizontalalignment='left')
        ax.set_xlabel('P Residuals [s]')
        ax.set_ylabel('Number obs')
        #ax.set_title(fig_title)
        ax.set_xlim(xbins[0],xbins[-1])
        ax = plt.subplot(2, 1,2)
        ax.hist(resid_s,xbins,color='0.75',edgecolor='white')
        ax.text(0.1, 0.8, 'mean = {:.3f}\n$\sigma$ = {:.3f}\n$2\sigma$ = {:.3f}'.format(
                indict['mean_S'], indict['std_S'],indict['two_sigma_S']),transform=ax.transAxes,horizontalalignment='left')
        ax.set_ylabel('Number obs')
        ax.set_xlabel('S Residuals [s]')
        ax.set_xlim(xbins[0],xbins[-1])
        
        #fig.savefig(output_figure)
        
    def select_ray(self,station_code=None,phase_code=None):
        """ Function made to select a subcatalog based on stations and phase codes
            station_code=[1,2,3,4]
            phase_code=[1] # only P

        """
    
        ### Handle default 
            
        if station_code is None:
            station_code=list(range(1,1000))
        
        if phase_code is None:
            phase_code=list(range(1,3)) # 2 elemnents in it, P or S
            
        ### Put paramerters in list
        
        if not isinstance(station_code,list):
            station_code=[station_code]
        
        if not isinstance(phase_code,list):
            phase_code=[phase_code]
        
        ### Read ray file
        
        Ray=self
        NewRay=deepcopy(self)
        events=Ray.events
        
        ### Start loop
        
        events_select=[]
        initial_obs=0
        final_obs=0
        
        for event in events:
            phases=event.phases
            
            ### Counter
            
            initial_obs=initial_obs+len(phases)
            
            ### Duplicate event
            
            tmp_event=deepcopy(event)
            
            ### Conditions 

            phases_select=[phase for phase in phases 
                           if phase.station in station_code
                           if phase.type in phase_code]
            
            if not phases_select:
                continue
            
            final_obs=final_obs+len(phases_select)
            tmp_event.phases=phases_select
            tmp_event.num_phase=len(phases_select)
            events_select.append(tmp_event)
            
        ### Print
        
        logging.info('Initial number of obs = %i'%initial_obs)
        logging.info('  Final number of obs = %i'%final_obs)
        
        ### Assign to new Catalog 
            
        NewRay.events=events_select
        
        return NewRay
    
    def in_box(self,center,angle_deg,len_prof,width_prof,map_xlim=None,map_ylim=None,flag_plot=False):
        
        NewCat=deepcopy(self)
        NewCat.events=[]
    
        for event in self.events:
            data=np.array([[event.x,event.y,1]])
            _,bool_sel=projection.project(data,center,angle_deg,len_prof,width_prof)
            if bool_sel[0]:
                NewCat.events.append(event)
                
        ### Plot if asked
        
        if flag_plot:
            fig,ax=self.plot_map(map_xlim=map_xlim,map_ylim=map_ylim)
            NewCat.plot_map(fig=fig,ax=ax,color='r',map_xlim=map_xlim,map_ylim=map_ylim)
            
        return NewCat        
        
    
    def get_statinfo(self,station_code=None):
        """function made to return statistical infos for all the stations 
        defined in station_code=[1,2,4] for example
        """
        
        ### Handle default 
            
        if station_code is None:
            station_code=self.__default_station_code()
      
        ### Start process through all functions

        info_stat=[]
        
        for kk in range(len(station_code)):
            
            tmp_stacode=station_code[kk]
            Tmp_Cat=self.select_ray(tmp_stacode)
            tmp_stat,_,_=Tmp_Cat.get_stat()
            
            if tmp_stat['num_P_all']==0 and tmp_stat['num_S_all']==0:
                logging.warning("No obs for station %i"%(tmp_stacode))
                continue
            
            tmp_stat['station_code']=tmp_stacode
            
            info_stat.append(tmp_stat)

        ### Save if asked

        ### Return
        
        return info_stat
    
    def plot_statinfo(self,station_code=None,fig_out=None):
        """
        Function made to plot all the statistics of the stations
        """ 
        
        ### Get statistics
        
        A=self.get_statinfo(station_code)

        ### Prepare Plot
        
        bar_width=0.35
        station_list=[tmp['station_code'] for tmp in A]
        
        x_bar=np.arange(1,len(station_list)+1)
        
        y_bar_nump=[tmp['num_P_all'] for tmp in A]
        y_bar_nums=[tmp['num_S_all'] for tmp in A]
        
        y_plot_meanp=[tmp['rms_P'] for tmp in A]
        y_plot_means=[tmp['rms_S'] for tmp in A]
        
        names=self.code2station(station_list)
   
        ### Plot
        
        fig=plt.figure()
        
        ax=fig.add_subplot(211)
        
        ax.set_xticks(x_bar)
        ax.set_xticklabels(names)
        ax.bar(x_bar-bar_width/2,y_bar_nump,bar_width,align='center',alpha=0.4,label='P')
        ax.bar(x_bar+bar_width/2,y_bar_nums,bar_width,color='r',align='center',alpha=0.4,label='S')
        
        ax.set_ylabel('Number obs')
        ax.legend()
        axlim=ax.get_xlim()
        
        ax=fig.add_subplot(212)
        ax.set_xticks(x_bar)
        ax.plot(axlim,[0,0],':',color='0.8')
        ax.xaxis.grid(True)
        
        ax.set_xticklabels(names)
        ax.plot(x_bar,y_plot_meanp,'bo-',alpha=0.4)
        ax.plot(x_bar,y_plot_means,'ro-',alpha=0.4)
        ax.set_xlabel('Stations code')
        
        ax.set_ylabel('RMS [s]')
        ax.set_xlim(axlim)

        
        ### save if asked
        
        if fig_out:
            logging.info('Print figure to %s'%(fig_out))
            plt.savefig(fig_out)


    def code2station(self,station_code):
        """
        Convert code to station realname
        """
        
        station_list=[]
        
        if self.stations_realname is None:
            print('Station conversion from code to realname not possible'
                  'attribute station_realnames not defined')
            station_list=[str(code) for code in station_code]
            return station_list

        for code in station_code:
            for keys in self.stations_realname.keys():
                if self.stations_realname[keys]==code:
                    station_list.append(keys)
        
        return station_list
    
#########################################
        ### Plot Methods
        
    def conv_station(self):
        """
        Function made to output x,y ftom lon/lat
        """
        
        if (self.station_file is None or
            self.ini_lat is None or
            self.ini_lon is None):
            print('sdfs')
            return
        
        A=lotos_util.read_station(self.station_file)

        #### Get lon,lat 
        
        lon=[]
        lat=[]
        
        for station in A:
            lon.append(station.longitude)
            lat.append(station.latitude)
            
            
        x,y=projection.ll2xy(lon,lat,self.ini_lon,self.ini_lat)
        
        return x,y
        
         
    def plot_map(self,fig=None,ax=None,map_xlim=None,map_ylim=None,color='0.9'):
        """
        Function made to plot the locations given in ray file in map view(x,y)
        """
        
        
        x_sta,y_sta=self.conv_station()
            
        ### Get locations
        
        x,y,z=self.get_xyz()        
        logging.info('%i locations on map'%len(x))        
        data=np.column_stack((x,y,z))
        
        ### Plotting
        
        ### Check figure status
        
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            fig = ax.get_figure()
        elif ax is None:
            ax = fig.gca()
        
        #### Start Plotting
        ax.plot(data[:,0],data[:,1],'o',color=color,mec='k',alpha=1,markersize=3,mew=0.5)
        ax.plot(x_sta,y_sta,'^',markersize=10,color='w',markeredgecolor='k')
        
        #### Plot lines files
        
        for line in self.lines:
            ax.plot(line[:,0],line[:,1],linestyle=':',color='k',lw=1.5)

        
        if map_xlim is not None:
            ax.set_xlim(tuple(map_xlim))
        if map_ylim is not None:
            ax.set_ylim(tuple(map_ylim))
        
        plt.axes(ax)
        plt.axis('scaled')
        plt.title('%i events on map'%(len(x)))
        
        return fig,ax
    
    def plot_cross(self,center,angle_deg,len_prof,width_prof,fig=None,ax=None,
                   output_fig=None,map_xlim=None,map_ylim=None,**kwargs):
        """
        Made to plot cross sections
        """
        
        
        ### Read surfaces and lines
        
        self.read_surf()
        self.read_line()
        
        ### Get locations
        
        x,y,z=self.get_xyz()        
        logging.info('%i locations on map'%len(x))        
        data=np.column_stack((x,y,z))
        
        ### Profile
        
        x_prof,y_prof=getprojline(center,angle_deg,len_prof)
        
        #### Do the projection
        
        proj_data,select_boolean=projection.project(data,center,angle_deg,len_prof,width_prof)
        
        data_select=data[select_boolean,:]
        
        #### Project suface if not empty
        
        if self.surf:
            proj_surf=[]
            for surface in self.surf:
                tmp_proj_surf,_=projection.project(surface,center,angle_deg,len_prof,width_prof)
                
                #### Smooth
                
                x_surf,y_surf=math.average_data(tmp_proj_surf[:,0],tmp_proj_surf[:,2],num_points=100,mode='max',flag_plot=False)
                tmp_proj_surf=np.column_stack((x_surf,y_surf))
                proj_surf.append(tmp_proj_surf)
            
        #### Project lines crossing as points
        
        if self.lines:
            proj_line=[]
            for line in self.lines:
                line=np.column_stack((line,np.zeros(line.shape[0])))
                tmp_proj_line,_=projection.project(line,center,angle_deg,len_prof,width_prof)
                
                #### Cluster the points
                
                x_line=math.cluster_1d(tmp_proj_line[:,0],1,flag_plot=False)
                tmp_proj_line=np.column_stack((x_line,np.zeros(x_line.shape)))
                
                proj_line.append(tmp_proj_line)
                
        
    
        z_lim=kwargs.get('z_lim')
    
        
        #### Plot Map
        
        if fig is None and ax is None:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 8))
        elif fig is None:
            fig = ax[0].get_figure()
        elif ax is None:
            plt.figure(fig.number)
            fig, (ax1,ax2) = plt.subplots(1,2)
            
            
        self.plot_map(fig,ax1,map_xlim=map_xlim,map_ylim=map_ylim)        
        
    
        plt.plot(data_select[:,0],data_select[:,1],'or',mec='k',alpha=1,markersize=3,mew=0.5)
        plt.plot(x_prof,y_prof,'k',lw=1)
        plt.plot(x_prof,y_prof,'k',marker=(2,0,angle_deg),lw=1)
        plt.plot(center[0],center[1],'k',marker=(2,0,angle_deg),lw=2)
        x_sta,y_sta=self.conv_station()
        plt.plot(x_sta,y_sta,'^',markersize=10,color='w',markeredgecolor='k')

        if map_xlim is not None:
            ax1.set_xlim(tuple(map_xlim))
        if map_ylim is not None:
            ax1.set_ylim(tuple(map_ylim))
        
        ### Plot Cross section
        
        ### Plot data
        
        plt.axes(ax2)
        

        plt.plot(proj_data[:,0],proj_data[:,2],'or',mec='k',alpha=1,markersize=3,mew=0.5)
        
        ### Plot surface if any
        
        if proj_surf:
            for data_surf in proj_surf:
                plt.plot(data_surf[:,0],data_surf[:,1],':k',lw=1)
                
        ### Plot lines crossing if any
        
        if proj_line:
            for data_line in proj_line:
                plt.plot(data_line[:,0],data_line[:,1],'v',mfc='k',markersize=10,mec='k')

 
        ax2.grid('on')
        ax2.axis('scaled')
        ini_ylim=ax2.get_ylim()
        ax2.set_ylim([0,ini_ylim[1]])
       
        
        
        if z_lim:
            ax2.set_ylim([0,z_lim])
            
        ax2.invert_yaxis()
  
      
        #ax2.invert_yaxis()
        
        plt.xlabel('X [km]')
        plt.ylabel('Z [km]')
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        plt.tight_layout()
        
        ### save if asked
        
        if output_fig:
            plt.savefig(output_fig)
        
        
        return fig,[ax1,ax2]
    
    def get_t_wadati(self):
        """
        Method made to get tobs times for both P and S waves if both obs
        are present for a same station
        
        Results
        -------
        
        tp : np.ndarray
        ts : np.ndarray
        """
                
        tp=[]
        ts=[]
        st=[]
        
        #for event in A.events:
        
        for event in self.events:
        
            un_station=[x.station for x in event.phases]
            un_station=list(set(un_station))
            
            for station in un_station:
                phase_P=event.get_phase(station,1)
                phase_S=event.get_phase(station,2)
                
                if (phase_P is not None) and (phase_S is not None):
                    tp.append(phase_P.t_obs)
                    ts.append(phase_S.t_obs)
                    st.append(station)
                    
        tp=np.array(tp)
        ts=np.array(ts)
        st=np.array(st)

        return tp,ts,st
    
    def get_dt_wadati(self,absolute=False):
        """
        Method made to get delta times for both P and S waves if both obs
        are present for a same station, this is useful to plot
        the modified wadati diagram 
        Pierri, P., de Lorenzo, S., and Calcagnile, G., 2013, 
        Analysis of the Low-Energy Seismic Activity in the Southern Apulia (Italy): 
            Open Journal of Earthquake Research, v. 2, p. 91–105, 
            doi: 10.4236/ojer.2013.24010.
        
        Results
        -------
        
        dtp : np.ndarray
        dts : np.ndarray
        min_delta : np.ndarray
        """
        
        dtp=[]
        dts=[]
        min_delta=[]
        
        for event in self.events:
        
            un_station=[x.station for x in event.phases]
            un_station=list(set(un_station))
            un_station.sort()
        

            tpp=[]
            tss=[]
          
            
            for station in un_station:
                phase_P=event.get_phase(station,1)
                phase_S=event.get_phase(station,2)
                
                if (phase_P is not None) and (phase_S is not None):
                    tpp.append(phase_P.t_obs)
                    tss.append(phase_S.t_obs)
                    
                else:
                    continue
            
            #### Do the combinations
            ind_combi=list(itertools.combinations(list(range(len(tpp))), 2))
            
            for index in ind_combi:
                dpp=tpp[index[0]]-tpp[index[1]]
                dss=tss[index[0]]-tss[index[1]]
                delta_1=tss[index[0]]-tpp[index[0]]
                delta_2=tss[index[1]]-tpp[index[1]]
                delta=np.min([delta_1,delta_2])
                dtp.append(dpp)
                dts.append(dss)
                min_delta.append(delta)
                    
                
        dtp=np.array(dtp)
        dts=np.array(dts)
        if absolute:
            dtp=np.abs(dtp)
            dts=np.abs(dts)
        min_delta=np.array(min_delta)
        
        return dtp,dts,min_delta
                            
#############################################
####### PRIVATE METHODS #####################    
    
    def __default_station_code(self):
        """
        Function made to return the list of all stations in the Catalog
        """
        b=[phase.station for event in self.events for phase in event.phases]
        default_station=list(np.unique(np.array(b)))
        
        return default_station


def getprojline(center,angle_deg,len_prof):
    
    end_prof_x=center[0]+len_prof[1]*np.cos(angle_deg*np.pi/180)
    end_prof_y=center[1]+len_prof[1]*np.sin(angle_deg*np.pi/180)

    start_prof_x=center[0]-len_prof[0]*np.cos(angle_deg*np.pi/180)
    start_prof_y=center[1]-len_prof[0]*np.sin(angle_deg*np.pi/180)
    
    x_prof=[start_prof_x,end_prof_x]
    y_prof=[start_prof_y,end_prof_y]
    
    logging.debug('Profile parameters %s %s'%(str(x_prof),str(y_prof)))
    
    return x_prof,y_prof
        


            
def statistics(data_array,type_stat=None):
    """
    Problem needs to be changed as the formula for absolute values is false,
    please see half folded ditributions
    """
    
    if len(data_array)==0:
        mean_D,std_D,rms_D,two_sigma_D=0,0,0,0
        return mean_D,std_D,rms_D,two_sigma_D

    if type_stat=='abs':
        data_array=np.abs(data_array)
        
    mean_D=np.mean(data_array)
    #mean_D=np.median(data_array)
    std_D=np.std(data_array)
    rms_D=np.sqrt(np.mean(np.array(data_array)**2))
    two_sigma_D=((np.percentile(data_array,97.72)-np.percentile(data_array,2.28))-mean_D)/2
        
    logging.info('mean=%.3f,sigma=%.3f,2sigma=%.3f'%(mean_D,std_D,two_sigma_D))
    return mean_D,std_D,rms_D,two_sigma_D
    
        

class Event(object):
    
    def __init__(self):
        """
        """
        self.phases=[]
        self.x=None
        self.y=None
        self.z=None
        self.num_phase=None
        self.id=None
        self.ot=None
        
    def read_header(self,header):
        
        self.x,self.y,self.z,self.num_phase=\
        [float(x) for x in header.split()]
    
    def write(self,fic):
        
        format_str='%.4f %.4f %.4f %i\n'
        fic.write(format_str%(self.x,self.y,self.z,self.num_phase))
        
        for phase in self.phases:
            phase.write(fic)
        
    def __repr__(self):
        return "X:%10.3f Y:%10.3f Z:%7.3f id=%s" % (self.x, self.y,self.z,self.id)
        
    def get_stat(self):
        
        phases=self.phases[:]
        num_P=0
        num_S=0
        
        resid_P=[]
        resid_S=[]
        resid_all=[]
        
        for phase in phases:
            resid=phase.t_obs-phase.t_tho
            resid_all.append(resid)
            if phase.type==1:
                resid_P.append(resid)
                num_P+=1
            else:
                resid_S.append(resid)
                num_S+=1
        
        return resid_P,num_P,resid_S,num_S,resid_all
    
    def get_phase(self,station_key,type_key):

        for phase in self.phases:
            if (phase.station==station_key) and (phase.type==type_key):
                return(phase)
                    
            
        
        
class Phase(object):
    
    def __init__(self,line=None):
        """
        Define attributes
        """
        self.type=None
        self.station=None
        self.t_obs=None
        self.t_tho=None
        self.read(line)
        
    def read(self,phase_string):
        
        if phase_string is None:
            return
        
        if len(phase_string.split())==4:
            self.type,self.station,self.t_obs,self.t_tho=\
            [float(x) for x in phase_string.split()]
        else:
            self.type,self.station,self.t_obs=\
            [float(x) for x in phase_string.split()]
            
    def write(self,fic):
        """
        write into opened file object
        """
        
        if self.t_tho is not None:
            format_str='%4i %4i %.4f %.4f\n'
            fic.write(format_str%(self.type,self.station,self.t_obs,self.t_tho))
        else:
            format_str='%4i %4i %.4f\n'
            fic.write(format_str%(self.type,self.station,self.t_obs))

    def __repr__(self):
        if self.t_tho is not None:
            if isinstance(self.station,str):
                 return "Phase:%1d Station:%3s t_obs: %10.3f t_tho: %10.3f" \
             % (self.type, self.station,self.t_obs,self.t_tho)
            else:
                 return "Phase:%1d Station:%3d t_obs: %10.3f t_tho: %10.3f" \
             % (self.type, self.station,self.t_obs,self.t_tho)
        else: 
            return "Phase:%1d Station:%3d t_obs: %10.3f" \
        % (self.type, self.station,self.t_obs)
            
            
        



