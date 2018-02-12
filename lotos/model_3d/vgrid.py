#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:18:51 2017

@author: baillard
"""

import numpy as np
import struct
import logging
import copy

import matplotlib.pyplot as plt
from lotos.model_1d import util as util1d
from lotos.model_3d import util as util3d
from general import cube,projection
from lotos import lotos_util 


from nlloc import util as nllocutil 

class VGrid(object):
    
    def __init__(self,num_elem=None):
        """
        Define attributes
        """

        self.name='hello'
        self.coord_sys='xy'
        self.lonlat_o=[-130.1,45.9]
        self.data=np.array([])
        self.mask=None
        self.num_elem=num_elem
        ### Initialize grid spec in dict
        self.grid_spec=dict.fromkeys(['xo','yo','zo','dx','dy','dz','nx','ny','nz'])
        self.stations=None
                 
        
    def write_lotos_header(self,output_file):
        """
        Write lotos header to file
        """
        
        fic=open(output_file,'wb')
        ### First write header
        fic.write(struct.pack('f',self.grid_spec['xo']))
        fic.write(struct.pack('i',self.grid_spec['nx']))
        fic.write(struct.pack('f',self.grid_spec['dx']))
        
        fic.write(struct.pack('f',self.grid_spec['yo']))
        fic.write(struct.pack('i',self.grid_spec['ny']))
        fic.write(struct.pack('f',self.grid_spec['dy']))
        
        fic.write(struct.pack('f',self.grid_spec['zo']))
        fic.write(struct.pack('i',self.grid_spec['nz']))
        fic.write(struct.pack('f',self.grid_spec['dz']))
        ### Close file
        fic.close()
        
    def write(self,output_file,output_type=None,phase='P',unit='VELOCITY'):
        """
        Write to file, choose between 'txt' or 'bin' or 'lotos' or 'nlloc'
        """
        
        if output_type=='lotos':
            ### Check first that grid_spec elements do not contain None (otherwise can't write lotos type)
            
            if None in [self.grid_spec[key] for key in self.grid_spec]:
                print('Grid specificators are not implemented, could not write in lotos format, abort')
                return
            
            ### Rearrange DATA to follows Lotos
            nx,ny,nz=self.grid_spec['nx'],self.grid_spec['ny'],self.grid_spec['nz']
            v=cube.xyz2matrix(nx,ny,nz,self.data[:,3])
            new_v=np.empty(nx*ny*nz)
            k=0
            for iz in range(0,nz):
                for iy in range(0,ny):
                    for ix in range(0,nx):
                        new_v[k]=v[ix,iy,iz]
                        k=k+1
        
            ### Write header
            self.write_lotos_header(output_file)
            ### Write array containing velocities
            fic=open(output_file,'ab')
            new_v.astype(dtype=np.float32).tofile(fic)
            fic.close()
        elif output_type=='nlloc':
            
            print('Make sure to choose the proper unit SLOW_LEN, VELOCITY...')
            
            prefix_file=copy.deepcopy(output_file)
            output_file='{}.{}.mod.buf'.format(output_file,phase)
            
            ### Write NLLOC header file
            
            nllocutil.write_nlloc_header(self.grid_spec,self.lonlat_o[0],self.lonlat_o[1],prefix_file,phase,unit)
                        
            ### Write NLLOC buf file
            
            fic=open(output_file,'wb')
            self.data[:,3].astype(dtype=np.float32).tofile(fic)
            fic.close()
        else:    
            util3d.write(self.data,output_file,output_type=output_type)
        
    def plot_slice(self,slice_param,c_center=None,coef_std=None,color_range=None,ax=None,cmap=plt.cm.get_cmap('jet_r')):
        """
        Plot cross section, slice_param must be "x=4"
        """

        (ax,h1)=cube.plot_slice(self.data,
                                 self.grid_spec['nx'],self.grid_spec['ny'],self.grid_spec['nz'],slice_param,c_center=c_center,
                                 ax=ax,color_range=color_range,cmap=cmap,coef_std=coef_std)
        
        
        #### ADD CUBE.PLOT_CONTOUR HERE IF MASK EXISTS
        return (ax,h1)
        
    def plot_contour(self,slice_param,ax=None,filled=False,**kwargs):
        """
        Plot contour
        """
        
        (ax,h1)=cube.plot_contour(self.data,
                                 self.grid_spec['nx'],self.grid_spec['ny'],self.grid_spec['nz'],slice_param,filled=filled,ax=ax,**kwargs)
        
        return (ax,h1)
    
    def write_slice(self,slice_param,output_file):
        """
        Write cross section to 3 columns X,Y,Z file
        """
        
        cube.write_slice(self.data,
                                 self.num_elem[0],self.num_elem[1],self.num_elem[2],slice_param,
                                 output_file)
        
    def xy2ll(self,ini_lon,ini_lat):
        """
        Convert data from xy to lon lat  
        """
        
        x,y=self.data[:,0],self.data[:,1]
        
        lon,lat=projection.xy2ll(x,y,ini_lon,ini_lat)
        
        self.data[:,0]=lon
        self.data[:,1]=lat
        
    def ll2xy(self,ini_lon,ini_lat):
        """
        Convert data from xy to lon lat  
        """
        
        lon,lat=self.data[:,0],self.data[:,1]
        
        x,y=projection.ll2xy(lon,lat,ini_lon,ini_lat)
        
        self.data[:,0]=x
        self.data[:,1]=y
    
    def extend_data_z(self,new_zo,location='top'):
        """
        Function made to extend the 3D cube at top or bottom 
        The function takes the last layer at top or bottom and repeats it a certain 
        times to reach the new_zo
        """
        
        nx,ny,nz=self.grid_spec['nx'],self.grid_spec['ny'],self.grid_spec['nz']
        zo=self.grid_spec['zo']
        dz=self.grid_spec['dz']
        data=self.data
        
        ### Check
        
        if new_zo>zo:
            raise ValueError('new_origin must be below(above) zo if top(bottom) chosen')
        
        ### Read
        xi,yi,zi=cube.get_xyz(nx,ny,nz,data)
        mv=cube.xyz2matrix(nx,ny,nz,data[:,3])

        ### Get number of slices to add
        
        nz_add=np.int((zo-new_zo)/dz)
        new_zo=zo-nz_add*dz
        nz_tot=nz_add+nz
        zi=np.arange(new_zo,new_zo+(nz_add+nz)*dz,dz)
        
        ### Get last slice
        
        if location=='top':
            slice_z=mv[:,:,0]
        else:
            slice_z=mv[:,:,-1]
        
        ### Repeat slice to get cube
        
        cube_add=np.repeat(slice_z[:, :, np.newaxis],nz_add,axis=2)
        
        ### Concatenate
        
        cube_new=np.concatenate((cube_add,mv),axis=2)
        
        ### transform to 1D array
        
        X_new,Y_new,Z_new=np.meshgrid(xi,yi,zi,indexing='ij') ## very important so that it works with reshape
         
        new_data=np.reshape(cube_new,(-1,1),order='C')
        x_final=np.reshape(X_new,(-1,1),order='C')
        y_final=np.reshape(Y_new,(-1,1),order='C')
        z_final=np.reshape(Z_new,(-1,1),order='C')
                     
        data_final=np.column_stack((x_final,y_final,z_final,new_data))
        
        self.data=data_final
        
        self.grid_spec.update({'nz':nz_tot,'zo':new_zo})
    
    def get_1D_velocity(self,xo,yo,radius,flag_plot=None,fig=None,labels=None):
        """
        Returns the velocity values under the block
        
        Parameters
        ----------
        
        xo,yo: float or list
            x and y coordinates where to extract the profile
            
        radius: float
            radius around the center to mean the velocity values
            
        flag_plot,fig: Boolean
            Want to plot and save into the filename
            
        Returns
        -------
        
        depth_array: np.ndarray
            depth values
        
        vel_array: np.ndarray
            velocity values
        """
        
        if isinstance(xo,(float,int)):
            xo=np.array([xo])
        else:
           xo=np.array(xo)
           
        if isinstance(yo,(float,int)):
            yo=np.array([yo])
        else:
           yo=np.array(yo) 
           
        if isinstance(radius,(float,int)):
            radius=np.array([radius])
        else:
           radius=np.array(radius) 

        
        ### Get parameters
        
        x=self.data[:,0]
        y=self.data[:,1]
        z=self.data[:,2]
        v=self.data[:,3]
        
        x_range,y_range,z_range=get_grid_range(self.grid_spec)
        
        if len(radius)==1:
            radius=np.ones(np.shape(xo))*radius
            
        
        ### Select
        
        ### Initialize
        
        velocities_arr=np.array([])
        depth_arr=np.array([])
        
        for j in np.arange(len(xo)):
            
            xo_single=xo[j]
            yo_single=yo[j]
            radius_single=radius[j]
            
            print(j,xo_single,yo_single,radius_single)
            x_sel,y_sel,boolean=projection.is_in_circle(x,y,xo_single,yo_single,radius_single,False)
        
            z_sel=z[boolean]
            v_sel=v[boolean]
            
            velocities=np.array([])
            depth=np.array([])
          
            for i,z_val in enumerate(z_range):
                if i>=len(z_range)-1:
                    bool_z=np.logical_and(z_sel>z_range[i-1],z_sel<=z_range[i])
                else:
                    bool_z=np.logical_and(z_sel>=z_range[i],z_sel<z_range[i+1])
                #print(i)
                velocity=v_sel[bool_z]
                velocities=np.append(velocities,np.mean(velocity))    
                depth=np.append(depth,z_val)
                
            if j==0:
                velocities_arr=velocities
            else:
                velocities_arr=np.column_stack((velocities_arr,velocities))
            depth_arr=depth
            
        ### Plot if asked
     
        if flag_plot:
            

            if fig==None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            else:
                ax=fig.axes[0]
                
            if velocities_arr.ndim==1:
                if labels:
                    ax.step(velocities_arr,depth,linewidth=1.5,c='r',label=labels[0])
                else:
                    ax.step(velocities_arr,depth,linewidth=1.5,c='r',label='xo = {:.1f}, yo= {:.1f}'.format(xo[0],yo[0]))
            else:
                
                cmap = plt.cm.get_cmap('winter', velocities_arr.shape[-1])
                
                for k in range(velocities_arr.shape[-1]):
                    rgb = cmap(k)[:3]
                    if labels:
                        ax.step(velocities_arr[:,k],depth,linewidth=1.5,c=rgb,label=labels[k])
                    else:
                        ax.step(velocities_arr[:,k],depth,linewidth=1.5,c=rgb,label='xo = {:.1f}, yo= {:.1f}'.format(xo[k],yo[k]))
                    
                    
            plt.legend()
            ax.set_ylim([0,4])
            
            ax.set_ylabel('Depth [km]')
            ax.set_xlabel('V [km/s]')
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 


        return depth_arr,velocities_arr,fig
    
    def write_1D_velocity(self,xo,yo,radius,output_file,format_file='txt'):
        """
        Write Velocity to file
        """
        
        if (not isinstance(xo,(float,int))) and (not isinstance(yo,(float,int))):
            raise ValueError('xo and yo should be float')
            
        depth,velocity,_=self.get_1D_velocity(xo,yo,radius)
        
        #### Stack elements
        
        data=np.column_stack((depth,velocity))
        
        #### Write to file
        
        if format_file=='lotos':
            util1d.write_1D_lotos(data,output_file)
        elif format_file=='txt':
            np.savetxt(output_file,np.column_stack((depth,velocity)),fmt='%.3f %.3f')
            
            
        ### Logging 
        logging.info('Printing velocity to file %s'%output_file)
        
        return
    
        
        
    def extend_1D_z(self,z,value):
        """
        Function made to extend the 1D velocity laterally to a 3D cube that has
        the same size as matrix self.data$. It returns a layer cake
        
        """
        
        _,_,zi=get_grid_range(self.grid_spec)
            
        ### interpolate
        
        valuei=np.interp(zi,z,value)
        
        nx=self.grid_spec['nx']
        ny=self.grid_spec['ny']
        new_cube=np.tile(valuei,(nx,ny,1))
        
        return new_cube
    
    def get_mean(self,axis='z'):
        """
        Function made to build a cube that is the same size as the initial cube
        but it will take the mean for each layer, depending on the axis given
        it handles nans
        
        Parameters
        ----------
        
        axis: {'x','y','z'}
            axis along which the mean will be computed
            
        """
        
        #### Initialize new grid
        
        NewGrid=copy.deepcopy(self)
        
        #### Choose axis
        
        dic_axis={'x':0,'y':1,'z':2}
        axis=dic_axis.get(axis,None)
        if axis ==None:
            raise ValueError('Wrong parameter given')
         
        #### Copy
        
        v_cube=self.data2matrix()
        v_cube_mean=copy.deepcopy(v_cube)
        
        for kk in range(v_cube.shape[axis]):
            if axis==0:
                slice_array=v_cube[kk,:,:]
                if np.isnan(slice_array).all():
                    continue
                val_mean=np.nanmean(slice_array)
                slice_mean=np.ones(slice_array.shape)*val_mean
                v_cube_mean[kk,:,:]=slice_mean
            elif axis==1:
                slice_array=v_cube[:,kk,:]
                if np.isnan(slice_array).all():
                    continue
                val_mean=np.nanmean(slice_array)
                slice_mean=np.ones(slice_array.shape)*val_mean
                v_cube_mean[:,kk,:]=slice_mean
            elif axis==2:
                slice_array=v_cube[:,:,kk]
                if np.isnan(slice_array).all():
                    continue
                val_mean=np.nanmean(slice_array)
                slice_mean=np.ones(slice_array.shape)*val_mean
                v_cube_mean[:,:,kk]=slice_mean
                
        #### update data
        
        NewGrid.matrix2data(v_cube_mean)
        
        ### Return
        
        return NewGrid       
        
    
    def apply_operator(self,fz,operator):
        """
        This function is made to transform the cube by applying another scalar,2*N array [z,values] or VGrid to it
        For example, given vp/vs(z) you can transform the Vp cube into a Vs cube
        of course if its a Vrgrid, it should have the same size as the previous one
        
        Parameters
        ----------
        
        fz: int,float,ndarray,Vgrid
            if fz a 2*N numpy array then 
            depth (don't need to be the same size as the cube, values will be interpolated linearly)
            and values to be applied to the cube [depths values]
        operator : {'add','sub','mul','div'}
            operator to be applied to the cube
        """
    
        ### Build dictionary
        
        dic_operator={'add':'+','sub':'-','mul':'*','div':'/'}
        
        ### Copy initial grid
    
        NewGrid=copy.deepcopy(self)
        
        ### Load one d file
        
        ini_cube=self.data2matrix()
            
        ### Interpolate operator
        if isinstance(fz,(float,int)):
            operator_cube=fz*np.ones(ini_cube.shape)
        elif isinstance(fz,np.ndarray):
            operator_cube=self.extend_1D_z(fz[:,0],fz[:,1])
        elif isinstance(fz,VGrid):
            operator_cube=fz.data2matrix()
    
            
        ### Transform cube
        
        cmd_str='ini_cube'+dic_operator[operator]+'operator_cube'
        fin_cube=eval(cmd_str)
        
        #### update data
        
        NewGrid.matrix2data(fin_cube)
        
        ### Return
        
        return NewGrid
            
    def data2matrix(self,unpack=False):
        
        nx=self.grid_spec['nx']
        ny=self.grid_spec['ny']
        nz=self.grid_spec['nz']
        matrx=cube.xyz2matrix(nx,ny,nz,self.data[:,3])
        X=cube.xyz2matrix(nx,ny,nz,self.data[:,0])
        Y=cube.xyz2matrix(nx,ny,nz,self.data[:,1])
        Z=cube.xyz2matrix(nx,ny,nz,self.data[:,2])
        
        if unpack is True:
            return (X,Y,Z,matrx)
        else:
            return matrx
    
    def matrix2data(self,matrx):
        
        new_data=cube.matrix2xyz(matrx)
        self.data[:,3]=new_data
        
        return self.data
        
    def read_stations(self,station_file=None):
        """
        Read station file of form x,y....
        """
        
        if station_file is None:
            station_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/STATIONS/stat_xy.dat'
        
        coord=np.loadtxt(station_file)
        
        self.stations=coord
    
    def plot_stations(self,ax=None):
        
        
        if self.stations is None:
            print("No stations stored, read file first")
            return
        
        coord=self.stations
        
            
        if ax is None:
            fig, ax = plt.subplots()
        
        plt.sca(ax)
        
            
        h1=ax.plot(coord[:,0],coord[:,1],'^',markerfacecolor='None',markeredgecolor='w')
        
        return (ax,h1)
        
    def plot_lines(self,line_file=None,coord_sys='lonlat',ax=None,color='w',lw=2):
        """
        Function made to plot lines form a xyz file, x (or lon) and y (or lat) will always
        be taken from the the first two columns
        
        Parameters
        ---------
        
        line_file: str
            path to the file 
        coord_sys: {'lonlat','xy'}
            set if coordinates are given in lonlat or xy
        """
        ### 
        
        if line_file==None:
            line_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/DATA/GRIDS/caldera_smooth.ll'
            
        coord=np.loadtxt(line_file)
        
        x=coord[:,0]
        y=coord[:,1]
        
        #### Convert lon lat to xy
        
        if coord_sys=='lonlat':
            new_x,new_y=projection.ll2xy(x,y,self.lonlat_o[0],self.lonlat_o[1])
            
        #### Start plotting
        
        if ax is None:
            fig, ax = plt.subplots()
        
        plt.sca(ax)
            
        h1=ax.plot(new_x,new_y,color=color,lw=lw)
        
        return (ax,h1)
        
        
##### Define function other than methods
       
def get_grid_range(gridspec_obj):
    
    xo=gridspec_obj['xo']
    nx=gridspec_obj['nx']
    dx=gridspec_obj['dx']
    
    x=np.linspace(xo,xo+(nx-1)*dx,nx)
    
    yo=gridspec_obj['yo']
    ny=gridspec_obj['ny']
    dy=gridspec_obj['dy']
    
    y=np.linspace(yo,yo+(ny-1)*dy,ny)
    
    zo=gridspec_obj['zo']
    nz=gridspec_obj['nz']
    dz=gridspec_obj['dz']
    
    z=np.linspace(zo,zo+(nz-1)*dz,nz)
    
    return x,y,z
    
def read(input_file,format_type):
    """
    Read cube and generate a VGrid object
    
    format_type : {'bin','lotos'}
    input_file : str
        grid filename
    """
    
    allow_format=['lotos','bin']
    
    if format_type not in allow_format:
        raise ValueError('Format is not lotos or bin')
    
    ### Call empty VGrid object
    
    VG=VGrid()

    ### Read depending on format
    
    if format_type=='lotos':
        VG.data,nx,ny,nz=util3d.read_dv_LOTOS(input_file)
        VG.num_elem=[int(nx),int(ny),int(nz)]
        
    else:
        VG.data=util3d.read_binary(input_file)
        
        ### check size
        if VG.num_elem==None:
            print('Give nx,ny, and nz as list [nx,ny,nz]')
            return
    
        nx,ny,nz=VG.num_elem
        if nx*ny*nz != VG.data.shape[0]:
            print('Number of element dont match length of array: %i vs %i'%(nx*ny*nz,VG.data.shape[0]))
            return
        
    print('Size of loaded array is %i %i'%(VG.data.shape))

    
    ### Assign arrays ranges and stuff grid_spec
    
    xi,yi,zi=cube.get_xyz(nx,ny,nz,VG.data)
    
    xo,yo,zo=xi[0],yi[0],zi[0]
    dx=float('%.2f'%np.mean(np.diff(xi)))
    dy=float('%.2f'%np.mean(np.diff(yi)))
    dz=float('%.2f'%np.mean(np.diff(zi)))
    
    VG.grid_spec.update({'xo':xo,'yo':yo,'zo':zo,
                           'dx':dx,'dy':dy,'dz':dz,
                           'nx':nx,'ny':ny,'nz':nz})


    return VG   


def read_raypaths(input_file,grid_spec):
    """
    Function made to transform a raypaths file generated by lotos, which gives for each
    ray the number of nodes and their position, into a vgrid type of cube with density=number of times
    a bin is illuminated. The function uses raypathstxt to work
    Parameters
    -----------
    
    input_file: str
        lotos raypath file
    
    grid_spec: dic
        same as in VGrid.grid_spec, contains info on the grid
    """
    
    ### Check that grid range is ok
    
    VG=VGrid()
    
    if set(grid_spec.keys()) != set(VG.grid_spec.keys()):
        raise ValueError('grid_range does not contain the appropriate keys')
    
    
    ### Read input file into x,y,z array
    
    (x,y,z,_)=lotos_util.read_raypaths(input_file)
    data=np.column_stack((x,y,z))
    
    ## Get density per bin
    
    x_range,y_range,z_range=get_grid_range(grid_spec)
    
    ### Histo takes the edges, so if we want the histo being centered we have to shift arrays by dx/2
    x_histo=np.append(x_range-grid_spec['dx']/2,x_range[-1]+grid_spec['dx']/2) 
    y_histo=np.append(y_range-grid_spec['dy']/2,y_range[-1]+grid_spec['dy']/2)
    z_histo=np.append(z_range-grid_spec['dz']/2,z_range[-1]+grid_spec['dz']/2)
    
    V,edges=np.histogramdd(data, (x_histo,y_histo,z_histo))
    
    X,Y,Z=np.meshgrid(x_range,y_range,z_range,indexing='ij')
    
    ### Rebuil data that can be stored into .data
    new_v=cube.matrix2xyz(V)
    new_x=cube.matrix2xyz(X)
    new_y=cube.matrix2xyz(Y)
    new_z=cube.matrix2xyz(Z)
    
    new_data=np.column_stack((new_x,new_y,new_z,new_v))
    
    ### Return a VGrid class
    
    density_cube=VGrid()
    density_cube.grid_spec=grid_spec
    density_cube.data=new_data
    
    return density_cube

        
        
        