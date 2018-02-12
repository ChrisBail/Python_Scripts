#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:40:09 2017

@author: baillard
"""

import numpy as np
import struct

from general import cube



def read_binary(input_binary):
    """
    Read simple binary file to data with x,y,z,v
    """
    
    data = np.fromfile(input_binary, dtype=np.float32, count=-1)
    data=np.reshape(data,(-1,4))
    return data


def write(data,output_file,output_type=None):
    
    if output_type is None:
        output_type='bin'
        
    if output_type=='bin':
        fic=open(output_file,'wb')
        np.reshape(data,(1,-1)).astype(dtype=np.float32).tofile(fic)
        fic.close()
    elif output_type=='txt':
        np.savetxt(output_file,data,fmt='%.3f %.3f %.3f %.3f')
    else:
        print('Wrong output_type')


def read_dv_LOTOS(dv_binary_file):
    """
    function made to read dv binary files in lotos format, LOTOS file contains 
    header with grid indications, xo, nx,dx see manual for details
    """
    k=0
    data = open(dv_binary_file,"rb").read()
    
    byf=[]
    for i in range(0, len(data), 4):
            scr=data[i:i+4]
            byf.append(scr)
    # Read x header
    
    xo = struct.unpack('f', byf[0])[0] 
    nx = struct.unpack('i', byf[1])[0]
    dx = struct.unpack('f', byf[2])[0] 

    
    # Read y header
    yo = struct.unpack('f', byf[3])[0] 
    ny = struct.unpack('i', byf[4])[0]
    dy = struct.unpack('f', byf[5])[0] 
    
    
    # Read z header
    zo = struct.unpack('f', byf[6])[0] 
    nz = struct.unpack('i', byf[7])[0]
    dz = struct.unpack('f', byf[8])[0] 
    
    # Read Data
    
    new_dv=np.empty((nx,ny,nz))
    
    dv=[]
    for i in range(9,len(byf)):
        
        dv.append(struct.unpack('f', byf[i])[0])
    
    
    dv=np.array(dv)
    # Put into correct order
    
    k=0
    while k<len(dv):
        for iz in range(0,nz):
            for iy in range(0,ny):
                for ix in range(0,nx):
                    new_dv[ix,iy,iz]=dv[k]
                    k=k+1
                    
    ### Reassign in good order
    
    xi=np.linspace(xo,xo+(nx-1)*dx,nx)
    yi=np.linspace(yo,yo+(ny-1)*dy,ny)
    zi=np.linspace(zo,zo+(nz-1)*dz,nz)
    
    x,y,z,v=np.zeros((4,nx*ny*nz))
    k=0
    
    for ix in range(0,nx):
        for iy in range(0,ny):
            for iz in range(0,nz):
                x[k]=xi[ix]
                y[k]=yi[iy]
                z[k]=zi[iz]
                v[k]=new_dv[ix,iy,iz]
                k=k+1
                
    data=None
    data=np.column_stack((x,y,z,v))
     
    return data,nx,ny,nz
            