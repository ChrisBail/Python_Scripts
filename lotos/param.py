#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:32:44 2017

@author: baillard
"""

import sys

def read_param(param_file):
    """
    Function made to read MAJOR_PARAM.dat files, output is a dictionary
    """
    
    ### Initialize
    
    param_dic={}

    ### Start feeding
    
    param_dic['general']=_get_general(param_file)
    param_dic['area_center']=_get_area(param_file)
    param_dic['1d_location_key']=_get_1d_location_key(param_file)
    param_dic['grid_orientations']=_get_grid_orientations(param_file)
    param_dic['inversion_parameters']=_get_inversion_parameters(param_file)
    param_dic['lin_loc_param']=_get_lin_loc_param(param_file)
    param_dic['3d_model_param']=_get_3d_model_param(param_file)
    param_dic['grid_param']=_get_grid_param(param_file)
    param_dic['loc_param']=_get_loc_param(param_file)
    param_dic['ref_param']=_get_ref_param(param_file)
    param_dic['model_1d']=_get_model_1d(param_file)

    ### Return
    
    return param_dic


def write_param(param_dic,param_file=None):
    """
    Function made to read MAJOR_PARAM.dat files, output is a dictionary
    """
    
    orig_stdout = sys.stdout
    
    ### Initialize
    
    if param_file is not None:
        sys.stdout = open(param_file, 'wt')
    
    ### Start printing
    
    _write_general(param_dic)
    _write_area(param_dic)
    _write_1d_location_key(param_dic)
    _write_grid_orientations(param_dic)
    _write_inversion_parameters(param_dic)
    _write_lin_loc_param(param_dic)
    _write_3d_model_param(param_dic)
    _write_grid_param(param_dic)
    _write_loc_param(param_dic)
    _write_ref_param(param_dic)
    _write_model_1d(param_dic)

    ### Return
#    
    if param_file is not None:
        sys.stdout.close()
        
    sys.stdout=orig_stdout
    


#######################################"

def _get_index(list_lines, pattern):
    index=[i for i,line in enumerate(list_lines) if pattern in line]
    if len(index)==1:
        index=index[0]
        
    return index


######################
    
def _get_general(param_file):
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'GENERAL INFORMATION')
    
    general=[]
    
    general=lines[kk:kk+5]
    
    return general

def _write_general(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['general'],list):
        for line in dic_param['general']:
            print(line)
            
    print('')

######################################
            
def _get_1d_location_key(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'1D LOCATION KEY')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+3]
    
    return block_dic

def _write_1d_location_key(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['1d_location_key'],list):
        for line in dic_param['1d_location_key']:
            print(line)
            
    print('')
            
#####################################
            
def _get_grid_orientations(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'ORIENTATIONS OF GRIDS')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+3]
    
    return block_dic

def _write_grid_orientations(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['grid_orientations'],list):
        for line in dic_param['grid_orientations']:
            print(line)

    print('')
#######################

def _get_inversion_parameters(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'INVERSION PARAMETERS')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+11]
    
    return block_dic

def _write_inversion_parameters(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['inversion_parameters'],list):
        for line in dic_param['inversion_parameters']:
            print(line)

    print('')
####################

def _get_lin_loc_param(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'LIN_LOC_PARAM')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+29]
    
    return block_dic

def _write_lin_loc_param(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['lin_loc_param'],list):
        for line in dic_param['lin_loc_param']:
            print(line)



#####################
            

def _get_3d_model_param(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'3D_MODEL PARAMETERS')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+7]
    
    return block_dic

def _write_3d_model_param(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['3d_model_param'],list):
        for line in dic_param['3d_model_param']:
            print(line)

    print('')

#####################
#####################
            

def _get_grid_param(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'GRID_PARAMETERS')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+6]
    
    return block_dic

def _write_grid_param(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['grid_param'],list):
        for line in dic_param['grid_param']:
            print(line)

    print('')

#####################
            
            
                       
def _get_loc_param(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'LOC_PARAMETERS')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+20]
    
    return block_dic

def _write_loc_param(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['loc_param'],list):
        for line in dic_param['loc_param']:
            print(line)

    print('')
#####################

                       
def _get_ref_param(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'REF_PARAM')
    
    block_dic=[]
    
    block_dic=lines[kk:kk+9]
    
    return block_dic

def _write_ref_param(dic_param):
    
    print('*********')
    
    if isinstance (dic_param['ref_param'],list):
        for line in dic_param['ref_param']:
            print(line)

    print('')
#####################            


def _get_area(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'AREA')
    
    area_center={}
    area_center['ini_lon']=float(lines[kk+1].split()[0])
    area_center['ini_lat']=float(lines[kk+1].split()[1])
    
    fic.close()
    
    return area_center

def _write_area(dic_param):
    print('*********')
    print('AREA_CENTER:')
    print('%.3f %.3f  ****'%(
            dic_param['area_center']['ini_lon'],
            dic_param['area_center']['ini_lat']
            ))
    
    print('')
######

def _get_model_1d(param_file):
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    
    ### Start feeding
    
    kk=_get_index(lines,'1D MODEL')
    
    model_1d={}
    model_1d['num_iter']=int(lines[kk+1].split()[0])
    
    model_1d['z_min']     =float(lines[kk+2].split()[0])
    model_1d['dz_step']   =float(lines[kk+2].split()[1])
    model_1d['num_events']=int(lines[kk+2].split()[2])
    
    model_1d['ds_min'] =float(lines[kk+3].split()[0])
    model_1d['dz_lay'] =float(lines[kk+3].split()[1])
    model_1d['zgr_max']=int(lines[kk+3].split()[2])
    
    model_1d['dz_par'] =float(lines[kk+4].split()[0])
    
    model_1d['ray_min']=float(lines[kk+5].split()[0])
    
    model_1d['sm_p']=float(lines[kk+6].split()[0])
    model_1d['sm_s']=float(lines[kk+6].split()[1])
    
    model_1d['rg_p']=float(lines[kk+7].split()[0])
    model_1d['rg_s']=float(lines[kk+7].split()[1])
    
    model_1d['w_hor'] =float(lines[kk+8].split()[0])
    model_1d['w_ver'] =float(lines[kk+8].split()[1])
    model_1d['w_time']=float(lines[kk+8].split()[2])
    
    model_1d['n_lsqr'] =int(lines[kk+9].split()[0])
    
    model_1d['n_sharp'] =int(lines[kk+10].split()[0])
    
    model_1d['z_sharp'] =[float(lines[kk+11].split()[n]) for n in range(model_1d['n_sharp'])]

    fic.close()
    
    return model_1d

def _write_model_1d(dic_param):
    print('*********')
    print('1D MODEL PARAMETERS :')
    print('%i  Iterations for 1D inversions'%(dic_param['model_1d']['num_iter']))
    print('%.1f %.1f %i  ****'%(
            dic_param['model_1d']['z_min'],
            dic_param['model_1d']['dz_step'],
            dic_param['model_1d']['num_events']
            ))
    
    print('%.1f %.1f %i  ****'%(
            dic_param['model_1d']['ds_min'],
            dic_param['model_1d']['dz_lay'],
            dic_param['model_1d']['zgr_max']
            ))
    
    print('%.1f  ****'%(
            dic_param['model_1d']['dz_par']
            ))
    
    print('%.1f  ****'%(
            dic_param['model_1d']['ray_min']
            ))
    
    print('%.1f %.1f  ****'%(
            dic_param['model_1d']['sm_p'],
            dic_param['model_1d']['sm_s']
            ))
    
    print('%.1f %.1f  ****'%(
            dic_param['model_1d']['rg_p'],
            dic_param['model_1d']['rg_s']
            ))
    
    print('%.1f %.1f %.1f ****'%(
            dic_param['model_1d']['w_hor'],
            dic_param['model_1d']['w_ver'],
            dic_param['model_1d']['w_time']
            ))
    
    print('%i ****'%(
            dic_param['model_1d']['n_lsqr']
            ))
    
    print('%i ****'%(
            dic_param['model_1d']['n_sharp']
            ))
    
    if not dic_param['model_1d']['z_sharp']:
        print('%i ****'%(
            999
            ))
    else:
        str_num = [str(x) for x in dic_param['model_1d']['z_sharp']]
        print('%s *** '%(
                str_num
                ))
        
    print('')


    
    