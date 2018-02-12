#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 08:52:10 2017

@author: baillard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:35:53 2017

@author: baillard
"""

import importlib
import os
import lotos
importlib.reload(lotos)
import datetime
import itertools

from lotos import param
importlib.reload(param)
from lotos import lotos_util
from lotos.model_1d import util


def run_1D():
    """
    Function made to run the 1D process using straigh lines,
    this Can be easily adapted to 3D
    """
    ### Paramters

    ref_param_file='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MAJOR_PARAM.DAT'
    model_path='/home/baillard/PROGRAMS/LOTOS13_unix/DATA/AXIALSEA/MODEL_01/'
    fortran_code='/home/baillard/PROGRAMS/LOTOS13_unix/PROGRAMS/1_PRELIM_LOC/START_1D/start_real.exe'
    tmp_directory='test_1D_01_bend'
    result_file='results_1D_01_bend.log'
    
    w_hor=[10]
    w_ver=[10]
    rg_p=[0.5,1,5]
    rg_s=rg_p
    sm_p=[0.5,1,2]
    sm_s=sm_p
    
    ########### PREPARE ############
    
    ##### Prepare directories for output
    
    if not os.path.exists(tmp_directory):
        os.makedirs(tmp_directory)
        
    model_name,data_dir=lotos_util.get_name(model_path)
    
    fic=open(result_file,'wt')
    fic.write('%15s %7s %7s %8s %6s %6s %6s %6s %6s %6s\n'%(
            'MODEL_KEY','RMS_P','RMS_S','NUM_OBS',
            'rg_p','rg_s','sm_p','sm_s','w_hor','w_ver'
            ))
    fic.close()
    ### Get all combinations
    
    
    ref_dic=param.read_param(ref_param_file)
    
    list_param = [rg_p,rg_s,sm_p,sm_s,w_hor,w_ver]
    combin=list(itertools.product(*list_param))
    
    ###### START LOOP #########
    
    
    for kk in range(len(combin)):
        
        ##### MESSAGE
        
        fic=open(result_file,'at')
        print('Run model %i / %i'%(kk,len(combin)))
        
        ####################
        ##### EDIT #########
    
        
        current_dic=ref_dic
        
        current_dic['model_1d']['rg_p']=combin[kk][0]
        current_dic['model_1d']['rg_s']=combin[kk][1]
        current_dic['model_1d']['sm_p']=combin[kk][2]
        current_dic['model_1d']['sm_s']=combin[kk][3]
        current_dic['model_1d']['w_hor']=combin[kk][4]
        current_dic['model_1d']['w_ver']=combin[kk][5]
             
        param.write_param(current_dic,model_path+'MAJOR_PARAM.DAT')
        
        id_key=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ####################
        ##### RUN #########
        
        max_iter=None

        try:
            lotos_util.run_exe(fortran_code,flag_out=True)
        except ValueError as vrr:
            print(vrr)
            max_iter=float(str(vrr).split('=')[1])-1
            
        
        ##################
        #### PLOT ########
        
        print(max_iter)
        util.plot_1DOPT(model_path,tmp_directory,id_key=id_key,max_iter=max_iter)
        
        #######################
        #### WRITE RESULTS ####
        
        resi_dic=util.read_resi_1DOPT(tmp_directory+'/resi_'+id_key+'.log')
        fic.write('%15s %7.3f %7.3f %8i %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f\n'%(
                id_key,resi_dic['RMS_P'][-1],resi_dic['RMS_S'][-1],resi_dic['num_obs'][-1],
                current_dic['model_1d']['rg_p'],current_dic['model_1d']['rg_s'],
                current_dic['model_1d']['sm_p'],current_dic['model_1d']['sm_s'],
                current_dic['model_1d']['w_hor'],current_dic['model_1d']['w_ver']
                ))
        
        #### END LOOP ####
        
        fic.close()
    
    
    #### CLOSE



