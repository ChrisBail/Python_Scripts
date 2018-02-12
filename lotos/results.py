#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:33:32 2017

@author: baillard

Set of functions made to read the results file after models have been processed
"""

import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copy2

def file2dic(result_file):
    """
    Function made to read the result.log file outputted by run_1D.py for example
    
    Parameters
    ---------
    
    result_file: str
        result file
        
    Returns
    -------
    
    dic_result: dic
        dictionary containing parameters and values 
    """
    
    #### Read file
    
    fic=open(result_file,'rt')
    lines=fic.readlines()
    fic.close()
    
    #### Feed structure
    
    dic_result={key_val:[] for key_val in lines[0].split()}
    keys=[key_val for key_val in lines[0].split()]
    param_keys=keys[4:]
    
    logging.info('Parameters in file %s'%(str(param_keys)))
    
    ### Start loop
    for line in lines[1:]:
        kk=0
        for key in dic_result:
            value=line.split()[kk]
            if key!='MODEL_KEY':
                value=float(value)
                
            dic_result[key]=np.append(dic_result[key],value)
            
            kk=kk+1
        
    return dic_result


def dic2file(dic_param,output_file=None):
    
    fic=None
    if output_file:
        fic=open(output_file,'wt')
        
    num_el=np.size(dic_param['MODEL_KEY'])
    
    param_keys=get_param_keys(dic_param)
    
    format_str='%15s %7s %7s %8s '+'%6s '*len(param_keys) 
    print(format_str%tuple([key for key in dic_param]),file=fic)
    
    format_str='%15s %7.3f %7.3f %8i '+'%6.1f '*len(param_keys) 
    
    for kk in range(num_el):
        value_list=[dic_param[key][kk] for key in dic_param]
        print(format_str%tuple(value_list),file=fic)
        
    if output_file: 
         fic.close()
        
def copy_files(dic_param,src_dir,dest_dir):
    """
    Function made to select plot files based on key
    """
    
    ### Make subdirectory
    if os.path.exists(dest_dir):
        answer=input('%s exists, want to remove what is inside? (y):\n'%(dest_dir))
        if answer=='y':
            os.system('rm -rf %s/*pdf'%dest_dir)
            os.system('rm -rf %s/*log'%dest_dir)
        else:
            return
    else:
        os.mkdir(dest_dir)
    
    for id_key in dic_param['MODEL_KEY']:
        list_file=glob.glob(src_dir+'/*%s*'%id_key)
        for file in list_file:
            copy2(file,dest_dir)


    
def get_param_keys(old_dic):
    param_keys=[key for key in old_dic if key not in ['MODEL_KEY','RMS_P','RMS_S','NUM_OBS']]
    
    return param_keys
    
    
def select_dic(old_dic,**kwargs):
    """
    Made to select only the models from a old dictionary that fulfills conditions
    based on keys i.e RMS_P...
    
    ex. select_dic(old_dic,'RMS_P'=[0.1,0.4])
    
    Returns:
    ----
    new_dic : dic
        elements selected
    bool_all: boolean
        boolean array of selected elemnents
    """
    
    
    bool_all=np.ones(old_dic['MODEL_KEY'].shape,dtype=bool)
    for key in old_dic:
        lim_val = kwargs.get(key,None)
        if lim_val is None:
            continue
        array_val=old_dic[key]
        bool_sel=(array_val<=lim_val[1]) & (array_val>=lim_val[0])
        
        bool_all=bool_all & bool_sel
        
    new_dic={key:old_dic[key][bool_all] for key in old_dic}
    
    logging.info('Initial number of elements is %i'%(len(old_dic['MODEL_KEY'])))
    logging.info('Final number of elements is %i'%(len(new_dic['MODEL_KEY'])))

    
    return new_dic,bool_all
    
def plot_residuals(result_file,output_fig=None,fig=None,**kwargs):
    """
    Function made to plot the P and S residuals 
    
    Parameters
    ---------
    
    result_file : str
    output_fig : str
        file where the plot should be saved
    """
    
    
    #### Read file
    
    dic_result=file2dic(result_file)

    ### Get xbin range
    
    rms_val=np.append(dic_result['RMS_P'],dic_result['RMS_S'])
    x_bins=np.linspace(np.min(rms_val),np.max(rms_val),100)
    
    ### Plots
    
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    else: 
        (ax1,ax2)=fig.get_axes()
    
    ax1.hist(dic_result['RMS_P'],x_bins,edgecolor='black',**kwargs)
    ax1.set_title('RMS for %i models tested'%(len(dic_result['RMS_P'])))
    ax1.set_ylabel('P')
    ax1.tick_params('x',direction='in')
    ax2.hist(dic_result['RMS_S'],x_bins,edgecolor='black',**kwargs)
    ax2.invert_yaxis()
    ax2.set_xlabel('RMS [s]')
    ax2.set_ylabel('S')
    
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    if output_fig:
        plt.savefig(output_fig)
    
    
    return fig

def histo_param(result_file,output_fig=None,**kwargs):
    """
    Function made to plot histograms 
    
    Parameters
    ---------
    
    result_file : str
    output_fig : str
        file where the plot should be saved
    """
    ###
    
    old_dic=file2dic(result_file)
    print(kwargs)
    new_dic=select_dic(old_dic,**kwargs)
    
    len_new=len(new_dic['MODEL_KEY'])
    len_old=len(old_dic['MODEL_KEY'])
    
    ### 
    
    param_keys=[key for key in old_dic if key not in ['MODEL_KEY','RMS_P','RMS_S','NUM_OBS']]
    
    ### plot
    
    f, ax = plt.subplots(int(len(param_keys)/2),2)
    ax=ax.reshape(-1)
    plt.suptitle('%i Models selected over %i'%(len_new,len_old))
    kk=0
    for key in param_keys:
    
        print(key)
        
        x_best=np.unique(old_dic[key])
        x_tick=[str(x) for x in x_best]
        x_hist=list(range(len(x_best)))
    
        diff_val=x_best[:-1]+np.diff(x_best)/2
        new_val=np.append(x_best[0],diff_val)
        new_val=np.append(new_val,x_best[-1])
    
        n_el,_=np.histogram(new_dic[key],new_val)
        plt.sca(ax[kk])
        plt.bar(x_hist,n_el,facecolor='0.9',width=1,edgecolor=['k' for i in range(len(n_el))])
        plt.xticks(x_hist,x_tick)
        plt.xlabel('Value')
        plt.ylabel('Counts')
        ax[kk].set_title('%s'%key)
        kk=kk+1
    
        
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    
def process_results_1D():
    
    result_file='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/results_1D_01.log'
    plot_direc='/home/baillard/Dropbox/_Moi/Projects/Axial/PROG/test_1D_01/'
    subdir_name='test'

    
    plt.close('all')
    ########
        
    logging.getLogger().setLevel(logging.INFO)
            
    RMS_P=[0,0.043]
    RMS_S=[0,0.129]
    
    old_dic=file2dic(result_file)
    new_dic=select_dic(old_dic,RMS_P=RMS_P,RMS_S=RMS_S)
    
    ##### dic2file
    
    
    #####
    
    copy_files(new_dic,plot_direc,plot_direc+'/'+subdir_name)
    dic2file(new_dic,plot_direc+'/'+subdir_name+'/subresults.log')
    
    histo_param(result_file,output_fig=None,RMS_P=RMS_P,RMS_S=RMS_S)
    plot_residuals(result_file,output_fig=None)  
    #
    #fig=results.plot_residuals(result_file,output_fig=None)