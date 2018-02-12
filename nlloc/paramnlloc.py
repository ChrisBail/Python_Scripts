#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:16:08 2017

@author: baillard
"""

import logging
import parse

class ParamNlloc(object):
        
    def __init__(self):
    
        self.param_dic={}
        self.format_dic=self._get_format_dic()
        
    def __repr__(self):
        
        out=''
        for key in self.param_dic:
            out+=key+':\n'+str(self.param_dic[key])+'\n'
        return out
        
    def _get_format_dic(self):
        
        format_dic={}
        format_dic['VGOUT']='{}'
        format_dic['VGTYPE']='{}'
        format_dic['VGRID']='{:d} '*3+'{:2f} '*3+'{:2f} '*3 + '{}'
        format_dic['LAYER']=7*'{:.3f} '.rstrip()
        format_dic['GTFILES']='{} {} {}'
        format_dic['GTMODE']='{} {}'
        format_dic['GTSRCE']='{} {} {:.5f} {:.5f} {:.5f} {:.5f}'
        format_dic['GT_PLFD']='{:.2e} {:d}'
        format_dic['EQFILES']='{} {}'
        format_dic['EQMECH']='{}'+ ' {:.1f}'*3
        format_dic['EQMODE']='{}'
        format_dic['EQEVENT']='{}'+ ' {:.3f}'*4
        format_dic['EQSTA']='{} {} {} {:.2f} {} {:.2f}'
        format_dic['EQVPVS']='{.2f}'
        format_dic['EQQUAL2ERR']=['{:.2f}','repeat']
        format_dic['LOCSIG']='{}'
        format_dic['LOCCOM']='{}'
        format_dic['LOCFILES']='{} {} {} {}'
        format_dic['LOCHYPOUT']=['{} ','repeat']
        format_dic['LOCSEARCH']={}
        format_dic['LOCSEARCH']['GRID']='{} {:d}'
        format_dic['LOCGRID']='{:d} '*3 +'{:f} '*6 + '{} {}'
        format_dic['LOCMETH']='{} {:f} '+3*'{:d} '+ '{:f} {:d} {:f} {:d}' 
        format_dic['LOCGAU']='{:f} {:f}'
        format_dic['LOCPHASEID']=['{} ','repeat']
        format_dic['LOCQUAL2ERR']=['{:f}','repeat']
        format_dic['LOCPHSTAT']='{:f} {:d}'+' {:f}'*6
        format_dic['LOCANGLES']='{} {:d}'
        format_dic['LOCMAG']='{} {:f} {:f} {:f}'
        format_dic['LOCDELAY']='{} {} {:d} {:f}'
        
        return format_dic
        
    
###### Functions
      
def read_param_nlloc(param_file):
    
    ### Initialize

    param_dic={}
    P=ParamNlloc()
    
    ### Read lines
    
    fic=open(param_file,'rt')
    lines=fic.read().splitlines()
    fic.close()
    
    ### Clean lines
    
    lines=[x for x in lines if (len(x)!=0) and (x[0]!='#') ]
    
    ### Get keys
    
    keys=[x.split()[0] for x in lines]
    keys=list(set(keys))
    
    ### Feed dictionnary
    
    for key in keys:
        
        lines_spe=[' '.join(x.split()[1:]) for x in lines if x.startswith(key)]
        
        ### Pass if no format specified
        
        if key not in P.format_dic:
            continue
        
        ### Read lines
        
        list_param=[]
        
        for line_spe in lines_spe:
            
            ### Get format
            
            format_str=P.format_dic[key]
            
            ### Check if subkey
            
            if type(format_str) is dict:
                sub_key=line_spe.split()[0]
                format_str=P.format_dic[key][sub_key]
                
            
            num_el=len(line_spe.split())    
    
            ### Check if repeat format
                
            if type(format_str) is list:
                format_str=(num_el*format_str[0]).rstrip()
             
    
            logging.info('%s'%(key))
            
            ### Feed
            
            list_param.append(list(parse.parse(format_str,line_spe)))
            
        if len(lines_spe)==1:
            list_param=list_param[0]
            
        param_dic[key]=list_param   
        
    P.param_dic=param_dic
    
    return P
        
            
                
                
    
            
            