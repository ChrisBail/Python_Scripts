#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:13:17 2017

@author: baillard
"""

import os 
from PyPDF2 import PdfFileMerger
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np


def full_path_list(directory_file,start_with='',end_with=''):
    """
    Funtion returning the list of files present in a given directory
    start_with and end_with could be arrays of string
    """
    
    ### Initialize
    
    file_list=[]
    
    ### Check
    
    if not isinstance(start_with,list) : start_with=[start_with]
    if not isinstance(end_with,list) : end_with=[end_with]
    
    ### Loop
    
    for start_w in start_with:
        for end_w in end_with:
            for file in os.listdir(directory_file):
                if file.startswith(start_w) and file.endswith(end_w):
                    file_list.append(directory_file+'/'+file)
    
  ### Sort
    file_list=sorted(set(file_list))
    return file_list

def merge_pdfs(list_pdf,output_pdf):
    merger = PdfFileMerger()
    
    for pdf in list_pdf:
        merger.append(open(pdf, 'rb'))
    
    
    with open(output_pdf, 'wb') as fout:
        merger.write(fout)
    
def get_max_field(elem_list):
    """
    function made to return the field width that should be used
    """
    
    max_field=max([len(str(x)) for x in elem_list])
    
    return max_field

def convert_values(input_array,thresholds,new_values):
    
    output_array=np.copy(input_array)
    
    min_val=np.min(input_array)
    max_val=np.max(input_array)
    
    thresholds=np.append(min_val,thresholds)
    thresholds=np.append(thresholds,max_val)
    
    for kk in range(len(thresholds)-1):
        output_array[ (input_array>=thresholds[kk]) & (input_array<=thresholds[kk+1]) ] = new_values[kk]
        
    return output_array


def get_index(list_lines, pattern):
    """
    Function made to retrieve index of lines that contain patterns
    """
    index=[i for i,line in enumerate(list_lines) if pattern in line]
    if len(index)==1:
        index=index[0]
        
    return index
        
def figs2pdf(pdf_file,figure_list=plt.get_fignums()):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_file)
    
    
    for fig in figure_list: ## will open an empty extra figure :(
        pdf.savefig( fig )
     
    pdf.close()
    
def get_colors(num_colors,colormap='jet',flag_plot=False):
    """
    Function made to generate a lits of RGB from a defined colormap
    """

    len_cmap=500
    cmap = plt.cm.get_cmap(colormap, len_cmap)
    
    step=int(len_cmap/(num_colors))
    
    
    rgb_list=[]
    index=int(step/2)
    for kk in range(num_colors):
        print(index)
        rgb_list.append(cmap(index)[:3])
        index+=step
        
     
    if flag_plot:
        
        x=np.linspace(1,10,num_colors)
        y=np.ones((num_colors,1))
        
        for kk in range(num_colors):
            plt.plot(x[kk],y[kk],'o',color=rgb_list[kk])
    
    return rgb_list
        
            
def getPickForArrival(picks, arrival):
    """
    searches list of picks for a pick that matches the arrivals pick_id
    and returns it (empty Pick object otherwise).
    """
    pick = None
    for p in picks:
        if arrival.pick_id == p.resource_id:
            pick = p
            break
    return pick           


