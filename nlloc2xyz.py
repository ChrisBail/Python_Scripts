#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:26:13 2018

@author: baillard
"""

import argparse


### Parse arguments

parser = argparse.ArgumentParser(description='Convert NLLoc hypocenter-phase file to simple event file')

parser.add_argument("file_nlloc", type=str,
                    help="NLLoc hypocenter-phase file")


args = parser.parse_args()

file_nlloc=args.file_nlloc


#file_nlloc='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc_3D_3082/sum.nlloc'


fic=open(file_nlloc,'rt')
lines=fic.read().splitlines()
fic.close()
 

lines_start = [i for i, line in enumerate(lines)
if line.startswith("NLLOC ")]
lines_end = [i for i, line in enumerate(lines) 
if line.startswith("END_NLLOC")]

print("%27s %12s %12s %11s %9s %3s %10s"%('OT','LON','LAT','Z','RMS','NP','ID'))

for start, end in zip(lines_start, lines_end):
        lines_event=lines[start:end + 1]

        
        for line in lines_event:
            if 'GEOGRAPHIC' in line:
                elements=line.split()
                OT="%4s-%2s-%2sT%2s:%2s:%09f"%(
                        elements[2],elements[3],elements[4],
                        elements[5],elements[6],float(elements[7])
                        )
                LON=elements[11]
                LAT=elements[9]
                Z=elements[13]
            if 'COMMENT' in line:
                elements=line.split('"')
                ID=elements[1]
            if 'QUALITY' in line:
                elements=line.split()
                RMS="%9.4f"%(float(elements[8]))
                NP="%3i"%(int(elements[10]))
                
            
        
        print("%27s %12s %12s %11s %9s %3s %10s"%(OT,LON,LAT,Z,RMS,NP,ID))
