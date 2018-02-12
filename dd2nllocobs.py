#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:20:34 2017

@author: baillard
"""


import argparse
import logging
import sys
from nlloc import util

### Parameters

#dd_file='/home/baillard/Dropbox/_Moi/Projects/Axial/DATA/CATALOG/Axial_hypoDDPhaseInput_20150201_20150210.dat'
#nlloc_file='20150201_20150210.nlloc'

### Parse arguments

parser = argparse.ArgumentParser(description='Convert hypodd format catalog file into NLLOC_OBS ray file')

parser.add_argument("dd_file", type=str,
                    help="hypodd catalog file")

parser.add_argument("nlloc_file", type=str,
                    help="nlloc_obs output file"
                    )

args = parser.parse_args()

dd_file=args.dd_file
nlloc_file=args.nlloc_file

### Logger

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
logging.info('Writing to file')

### Read/write

util.dd2nllocobs(dd_file,nlloc_file)



