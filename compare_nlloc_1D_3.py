#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:04:53 2018

@author: baillard
"""

from obspy.io.nlloc.core import read_nlloc_hyp

import matplotlib.pyplot as plt

from lotos.LOTOS_class import Catalog,Event,Phase
from general import util as gutil
from nlloc import util as nllocutil

plt.close('all')
#file_in='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc3/AXIAL.20170130.005908.grid0.loc.hyp'
file_3d='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc_3D_3082/sum.nlloc'
file_1d='/media/baillard/Shared/Dropbox/_Moi/Projects/Axial/PROG/NLLOC_AXIAL/loc_1D_3082/sum.nlloc'

Cat_1=nllocutil.read_nlloc_sum(file_1d)
Cat_3=nllocutil.read_nlloc_sum(file_3d)


Cat_1.ll2xy(Cat_1.ini_lon,Cat_1.ini_lat)

Cat_3.ll2xy(Cat_3.ini_lon,Cat_3.ini_lat)
#
#Cat_1.plot_cross(center=[5,4],angle_deg=25,len_prof=[0,6],width_prof=[0.5,0.5])
#Cat_3.plot_cross(center=[5,4],angle_deg=25,len_prof=[0,6],width_prof=[0.5,0.5])

Cat_1.plot_histo(station_code=[1,2,3,4,5,6,7],phase_code=[1,2])
fig=plt.gcf()
fig.suptitle('1D Model')

Cat_3.plot_histo(station_code=[1,2,3,4,5,6,7],phase_code=[1,2])

fig=plt.gcf()
fig.suptitle('3D Model')

Cat_1.plot_statinfo()

fig=plt.gcf()
fig.suptitle('1D Model')
[ax1,ax2]=fig.get_axes()
ax2.set_ylim([0,0.23])

Cat_3.plot_statinfo()

fig=plt.gcf()

fig.suptitle('3D Model')
[ax1,ax2]=fig.get_axes()
ax2.set_ylim([0,0.23])

gutil.figs2pdf('aj.pdf',figure_list=plt.get_fignums())


