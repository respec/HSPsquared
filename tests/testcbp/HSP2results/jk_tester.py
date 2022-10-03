#!/usr/bin/python3

# from re import S
# from numpy import float64, float32
# from pandas import DataFrame, date_range
# from pandas.tseries.offsets import Minute
# from datetime import datetime as dt
# import os
# from HSP2.utilities import versions, get_timeseries, expand_timeseries_names, save_timeseries, get_gener_timeseries
# from HSP2.configuration import activities, noop, expand_masslinks

# from HSP2IO.io import IOManager, SupportsReadTS, Category

print("Running jk_tester.py")

# from pandas import read_hdf

# from pandas import options, read_hdf, DataFrame
# import sys, os
# print '\n'.join(sys.path)

# home_dir = os.path.expanduser("~")
# print "My home directory:", home_dir

# python -m site

# sys.path.append('/usr/local/lib/python3.8')
# sys.path.append('/usr/local/lib/python3.8/dist-packages')
# print '\n'.join(sys.path)

################################

import h5py
filename = "PL3_5250_0001.h5" 

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())

# from HSP2 import versions, main as run
# from HSP2tools import read_UCI, read_WDM

################################################################
# from pandas import read_hdf

# HBN = 'PL3_5250_0001.h5'
# read_hdf(HBN,'PL3_5250_0001')


