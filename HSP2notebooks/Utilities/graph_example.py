''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

import networkx as nx
import pandas   as pd
import numpy as np


def graphtutoral_test10(hdfname):
    ''' TUTORIAL routine to color a watershed DAG'''

    df = pd.read_hdf(hdfname, 'PERLND/GENERAL_INFO')
    df['GISx'] = 7.0
    df['GISy'] = 10.0
    df['GIScolor'] = 'lightblue'
    df['GISarea'] = 6000.0
    df.to_hdf(hdfname, '/PERLND/GENERAL_INFO', data_columns=True, format='t')

    df = pd.read_hdf(hdfname, 'IMPLND/GENERAL_INFO')
    df['GISx'] = 3.0
    df['GISy'] = 1.0
    df['GIScolor'] = 'yellow'
    df['GISarea'] = 3000.0
    df.to_hdf(hdfname, '/IMPLND/GENERAL_INFO', data_columns=True, format='t')

    df = pd.read_hdf(hdfname, 'RCHRES/GENERAL_INFO')
    df['GISx'] = [5.0, 3.0, 7.0, 5.0, 5.0]
    df['GISy'] = [9.0, 6.0, 6.0, 3.5, 0.0]
    df['GIScolor'] = 'lawngreen'
    df.to_hdf(hdfname, '/RCHRES/GENERAL_INFO', data_columns=True, format='t')
