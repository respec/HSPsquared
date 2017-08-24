''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import sys
import IPython
import HSP2
import HSP2tools
import pandas
import numpy
import matplotlib
import h5py
import networkx
import platform
import tables
import numba
import wdmtoolbox  # modified version of Tim Cera's wdmtoolbox, BSD license, Copyright 2016 by Tim Cera, P.E.

def versions():
    ''' Returns the version of the Python and HSP2 library modules in a DataFrame'''
    packages = {
            'HSP2': HSP2.__version__,
	    'HSP2tools': HSP2tools.__version__,
            '  ':'',               # spacer
            'PYTHON': sys.version.replace('\n', ''),
            'IPYTHON': IPython.__version__,
            ' ':'',                # spacer

	    'H5PY': h5py.__version__,
            'MATPLOTLIB': matplotlib.__version__,
            'NETWORKX': networkx.__version__,
            'NUMBA': numba.__version__,
            'NUMPY': numpy.__version__,
            'PANDAS': pandas.__version__,
            'PYTABLES': tables.__version__,
            'WDMTOOLBOX': wdmtoolbox.__version__,  # doesn't provide version info - yet

            '   ':'',               # spacer
            'os': platform.platform(),
            'processor': platform.processor(),
            }

    cols = ['HSP2', 'HSP2tools','  ', 'PYTHON', 'IPYTHON', ' ',
    'H5PY', 'MATPLOTLIB','NETWORKX', 'NUMBA', 'NUMPY','PANDAS','PYTABLES',
    'WDMTOOLBOX', '   ',  'os','processor']

    return pandas.DataFrame(packages, index=['Version'], columns=cols).T
