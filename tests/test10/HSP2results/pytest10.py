# if testing manually you may need to os.chdir('./tests/test10specl/HSP2results')
import pytest
import os
import pandas as pd
from pandas.io.pytables import read_hdf
import HSP2.utilities
import HSP2IO
from HSP2IO.hdf import HDF5

def test_h5_file_exists():
    assert os.path.exists('test10.h5')

# uncomment if you want to see how to break the test
#def test_that_should_fail():
#    assert os.path.exists('test100000.h5')


# uncomment if you want to see how to break the test
#def test_that_should_fail():
# Code to evaluate difference between runs with and without specl
# not formatted in pytest friendly fashion yet, and would require oth h5 files to exist
"""

fpath = "./test10.h5"
dstore = pd.HDFStore(fpath)
sedtrn_r005 = read_hdf(dstore, '/RESULTS/RCHRES_R005/SEDTRN/table')
sedtrn_r005['RSED5'].mean()
dstore.close()

# Grab the other
os.chdir('../../test10specl/HSP2results')

fpath = "./test10specl.h5"
specl_dstore = pd.HDFStore(fpath)
specl_sedtrn_r005 = read_hdf(specl_dstore, '/RESULTS/RCHRES_R005/SEDTRN/table')
specl_sedtrn_r005['RSED5'].mean()
specl_dstore.close()

"""