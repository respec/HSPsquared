# import sys, os
# print '\n'.join(sys.path)

# home_dir = os.path.expanduser("~")
# print "My home directory:", home_dir



# sys.path.append('/usr/local/lib/python3.8/dist-packages')
# print '\n'.join(sys.path)

################################

# import h5py
# filename = "test10.h5"

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())

# from HSP2 import versions, main as run
# from HSP2tools import read_UCI, read_WDM
from HSP2.utilities import initm, make_numba_dict
versions()