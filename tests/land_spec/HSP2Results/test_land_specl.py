# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory

from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2.SPECL import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.hsp2tools.readUCI import *

from hsp2.state.state import *

th = "./tests/land_spec/HSP2Results/hwmA51800.h5"
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'

# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)

parameter_obj = io_manager.read_parameters()
opseq = parameter_obj.opseq
ddlinks = parameter_obj.ddlinks
ddmasslinks = parameter_obj.ddmasslinks
ddext_sources = parameter_obj.ddext_sources
ddgener = parameter_obj.ddgener
model = parameter_obj.model
siminfo = parameter_obj.siminfo
ftables = parameter_obj.ftables
specactions = parameter_obj.specactions
monthdata = parameter_obj.monthdata
conditions = specactions["conditions"]
actions = specactions["ACTIONS"]

start, stop = siminfo["start"], siminfo["stop"]
