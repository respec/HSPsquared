# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory

import os
import numpy
from hsp2.hsp2.main import *
from hsp2.state.state import *
from hsp2.hsp2.om import *
from hsp2.hsp2.SPECL import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.state.state import *
from hsp2.hsp2.om_timer import timer_class

fpath = "./tests/test10specl/HSP2results/test10specl.h5"
timer = timer_class()
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


# read user control, parameters, states, and flags parameters and map to local variables
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

start, stop = siminfo["start"], siminfo["stop"]

copy_instances = {}
gener_instances = {}
section_timing = {}

section_timing["io_manager.read_parameters() call and config"] = str(timer.split()) + "seconds"
#######################################################################################
# initialize STATE dicts
#######################################################################################
# Set up Things in state that will be used in all modular activities like SPECL
state = state_class(
    state_empty["state_ix"], state_empty["op_tokens"], state_empty["state_paths"], 
    state_empty["op_exec_lists"], state_empty["model_exec_list"], state_empty["dict_ix"], 
    state_empty["ts_ix"], state_empty["hsp_segments"]
)
om_operations = om_init_state()  # set up operational model specific containers
state_siminfo_hsp2(state, parameter_obj, siminfo, io_manager)
state_om_model_root_object(state, om_operations, siminfo)
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities, timer)
om_init_hsp2_segments(state, om_operations)
# now initialize all state variables for mutable variables
hsp2_domain_dependencies(state, opseq, activities, om_operations, False)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
# - finally stash specactions in state, not domain (segment) dependent so do it once
specl_load_om(om_operations, specactions)  # load traditional special actions
state_load_dynamics_om(
    state, io_manager, siminfo, om_operations
)  # operational model for custom python
# finalize all dynamically loaded components and prepare to run the model
state_om_model_run_prep(opseq, activities, state, om_operations, siminfo)
section_timing["state om initialization()"] = str(timer.split()) + "seconds"
statenb = state_class_lite(0)
state_copy(state, statenb)
#######################################################################################
