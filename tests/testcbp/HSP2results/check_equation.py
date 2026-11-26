# this is a simple tester that should be kept up to date with the code in main.py
# alternative to step debugger
fpath = "./tests/testcbp/HSP2results/JL1_6562_6560.h5"

# Third party imports
from hsp2.hsp2io.hdf import HDF5

# First party imports
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2io.io import IOManager
from hsp2.state.state import *

# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)
parameter_obj = io_manager.read_parameters()
siminfo = parameter_obj.siminfo
opseq = parameter_obj.opseq
# Note: now that the UCI is read in and hdf5 loaded, you can see things like:
# - hdf5_instance._store.keys() - all the paths in the UCI/hdf5
# - finally stash specactions in state, not domain (segment) dependent so do it once
# now load state and the special actions
state = init_state_dicts()
state_siminfo_hsp2(parameter_obj, siminfo, io_manager, state)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities)
# - finally stash specactions in state, not domain (segment) dependent so do it once
state["specactions"] = specactions  # stash the specaction dict in state
om_init_state(state)  # set up operational model specific state entries
specl_load_state(state, io_manager, siminfo)  # traditional special actions
state_load_dynamics_om(
    state, io_manager, siminfo
)  # operational model for custom python
# finalize all dynamically loaded components and prepare to run the model
state_om_model_run_prep(state, io_manager, siminfo)
