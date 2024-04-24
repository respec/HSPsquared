# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
import os
from HSP2.main import *
from HSP2.om import *
import HSP2IO
import numpy
from HSP2IO.hdf import HDF5
from HSP2IO.io import IOManager
fpath = './tests/test10/HSP2results/test10spec.h5' 
# try also:
# fpath = './tests/testcbp/HSP2results/PL3_5250_0001.h5' 
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)
uci_obj = io_manager.read_uci()
siminfo = uci_obj.siminfo
opseq = uci_obj.opseq
# - finally stash specactions in state, not domain (segment) dependent so do it once
# now load state and the special actions
state = init_state_dicts()
state_initialize_om(state)
state['specactions'] = uci_obj.specactions # stash the specaction dict in state

state_siminfo_hsp2(uci_obj, siminfo)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects 
state_load_dynamics_hsp2(state, io_manager, siminfo)
# Iterate through all segments and add crucial paths to state 
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities)
state_load_dynamics_specl(state, io_manager, siminfo) # traditional special actions
state_load_dynamics_om(state, io_manager, siminfo) # operational model for custom python
state_om_model_run_prep(state, io_manager, siminfo) # this creates all objects from the UCI and previous loads
# state['model_root_object'].find_var_path('RCHRES_R001')

# Aggregate the list of all SEDTRN end point dependencies
domain = '/STATE/RCHRES_R005'
ep_list = ['RSED4', 'RSED5', 'RSED6']
mello = model_domain_dependencies(state, domain, ep_list)
print("Dependency ordered execution for RSED constants and runnables influencing", domain, "=", mello)
mel_runnable = ModelObject.runnable_op_list(state['op_tokens'], mello)
print("Dependency ordered execution of RSED depemndencies (all)", domain, "=", 
model_element_paths(mello, state))
print("Dependency ordered execution of RSED runnables only for", domain, "=", 
model_element_paths(mel_runnable, state))


# Show order of ops based on dependencies
endpoint = state['model_object_cache']['/STATE/RCHRES_R005/RSED5']
mel = []
mtl = []
model_order_recursive(endpoint, state['model_object_cache'], mel, mtl)
print("Dependency ordered execution for constants and runnables influencing", endpoint.name)
model_element_paths(mel, state)
mel_runnable = ModelObject.runnable_op_list(state['op_tokens'], mel)
print("Dependency ordered execution of runnables only for", endpoint.name)
model_element_paths(mel_runnable, state)


# Just for grins, we can show the dependency using the special action as an end point
specl2 = state['model_object_cache']['/STATE/SPECACTION2']
mel = []
mtl = []
print("Dependency ordered execution for constants and runnables influencing ", rsed4.name)
model_order_recursive(specl2, state['model_object_cache'], mel, mtl)
model_element_paths(mel, state)
mel_runnable = ModelObject.runnable_op_list(state['op_tokens'], mel)
model_element_paths(mel_runnable, state)

