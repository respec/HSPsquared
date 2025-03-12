# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
import os
import numpy
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2.state import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager

fpath = "./tests/test10eq/HSP2results/test10eq.h5"
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)


io_manager = IOManager(hdf5_instance)
uci_obj = io_manager.read_uci()
siminfo = uci_obj.siminfo
opseq = uci_obj.opseq
# Note: now that the UCI is read in and hdf5 loaded, you can see things like:
# - hdf5_instance._store.keys() - all the paths in the UCI/hdf5
# - finally stash specactions in state, not domain (segment) dependent so do it once
# now load state and the special actions
state = init_state_dicts()
state_siminfo_hsp2(uci_obj, siminfo, io_manager, state)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities)
# - finally stash specactions in state, not domain (segment) dependent so do it once
state["specactions"] = uci_obj.specactions  # stash the specaction dict in state
om_init_state(state)  # set up operational model specific state entries
specl_load_state(state, io_manager, siminfo)  # traditional special actions
state_load_dynamics_om(
    state, io_manager, siminfo
)  # operational model for custom python
# finalize all dynamically loaded components and prepare to run the model
state_om_model_run_prep(state, io_manager, siminfo)

# check dependencies 
# get context first (sets up domain in state) 
# operation = RCHRES, segment = RCHRES_004, activity = SEDTRN
state_context_hsp2(state, "RCHRES", "R005", "SEDTRN")
ep_list = ["RSED4", "RSED5", "RSED6"]
domain = state['domain']
model_exec_list = model_domain_dependencies(state, state['domain'], ep_list)
rsed4 = state['model_object_cache'][domain + "/" + 'RSED4']
test10eq = state['model_object_cache']['/STATE/test10eq']
RCHRESR005 = state['model_object_cache']['/STATE/test10eq/RCHRES_R005']
RCHRESR005.inputs
hydr_get_ix(state['state_ix'], state['state_paths'], domain)
hvars = hydr_state_vars()
get_domain_state(state['state_paths'], state['state_ix'], domain, hvars)
dep,ivol,o1,o2,o3,ovol1,ovol2,ovol3,prsupy,ro,rovol,sarea,tau,ustar,vol,volev = get_domain_state(state['state_paths'], state['state_ix'], domain, hvars)
sedvars = sedtrn_state_vars()
rsed4,rsed5,rsed6 = get_domain_state(state['state_paths'], state['state_ix'], domain, sedvars)
start = time.time()
model_domain_dependencies(state, state["domain"], ep_list)
iterate_models(
    model_exec_list, state["op_tokens"], state["state_ix"], state["dict_ix"], state["ts_ix"], siminfo["steps"], -1)
end = time.time()
print(
    len(model_exec_list),
    "components iterated over state_ix",
    siminfo["steps"],
    "time steps took",
    end - start,
    "seconds",
)
