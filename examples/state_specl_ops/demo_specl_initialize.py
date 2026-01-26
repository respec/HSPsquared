
##########################################################################################
# LOAD HSP2 RUNTIME CODE AND UCI FILE
##########################################################################################
import os, numpy
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2.state import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import io_manager
hdf5_instance = HDF5("./tests/test10specl/HSP2results/test10specl.demo.h5")


io_manager = io_manager(hdf5_instance)
uci_obj = io_manager.read_uci()
siminfo = uci_obj.siminfo
opseq = uci_obj.opseq
state = init_state_dicts()
state_siminfo_hsp2(uci_obj, siminfo, io_manager, state)

# Add support for dynamic functions to operate on STATE
state_load_dynamics_hsp2(state, io_manager, siminfo)
state_init_hsp2(state, opseq, activities)
state["specactions"] = uci_obj.specactions  # stash the specaction dict in state
om_init_state(state)  # set up operational model specific state entries
specl_load_state(state, io_manager, siminfo)  # traditional special actions
state_load_dynamics_om(state, io_manager, siminfo) 
state_om_model_run_prep(state, io_manager, siminfo)
state_context_hsp2(state, "RCHRES", "R005", "SEDTRN")

domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens = state["domain"], state["state_paths"], state["state_ix"], state["dict_ix"], state["ts_ix"], state["op_tokens"]
ep_list = np.asarray(["RSED1", "RSED2", "RSED3", "RSED4", "RSED5", "RSED6"], dtype='U')
model_exec_list = model_domain_dependencies(state, domain, ep_list, True)
get_domain_state(state_paths, state_ix, domain, ep_list)


