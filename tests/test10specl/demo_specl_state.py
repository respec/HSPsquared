
##########################################################################################
# LOAD HSP2 RUNTIME CODE AND UCI FILE
##########################################################################################
import os, numpy
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2.state import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
hdf5_instance = HDF5("./tests/test10specl/HSP2results/test10specl.h5")


iomanager = IOManager(hdf5_instance)
uci_obj = iomanager.read_uci()
siminfo = uci_obj.siminfo
opseq = uci_obj.opseq
state = init_state_dicts()
state_siminfo_hsp2(uci_obj, siminfo, iomanager, state)

# Add support for dynamic functions to operate on STATE
state_load_dynamics_hsp2(state, iomanager, siminfo)
state_init_hsp2(state, opseq, activities)
state["specactions"] = uci_obj.specactions  # stash the specaction dict in state
om_init_state(state)  # set up operational model specific state entries
specl_load_state(state, iomanager, siminfo)  # traditional special actions
state_load_dynamics_om(state, iomanager, siminfo) 
state_om_model_run_prep(state, iomanager, siminfo)
state_context_hsp2(state, "RCHRES", "R005", "SEDTRN")

domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens = state["domain"], state["state_paths"], state["state_ix"], state["dict_ix"], state["ts_ix"], state["op_tokens"]
ep_list = np.asarray(["RSED1", "RSED2", "RSED3", "RSED4", "RSED5", "RSED6"], dtype='U')
model_exec_list = model_domain_dependencies(state, domain, ep_list, True)
get_domain_state(state_paths, state_ix, domain, ep_list)


##########################################################################################
# SAMPLE MAIN LOOP (does not include timestep changes)
##########################################################################################
start = time.time()
numsteps = siminfo['steps'] * 40
for step in range(numsteps):
    #step_hydr(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)
    #step_adcalc(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)
    #step_cons(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)
    #step_htrch(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)
    step_sedtrn(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list, step, ep_list)
    #step_gqual(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)
    #step_rqual(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list)

end = time.time()
print(
    len(model_exec_list), "components iterated over state_ix", numsteps,
    "time steps took", end - start, "seconds",
)

for key, value in state_paths.items():
    print(f"{key}: {value}")

state['model_root_object'].get_state("/STATE/test10specl/RCHRES_R005/RSED4")
state['model_root_object'].get_state("/STATE/test10specl/RCHRES_R005/RSED5")
state['model_root_object'].get_state("/STATE/test10specl/RCHRES_R005/RSED6")
