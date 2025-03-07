# MUST RUN CODE FROM demo_specl_iniitialize.py before running this.
from hsp2.hsp2.sedtrn_step import step_sedtrn
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

state['model_root_object'].get_state("/STATE/test10specl.demo/RCHRES_R005/RSED4")
state['model_root_object'].get_state("/STATE/test10specl.demo/RCHRES_R005/RSED5")
state['model_root_object'].get_state("/STATE/test10specl.demo/RCHRES_R005/RSED6")
state['model_root_object'].inputs
state['model_object_cache']['/STATE/test10specl.demo/RCHRES_R005'].inputs

# Show all state var paths and indices
##    print(f"{key}: {value}")
