from numpy import array, zeros, where, int64, asarray
from math import log10, exp
from numba import njit, types
from numba.types import List

# the following imports added to handle special actions
from hsp2.hsp2.state import sedtrn_get_ix, sedtrn_init_ix, get_domain_state, set_domain_state
from hsp2.hsp2.om import pre_step_model, step_model, model_domain_dependencies
from numba.typed import Dict


@njit
def step_sedtrn(domain, state_paths, state_ix, dict_ix, ts_ix, op_tokens, model_exec_list, step, ep_list):
    # model_exec_list: a list of elements (specl etc.) that influence these SEDTRN end points
    # NOTE: this could be cached in dict_ix
    # call related specl/ops pre-steps, such as loading timeseries values
    pre_step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
    # call related specl/ops steps
    step_model( model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
    # get state value at beginning of timestep - python experts will no doubt have a more code efficient method than this
    sand_rsed1, silt_rsed2, clay_rsed3, sand_wt_rsed4, silt_wt_rsed5, clay_wt_rsed6 = get_domain_state(state_paths, state_ix, domain, ep_list)
    
    # now, do sedtrn (simplified for demo purposes)
    tsed1 = sand_rsed1 + silt_rsed2 + clay_rsed3
    tsed2 = sand_wt_rsed4 + silt_wt_rsed5 + clay_wt_rsed6
    sand_t_rsed7 = sand_rsed1 + sand_wt_rsed4 
    silt_t_rsed8 = silt_rsed2 + silt_wt_rsed5
    clay_t_rsed9 = clay_rsed3 + clay_wt_rsed6
    tsed3 = sand_t_rsed7 + silt_t_rsed8 + clay_t_rsed9
    
    # pass values back to state
    state_vals = [sand_rsed1, silt_rsed2, clay_rsed3, sand_wt_rsed4, silt_wt_rsed5, clay_wt_rsed6]
    set_domain_state(state_paths, state_ix, domain, ep_list, state_vals)
    return
