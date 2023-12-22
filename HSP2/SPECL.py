''' process special actions in this domain

CALL: specl(io_manager, siminfo, uci, ts, state, specl_actions)
    store is the Pandas/PyTable open store
    siminfo is a dictionary with simulation level infor (OP_SEQUENCE for example)
    ui is a dictionary with RID specific HSPF UCI like data
    ts is a dictionary with RID specific timeseries
    state is a dictionary with value of variables at ts[step - 1]
    specl_actions is a dictionary with all SPEC-ACTIONS entries
'''

from numba import njit
import h5py

def specl_load_actions(state, io_manager, siminfo)
    for speca in state['specactions']['ACTIONS'].items():
        

def state_load_dynamics_specl(state, io_manager, siminfo):
    specl_load_actions(state, io_manager, siminfo)
    # others defined below, like:
    # specl_load_uvnames(state, io_manager, siminfo)
    # ...
    return

@njit
def specl(ui, ts, step, state_info, state_paths, state_ix, specactions):
    # ther eis no need for _specl_ because this code must already be njit
    return
    