''' process special actions in this domain

CALL: specl(io_manager, siminfo, uci, ts, state, specl_actions)
    store is the Pandas/PyTable open store
    siminfo is a dictionary with simulation level infor (OP_SEQUENCE for example)
    ui is a dictionary with RID specific HSPF UCI like data
    ts is a dictionary with RID specific timeseries
    state is a dictionary with value of ts[step - 1]
    specl_actions is a dictionary with all SPEC-ACTIONS entries
'''

from numba import njit

@njit
# def specl(io_manager, siminfo, uci, ts, step, specl_actions):
def specl(ui, ts, step, state_info, state_paths, state_ix, specactions):

    # print('Made it to specl()')
    ts = _specl_(ui, ts, step, specactions)
    
    # return errors, ERRMSGS
    # return ts



# def _specl_(ui, ts, COLIND, OUTDGT, rowsFT, funct, Olabels, OVOLlabels):
@njit
def _specl_(ui, ts, step, state_info, state_paths, state_ix, specactions):
    return
    