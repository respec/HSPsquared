''' process special actions in this domain

CALL: specl(ui, ts, step, specactions)
    ui is a dictionary with RID specific HSPF UCI like data
    ts is a dictionary with RID specific timeseries
    step is the current simulation step
    specactions is a dictionary with all SPEC-ACTIONS entries
'''

from numba import njit

@njit
def specl(ui, ts, step, state_info, state_paths, state_ix, specactions):
    # there is no need for _specl_ because this code must already be njit
    return