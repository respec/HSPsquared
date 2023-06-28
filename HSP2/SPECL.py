''' process special actions in this domain

CALL: specl(ui, ts, step, specactions)
    ui is a dictionary with RID specific HSPF UCI like data
    ts is a dictionary with RID specific timeseries
    step is the current simulation step
    specactions is a dictionary with all SPEC-ACTIONS entries
'''

from numba import njit

@njit
def specl(ui, ts, step, specactions):

    errors_specl = _specl_(ui, ts, step, specactions)
    
    return errors_specl


@njit
def _specl_(ui, ts, step, specactions):
    # todo determine best way to do error handling in specl
    errors_specl = zeros(int(1)).astype(int64)

    return errors_specl    