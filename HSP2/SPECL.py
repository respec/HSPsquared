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
def specl(io_manager, siminfo, uci, ts, step, specl_actions):
#def specl(test_param):

    print('Made it to specl()')
    ts['VOL'][step - 1] = ts['VOL'][step - 1] * 5.0
    # run _specl_()
    ###########################################################################
#    test_run = _specl_(test_param)
    ###########################################################################

    # return errors, ERRMSGS
 #   return test_run



# def _specl_(ui, ts, COLIND, OUTDGT, rowsFT, funct, Olabels, OVOLlabels):
@njit
def _specl_(test_param):
    
    print('Made it to _specl_()')
    paramx2 = test_param * 2

    # return errors
    return paramx2
