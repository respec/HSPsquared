''' process special actions in this domain
Notes:
  - code for parsing UCI SPEC-ACTIONS is in HSP2tools/readUCI.py
  - code for object classes that transform parsed data into OP codes for OM and STATE support
    is in this directory tree as om_special_[action type].py,
    - Ex: om_special_action.py contains object support and runtime functions for classic ACTIONS
'''

from numba import njit
from pandas import DataFrame, date_range
import h5py

def specl_load_actions(state, io_manager, siminfo):
    if 'ACTIONS' in state['specactions']:
        dc = state['specactions']['ACTIONS']
        for ix in dc.index:
            # add the items to the state['model_data'] dict
            speca = dc[ix:(ix+1)]
            # need to add a name attribute
            opname = 'SPEC' + 'ACTION' + str(ix)
            state['model_data'][opname] = {}
            state['model_data'][opname]['name'] = opname
            for ik in speca.keys():
                #print("looking for speca key ", ik)
                state['model_data'][opname][ik] = speca.to_dict()[ik][ix]   # add subscripts?
                if ik == 'VARI':
                    if len(speca.to_dict()['S1'][ix]) > 0:
                        state['model_data'][opname][ik] += speca.to_dict()['S1'][ix]
                    if len(speca.to_dict()['S2'][ix]) > 0:
                        state['model_data'][opname][ik] += speca.to_dict()['S2'][ix]
            state['model_data'][opname]['object_class'] = 'SpecialAction'
            #print("model_data", ix, " = ", state['model_data'][opname])
    return

def state_load_dynamics_specl(state, io_manager, siminfo):
    specl_load_actions(state, io_manager, siminfo)
    # others defined below, like:
    # specl_load_uvnames(state, io_manager, siminfo)
    # ...
    return

'''
# the code specl() is deprecated in favor of execution inside OM
# see om_special_action.py for example of object support and runtime functions for classic ACTIONS
CALL: specl(ui, ts, step, state_info, state_paths, state_ix, specactions)
    store is the Pandas/PyTable open store
    siminfo is a dictionary with simulation level infor (OP_SEQUENCE for example)
    ui is a dictionary with RID specific HSPF UCI like data
    ts is a dictionary with RID specific timeseries
    state is a dictionary with value of variables at ts[step - 1]
    specl_actions is a dictionary with all SPEC-ACTIONS entries
'''

@njit
def specl(ui, ts, step, state_info, state_paths, state_ix, specactions):
    # ther eis no need for _specl_ because this code must already be njit
    return
    