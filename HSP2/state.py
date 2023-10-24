''' General routines for SPECL '''

import numpy as np
import time
from pandas import DataFrame, date_range
from pandas.tseries.offsets import Minute
from numba.typed import Dict
from numpy import zeros
from numba import int8, float32, njit, types, typed # import the types
import os
import importlib.util
import sys

def init_state_dicts():
    """
    This contains the base dictionaries used to pass model state amongst modules and custom code plugins
    """
    state = {} # shared state Dictionary, contains numba-ready Dicts 
    state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
    dict_ix = Dict.empty(key_type=types.int64, value_type=types.float64[:,:])
    ts_ix = Dict.empty(key_type=types.int64, value_type=types.float64[:])
    # initialize state for hydr
    # now put all of these Dicts into the state Dict 
    state['state_paths'], state['state_ix'], state['dict_ix'], state['ts_ix'] = state_paths, state_ix, dict_ix, ts_ix
    return state


def find_state_path(state_paths, parent_path, varname):
    """
    We should get really good at using docstrings...
    """
    # this is a bandaid, we should have an object routine that searches the parent for variables or inputs
    var_path = parent_path + "/states/" + str(varname)
    return var_path

def op_path_name(operation, id):
    """
    Used to generate hdf5 operation name in a central fashion to avoid naming convention slip-ups
    """
    tid = str(id).zfill(3)
    path_name = f'{operation}_{operation[0]}{tid}'
    return path_name

def get_op_state_path(operation, id, activity = ''):
    """
    Used to generate hdf5 paths in a central fashion to avoid naming convention slip-ups
    """
    op_name = op_path_name(operation, id) 
    if activity == '':
        op_path = f'/STATE/{op_name}'
    else:
        op_path = f'/STATE/{op_name}/{activity}'
    return op_path


def get_state_ix(state_ix, state_paths, var_path):
    """
    Find the integer key of a variable name in state_ix 
    """
    if not (var_path in list(state_paths.keys())):
        # we need to add this to the state 
        return False # should throw an error 
    var_ix = state_paths[var_path]
    return var_ix


def get_ix_path(state_paths, var_ix):
    """
    Find the integer key of a variable name in state_ix 
    """
    for spath, ix in state_paths.items():
        if var_ix == ix:
            # we need to add this to the state 
            return spath 
    return False

def set_state(state_ix, state_paths, var_path, default_value = 0.0, debug = False):
    """
    Given an hdf5 style path to a variable, set the value 
    If the variable does not yet exist, create it.
    Returns the integer key of the variable in the state_ix Dict
    """
    if not (var_path in state_paths.keys()):
        # we need to add this to the state 
        state_paths[var_path] = append_state(state_ix, default_value)
    var_ix = get_state_ix(state_ix, state_paths, var_path)
    if (debug == True):
        print("Setting state_ix[", var_ix, "], to", default_value)
    state_ix[var_ix] = default_value
    return var_ix


def set_dict_state(state_ix, dict_ix, state_paths, var_path, default_value = {}):
    """
    Given an hdf5 style path to a variable, set the value in the dict
    If the variable does not yet exist, create it.
    Returns the integer key of the variable in the state_ix Dict
    """
    if not (var_path in state_paths.keys()):
        # we need to add this to the state 
        state_paths[var_path] = append_state(state_ix, default_value)
    var_ix = get_state_ix(state_ix, state_paths, var_path)
    return var_ix


def append_state(state_ix, var_value):
    """
    Add a new variable on the end of the state_ix Dict
    Return the key of this new variable
    """
    if (len(state_ix) == 0):
      val_ix = 1
    else:
        val_ix = max(state_ix.keys()) + 1 # next ix value
    state_ix[val_ix] = var_value
    return val_ix

def state_context_hsp2(state, operation, segment, activity):
    state['operation'] = operation 
    state['segment'] = segment # 
    state['activity'] = activity
    # give shortcut to state path for the upcoming function 
    state['domain'] = "/STATE/" + operation + "_" + segment + "/" + activity 

def state_siminfo_hsp2(uci_obj, siminfo):
    # Add crucial simulation info for dynamic operation support
    delt = uci_obj.opseq.INDELT_minutes[0] # get initial value for STATE objects
    siminfo['delt'] = delt
    siminfo['tindex'] = date_range(siminfo['start'], siminfo['stop'], freq=Minute(delt))[1:]
    siminfo['steps'] = len(siminfo['tindex'])

def state_load_dynamics_hsp2(state, io_manager, siminfo):
    # Load any dynamic components if present, and store variables on objects 
    hsp2_local_py = load_dynamics(io_manager, siminfo)
    # if a local file with state_step_hydr() was found in load_dynamics(), we add it to state 
    state['state_step_hydr'] = siminfo['state_step_hydr'] # enabled or disabled 
    state['hsp2_local_py'] = hsp2_local_py # Stores the actual function in state

def hydr_init_ix(state_ix, state_paths, domain):
    # get a list of keys for all hydr state variables
    hydr_state = ["DEP","IVOL","O1","O2","O3","OVOL1","OVOL2","OVOL3","PRSUPY","RO","ROVOL","SAREA","TAU","USTAR","VOL","VOLEV"]
    hydr_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        #var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        hydr_ix[i] = set_state(state_ix, state_paths, var_path, 0.0)
    return hydr_ix    
    
@njit
def hydr_get_ix(state_ix, state_paths, domain):
    # get a list of keys for all hydr state variables
    hydr_state = ["DEP","IVOL","O1","O2","O3","OVOL1","OVOL2","OVOL3","PRSUPY","RO","ROVOL","SAREA","TAU","USTAR","VOL","VOLEV"]
    hydr_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        #var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        hydr_ix[i] = state_paths[var_path]
    return hydr_ix    

# function to dynamically load module, based on "Using imp module" in https://www.tutorialspoint.com/How-I-can-dynamically-import-Python-module#
#def dynamic_module_import(module_name, class_name):
def dynamic_module_import(local_name, local_path, module_name):
    # find_module() is used to find the module in current directory
    # it gets the pointer, path and description of the module
    module = False
    local_spec = False
    try:
        # print ("Looking for local_name, local_path", local_name, local_path)
        local_spec = importlib.util.spec_from_file_location(local_name, local_path)
    except ImportError:
        print ("Imported module {} not found".format(local_name))
    try:
        # load_module dynamically loads the module
        # the parameters are pointer, path and description of the module 
        if (local_spec != False):
            module = importlib.util.module_from_spec(local_spec)
            sys.modules[local_spec.name] = module
            sys.modules[module_name] = module
            local_spec.loader.exec_module(module)
            print("Imported custom module {}".format(local_path))
    except Exception as e:
        # print(e)  this isn't really an exception, it's legit to have no custom python code
        pass
    return module


def load_dynamics(io_manager, siminfo):
    local_path = os.getcwd()
    # try this
    hdf5_path = io_manager._input.file_path
    (fbase, fext) = os.path.splitext(hdf5_path)
    # see if there is a code module with custom python 
    # print("Looking for SPECL with custom python code ", (fbase + ".py"))
    hsp2_local_py = dynamic_module_import(fbase, fbase + ".py", "hsp2_local_py")
    siminfo['state_step_hydr'] = 'disabled'
    if 'state_step_hydr' in dir(hsp2_local_py):
        siminfo['state_step_hydr'] = 'enabled'
        print("state_step_hydr function defined, using custom python code")
    else:
        # print("state_step_hydr function not defined. Using default")
        return False
    return hsp2_local_py