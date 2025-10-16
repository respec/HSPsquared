"""General routines for SPECL"""

import numpy as np
from pandas import date_range
from pandas.tseries.offsets import Minute
from numba.typed import Dict
from numba.experimental import jitclass
from numpy import zeros, int32, asarray as npasarray
from numba import njit, types, typeof  # import the types
import os
import importlib.util
import sys
from hsp2.hsp2.utilities import make_class_spec


# this is temporary, these will be merged with state soon
state_ix = npasarray(zeros(1), dtype="float64")
model_exec_list = npasarray(zeros(1), dtype="int64")
op_tokens = int32(zeros((1,64)))
op_exec_lists = int32(zeros((1,1024)))
# note: tested 32-bit key and saw absolutely no improvement, so test 32bit value
state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
hsp_segments = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type)
ts_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
ts_ix = Dict.empty(key_type=types.int64, value_type=types.float64[:])

state_paths_ty = ('state_paths', typeof(state_paths))
model_exec_list_ty = ('model_exec_list', typeof(model_exec_list))
hsp_segments_ty = ('hsp_segments', typeof(hsp_segments))
op_tokens_ty = ('op_tokens', typeof(op_tokens))
state_ix_ty = ('state_ix', typeof(state_ix))
ts_ix_ty = ('ts_ix', typeof(ts_ix))
ts_paths_ty = ('ts_paths', typeof(ts_paths))
model_root_name_ty = ('model_root_name', types.unicode_type)
state_step_hydr_ty = ('state_step_hydr', types.unicode_type)
hsp2_local_py_ty = ('hsp2_local_py', types.boolean)
op_exec_lists_ty = ('op_exec_lists', typeof(op_exec_lists))
state_spec = [state_paths_ty, state_ix_ty, ts_paths_ty, ts_ix_ty,
              model_root_name_ty, state_step_hydr_ty, hsp2_local_py_ty,
              hsp_segments_ty, op_tokens_ty, op_exec_lists_ty, model_exec_list_ty]

@jitclass(state_spec)
class state_object:
    def __init__(self, num_ops=5000):
        self.state_ix = zeros(num_ops)
        self.state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        self.hsp_segments = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type)
        self.ts_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        self.ts_ix = Dict.empty(key_type=types.int64, value_type=types.float64[:])
        self.state_step_hydr = "disabled"
        self.model_root_name = ""
        self.hsp2_local_py = False
        # Note: in the type declaration above we are alloweed to use the shortened form
        #         op_tokens = int32(zeros((1,64)))
        #       but in jited class that throws an error and we have to use the 
        #       form op_tokens.astype(int32) to do the type cast
        op_tokens = zeros( (num_ops,64) )
        self.op_tokens = op_tokens.astype(int32)
        op_exec_lists = zeros( (num_ops,1024) )
        self.op_exec_lists = op_exec_lists.astype(int32)
        return



def op_path_name(operation, id):
    """
    Used to generate hdf5 operation name in a central fashion to avoid naming convention slip-ups
    """
    tid = str(id).zfill(3)
    path_name = f"{operation}_{operation[0]}{tid}"
    return path_name


def get_state_ix(state_ix, state_paths, var_path):
    """
    Find the integer key of a variable name in state_ix
    """
    if var_path not in list(state_paths.keys()):
        # we need to add this to the state
        return False  # should throw an error
    var_ix = state_paths[var_path]
    return var_ix


def get_ix_path(state_paths, var_ix):
    """
    Find the path of a variable with integer key in state_ix
    """
    for spath, ix in state_paths.items():
        if var_ix == ix:
            # we need to add this to the state
            return spath
    return False


def set_state(state_ix, state_paths, var_path, default_value=0.0, debug=False):
    """
    Given an hdf5 style path to a variable, set the value
    If the variable does not yet exist, create it.
    Returns the integer key of the variable in the state_ix Dict
    """
    if var_path not in state_paths.keys():
        # we need to add this to the state
        state_paths[var_path] = append_state(state_ix, default_value)
    var_ix = get_state_ix(state_ix, state_paths, var_path)
    if debug == True:
        print("Setting state_ix[", var_ix, "], to", default_value)
    state_ix[var_ix] = default_value
    return var_ix


def state_add_ts(state, var_path, default_value=0.0, debug=False):
    """
    Given an hdf5 style path to a variable, set the value
    If the variable does not yet exist, create it.
    Returns the integer key of the variable in the state_ix Dict
    """
    if var_path not in state.state_paths.keys():
        # we need to add this to the state
        state.state_paths[var_path] = append_state(state.state_ix, default_value)
    var_ix = get_state_ix(state.state_ix, state.state_paths, var_path)
    if debug == True:
        print("Setting state_ix[", var_ix, "], to", default_value)
    # siminfo needs to be in the model_data array of state.  Can be populated by HSP2 or standalone by ops model
    state["ts_ix"][var_ix] = np.full_like(
        zeros(state["model_data"]["siminfo"]["steps"]), default_value
    )
    return var_ix


def set_dict_state(state_ix, dict_ix, state_paths, var_path, default_value={}):
    """
    Given an hdf5 style path to a variable, set the value in the dict
    If the variable does not yet exist, create it.
    Returns the integer key of the variable in the state_ix Dict
    """
    if var_path not in state_paths.keys():
        # we need to add this to the state
        state_paths[var_path] = append_state(state_ix, default_value)
    var_ix = get_state_ix(state_ix, state_paths, var_path)
    return var_ix


def append_state(state_ix, var_value):
    """
    Add a new variable on the end of the state_ix Dict
    Return the key of this new variable
    """
    if len(state_ix) == 0:
        val_ix = 1
    else:
        val_ix = max(state_ix.keys()) + 1  # next ix value
    state_ix[val_ix] = var_value
    return val_ix


def state_siminfo_hsp2(state, parameter_obj, siminfo, io_manager):
    # Add crucial simulation info for dynamic operation support
    delt = parameter_obj.opseq.INDELT_minutes[0]  # get initial value for STATE objects
    siminfo["delt"] = delt
    siminfo["tindex"] = date_range(
        siminfo["start"], siminfo["stop"], freq=Minute(delt)
    )[1:]
    siminfo["steps"] = len(siminfo["tindex"])
    hdf5_path = io_manager._input.file_path
    (fbase, fext) = os.path.splitext(hdf5_path)
    state.model_root_name = os.path.split(fbase)[1]  # takes the text before .h5


def state_context_hsp2(state, operation, segment, activity):
    # this establishes domain info so that a module can know its paths
    state["operation"] = operation
    state["segment"] = segment  #
    state["activity"] = activity
    # give shortcut to state path for the upcoming function
    # insure that there is a model object container
    seg_name = op_path_name(operation, segment)
    seg_path = "/STATE/" + state.model_root_name + "/" + seg_name
    if seg_name not in state.hsp_segments.keys():
        state.hsp_segments[seg_name] = seg_path
    state.domain = seg_path  # + "/" + activity   # may want to comment out activity?

def state_init_hsp2(state, opseq, activities):
    # This sets up the state entries for all state compatible HSP2 model variables
    # print("STATE initializing contexts.")
    for _, operation, segment, delt in opseq.itertuples():
        if operation != "GENER" and operation != "COPY":
            for activity, function in activities[operation].items():
                # set up named paths for model operations
                seg_name = op_path_name(operation, segment)
                seg_path = "/STATE/" + state.model_root_name + "/" + seg_name
                segid = set_state(
                    state.state_ix, state.state_paths, seg_path, 0.0
                )
                ep_list = []
                if activity == "HYDR":
                    state_context_hsp2(state, operation, segment, activity)
                    ep_list = hydr_init_ix(state, state.domain)
                elif activity == "SEDTRN":
                    state_context_hsp2(state, operation, segment, activity)
                    ep_list = sedtrn_init_ix(state, state.domain)
                elif activity == "SEDMNT":
                    state_context_hsp2(state, operation, segment, activity)
                    ep_list = sedmnt_init_ix(state, state.domain)
                elif activity == "RQUAL":
                    state_context_hsp2(state, operation, segment, activity)
                    ep_list = rqual_init_ix(state, state.domain)
                # Register list of elements to execute if any
                op_exec_list = model_domain_dependencies(
                    state, state.domain, ep_list, True
                )
                state.op_exec_lists[segid] = op_exec_list

def load_dynamics_hsp2(state, io_manager, siminfo):
    # Load any dynamic components if present, and store variables on objects
    # if a local file with state_step_hydr() was found in load_dynamics(), we add it to state
    state.state_step_hydr = siminfo.state_step_hydr  # enabled or disabled
    state.hsp2_local_py = load_dynamics(io_manager, siminfo)  # Stores the actual function in state

def state_load_hdf5_components(
    io_manager,
    siminfo,
    op_tokens,
    state_paths,
    state_ix,
    dict_ix,
    ts_ix,
    model_object_cache,
):
    # Implement population of model_object_cache etc from components in a hdf5 such as Special ACTIONS
    return



@njit
def get_domain_state(state_paths, state_ix, domain, varkeys):
    # get values for a set of variables in a domain
    # will not check for the index in state_ix, and will fail if a non-scalar value is needed (like from dict_ix)
    # if varkeys = False, assume that we want all the variables
    # from the domain, that are predetermined ahead of time, and should save performance
    ret_vals = np.zeros(len(varkeys))
    j = 0
    for i in varkeys:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        # print(var_path)
        ix = state_paths[var_path]
        # print("ix",ix)
        ret_vals[j] = state_ix[ix]
        j += 1
    return ret_vals


@njit
def set_domain_state(state_paths, state_ix, domain, varkeys, state_vals):
    # get values for a set of variables in a domain
    # will not check for the index in state_ix, and will fail if a non-scalar value is needed (like from dict_ix)
    # if varkeys = False, assume that we want all the variables
    # from the domain, that are predetermined ahead of time, and should save performance
    j = 0
    for i in varkeys:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        # print(var_path)
        ix = state_paths[var_path]
        state_ix[ix] = state_vals[j]
        j += 1
    return True


def hydr_state_vars():
    return [
        "DEP",
        "IVOL",
        "O1",
        "O2",
        "O3",
        "OVOL1",
        "OVOL2",
        "OVOL3",
        "PRSUPY",
        "RO",
        "ROVOL",
        "SAREA",
        "TAU",
        "USTAR",
        "VOL",
        "VOLEV",
    ]


def hydr_init_ix(state, domain):
    # get a list of keys for all hydr state variables
    hydr_state = hydr_state_vars()
    hydr_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        hydr_ix[i] = set_state(state.state_ix, state.state_paths, var_path, 0.0)
    return hydr_ix


def sedtrn_state_vars():
    sedtrn_state = ["RSED1", "RSED2", "RSED3", "RSED4", "RSED5", "RSED6"]
    return sedtrn_state


def sedtrn_init_ix(state, domain):
    # get a list of keys for all sedtrn state variables
    sedtrn_state = sedtrn_state_vars()
    sedtrn_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedtrn_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        sedtrn_ix[i] = set_state(state.state_ix, state.state_paths, var_path, 0.0)
    return sedtrn_ix


def sedmnt_state_vars():
    sedmnt_state = ["DETS"]
    return sedmnt_state


def sedmnt_init_ix(state, domain):
    # get a list of keys for all sedmnt state variables
    sedmnt_state = sedmnt_state_vars()
    sedmnt_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedmnt_state:
        var_path = domain + "/" + i
        sedmnt_ix[i] = set_state(state.state_ix, state.state_paths, var_path, 0.0)
    return sedmnt_ix


def rqual_state_vars():
    rqual_state = [
        "DOX",
        "BOD",
        "NO3",
        "TAM",
        "NO2",
        "PO4",
        "BRTAM1",
        "BRTAM2",
        "BRPO41",
        "BRPO42",
        "CFOREA",
    ]
    return rqual_state


def rqual_init_ix(state, domain):
    # get a list of keys for all rqual state variables
    rqual_state = rqual_state_vars()
    rqual_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in rqual_state:
        var_path = domain + "/" + i
        rqual_ix[i] = set_state(state.state_ix, state.state_paths, var_path, 0.0)
    return rqual_ix


@njit
def hydr_get_ix(state_ix, state_paths, domain):
    # get a list of keys for all hydr state variables
    hydr_state = [
        "DEP",
        "IVOL",
        "O1",
        "O2",
        "O3",
        "OVOL1",
        "OVOL2",
        "OVOL3",
        "PRSUPY",
        "RO",
        "ROVOL",
        "SAREA",
        "TAU",
        "USTAR",
        "VOL",
        "VOLEV",
    ]
    hydr_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        hydr_ix[i] = state_paths[var_path]
    return hydr_ix


@njit
def sedtrn_get_ix(state_ix, state_paths, domain):
    # get a list of keys for all sedtrn state variables
    sedtrn_state = ["RSED4", "RSED5", "RSED6"]
    sedtrn_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedtrn_state:
        var_path = domain + "/" + i
        sedtrn_ix[i] = state_paths[var_path]
    return sedtrn_ix


@njit
def sedmnt_get_ix(state_ix, state_paths, domain):
    # get a list of keys for all sedmnt state variables
    sedmnt_state = ["DETS"]
    sedmnt_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedmnt_state:
        var_path = domain + "/" + i
        sedmnt_ix[i] = state_paths[var_path]
    return sedmnt_ix


@njit
def rqual_get_ix(state_ix, state_paths, domain):
    # get a list of keys for all sedmnt state variables
    rqual_state = [
        "DOX",
        "BOD",
        "NO3",
        "TAM",
        "NO2",
        "PO4",
        "BRTAM1",
        "BRTAM2",
        "BRPO41",
        "BRPO42",
        "CFOREA",
    ]
    rqual_ix = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in rqual_state:
        var_path = domain + "/" + i
        rqual_ix[i] = state_paths[var_path]
    return rqual_ix


# function to dynamically load module, based on "Using imp module" in https://www.tutorialspoint.com/How-I-can-dynamically-import-Python-module#
# def dynamic_module_import(module_name, class_name):
def dynamic_module_import(local_name, local_path, module_name):
    # find_module() is used to find the module in current directory
    # it gets the pointer, path and description of the module
    module = False
    local_spec = False
    try:
        # print ("Looking for local_name, local_path", local_name, local_path)
        local_spec = importlib.util.spec_from_file_location(local_name, local_path)
    except ImportError:
        print("Imported module {} not found".format(local_name))
    try:
        # load_module dynamically loads the module
        # the parameters are pointer, path and description of the module
        if local_spec != False:
            module = importlib.util.module_from_spec(local_spec)
            sys.modules[local_spec.name] = module
            sys.modules[module_name] = module
            local_spec.loader.exec_module(module)
            print("Imported custom module {}".format(local_path))
    except Exception:
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
    siminfo.state_step_hydr = "disabled"
    if "state_step_hydr" in dir(hsp2_local_py):
        siminfo.state_step_hydr = "enabled"
        print("state_step_hydr function defined, using custom python code")
    else:
        # print("state_step_hydr function not defined. Using default")
        return False
    return hsp2_local_py
