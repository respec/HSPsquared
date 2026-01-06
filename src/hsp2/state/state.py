"""General routines for SPECL"""

import importlib.util
import os
import sys

import numpy as np
from numba import njit, types, typeof  # import the types supplies int64, float64
from numba.experimental import jitclass
from numba.typed import Dict as ntdict
from numpy import zeros, float64 as npfloat64, int64 as npint64
from pandas import date_range
from pandas.tseries.offsets import Minute

# Beginning in operation these are likely to be located in model objects when we go fully to that level.
# But for now, they are here to maintain compatiility with the existing code base
# Combine these into a spec to create the class
tindex = date_range("1984-01-01", "2020-12-31", freq=Minute(60))

state_spec = [
    # the first entries here are NP arrays, fixed dimenstions, and fast
    ("state_ix", typeof(np.asarray(zeros(1), dtype="float64")) ),
    ("op_tokens", typeof(types.int64(zeros((1, 64)))) ),
    ("op_exec_lists", typeof(types.int64(zeros((1, 1024)))) ),
    ("model_exec_list", typeof(np.asarray(zeros(1), dtype="int64")) ),
    ("tindex", typeof(tindex.to_numpy()) ),
    # dict_ix SHOULD BE an array, this is TBD.  Likely defer till OM class runtimes
    ("dict_ix", types.DictType(types.int64, types.float64[:, :]) ),
    # below here are dictionaries as they are not used in runtime and can be slow
    ("state_paths", types.DictType(types.unicode_type, types.int64) ),
    ("ts_paths", types.DictType(types.unicode_type, types.float64[:]) ),
    ("ts_ix", types.DictType(types.int64, types.float64[:]) ),
    ("last_id", types.int64),
    ("model_root_name", types.unicode_type),
    ("state_step_hydr", types.unicode_type),
    ("hsp2_local_py", types.boolean),
    ("num_ops", types.int64),
    ("operation", types.unicode_type),
    ("segment", types.unicode_type),
    ("activity", types.unicode_type),
    ("domain", types.unicode_type),
    ("state_step_om", types.unicode_type),
    ("hsp_segments", types.DictType(types.unicode_type, types.unicode_type) )
]


@jitclass(state_spec)
class state_class:
    def __init__(self):
        self.num_ops = 0
        # IMPORTANT these are handled as nparray as numba Dict would be super slow. 
        # Note: in the type declaration above we are alloweed to use the shortened form op_tokens = int64(zeros((1,64)))
        #       but in jited class that throws an error and we have to use the form op_tokens.astype(int64)
        #       to do the type cast
        return
    
    @property
    def size(self):
        return self.state_ix.size
    
    def append_state(self, var_value):
        val_ix = self.size  # next ix value= size since ix starts from zero
        self.state_ix = np.append(self.state_ix, var_value)
        self.last_id = val_ix
        self.resize()
        return val_ix
    
    def set_token(self, var_ix, tokens, debug=False):
        if var_ix not in range(len(self.state_ix)):
            if debug:
                print("Undefined index value,", var_ix, ", provided for set_token()")
            return False
        if var_ix not in range(np.shape(self.op_tokens)[0]):
            if debug:
                print("set_token called for ix", var_ix, ", need to expand")
            self.resize()
        # in a perfect world we would insure that the length of tokens is correct
        # and if not, we would resize.  But this is only called from ModelObject
        # and its methods add_op_tokens() and model_format_ops(ops) enforce the
        # length limit described by ModelObject.max_token_length (64) which must match
        self.op_tokens[var_ix] = tokens
    
    def resize(self, debug=False):
        num_ops = self.size
        # print("state_ix has", num_ops, "elements")
        ops_needed = num_ops - np.shape(self.op_tokens)[0]
        if ops_needed == 0:
            # print("resize op_tokens unneccesary, state has", self.size,"indices and op_tokens has", np.shape(self.op_tokens)[0], "elements")
            return
        if debug:
            print("op_tokens needs", ops_needed, "slots")
        add_ops = zeros((ops_needed, 64))
        # print("Created add_ops with", ops_needed, "slots")
        # we use the 3rd param "axis=1" to prevent flattening of array
        if self.op_tokens.size == 0:
            if debug:
                print("Creating op_tokens")
            self.op_tokens = add_ops.astype(npint64)
        else:
            if debug:
                print("Merging op_tokens")
            add_ops = np.append(self.op_tokens, add_ops, 0)
            self.op_tokens = add_ops.astype(npint64)
        ops_needed = num_ops - np.shape(self.op_exec_lists)[0]
        el_width = np.shape(self.op_exec_lists)[1]
        if debug:
            print("op_exec_lists needs", ops_needed, "slots")
        if ops_needed == 0:
            return
        add_ops = zeros((ops_needed, el_width))
        # we use the 3rd param "axis=1" to prevent flattening of array
        if self.op_exec_lists.size == 0:
            if debug:
                print("Creating op_exec_lists")
            self.op_exec_lists = add_ops.astype(npint64)
        else:
            if debug:
                print("Merging op_exec_lists")
            add_ops = np.append(self.op_exec_lists, add_ops, 0)
            self.op_exec_lists = add_ops.astype(npint64)
        return
    
    def set_exec_list(self, ix, op_exec_list):
        for i in range(len(op_exec_list)):
            self.op_exec_lists[ix][i] = op_exec_list[i]
    
    def set_state(self, var_path, var_value=0.0, debug=False):
        """
        Given an hdf5 style path to a variable, set the value
        If the variable does not yet exist, create it.
        Returns the integer key of the variable in the state_ix Dict
        """
        if var_path not in self.state_paths:
            # we need to add this to the state
            var_ix = self.append_state(var_value)
            self.state_paths[var_path] = var_ix
        else:
            var_ix = self.get_state_ix(var_path)
            self.state_ix[var_ix] = var_value
        if debug:
            print("Setting state_ix[", var_ix, "], to", var_value)
        return var_ix
    
    def get_state_ix(self, var_path):
        """
        Find the integer key of a variable name in state_ix
        """
        if var_path == False:
            # handle a bad path with False
            return False
        if var_path not in self.state_paths:
            # we need to add this to the state
            return False  # should throw an error
        var_ix = self.state_paths[var_path]
        return var_ix
    
    def get_ix_path(self, var_ix):
        """
        Find the path of a variable with integer key in state_ix
        """
        spath = None
        for spath, ix in self.state_paths.items():
            if var_ix == ix:
                # we need to add this to the state
                return spath
        return spath

def init_state(state_class):
    state_ix = zeros(state_class.num_ops)
    state_class.state_ix = state_ix.astype(npfloat64)
    op_tokens = zeros((state_class.num_ops, 64))
    state_class.op_tokens = op_tokens.astype(npint64)
    # TODO: move to individual objects in OM/RCHRES/PERLND/...
    op_exec_lists = zeros((state_class.num_ops, 1024))
    state_class.op_exec_lists = op_exec_lists.astype(npint64)
    # TODO: is this even needed? Since each domain has it's own exec list?
    model_exec_list = zeros(state_class.num_ops)
    state_class.model_exec_list = model_exec_list.astype(npint64)
    # Done with nparray initializations
    # this dict_ix approach is inherently slow, and should be replaced by some other np table type
    # on an as-needed basis if possible.  Especially for dataMatrix types which are supposed to be fast
    # state can still get values via get_state, by grabbing a reference object and then accessing it's storage
    state_class.dict_ix = ntdict.empty(key_type=types.int64, value_type=types.float64[:, :])
    state_class.dict_ix = ntdict.empty(key_type=types.int64, value_type=types.float64[:, :])
    state_class.state_paths = ntdict.empty(
        key_type=types.unicode_type, value_type=types.int64
    )
    state_class.hsp_segments = ntdict.empty(
        key_type=types.unicode_type, value_type=types.unicode_type
    )
    state_class.ts_paths = ntdict.empty(
        key_type=types.unicode_type, value_type=types.float64[:]
    )
    state_class.ts_ix = ntdict.empty(key_type=types.int64, value_type=types.float64[:])
    state_class.state_step_om = "disabled"
    state_class.state_step_hydr = "disabled"
    state_class.model_root_name = ""
    state_class.operation = ""
    state_class.segment = ""
    state_class.activity = ""
    state_class.domain = ""
    state_class.last_id = 0
    state_class.hsp2_local_py = False


@njit(cache=True)
def make_state_class():
    sc = state_class()
    return(sc)

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
    state.ts_ix[var_ix] = np.full_like(
        zeros(om_operations["model_data"]["steps"]), default_value
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
    state.tindex = siminfo["tindex"].to_numpy()
    hdf5_path = io_manager._input.file_path
    (fbase, fext) = os.path.splitext(hdf5_path)
    state.model_root_name = os.path.split(fbase)[1]  # takes the text before .h5


def state_context_hsp2(state, operation, segment, activity):
    # this establishes domain info so that a module can know its paths
    state.operation = operation
    state.segment = segment  #
    state.activity = activity
    # give shortcut to state path for the upcoming function
    # insure that there is a model object container
    seg_name = operation + "_" + segment
    seg_path = "/STATE/" + state.model_root_name + "/" + seg_name
    if seg_name not in state.hsp_segments.keys():
        state.hsp_segments[seg_name] = seg_path
    state.domain = seg_path  # + "/" + activity   # may want to comment out activity?


def state_init_hsp2(state, opseq, activities, om_operations):
    # This sets up the state entries for all state compatible HSP2 model variables
    # print("STATE initializing contexts.")
    for _, operation, segment, delt in opseq.itertuples():
        if operation != "GENER" and operation != "COPY":
            for activity, function in activities[operation].items():
                # set up named paths for model operations
                seg_name = operation + "_" + segment
                seg_path = "/STATE/" + state.model_root_name + "/" + seg_name
                state.set_state(seg_path, 0.0)
                if activity == "HYDR":
                    state_context_hsp2(state, operation, segment, activity)
                elif activity == "SEDTRN":
                    state_context_hsp2(state, operation, segment, activity)
                elif activity == "SEDMNT":
                    state_context_hsp2(state, operation, segment, activity)
                elif activity == "RQUAL":
                    state_context_hsp2(state, operation, segment, activity)


def state_load_dynamics_hsp2(state, io_manager, siminfo):
    # Load any dynamic components if present, and store variables on objects
    # if a local file with state_step_hydr() was found in load_dynamics(), we add it to state
    state.hsp2_local_py = load_dynamics(
        io_manager, siminfo
    )  # Stores the actual function in state
    state.state_step_hydr = siminfo["state_step_hydr"]  # enabled or disabled


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


def hydr_init_ix(state, domain, debug = False):
    # get a list of keys for all hydr state variables
    hydr_state = hydr_state_vars()
    hydr_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        if debug:
            print("initializing", var_path)
        hydr_ix[i] = state.set_state(var_path, 0.0)
    return hydr_ix


def sedtrn_state_vars():
    sedtrn_state = ["RSED1", "RSED2", "RSED3", "RSED4", "RSED5", "RSED6"]
    return sedtrn_state


def sedtrn_init_ix(state, domain):
    # get a list of keys for all sedtrn state variables
    sedtrn_state = sedtrn_state_vars()
    sedtrn_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedtrn_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        sedtrn_ix[i] = state.set_state(var_path, 0.0)
    return sedtrn_ix


def sedmnt_state_vars():
    sedmnt_state = ["DETS"]
    return sedmnt_state


def sedmnt_init_ix(state, domain):
    # get a list of keys for all sedmnt state variables
    sedmnt_state = sedmnt_state_vars()
    sedmnt_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedmnt_state:
        var_path = domain + "/" + i
        sedmnt_ix[i] = state.set_state(var_path, 0.0)
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
    rqual_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in rqual_state:
        var_path = domain + "/" + i
        rqual_ix[i] = state.set_state(var_path, 0.0)
    return rqual_ix


@njit
def hydr_get_ix(state, domain):
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
    # print(state.state_paths)
    hydr_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in hydr_state:
        # var_path = f'{domain}/{i}'
        var_path = domain + "/" + i
        # print("looking for:", var_path)
        hydr_ix[i] = state.get_state_ix(var_path)
    return hydr_ix


@njit
def sedtrn_get_ix(state, domain):
    # get a list of keys for all sedtrn state variables
    sedtrn_state = ["RSED4", "RSED5", "RSED6"]
    sedtrn_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedtrn_state:
        var_path = domain + "/" + i
        sedtrn_ix[i] = state.get_state_ix(var_path)
    return sedtrn_ix


@njit
def sedmnt_get_ix(state, domain):
    # get a list of keys for all sedmnt state variables
    sedmnt_state = ["DETS"]
    sedmnt_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in sedmnt_state:
        var_path = domain + "/" + i
        sedmnt_ix[i] = state.get_state_ix(var_path)
    return sedmnt_ix


@njit
def rqual_get_ix(state, domain):
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
    rqual_ix = ntdict.empty(key_type=types.unicode_type, value_type=types.int64)
    for i in rqual_state:
        var_path = domain + "/" + i
        rqual_ix[i] = state.get_state_ix(var_path)
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
    siminfo["state_step_hydr"] = "disabled"
    if "state_step_hydr" in dir(hsp2_local_py):
        siminfo["state_step_hydr"] = "enabled"
        print("state_step_hydr function defined, using custom python code")
    else:
        # print("state_step_hydr function not defined. Using default")
        return False
    return hsp2_local_py
