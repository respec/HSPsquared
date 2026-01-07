# null function to be loaded when not supplied by user
from numba import njit  # import the types
from numba.typed import Dict
from numba import types  # import the types

state_empty = {}  # shared state Dictionary, contains numba-ready Dicts
state_empty["state_paths"] = Dict.empty(
    key_type=types.unicode_type, value_type=types.int64
)
state_empty["state_ix"] = types.float64(zeros(0))
state_empty["dict_ix"] = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
state_empty["ts_ix"] = Dict.empty(key_type=types.int64, value_type=types.float64[:])
state_empty["hsp_segments"] = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type)
state_empty["op_tokens"] = types.int64(zeros((0, 64)))
state_empty["model_exec_list"] = types.int64(zeros(0))
state_empty["op_exec_lists"] = types.int64(zeros((0, 1024)))

# initialize state for hydr
# add a generic place to stash model_data for dynamic components
state_empty["model_data"] = {}

# variables: these could go into individual files later or in object defs
rqual_state_vars = [ 
    "DOX", "BOD", "NO3", "TAM", "NO2", "PO4", "BRTAM1",
    "BRTAM2", "BRPO41", "BRPO42", "CFOREA"
]

@njit
def state_step_hydr(state_info, state_paths, state_ix, dict_ix, ts_ix, hydr_ix, step):
    return
