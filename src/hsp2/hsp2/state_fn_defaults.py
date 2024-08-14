# null function to be loaded when not supplied by user
from numba import float32, int8, njit, typed, types  # import the types


@njit
def state_step_hydr(state_info, state_paths, state_ix, dict_ix, ts_ix, hydr_ix, step):
    return
