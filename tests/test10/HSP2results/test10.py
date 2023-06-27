import sys
from numba import int8, float32, njit, types, typed # import the types

print("Loaded a set of  HSP2 code!")

@njit
def state_step_hydr(state_info, state_ix, dict_ix, ts_ix, hydr_ix, step):
    fn_defined = False
    state_ix[hydr_ix['O1']] = 10.0 * 1.547
    if (step <= 1):
        print("Custom state_step_hydr() called")
        print("state at start", state_ix)
        print("domain info", state_info)
    return
