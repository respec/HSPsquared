# null function to be loaded when not supplied by user
@njit
def state_step_hydr(state_ix, dict_ix, ts_ix, hydr_ix, step):
    fn_defined = False
    return