import sys
from numba import int8, float32, njit, types, typed # import the types

print("Loaded a set of  HSP2 code!")

@njit
def state_step_hydr(state_info, state_paths, state_ix, dict_ix, ts_ix, hydr_ix, step):
    if (step <= 1):
        print("Custom state_step_hydr() called for ", state_info['segment'])
        print("operation", state_info['operation'])
        print("state at start", state_ix)
        print("domain info", state_info)
        print("state_paths", state_paths)
    # Do a simple withdrawal of 10 MGGD from R001
    if (state_info['segment'] == 'R001') and (state_info['activity'] == 'HYDR'):
        state_ix[hydr_ix['O1']] = 10.0 * 1.547
    # Route point source return from segment R001 demand to R005 inflow (IVOL)
    # For demo purposes this will only use the last state_ix value for R001 demand
    # Realistic approach would run all segments simultaneously or use value from ts_ix (ts_ix loading TBD)
    if (state_info['segment'] == 'R005') and (state_info['activity'] == 'HYDR'):
        state_ix[hydr_ix['IVOL']] += 0.85 * state_ix[state_paths['/STATE/RCHRES_R001/HYDR/O1']]
        if (step <= 1):
            print("IVOL after", state_ix[hydr_ix['IVOL']])
    return

