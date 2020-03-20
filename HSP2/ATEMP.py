''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of HSPF HPERAIR.FOR module. '''


from numba import njit, types
from numba.typed import Dict
from numpy import empty, zeros, int64
from utilities import hoursval

ERRMSGS = ()


def atemp(store, siminfo, uic, ts):
    ''' high level driver for air temperature module'''

    segment = siminfo['segment']

    ts['LAPSE'] = hoursval(siminfo, store['TIMESERIES/LAPSE_Table'])

    ui = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for name in set(uic.keys()) & {'FLAGS', 'PARAMETERS', 'STATES'}:
        for key, value in uic[name][segment].items():
            if type(value) in {int, float}:
                ui[key] = float(value)

    ui['k']      = siminfo['delt']  * 0.000833        # convert to in/timestep
    ui['steps']  = siminfo['steps']
    ui['errlen'] = len(ERRMSGS)

    ############################################################################
    errors = atemp_(ui, ts)                     # run ATEMP_ simulation code
    ############################################################################

    return errors, ERRMSGS


@njit(cache=True)
def atemp_(ui, ts):
    ''' computes airtemp by correcting gage temp with prec and elevation
    general, ui, ts are Python dictionaries for user input and time series,
    gatmp is the reference gauge temperature time series, required,
    prec  is the preciptation time series,required,
    eldat is difference in elevation between LS and air temp gage, feet,
    lapse is dry air lapse rate for each hour of the day, optional.'''

    errors = zeros(int(ui['errlen'])).astype(int64)

    # pay for lookup once
    k     = ui['k']      # calculated in atemp()
    eldat = ui['eldat']
    steps = int(ui['steps'])

    LAPSE = ts['LAPSE']   # already extended to the entire simulation
    PREC  = ts['PREC']
    GATMP = ts['GATMP']

    # like MATLAB, faster to preallocate array storage, stores in Dict automatically
    ts['AIRTMP'] = AIRTMP = empty(steps)

    for step in range(steps):
        lapse = 0.0035 if PREC[step] > k else LAPSE[step]  # use wet lapse if prec
        AIRTMP[step] = GATMP[step] - lapse * eldat
    return errors
