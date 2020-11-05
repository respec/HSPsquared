''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of HSPF HPERAIR.FOR module. '''


from numba import njit
from numpy import empty, zeros, int64
from HSP2.utilities import hoursval, make_numba_dict

ERRMSGS = ()


def atemp(store, siminfo, uci, ts):
    ''' high level driver for air temperature module'''

    ts['LAPSE'] = hoursval(siminfo, store['TIMESERIES/LAPSE_Table'], lapselike=True)

    ui = make_numba_dict(uci)   # Note: all values coverted to float automatically
    ui['k']      = siminfo['delt']  * 0.000833        # convert to in/timestep
    ui['steps']  = siminfo['steps']
    ui['errlen'] = len(ERRMSGS)

    ############################################################################
    errors = _atemp_(ui, ts)                     # run ATEMP_ simulation code
    ############################################################################

    return errors, ERRMSGS


# @njit(cache=True)
def _atemp_(ui, ts):
    ''' computes airtemp by correcting gage temp with prec and elevation
    general, ui, ts are Python dictionaries for user input and time series,
    gatmp is the reference gauge temperature time series, required,
    prec  is the preciptation time series,required,
    eldat is difference in elevation between LS and air temp gage, feet,
    lapse is dry air lapse rate for each hour of the day, optional.'''
    errors = zeros(int(ui['errlen'])).astype(int64)

    # pay for lookup once
    k     = ui['k']      # calculated in atemp()
    eldat = ui['ELDAT']
    steps = int(ui['steps'])

    # all series already aggregated/disaggregated to runtime delt frequecy
    LAPSE = ts['LAPSE']
    PREC  = ts['PREC']
    GATMP = ts['GATMP']

    # like MATLAB, faster to preallocate array storage, stores in Dict automatically
    ts['AIRTMP'] = AIRTMP = empty(steps)

    for step in range(steps):
        lapse = 0.0035 if PREC[step] > k else LAPSE[step]  # use wet lapse if prec
        AIRTMP[step] = GATMP[step] - lapse * eldat
    return errors
