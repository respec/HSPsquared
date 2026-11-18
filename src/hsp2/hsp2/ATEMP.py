"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of HSPF HPERAIR.FOR module."""

from numba import njit
from numpy import empty, int64, ones, zeros

from hsp2.hsp2.utilities import LAPSE, hoursval, make_numba_dict
from hsp2.hsp2io.protocols import SupportsReadTS

ERRMSGS = ()


def atemp(io_manager: SupportsReadTS, siminfo, parameters, ts):
    """high level driver for air temperature module"""

    ts["LAPSE"] = hoursval(siminfo, LAPSE, lapselike=True)
    # first interval of each hour
    ts["HRFG"] = hoursval(siminfo, ones(24), dofirst=True).astype(float)

    ui = make_numba_dict(parameters)  # Note: all values coverted to float automatically
    ui["k"] = siminfo["delt"] * 0.000833  # convert to in/timestep
    ui["steps"] = siminfo["steps"]
    ui["errlen"] = len(ERRMSGS)

    ############################################################################
    errors = _atemp_(ui, ts)  # run ATEMP_ simulation code
    ############################################################################

    return errors, ERRMSGS


@njit(cache=True)
def _atemp_(ui, ts):
    """computes airtemp by correcting gage temp with prec and elevation
    general, ui, ts are Python dictionaries for user input and time series,
    gatmp is the reference gauge temperature time series, required,
    prec  is the precipitation time series,required,
    eldat is difference in elevation between land surface and air temp gage, feet,
    lapse is dry air lapse rate for each hour of the day, optional."""
    errors = zeros(int(ui["errlen"])).astype(int64)

    # pay for lookup once
    eldat = ui["ELDAT"]
    k = ui["k"]  # calculated in atemp()

    steps = int(ui["steps"])

    # all series already aggregated/disaggregated to runtime delt frequency
    LAPSE = ts["LAPSE"]
    PREC = ts["PREC"]
    GATMP = ts["GATMP"]
    HRFG = ts["HRFG"]

    # like MATLAB, faster to preallocate array storage, stores in Dict automatically
    ts["AIRTMP"] = AIRTMP = empty(steps)
    ts["GATMP"] = GATMPH = empty(steps)

    # as in ATEMP of HSPF, the gage temperature is read and the air temperature
    # updated in the first interval of each hour, and both are held for the rest
    # of the hour; the first interval of the run always updates them
    gatmp = 0.0
    airtmp = 0.0
    for step in range(steps):
        if step == 0 or HRFG[step] > 0.0:
            gatmp = GATMP[step]
            lapse = 0.0035 if PREC[step] > k else LAPSE[step]  # use wet lapse if prec
            airtmp = gatmp - lapse * eldat
        GATMPH[step] = gatmp
        AIRTMP[step] = airtmp
    return errors
