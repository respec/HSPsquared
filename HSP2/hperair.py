''' Copyright 2017 by RESPEC, INC.- see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HPERAIR.FOR module.  PATEMP folded into ATEMP. '''           #$$HPERAIR.FOR


from numpy import zeros, where
from HSP2 import transform

ERRMSG = ['AIR TEMP MODULE (HPERAIR): Required timeseries missing']  #ERRMSG0
ERRCNT = zeros(len(ERRMSG), dtype=int)


def atemp(store, general, ui, ts):
    ''' high level driver for air temperature module
    CALL: atemp(store, general, ui, ts)
       store is the Pandas/PyTable store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with specific HSPF UCI like data
       ts is a dictionary with specific tim'''

    ts['LAPSE24'] = store['/TIMESERIES/LAPSE24']  # avoid adding this to EXT_SOURCES
    # user can replace LAPSE24 data in HDF5 file, if desired

    ############################################################################
    errorsV = atemp_(general, ui, ts)              # run ATEMP_ simulation code
    ############################################################################

    return errorsV, ERRMSG


def atemp_(general, ui, ts):
    ''' computes airtemp by correcting gage temp with prec and elevation
    general, ui, ts are Python dictionaries for user input and time series,
    gatmp is the reference gauge temperature time series, required,
    prec  is the preciptation time series,required,
    eldat is difference in elevation between LS and air temp gage, feet,
    lapse is dry air lapse rate for each hour of the day, optional.'''

    if 'LAPSE' in ts:    # allow user to provide complete timeseries, however unlikely
        lapse = transform(ts['LAPSE'], general['tindex'], 'MEAN')
    else:                # build lapse from 24 hour array for DELT in current tindex
        lapse = transform(ts['LAPSE24'], general['tindex'], 'LAPSE')

    k = 0.000833 * general['sim_delt']          # convert to in/timestep        #$125,127
    laps = where(ts['PREC'] > k, 0.0035, lapse) # use wet lapse when prec       #$129,122

    eldat = ui['ELDAT']
    ts['AIRTMP'] = ts['GATMP'] - laps * eldat  # does entire vector calculation #$145,68,139,269-285
    return ERRCNT
