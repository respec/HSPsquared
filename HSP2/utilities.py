''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
General routines for HSP2 '''


from pandas import Series, date_range, Timedelta
from pandas.tseries.offsets import Minute
from numpy import zeros, full, tile
from numba import types
from numba.typed import Dict


def make_numba_dict(uci):
    '''
    Move UCI dictionary data to Numba dict for FLAGS, STATES, PARAMETERS.

    Parameters
    ----------
    uci : Python dictionary
        The uci dictionary contains xxxx.uci file data

    Returns
    -------
    ui : Numba dictionary
        Same content as uci except for strings

    '''

    ui = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for name in set(uci.keys()) & {'FLAGS', 'PARAMETERS', 'STATES'}:
        for key, value in uci[name].items():
            if type(value) in {int, float}:
                ui[key] = float(value)
    return ui


def transform(ts, how, siminfo):
    '''
     upsample (disaggregate) /downsample (aggregate) ts to freq and trim to [start:stop]
     how methods (default is SAME)
         disaggregate: LAST, SAME, DIV, ZEROFILL, INTERPOLATE
         aggregate: MEAN, SUM, MAX, MIN
    NOTE: these routines work for both regular and sparse timeseries input
    '''

    if not how:
        how = 'SAME'

    start, stop, steps = siminfo['start'], siminfo['stop'], siminfo['steps']
    freq = Minute(siminfo['delt'])
    if ts.index.freq == freq and ts.index[0] == start and ts.index[-1] <= stop:
        return ts[0:steps]

    # need to append duplicate of last point to force processing last full interval
    if ts.index.freq in ('1M', '1MS'):
        ts[ts.index[-1] + Timedelta('31D')] = ts[-1]
    else:
        ts[ts.index[-1] + Timedelta(ts.index.freq)] = ts[-1]

    if how == 'MEAN':          ts = ts.resample(freq).mean()
    elif how == 'SUM':         ts = ts.resample(freq).sum()
    elif how == 'MAX':         ts = ts.resample(freq).max()
    elif how == 'MIN':         ts = ts.resample(freq).min()
    elif how == 'LAST':        ts = ts.resample(freq).ffill()
    elif how == 'SAME':        ts = ts.resample(freq).ffill()
    elif how == 'DIV':         ts = (ts * (freq / ts.index.freq)).resample(freq).ffill()
    elif how == 'ZEROFILL':    ts = ts.resample(freq).fillna(0.0)
    elif how == 'INTERPOLATE': ts = ts.resample(freq).interpolate()

    else:
        print(f'UNKNOWN method in TRANS, {how}')
        return zeros(1)

    return ts[start:stop][0:steps]


def hoursval(siminfo, hours24, dofirst=False, lapselike=False):
    '''create hours flags, flag on the hour or lapse table over full simulation'''
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = Minute(siminfo['delt'])

    dr = date_range(start=f'{start.year}-01-01', end=f'{stop.year}-12-31', freq=Minute(60))
    hours = tile(hours24, (len(dr) + 23) // 24).astype(float)
    if dofirst:
        hours[0] = 1

    ts = Series(hours[0:len(dr)], dr)
    if lapselike:
        if ts.index.freq > freq:     # upsample
            ts = ts.resample(freq).asfreq().ffill()
        elif ts.index.freq < freq:   # downsample
            ts = ts.resample(freq).mean()
    else:
        if ts.index.freq > freq:     # upsample
            ts = ts.resample(freq).asfreq().fillna(0.0)
        elif ts.index.freq < freq:   # downsample
            ts = ts.resample(freq).max()
    return ts.truncate(start, stop).to_numpy()


def hourflag(siminfo, hourfg, dofirst=False):
    '''timeseries with 1 at desired hour and zero otherwise'''
    hours24 = zeros(24)
    hours24[hourfg] = 1.0
    return hoursval(siminfo, hours24, dofirst)


def monthval(siminfo, monthly):
    ''' returns value at start of month for all times within the month'''
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = Minute(siminfo['delt'])

    months = tile(monthly, stop.year - start.year + 1).astype(float)
    dr = date_range(start=f'{start.year}-01-01', end=f'{stop.year}-12-31',
     freq='MS')
    ts = Series(months, index=dr).resample('D').ffill()

    if ts.index.freq > freq:     # upsample
        ts = ts.resample(freq).asfreq().ffill()
    elif ts.index.freq < freq:   # downsample
        ts = ts.resample(freq).mean()
    return ts.truncate(start, stop).to_numpy()


def dayval(siminfo, monthly):
    '''broadcasts HSPF monthly data onto timeseries at desired freq with HSPF
    interpolation to day, but constant within day'''
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = Minute(siminfo['delt'])

    months = tile(monthly, stop.year - start.year + 1).astype(float)
    dr = date_range(start=f'{start.year}-01-01', end=f'{stop.year}-12-31',
     freq='MS')
    ts = Series(months, index=dr).resample('D').interpolate('time')

    if ts.index.freq > freq:     # upsample
        ts = ts.resample(freq).ffill()
    elif ts.index.freq < freq:   # downsample
        ts = ts.resample(freq).mean()
    return ts.truncate(start, stop).to_numpy()


def initm(siminfo, ui, flag, monthly, default):
    ''' initialize timeseries with HSPF interpolation of monthly array or with fixed value'''
    if flag and monthly in ui:
        month = ui[monthly].values()
        return dayval(siminfo, list(month))
    else:
        return full(siminfo['steps'], default)


def versions(import_list=[]):
    '''
    Versions of libraries required by HSP2

    Parameters
    ----------
    import_list : list of strings, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    Pandas DataFrame
        Libary verson strings.
    '''

    import sys
    import platform
    import pandas
    import importlib
    import datetime

    names = ['Python']
    data  = [sys.version]
    import_list = ['HSP2', 'numpy', 'numba', 'pandas'] + list(import_list)
    for import_ in import_list:
        imodule = importlib.import_module(import_)
        names.append(import_)
        data.append(imodule.__version__)
    names.extend(['os', 'processor', 'Date/Time'])
    data.extend([platform.platform(), platform.processor(),
      str(datetime.datetime.now())[0:19]])
    return pandas.DataFrame(data, index=names, columns=['version'])
