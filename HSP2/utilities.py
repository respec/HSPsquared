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
    ''' move UCI like data from uci dictionary to Numba dict '''
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

    freq = Minute(siminfo['delt'])
    if ts.index.freq == freq and how == 'SAME':
        return ts

    # need to append duplicate of last point to force processing last full interval
    ts[ts.index[-1] + Timedelta(ts.index.freq)] = ts[-1]

    if how == 'MEAN':
        return ts.resample(freq).mean()
    elif how == 'SUM':
        return ts.resample(freq).sum()
    elif how == 'MAX':
        return ts.resample(freq).max()
    elif how == 'MIN':
        return ts.resample(freq).min()

    elif how == 'LAST':
        return ts.resample(freq).ffill()
    elif how == 'SAME':
        return ts.resample(freq).ffill()

    elif how == 'DIV':
        ts = ts * (freq / ts.index.freq)
        return ts.resample(freq).ffill()
    elif how == 'ZEROFILL':
        return ts.resample(freq).asfreq().fillna(0.0)
    elif how == 'INTERPOLATE':
        return ts.resample(r=freq).interpolate()

    else:
        print(f'UNKNOWN method in TRANS, {how}')
        return zeros(1)


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


def versions(import_list):
    '''prints versions of libraries in import_lint as a pandas Dataframe'''
    import sys
    import platform
    import pandas
    import importlib
    import datetime

    names = ['Python']
    data  = [sys.version]
    for import_ in import_list:
        imodule = importlib.import_module(import_)
        names.append(import_)
        data.append(imodule.__version__)
    names.extend(['os', 'processor', 'Date/Time'])
    data.extend([platform.platform(), platform.processor(),
                 str(datetime.datetime.now())[0:19]])
    return pandas.DataFrame(data, index=names, columns=['version'])
