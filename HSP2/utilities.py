''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
General routines for HSP2 '''

import sys
import platform
import pandas
import importlib
import datetime
from pandas import Series, date_range, Timedelta
from numpy import empty, ravel, zeros, full


def transform(ts, how, siminfo):
    '''
     upsample (disaggregate) /downsample (aggregate) ts to freq and trim to [start:stop]
     how methods (default is SAME)
         disaggregate: LAST, SAME, DIV, ZEROFILL, INTERPOLATE
         aggregate: MEAN, SUM, MAX, MIN
    NOTE: these routines work for both regular and sparse timeseries input
    '''

    freq  = siminfo['freq']
    if ts.index.freq == freq and how == 'SAME':
        return ts

    # need to append duplicate of last point to force processing last full day
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


def hoursval(siminfo, hours24):
    '''create hours flags, flag on the hour, lapse table over full simulation'''
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = siminfo['freq']

    dr = date_range(start=f'{start.year}-01-01', end=f'{stop.year}-12-31', freq='60T')

    hours = empty(((len(dr) + 23) // 24, 24))
    hours[:] = hours24   # broadcast daily hourly values onto simulation time
    hours = ravel(hours)

    ts = Series(hours[0:len(dr)], dr)
    if ts.index.freq > freq:     # upsample
        ts = ts.resample(freq).asfreq().fillna(0.0)
    elif ts.index.freq < freq:   # downsample
        ts = ts.resample(freq).max()
    return ts.truncate(start, stop).to_numpy()


def hourflag(siminfo, hourfg):
    '''timeseries with 1 at desired hour and zero otherwise'''
    hours24 = zeros(24)
    hours24[hourfg] = 1.0
    return hoursval(siminfo, hours24)


def monthval(siminfo, monthly):
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = siminfo['freq']

    months = empty((stop.year - start.year + 1, 12))
    months[:] = monthly            # broadcast monthly values into each year
    months = ravel(months)

    # Numba doesn't support tile() - although not in Numba @njit code ???
    #months = tile(monthly, stop.year - start.year + 1)


    dr = date_range(start=f'{start.year}-01-01', end=f'{stop.year}-12-31',
     freq='MS')
    ts = Series(months, index=dr).resample(freq).ffill()
    return ts.truncate(start, stop).to_numpy()


def dayval(siminfo, monthly):
    '''broadcasts HSPF monthly data onto timeseries at desired freq with HSPF
    interpolation to day, but constant within day'''
    start = siminfo['start']
    stop  = siminfo['stop']
    freq  = siminfo['freq']

    months = empty((stop.year - start.year + 1, 12))
    months[:] = monthly            # broadcast monthly values into each year
    months = ravel(months)

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
        month = ui[monthly][siminfo['segment']].values()
        return dayval(siminfo, list(month))
    else:
        return full(siminfo['steps'], default)


def flows(ts, tvol, tvolno, flag, flowdata, siminfo, store):
    xflowdd = flowdata['xflowdd']
    fetchlist = []
    for lnk in flowdata['linkdd'][tvol, tvolno]:
        for infl in flowdata['lookup'][flag]:
            if not lnk.MLNO:
                dd = lnk.to_dict()
                if not dd['SMEMN']:
                    dd['SMEMN'] = xflowdd[flag, infl][dd['SGRPN']]
                    dd['TMEMN'] = xflowdd[flag, infl][dd['TGRPN']]
                    dd['SGRPN'] = xflowdd[flag, infl]['Name']
                fetchlist.append(dd)
            else:
                mlno = lnk.MLNO
                for x in flowdata['mldd'][mlno]:
                    if not x.TMEMN or x.TMEMN == infl:
                        dd = lnk.to_dict()
                        dd.update(x)
                        if not dd['SMEMN']:
                            dd['SMEMN'] = xflowdd[flag, infl][dd['SGRPN']]
                            dd['TMEMN'] = xflowdd[flag, infl][dd['TGRPN']]
                            dd['SGRPN'] = xflowdd[flag, infl]['Name']
                        fetchlist.append(dd)

    for x in fetchlist:
        path = '/RESULTS/' + x['SVOL'] + '_' + x['SVOLNO'] + '/' + x['SGRPN']
        t = store[path][x['SMEMN'] + x['SMEMSB']]

        if type(x['AFACTR']) == str and x['AFACTR'][0] == '*':
            afactr = store['TIMESERIES/' + x['AFACTR'][1:]]
            afactr = transform(afactr, 'SAME', siminfo)
        else:
            afactr = float(x['AFACTR'])

        if type(x['MFACTOR']) == str and x['MFACTOR'][0] == '*':
            mfactor = store['TIMERSERIES/' + x['MFACTOR'][1:]]
            mfactor = transform(mfactor, 'SAME', siminfo)
        else:
            mfactor = float(x['MFACTOR'])

        t = transform(t, 'SAME', siminfo) * mfactor * afactr
        if x['TMEMN'] in ts:
            ts[x['TMEMN']] += t.values
        else:
            ts[x['TMEMN']]  = t.values
    return


def versions(import_list):
    '''prints versions of libraries in import_lint as a pandas Dataframe'''
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
