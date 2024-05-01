''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
General routines for HSP2 '''

import pandas as pd
import numpy as np
from pandas import Series, date_range
from pandas.tseries.offsets import Minute
from numpy import zeros, full, tile, float64
from numba import types
from numba.typed import Dict

from HSP2IO.protocols import Category, SupportsReadTS, SupportsWriteTS
from typing import List


flowtype = {
  # EXTERNAL FLOWS
  'PREC','WIND','WINMOV','SOLRAD','PETINP','POTEV','SURLI','IFWLI','AGWLI',
  'SLSED','IVOL','ICON',
  # COMPUTED FLOWS
  'PRECIP','SNOWF','PRAIN','SNOWE','WYIELD','MELT',                     #SNOW
  'SUPY','SURO','IFWO','AGWO','PERO','IGWI','PET','CEPE','UZET','LZET', #PWATER
  'AGWET','BASET','TAET','IFWI','UZI','INFIL','PERC','LZI','AGWI',      #PWATER
  'SOHT','IOHT','AOHT','POHT','SODOXM','SOCO2M','IODOXM','IOCO2M',      #PWTGAS
  'AODOXM','AOCO2M','PODOXM','POCO2M',                                  #PWTGAS
  'SUPY','SURO','PET','IMPEV'                                           #IWATER
  'SOSLD',                                                              #SOLIDS
  'SOHT','SODOXM','SOCO2M',                                             #IWTGAS
  'SOQS','SOQO','SOQUAL',                                               #IQUAL
  'IVOL','PRSUPY','VOLEV','ROVOL','POTEV',                              #HYDR
  'ICON','ROCON',                                                       #CONS
  'IHEAT','HTEXCH','ROHEAT','QTOTAL','QSOLAR','QLONGW','QEVAP','QCON',  #HTRCH
  'QPREC','QBED',                                                       #HTRCH
  }

# These are hardcoded series in HSPF that are used various modules
# Rather than have them become a IO requirement, carry them over as 
# hard coded variables for the time being.   
LAPSE = Series([0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0037,
 0.0040, 0.0041, 0.0043, 0.0046, 0.0047, 0.0048, 0.0049, 0.0050, 0.0050,
 0.0048, 0.0046, 0.0044, 0.0042, 0.0040, 0.0038, 0.0037, 0.0036])

SEASONS = Series([0,0,0,1,1,1,1,1,1,0,0,0]).astype(bool)

SVP = Series([1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005,
 1.005, 1.01, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.1, 1.29, 1.66,
 2.13, 2.74,3.49, 4.40, 5.55,6.87, 8.36, 10.1,12.2,14.6, 17.5, 20.9, 24.8,
 29.3, 34.6, 40.7, 47.7, 55.7, 64.9]).to_numpy()


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



def transform(ts, name, how, siminfo):
    '''
     upsample (disaggregate) /downsample (aggregate) ts to freq and trim to [start:stop]
     how methods (default is SAME)
         disaggregate: LAST, SAME, DIV, ZEROFILL, INTERPOLATE
         aggregate: MEAN, SUM, MAX, MIN
    NOTE: these routines work for both regular and sparse timeseries input
    '''

    tsfreq = ts.index.freq
    freq   = Minute(siminfo['delt'])
    stop   = siminfo['stop']

    # append duplicate of last point to force processing last full interval
    if ts.index[-1] < stop:
        ts[stop] = ts[-1]

    if freq == tsfreq:
        pass
    elif tsfreq == None:     # Sparse time base, frequency not defined
         ts = ts.reindex(siminfo['tbase']).ffill().bfill()
    elif how == 'SAME':
        ts = ts.resample(freq).ffill()  # tsfreq >= freq assumed, or bad user choice
    elif not how:
        if name in flowtype:
            if 'Y' in str(tsfreq) or 'M' in str(tsfreq) or tsfreq > freq:
                if   'M' in str(tsfreq):  ratio = 1.0/730.5
                elif 'Y' in str(tsfreq):  ratio = 1.0/8766.0
                else:                     ratio = freq / tsfreq
                ts = (ratio * ts).resample(freq).ffill()   # HSP2 how = div
            else:
                ts = ts.resample(freq).sum()
        else:
            if 'Y' in str(tsfreq) or 'M' in str(tsfreq) or tsfreq > freq:
                ts = ts.resample(freq).ffill()
            else:
                ts = ts.resample(freq).mean()
    elif how == 'MEAN':        ts = ts.resample(freq).mean()
    elif how == 'SUM':         ts = ts.resample(freq).sum()
    elif how == 'MAX':         ts = ts.resample(freq).max()
    elif how == 'MIN':         ts = ts.resample(freq).min()
    elif how == 'LAST':        ts = ts.resample(freq).ffill()
    elif how == 'DIV':         ts = (ts * (freq / ts.index.freq)).resample(freq).ffill()
    elif how == 'ZEROFILL':    ts = ts.resample(freq).fillna(0.0)
    elif how == 'INTERPOLATE': ts = ts.resample(freq).interpolate()
    else:
        print(f'UNKNOWN method in TRANS, {how}')
        return zeros(1)

    start, steps = siminfo['start'], siminfo['steps']
    return ts[start:stop].to_numpy().astype(float64)[0:steps]


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


def initmdiv(siminfo, ui, flag, monthly1, monthly2, default1, default2):
    ''' initialize timeseries with HSPF interpolation of monthly array or with fixed value'''
    # special case for 'ACQOP' divided by 'SQOLIM'
    if flag and monthly1 and monthly2 in ui:
        month1 = list(ui[monthly1].values())
        month2 = list(ui[monthly2].values())
        month = zeros(12)
        for m in range(0, 12):
            month[m] = month1[m] / month2[m]
        return dayval(siminfo, list(month))
    else:
        return full(siminfo['steps'], default1 / default2)


def initmd(siminfo, monthdata, monthly, default):
    ''' initialize timeseries from HSPF month data table'''
    if monthly in monthdata:
        month = monthdata[monthly].values[0]
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

def get_timeseries(timeseries_inputs:SupportsReadTS, ext_sourcesdd, siminfo):
    ''' makes timeseries for the current timestep and trucated to the sim interval'''
    # explicit creation of Numba dictionary with signatures
    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for row in ext_sourcesdd:
        data_frame = timeseries_inputs.read_ts(category=Category.INPUTS,segment=row.SVOLNO)

        if row.MFACTOR != 1.0:
            data_frame *= row.MFACTOR
        t = transform(data_frame, row.TMEMN, row.TRAN, siminfo)

        tname = clean_name(row.TMEMN,row.TMEMSB)
        if tname in ts:
            ts[tname] += t
        else:
            ts[tname]  = t
    return ts

def save_timeseries(timeseries:SupportsWriteTS, ts, savedict, siminfo, saveall, operation, segment, activity, compress=True, outstep=2):
    df = pd.DataFrame(index=siminfo['tindex'])
    if (operation == 'IMPLND' and activity == 'IQUAL') or (operation == 'PERLND' and activity == 'PQUAL'):
        for y in savedict.keys():
            for z in set(ts.keys()):
                if '/' + y in z:
                    zrep = z.replace('/','_')
                    zrep2 = zrep.replace(' ', '')
                    df[zrep2] = ts[z]
                if '_' + y in z:
                    df[z] = ts[z]
    elif (operation == 'RCHRES' and (activity == 'CONS' or activity == 'GQUAL')):
        for y in savedict.keys():
            for z in set(ts.keys()):
                if '_' + y in z:
                    df[z] = ts[z]
        for y in (savedict.keys() & set(ts.keys())):
            df[y] = ts[y]
    else:
        for y in (savedict.keys() & set(ts.keys())):
            df[y] = ts[y]
    df = df.astype(np.float32).sort_index(axis='columns')

    if saveall:
        save_columns = df.columns
    else:
        save_columns = [key for key,value in savedict.items() if value or saveall]

    if not df.empty:
        timeseries.write_ts(
            data_frame=df,
            save_columns=save_columns,
            category = Category.RESULTS,
            operation=operation,
            segment=segment,
            activity=activity,
            compress=compress,
            outstep=outstep
        )
    else:
        print(f'DataFrame Empty for {operation}|{activity}|{segment}')
    return

def expand_timeseries_names(sgrp, smemn, smemsb1, smemsb2, tmemn, tmemsb1, tmemsb2):
    #special cases to expand timeseries names to resolve with output names in hdf5 file
    if tmemn == 'ICON':
        if tmemsb1 == '':
            tmemn = 'CONS1_ICON'
        else:
            tmemn = 'CONS' + tmemsb1 + '_ICON'
    if smemn == 'OCON':
        if smemsb2 == '':
            smemn = 'CONS1_OCON' + smemsb1
        else:
            smemn = 'CONS' + smemsb2 + '_OCON' + smemsb1
    if smemn == 'ROCON':
        if smemsb1 == '':
            smemn = 'CONS1_ROCON'
        else:
            smemn = 'CONS' + smemsb1 + '_ROCON'

    # GQUAL:
    if tmemn == 'IDQAL':
        if tmemsb1 == '':
            tmemn = 'GQUAL1_IDQAL'
        else:
            tmemn = 'GQUAL' + tmemsb1 + '_IDQAL'
    if tmemn == 'ISQAL1' or tmemn == 'ISQAL2' or tmemn == 'ISQAL3':
        if tmemsb2 == '':
            tmemn = 'GQUAL1_' + tmemn
        else:
            tmemn = 'GQUAL' + tmemsb2 + '_' + tmemn
    if tmemn == 'ISQAL':
        if tmemsb2 == '':
            tmemn = 'GQUAL1_' + 'ISQAL' + tmemsb1
        else:
            tmemn = 'GQUAL' + tmemsb2 + '_' + 'ISQAL' + tmemsb1
    if smemn == 'ODQAL':
        smemn = 'GQUAL' + smemsb1 + '_ODQAL' + smemsb2  # smemsb2 is exit number
    if smemn == 'OSQAL':
        smemn = 'GQUAL' + smemsb1 + '_OSQAL' + smemsb2  # smemsb2 is ssc plus exit number
    if smemn == 'RODQAL':
        smemn = 'GQUAL' + smemsb1 + '_RODQAL'
    if smemn == 'ROSQAL':
        smemn = 'GQUAL' + smemsb2 + '_ROSQAL' + smemsb1  # smemsb1 is ssc
    if smemn == 'RSQAL':
        smemn = 'GQUAL' + smemsb2 + '_RSQAL' + smemsb1

    # OXRX:
    if smemn == 'OXCF1':
        smemn = 'OXCF1_' + smemsb1
    if smemn == 'OXCF2':
        smemn = 'OXCF2_' + smemsb1 + smemsb2   # smemsb1 is exit #
    if tmemn == 'OXIF':
        tmemn = 'OXIF' + tmemsb1
        if sgrp == "PQUAL" or sgrp == "IQUAL":  # could be from pqual or iqual
            if smemsb1 == '':
                smemsb1 = '1'
            smemn = sgrp + smemsb1 + '_' + smemn

    # NUTRX - dissolved species:
    if smemn == 'NUCF1':                            # total outflow
        smemn = 'NUCF1_' + smemsb1

    if smemn == 'NUCF9':                            # exit-specific outflow
        smemn = 'NUCF9_' + smemsb1 + smemsb2        # smemsb1 is exit #

    if tmemn == 'NUIF1':
        tmemn = 'NUIF1_' + tmemsb1
        if sgrp == "PQUAL" or sgrp == "IQUAL":  # could be from pqual or iqual
            if smemsb1 == '':
                smemsb1 = '1'
            smemn = sgrp + smemsb1 + '_' + smemn

    # NUTRX - particulate species:
    if smemn == 'NUCF2':                            # total outflow
        smemn = 'NUCF2_' + smemsb1 + smemsb2        # smemsb1 is sediment class

    if smemn == 'OSNH4' or smemn == 'OSPO4':        # exit-specific outflow
        smemn = smemn + '_' + smemsb1 + smemsb2     # smemsb1 is exit #, smemsb2 is sed class

    if tmemn == 'NUIF2':
        tmemn = 'NUIF2_' + tmemsb1 + tmemsb2
        if sgrp == "PQUAL" or sgrp == "IQUAL":  # could be from pqual or iqual
            if smemsb1 == '':
                smemsb1 = '1'
            smemn = sgrp + smemsb1 + '_' + smemn

    # PLANK:
    if smemn == 'PKCF1':                            # total outflow
        smemn = 'PKCF1_' + smemsb1                  # smemsb1 is species index

    if smemn == 'PKCF2':                            # exit-specific outflow
        smemn = 'PKCF2_' + smemsb1 + smemsb2        # smemsb1 is exit #, smemsb2 is species index

    if tmemn == 'PKIF':
        tmemn = 'PKIF' + tmemsb1                    # tmemsb1 is species index
        if sgrp == "PQUAL" or sgrp == "IQUAL":      # could be from pqual or iqual
            if smemsb1 == '':
                smemsb1 = '1'
            smemn = sgrp + smemsb1 + '_' + smemn

    # PHCARB:
    if smemn == 'PHCF1' and smemsb1 == 1:           # total outflow
        smemn = 'ROTIC'
    if smemn == 'PHCF1' and smemsb1 == 2:           # total outflow
        smemn = 'ROCO2'

    if smemn == 'PHCF2' and smemsb2 == 1:           # exit-specific outflow
        smemn = 'OTIC' + smemsb1                    # smemsb1 is exit #, smemsb2 is species index
    if smemn == 'PHCF2' and smemsb2 == 2:           # exit-specific outflow
        smemn = 'OCO2' + smemsb1                    # smemsb1 is exit #, smemsb2 is species index

    if tmemn == 'PHIF':
        tmemn = 'PHIF' + tmemsb1                    # tmemsb1 is species index

    return smemn, tmemn

def get_gener_timeseries(ts: Dict, gener_instances: Dict, ddlinks: List, ddmasslinks) -> Dict:
    """
    Uses links tables to load necessary TimeSeries from Gener class instances to TS dictionary
    """
    for link in ddlinks:
        if link.SVOL == 'GENER':
            if link.SVOLNO in gener_instances:
                gener = gener_instances[link.SVOLNO]
                series = zeros(len(gener.ts_output)) + gener.ts_output
                if type(link.MFACTOR) == float and link.MFACTOR != 1:
                    series *= link.MFACTOR

                key = f'{link.TMEMN}{link.TMEMSB1} {link.TMEMSB2}'.rstrip()
                if key != '':
                    key = clean_name(link.TMEMN,link.TMEMSB1 + link.TMEMSB2)
                    if key in ts:
                        ts[key] = ts[key] + series
                    else:
                        ts[key] = series
                else:
                    # have to use ML
                    mldata = ddmasslinks[link.MLNO]
                    for dat in mldata:
                        mfactor = dat.MFACTOR
                        tmemn = dat.TMEMN
                        tmemsb1 = dat.TMEMSB1
                        tmemsb2 = dat.TMEMSB2

                        afactr = link.AFACTR
                        factor = afactr * mfactor

                        # may need to do something in here for special cases like in get_flows
                        if tmemn != 'ONE' and tmemn != 'TWO':
                            tmemn = clean_name(tmemn, tmemsb1 + tmemsb2)

                        t = series * factor

                        if tmemn in ts:
                            ts[tmemn] += t
                        else:
                            ts[tmemn] = t

    return ts

def clean_name (TMEMN,TMEMSB):
    # in some cases the subscript is irrelevant, like '1' or '1 1', and we can leave it off.
    # there are other cases where it is needed to distinguish, such as ISED and '1' or '1 1'.
    tname = f'{TMEMN}{TMEMSB}'
    if TMEMN in {'GATMP', 'PREC', 'DTMPG', 'WINMOV', 'DSOLAR', 'SOLRAD', 'CLOUD', 'PETINP', 'IRRINP', 'POTEV',
                     'DEWTMP', 'WIND',
                     'IVOL', 'IHEAT'}:
        tname = f'{TMEMN}'
    elif TMEMN == 'ISED':
        if TMEMSB == '1 1' or TMEMSB == '1' or TMEMSB == '':
            tname = 'ISED1'
        else:
            tname = 'ISED' + TMEMSB[0]
    elif TMEMN == 'NUIF1':
        if len(TMEMSB) > 0:
            tname = TMEMN + '_' + TMEMSB[0]
        else:
            tname = TMEMN + '_1'
    elif TMEMN in {'ICON', 'IDQAL', 'ISQAL'}:
        tmemsb1 = '1'
        tmemsb2 = '1'
        if len(TMEMSB) > 0:
            tmemsb1 = TMEMSB[0]
        if len(TMEMSB) > 2:
            tmemsb2 = TMEMSB[-1]
        sname, tname = expand_timeseries_names('', '', '', '', TMEMN, tmemsb1, tmemsb2)
    elif TMEMN == 'PKIF':
        if len(TMEMSB) > 0:
            tname = TMEMN + TMEMSB[0]
        else:
            tname = TMEMN + '1'

    return tname