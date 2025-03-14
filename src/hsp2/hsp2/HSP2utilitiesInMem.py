# in memory versions of HSP2 utilities get_timeseries and save_timeseries

from typing import Protocol, Dict, Any, List, Union, runtime_checkable
from numba import types
from numba.typed import Dict
from enum import Enum
from hsp2.hsp2.utilities import transform, clean_name
import numpy as np
import pandas as pd

def get_timeseries_InMem(timeseries_inputs, ext_sourcesdd, siminfo):
    ''' makes timeseries for the current timestep and trucated to the sim interval'''
    # explicit creation of Numba dictionary with signatures
    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for row in ext_sourcesdd:
        path = f'/TIMESERIES/{row.SVOLNO}'
        if path in timeseries_inputs:
            data_frame = timeseries_inputs[path]
            temp_data_frame = data_frame.copy()

        if row.MFACTOR != 1.0:
            temp_data_frame *= row.MFACTOR
        t = transform(temp_data_frame, row.TMEMN, row.TRAN, siminfo)

        tname = clean_name(row.TMEMN,row.TMEMSB)
        if tname in ts:
            ts[tname] += t
        else:
            ts[tname]  = t
    return ts

def save_timeseries_InMem(timeseries, ts, savedict, siminfo, saveall, operation, segment, activity, compress=True, outstep=2):
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
        path = f'{operation}_{segment}/{activity}'
        #if category:
        path = '/RESULTS/' + path
        timeseries[path] = df
    else:
        print(f'DataFrame Empty for {operation}|{activity}|{segment}')
    return