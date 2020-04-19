''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros, float64
from pandas import HDFStore, Timestamp, read_hdf, DataFrame, date_range
from pandas.tseries.offsets import Minute
from numba import types
from numba.typed import Dict
from collections import defaultdict
from datetime import datetime as dt
import os
from copy import deepcopy
from HSP2.utilities import transform, versions

def noop (store, siminfo, ui, ts):
    ERRMSGS = []
    errors = zeros(len(ERRMSGS), dtype=int)
    return errors, ERRMSGS

# new activity modules must be added here and in *activites* below
from HSP2.ATEMP  import atemp
from HSP2.SNOW   import snow
from HSP2.PWATER import pwater
from HSP2.IWATER import iwater
from HSP2.HYDR   import hydr

# Note: This is the ONLY place in HSP2 that defines activity execution order
activities = {
  'PERLND': {'ATEMP':atemp, 'SNOW':snow, 'PWATER':pwater, 'SEDMNT':noop,
     'PSTEMP':noop, 'PWTGAS':noop, 'PQUAL':noop, 'MSTLAY':noop, 'PEST':noop,
     'NITR':noop, 'PHOS':noop, 'TRACER':noop},
  'IMPLND': {'ATEMP':atemp, 'SNOW':snow, 'IWATER':iwater, 'SOLIDS':noop,
     'IWTGAS':noop, 'IQUAL':noop},
  'RCHRES': {'HYDR':hydr, 'ADCALC':noop, 'CONS':noop, 'HTRCH':noop,
     'SEDTRN':noop, 'GQUAL':noop, 'OXRX':noop, 'NUTRX':noop, 'PLANK':noop,
     'PHCARB':noop}}


def main(hdfname, doe, doename='DOE_RESULTS', saveall=False):
    '''
    Runs main HSP2 program with a Design of Experiments.

    Parameters
    ----------
    hdfname: string
        HDF5 filename used for both input and output.
    doe : List of run lines. Best explaned by example:
        doe = [
         [1, 'PERLND/PWATER/PARAMETERS', 'P001', 'FOREST', 0.02],
         [1, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.15],
         [2, 'PERLND/PWATER/PARAMETERS', 'P001', 'FOREST', 0.01],
         [2, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.20],
         [3, 'PERLND/PWATER/PARAMETERS', 'P001', 'FOREST', 0.02],
         [3, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.15],
         [4, 'PERLND/PWATER/PARAMETERS', 'P001', 'FOREST', 0.01],
         [4, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.20],

         [5, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.05],
         [6, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.10],
         [7, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.15],
         [8, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.20],
         [9, 'PERLND/PWATER/PARAMETERS', 'P001', 'INFILT', 0.25],

         [10, 'PERLND/SNOW/PARAMETERS', 'P001', 'MWATER', 0.09],
         [10, 'IMPLND/SNOW/PARAMETERS', 'I001', 'MWATER', 0.09],

         [11, 'PERLND/SNOW/PARAMETERS',   'P001', 'MWATER', 0.07],
         [11, 'PERLND/PWATER/PARAMETERS', 'P001', 'FOREST', 0.02],

         [12, 'PERLND/PWATER/MONTHLY/CEPSC', 'P001', 'MAR', 0.04],
         [13, 'PERLND/PWATER/MONTHLY/CEPSC', 'P001', 'MAY', 0.05],

         [14, 'PERLND/SNOW/PARAMETERS', 'P001', 'MWATER', 0.09],
         [14, 'IMPLND/SNOW/PARAMETERS', 'I001', 'MWATER', 0.09],

         [14, 'PERLND/SNOW/FLAGS', 'P001', 'ICEFG', 0],
         [14, 'IMPLND/SNOW/FLAGS', 'I001', 'ICEFG', 0],
         [15, 'PERLND/SNOW/FLAGS', 'P001', 'ICEFG', 1],
         [15, 'IMPLND/SNOW/FLAGS', 'I001', 'ICEFG', 1]]


        Each line has the stucture:
         [runnumber, pathtodatatable, segment, name, value]
        All lines with the same runnumber are combined in that run.
        The original UCI dictionary is used at the start of each run.

    doename : string, optional
        Prefix for all run results.
        The default is 'DOE_RESULTS'.
        Resulting HDF5 structure is
            'DOE_RESULTS'
                RUN1
                    (contents like normal /RESULTS with parameters from first line of DOE)
                ***
                RUNx
                    (contents like norma /RESULTS with parameters from the x line of DOE)
    saveall: Boolean
        [optional] Default is False.
        Saves all calculated data ignoring SAVE tables.

    Returns
    -------
    None.
    '''


    if not os.path.exists(hdfname):
        print(f'{hdfname} HDF5 File Not Found, QUITTING')
        return

    with HDFStore(hdfname) as store:
        msg = messages()
        msg(1, f'Processing started for file {hdfname}')

        # read user control, parameters, states, and flags  from HDF5 file
        opseq, ddlinks, ddmasslinks, ddext_sources, ucioriginal, siminfo = get_uci(store)
        start, stop = siminfo['start'], siminfo['stop']

        # construct dictionary parallel in form to uciorginal from doe
        rundict = make_runlist(store, doe, doename)

        # main processing loop
        msg(1, f'Simulation Start: {start}, Stop: {stop}')
        for run in rundict:
            msg(2, f'Starting Run {run}')
            savepath = f'{doename}/RUN{run}'

            uci = deepcopy(ucioriginal)
            for _, operation, segment, delt in opseq.itertuples():
                msg(3, f'{operation} {segment} DELT(minutes): {delt}')
                siminfo['delt']      = delt
                siminfo['tindex']    = date_range(start, stop, freq=Minute(delt))[0:-1]
                siminfo['steps']     = len(siminfo['tindex'])

                # now conditionally execute all activity modules for the op, segment
                ts = get_timeseries(store,ddext_sources[(operation,segment)],siminfo)
                flags = uci[(operation, 'GENERAL', segment)]['ACTIVITY']
                for activity, function in activities[operation].items():
                    if function == noop or not flags[activity]:
                        continue

                    msg(4, f'{activity}')
                    if operation == 'RCHRES':
                        get_flows(store,ts,activity,segment,ddlinks,ddmasslinks)
                    ui = uci[operation, activity, segment] # ui is a dictionary

                    # update deep copy of UCI dict with run dict
                    ruci = rundict[run]
                    if (operation, activity, segment) in ruci:
                        for table in ruci[operation, activity, segment]:
                            ui[table].update(ruci[operation, activity, segment][table])

                    ############ calls activity function like snow() ##############
                    errors, errmessages = function(store, siminfo, ui, ts)
                    ###############################################################

                    for errorcnt, errormsg in zip(errors, errmessages):
                        if errorcnt > 0:
                            msg(5, f'Error count {errorcnt}: {errormsg}')
                    save_timeseries(store,ts,ui['SAVE'],siminfo,saveall,operation,segment,activity, savepath)

        # print Done message with timing and write logfile to HDF5 file
        msglist = msg(1, 'Done', final=True)
        df = DataFrame(msglist, columns=['logfile'])
        df.to_hdf(store, 'RUN_INFO/LOGFILE', data_columns=True, format='t')

        df = versions(['jupyterlab', 'notebook'])
        df.to_hdf(store, 'RUN_INFO/VERSIONS', data_columns=True, format='t')
        print('\n\n', df)
    return


def messages():
    '''Closure routine; msg() prints messages to screen and run log'''
    start = dt.now()
    mlist = []
    def msg(indent, message, final=False):
        now = dt.now()
        m = str(now)[:22] + '   ' * indent + message
        if final:
            mn,sc = divmod((now-start).seconds, 60)
            ms = (now-start).microseconds // 100_000
            m = '; '.join((m, f'Run time is about {mn:02}:{sc:02}.{ms} (mm:ss)'))
        print(m)
        mlist.append(m)
        return mlist
    return msg


def get_uci(store):
    # read user control and user data from HDF5 file
    uci = defaultdict(dict)
    siminfo = {}
    for path in store.keys():   # finds ALL data sets into HDF5 file
        op, module, *other = path[1:].split(sep='/', maxsplit=3)
        s = '_'.join(other)
        if op == 'CONTROL':
            if module =='GLOBAL':
                temp = store[path].to_dict()['Info']
                siminfo['start'] = Timestamp(temp['Start'])
                siminfo['stop']  = Timestamp(temp['Stop'])
            elif module == 'LINKS':
                ddlinks = defaultdict(list)
                for row in store[path].itertuples():
                    ddlinks[row.TVOLNO].append(row)
            elif module == 'MASS_LINKS':
                ddmasslinks = defaultdict(list)
                for row in store[path].itertuples():
                    ddmasslinks[row.MLNO].append(row)
            elif module == 'EXT_SOURCES':
                ddext_sources = defaultdict(list)
                for row in store[path].itertuples():
                    ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
            elif module == 'OP_SEQUENCE':
                opseq = store[path]
        elif op in {'PERLND', 'IMPLND', 'RCHRES'}:
            for id, vdict in store[path].to_dict('index').items():
                uci[(op, module, id)][s] = vdict
    return opseq, ddlinks, ddmasslinks, ddext_sources, uci, siminfo


def make_runlist(store, doe, doename):
    df = DataFrame(doe, columns=['Run', 'DataPath', 'Segment', 'Name', 'Value'])
    df.to_hdf(store, f'{doename}/DoE', format='t', data_columns=True
             )
    rundict = defaultdict(defaultdict)
    for line in doe:
        run, path, segment, name, value = line[:]
        operation, module, *temp = path.split(sep='/', maxsplit=3)
        table = '_'.join(temp)
        runstr = f'Run{run}'

        if (operation, module, segment) not in rundict[runstr]:
            rundict[runstr][operation, module, segment] = defaultdict(dict)
        rundict[runstr][operation, module, segment][table]  [name] = value
    return rundict


def get_timeseries(store, ext_sourcesdd, siminfo):
    ''' makes timeseries for the current timestep and trucated to the sim interval'''
    # explicit creation of Numba dictionary with signatures
    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for row in ext_sourcesdd:
        path = f'TIMESERIES/{row.SVOLNO}'
        temp1 = store[path] if row.SVOL == '*' else read_hdf(row.SVOL, path)

        if row.MFACTOR != 1.0:
            temp1 *= row.MFACTOR
        name = f'{row.TMEMN}{row.TMEMSB}'
        if name in ts:
            ts[name] += transform(temp1, row.TRAN, siminfo).to_numpy().astype(float)
        else:
            ts[name]  = transform(temp1, row.TRAN, siminfo).to_numpy().astype(float)
    return ts


def save_timeseries(store, ts, savedict, siminfo, saveall, operation, segment, activity,savepath):
    # save computed timeseries (at computation DELT)
    save = {k for k,v in savedict.items() if v or saveall}
    df = DataFrame(index=siminfo['tindex'])
    for y in (save & set(ts.keys())):
        df[y] = ts[y]
    df = df.astype('float32').sort_index(axis='columns')

    path = f'{savepath}/RESULTS/{operation}_{segment}/{activity}'
    df.to_hdf(store, path, data_columns=True, format='t')
    return


def get_flows(store, ts, activity, segment, ddlinks, ddmasslinks):
    for x in ddlinks[segment]:
        mldata = ddmasslinks[x.MLNO]
        for dat in mldata:
            if x.MLNO == '':  # Data from NETWORK part of Links table
                factor = x.MFACTOR if x.AFACTR == '' else x.FACTOR * x.AFACTR
                sgrpn  = x.SGRPN
                smemn  = x.SMEMN
                smemsb = x.SMEMSB
                tmemn  = x.TMEMN
                tmemsb = x.TMEMSB
            else:   # Data from SCHEMATIC part of Links table
                factor = dat.MFACTOR * x.AFACTR
                sgrpn  = dat.SGRPN
                smemn  = dat.SMEMN
                smemsb = dat.SMEMSB
                tmemn  = dat.TMEMN
                tmemsb = dat.TMEMSB

            # KLUDGE until remaining HSP2 modules are available.
            if tmemn not in {'IVOL', ''}:
                continue
            if sgrpn == 'OFLOW' and dat.SVOL == 'RCHRES':
                tmemn = 'IVOL'
                smemn = 'OVOL'
                sgrpn = 'HYDR'
            if sgrpn == 'ROFLOW' and dat.SVOL == 'RCHRES':
                tmemn = 'IVOL'
                smemn = 'ROVOL'
                sgrpn = 'HYDR'

            path = f'RESULTS/{x.SVOL}_{x.SVOLNO}/{sgrpn}'
            data = f'{smemn}{smemsb}'

            t = factor * store[path][data].astype(float64).to_numpy()
            # ??? ISSUE: can fetched data be at different frequency - don't know how to transform.
            if tmemn in ts:
                ts[tmemn] += t
            else:
                ts[tmemn] = t
    return


'''

    # This table defines the expansion to INFLOW, ROFLOW, OFLOW for RCHRES networks
    d = [
        ['IVOL',  'ROVOL',  'OVOL',  'HYDRFG', 'HYDR'],
        ['ICON',  'ROCON',  'OCON',  'CONSFG', 'CONS'],
        ['IHEAT', 'ROHEAT', 'OHEAT', 'HTFG',   'HTRCH'],
        ['ISED',  'ROSED',  'OSED',  'SEDFG',  'SEDTRN'],
        ['IDQAL', 'RODQAL', 'ODQAL', 'GQALFG', 'GQUAL'],
        ['ISQAL', 'ROSQAL', 'OSQAL', 'GQALFG', 'GQUAL'],
        ['OXIF',  'OXCF1',  'OXCF2', 'OXFG',   'OXRX'],
        ['NUIF1', 'NUCF1',  'NUCF1', 'NUTFG',  'NUTRX'],
        ['NUIF2', 'NUCF2',  'NUCF9', 'NUTFG',  'NUTRX'],
        ['PKIF',  'PKCF1',  'PKCH2', 'PLKFG',  'PLANK'],
        ['PHIF',  'PHCF1',  'PHCF2', 'PHFG',   'PHCARB']]
    df = pd.DataFrame(d, columns=['INFLOW', 'ROFLOW', 'OFLOW', 'Flag', 'Name'])
    df.to_hdf(h2name, '/FLOWEXPANSION', format='t', data_columns=True)


'''