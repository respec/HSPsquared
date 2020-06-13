''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import float64, float32
from pandas import HDFStore, Timestamp, read_hdf, DataFrame, date_range
from pandas.tseries.offsets import Minute
from numba import types
from numba.typed import Dict
from collections import defaultdict
from datetime import datetime as dt
import os
from copy import deepcopy
from HSP2.utilities import transform, versions
from HSP2.configuration import activities, noop


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
        msg(1, f'Processing started for file {hdfname}; saveall={saveall}')

        # read user control, parameters, states, and flags  from HDF5 file
        opseq, ddlinks, ddmasslinks, ddext_sources, originaluci, siminfo = get_uci(store)
        start, stop = siminfo['start'], siminfo['stop']

        # construct dictionary parallel in form to uciorginal from doe
        rundict = make_runlist(store, doe, doename)

        # main processing loop
        msg(1, f'Simulation Start: {start}, Stop: {stop}')
        for run in rundict:
            savepath = f'{doename}/RUN{run}'
            msg(2, f'Starting Run {run}; saving as {savepath}')

            uci = deepcopy(originaluci)
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
                        get_flows(store,ts,activity,segment,ddlinks,ddmasslinks,siminfo['steps'],msg,savepath)
                    ui = uci[operation, activity, segment] # ui is a dictionary

                    # update deep copy of UCI dict with run dict
                    ruci = rundict[run]
                    if (operation, activity, segment) in ruci:
                        for table in ruci[operation, activity, segment]:
                            msg(5, str(ruci[operation, activity, segment][table]))
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
        runstr = f'{run}'

        if (operation, module, segment) not in rundict[runstr]:
            rundict[runstr][operation, module, segment] = defaultdict(dict)
        rundict[runstr][operation, module, segment][table]  [name] = float(value)
    return rundict


def get_timeseries(store, ext_sourcesdd, siminfo):
    ''' makes timeseries for the current timestep and trucated to the sim interval'''
    # explicit creation of Numba dictionary with signatures
    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for row in ext_sourcesdd:
        if row.SVOL == '*':
            path = f'TIMESERIES/{row.SVOLNO}'
            if path in store:
                temp1 = store[path]
            else:
                print('Get Timeseries ERROR for', path)
                continue
        else:
            temp1 = read_hdf(row.SVOL, path)

        if row.MFACTOR != 1.0:
            temp1 *= row.MFACTOR
        t = transform(temp1, row.TMEMN, row.TRAN, siminfo)

        tname = f'{row.TMEMN}{row.TMEMSB}'
        if tname in ts:
            ts[tname] += t
        else:
            ts[tname]  = t
    return ts


def save_timeseries(store,ts,savedict,siminfo,saveall,operation,segment,activity,savepath):
    # save computed timeseries (at computation DELT)
    save = {k for k,v in savedict.items() if v or saveall}
    df = DataFrame(index=siminfo['tindex'])
    for y in (save & set(ts.keys())):
        df[y] = ts[y]
    df = df.astype(float32).sort_index(axis='columns')
    path = f'{savepath}/{operation}_{segment}/{activity}'
    if not df.empty:
        store.put(path, df)
    else:
        print('Save DataFrame Empty for', path)
    return


def get_flows(store, ts, activity, segment, ddlinks, ddmasslinks, steps, msg, savepath):
    for x in ddlinks[segment]:
        mldata = ddmasslinks[x.MLNO]
        for dat in mldata:
            if x.MLNO == '':  # Data from NETWORK part of Links table
                mfactor = x.MFACTOR
                sgrpn   = x.SGRPN
                smemn   = x.SMEMN
                smemsb  = x.SMEMSB
                tmemn   = x.TMEMN
                tmemsb  = x.TMEMSB
            else:   # Data from SCHEMATIC part of Links table
                mfactor = dat.MFACTOR
                sgrpn   = dat.SGRPN
                smemn   = dat.SMEMN
                smemsb  = dat.SMEMSB
                tmemn   = dat.TMEMN
                tmemsb  = dat.TMEMSB

            afactr = x.AFACTR
            factor = afactr * mfactor

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

            path = f'{savepath}/{x.SVOL}_{x.SVOLNO}/{sgrpn}'
            MFname = f'{x.SVOL}{x.SVOLNO}_MFACTOR'
            AFname = f'{x.SVOL}{x.SVOLNO}_AFACTR'
            data = f'{smemn}{smemsb}'

            if path in store:
                t = store[path][data].astype(float64).to_numpy()[0:steps]
                if MFname in ts and AFname in ts:
                    t *= ts[MFname][:steps] * ts[AFname][0:steps]
                    msg(4, f'MFACTOR modified by timeseries {MFname}')
                    msg(4, f'AFACTR modified by timeseries {AFname}')
                elif MFname in ts:
                    t *= afactr * ts[MFname][0:steps]
                    msg(4, f'MFACTOR modified by timeseries {MFname}')
                elif AFname in ts:
                    t *= mfactor * ts[AFname][0:steps]
                    msg(4, f'AFACTR modified by timeseries {AFname}')
                else:
                    t *= factor

                # ??? ISSUE: can fetched data be at different frequency - don't know how to transform.
                if tmemn in ts:
                    ts[tmemn] += t
                else:
                    ts[tmemn] = t
            else:
                print('ERROR in FLOWS for', path)
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