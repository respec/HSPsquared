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
from HSP2.utilities import transform, versions
from HSP2.configuration import activities, noop


def main(hdfname, saveall=False, jupyterlab=True):
    '''Runs main HSP2 program.

    Parameters
    ----------
    hdfname: str
        HDF5 (path) filename used for both input and output.
    saveall: Boolean
        [optional] Default is False.
        Saves all calculated data ignoring SAVE tables.
    '''

    if not os.path.exists(hdfname):
        print(f'{hdfname} HDF5 File Not Found, QUITTING')
        return

    with HDFStore(hdfname, 'a') as store:
        msg = messages()
        msg(1, f'Processing started for file {hdfname}; saveall={saveall}')

        # read user control, parameters, states, and flags  from HDF5 file
        opseq, ddlinks, ddmasslinks, ddext_sources, uci, siminfo = get_uci(store)
        start, stop = siminfo['start'], siminfo['stop']

        # main processing loop
        msg(1, f'Simulation Start: {start}, Stop: {stop}')
        for _, operation, segment, delt in opseq.itertuples():
            msg(2, f'{operation} {segment} DELT(minutes): {delt}')
            siminfo['delt']      = delt
            siminfo['tindex']    = date_range(start, stop, freq=Minute(delt))[0:-1]
            siminfo['steps']     = len(siminfo['tindex'])

            # now conditionally execute all activity modules for the op, segment
            ts = get_timeseries(store,ddext_sources[(operation,segment)],siminfo)
            flags = uci[(operation, 'GENERAL', segment)]['ACTIVITY']
            if operation == 'RCHRES':
                get_flows(store, ts, flags, segment, ddlinks, ddmasslinks, siminfo['steps'], msg)

            for activity, function in activities[operation].items():
                if function == noop or not flags[activity]:
                    continue

                msg(3, f'{activity}')

                ui = uci[(operation, activity, segment)]   # ui is a dictionary
                if operation == 'PERLND' and activity == 'SEDMNT':
                    # special exception here to make CSNOFG available
                    ui['PARAMETERS']['CSNOFG'] = uci[(operation, 'PWATER', segment)]['PARAMETERS']['CSNOFG']
                if operation == 'PERLND' and activity == 'PSTEMP':
                    # special exception here to make AIRTFG available
                    ui['PARAMETERS']['AIRTFG'] = flags['ATEMP']
                if operation == 'PERLND' and activity == 'PWTGAS':
                    # special exception here to make CSNOFG available
                    ui['PARAMETERS']['CSNOFG'] = uci[(operation, 'PWATER', segment)]['PARAMETERS']['CSNOFG']
                if operation == 'RCHRES':
                    if not 'PARAMETERS' in ui:
                        ui['PARAMETERS'] = {}
                    ui['PARAMETERS']['NEXITS'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['NEXITS']
                    if activity == 'ADCALC':
                        ui['PARAMETERS']['ADFG'] = flags['ADCALC']
                        ui['PARAMETERS']['KS']   = uci[(operation, 'HYDR', segment)]['PARAMETERS']['KS']
                        ui['PARAMETERS']['VOL']  = uci[(operation, 'HYDR', segment)]['STATES']['VOL']
                    if activity == 'HTRCH':
                        ui['PARAMETERS']['ADFG'] = flags['ADCALC']
                        ui['advectData'] = uci[(operation, 'ADCALC', segment)]['adcalcData']
                        # ui['STATES']['VOL'] = uci[(operation, 'HYDR', segment)]['STATES']['VOL']
                    if activity == 'CONS':
                        ui['advectData'] = uci[(operation, 'ADCALC', segment)]['adcalcData']
                    if activity == 'SEDTRN':
                        ui['PARAMETERS']['ADFG'] = flags['ADCALC']
                        ui['advectData'] = uci[(operation, 'ADCALC', segment)]['adcalcData']
                        # ui['STATES']['VOL'] = uci[(operation, 'HYDR', segment)]['STATES']['VOL']
                        ui['PARAMETERS']['HTFG'] = flags['HTRCH']
                        if flags['HYDR']:
                            ui['PARAMETERS']['LEN'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LEN']
                            ui['PARAMETERS']['DELTH'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DELTH']
                            ui['PARAMETERS']['DB50'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DB50']

                ############ calls activity function like snow() ##############
                errors, errmessages = function(store, siminfo, ui, ts)
                ###############################################################

                for errorcnt, errormsg in zip(errors, errmessages):
                    if errorcnt > 0:
                        msg(4, f'Error count {errorcnt}: {errormsg}')
                if 'SAVE' in ui:
                    save_timeseries(store,ts,ui['SAVE'],siminfo,saveall,operation,segment,activity,jupyterlab)
        msglist = msg(1, 'Done', final=True)

        df = DataFrame(msglist, columns=['logfile'])
        df.to_hdf(store, 'RUN_INFO/LOGFILE', data_columns=True, format='t')

        if jupyterlab:
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
    ddlinks = defaultdict(list)
    ddmasslinks = defaultdict(list)
    ddext_sources = defaultdict(list)
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
                for row in store[path].replace('na','').itertuples():
                    ddlinks[row.TVOLNO].append(row)
            elif module == 'MASS_LINKS':
                for row in store[path].replace('na','').itertuples():
                    ddmasslinks[row.MLNO].append(row)
            elif module == 'EXT_SOURCES':
                for row in store[path].replace('na','').itertuples():
                    ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
            elif module == 'OP_SEQUENCE':
                opseq = store[path]
        elif op in {'PERLND', 'IMPLND', 'RCHRES'}:
            for id, vdict in store[path].to_dict('index').items():
                uci[(op, module, id)][s] = vdict
    return opseq, ddlinks, ddmasslinks, ddext_sources, uci, siminfo


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


def save_timeseries(store, ts, savedict, siminfo, saveall, operation, segment, activity, jupyterlab=True):
    # save computed timeseries (at computation DELT)
    save = {k for k,v in savedict.items() if v or saveall}
    df = DataFrame(index=siminfo['tindex'])
    if (operation == 'IMPLND' and activity == 'IQUAL') or (operation == 'PERLND' and activity == 'PQUAL'):
        for y in save:
            for z in set(ts.keys()):
                if '/' + y in z:
                    zrep = z.replace('/','_')
                    zrep2 = zrep.replace(' ', '')
                    df[zrep2] = ts[z]
        df = df.astype(float32).sort_index(axis='columns')
    elif (operation == 'RCHRES' and activity == 'CONS'):
        for y in save:
            for z in set(ts.keys()):
                if '_' + y in z:
                    df[z] = ts[z]
        for y in (save & set(ts.keys())):
            df[y] = ts[y]
        df = df.astype(float32).sort_index(axis='columns')
    else:
        for y in (save & set(ts.keys())):
            df[y] = ts[y]
        df = df.astype(float32).sort_index(axis='columns')
    path = f'RESULTS/{operation}_{segment}/{activity}'
    if not df.empty:
        if jupyterlab:
            df.to_hdf(store, path, complib='blosc', complevel=9) # This is the official version
        else:
            df.to_hdf(store, path, format='t', data_columns=True)  # show the columns in HDFView
    else:
        print('Save DataFrame Empty for', path)
    return


def get_flows(store, ts, flags, segment, ddlinks, ddmasslinks, steps, msg):
    # get inflows to this operation
    for x in ddlinks[segment]:
        mldata = ddmasslinks[x.MLNO]
        for dat in mldata:
            recs = []
            if x.MLNO == '':  # Data from NETWORK part of Links table
                rec = {}
                rec['MFACTOR'] = x.MFACTOR
                rec['SGRPN'] = x.SGRPN
                rec['SMEMN'] = x.SMEMN
                rec['SMEMSB'] = x.SMEMSB
                rec['TMEMN'] = x.TMEMN
                rec['TMEMSB'] = x.TMEMSB
                rec['SVOL'] = x.SVOL
                recs.append(rec)
            else:  # Data from SCHEMATIC part of Links table
                if dat.SMEMN != '':
                    rec = {}
                    rec['MFACTOR'] = dat.MFACTOR
                    rec['SGRPN'] = dat.SGRPN
                    rec['SMEMN'] = dat.SMEMN
                    rec['SMEMSB'] = dat.SMEMSB
                    rec['TMEMN'] = dat.TMEMN
                    rec['TMEMSB'] = dat.TMEMSB
                    rec['SVOL'] = dat.SVOL
                    recs.append(rec)
                else:
                    # this is the kind that needs to be expanded
                    if dat.SGRPN == "ROFLOW" or dat.SGRPN == "OFLOW":
                        if flags['HYDR']:
                            # IVOL
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'HYDR'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROVOL'
                            else:
                                rec['SMEMN'] = 'OVOL'
                            rec['SMEMSB'] = dat.SMEMSB
                            rec['TMEMN'] = 'IVOL'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)
                        if flags['CONS']:
                            # ICONS
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'CONS'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROCON'
                            else:
                                rec['SMEMN'] = 'OCON'
                            rec['SMEMSB'] = dat.SMEMSB  # first sub is exit number
                            rec['TMEMN'] = 'ICON'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)
                        if flags['HTRCH']:
                            # IHEAT
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'HTRCH'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROHEAT'
                            else:
                                rec['SMEMN'] = 'OHEAT'
                            rec['SMEMSB'] = dat.SMEMSB
                            rec['TMEMN'] = 'IHEAT'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)
                        if flags['SEDTRN']:
                            # ISED1
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'SEDTRN'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROSED'
                                rec['SMEMSB'] = '1'
                            else:
                                rec['SMEMN'] = 'OSED'
                                rec['SMEMSB'] = '1' + dat.SMEMSB
                            rec['TMEMN'] = 'ISED1'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)
                            # ISED2
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'SEDTRN'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROSED'
                                rec['SMEMSB'] = '2'
                            else:
                                rec['SMEMN'] = 'OSED'
                                rec['SMEMSB'] = '2' + dat.SMEMSB
                            rec['TMEMN'] = 'ISED2'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)
                            # ISED3
                            rec = {}
                            rec['MFACTOR'] = dat.MFACTOR
                            rec['SGRPN'] = 'SEDTRN'
                            if dat.SGRPN == "ROFLOW":
                                rec['SMEMN'] = 'ROSED'
                                rec['SMEMSB'] = '3'
                            else:
                                rec['SMEMN'] = 'OSED'
                                rec['SMEMSB'] = '3' + dat.SMEMSB
                            rec['TMEMN'] = 'ISED3'
                            rec['TMEMSB'] = dat.TMEMSB
                            rec['SVOL'] = dat.SVOL
                            recs.append(rec)

            for rec in recs:
                mfactor = rec['MFACTOR']
                sgrpn   = rec['SGRPN']
                smemn   = rec['SMEMN']
                smemsb  = rec['SMEMSB']
                tmemn   = rec['TMEMN']
                tmemsb  = rec['TMEMSB']

                afactr = x.AFACTR
                factor = afactr * mfactor

                # KLUDGE until remaining HSP2 modules are available.
                if tmemn not in {'IVOL', 'ICON', 'IHEAT', 'ISED', 'ISED1', 'ISED2', 'ISED3'}:
                    continue
                if (sgrpn == 'OFLOW' and smemn == 'OVOL') or (sgrpn == 'ROFLOW' and smemn == 'ROVOL'):
                     sgrpn = 'HYDR'
                if (sgrpn == 'OFLOW' and smemn == 'OHEAT') or (sgrpn == 'ROFLOW' and smemn == 'ROHEAT'):
                     sgrpn = 'HTRCH'
                if (sgrpn == 'OFLOW' and smemn == 'OSED') or (sgrpn == 'ROFLOW' and smemn == 'ROSED'):
                     sgrpn = 'SEDTRN'
                if tmemn == 'ISED':
                    tmemn = tmemn + tmemsb
                if tmemn == 'ICON':
                    if tmemsb == '':
                        tmemn = 'CONS1_ICON'
                    else:
                        tmemn = 'CONS' + tmemsb + '_ICON'
                if smemn == 'OCON':
                    if smemsb == '':
                        smemn = 'CONS1_OCON1'
                    else:
                        smemn = 'CONS1' + '_OCON' + smemsb
                if smemn == 'ROCON':
                    if smemsb == '':
                        smemn = 'CONS1_ROCON'
                    else:
                        smemn = 'CONS' + smemsb + '_ROCON'

                path = f'RESULTS/{x.SVOL}_{x.SVOLNO}/{sgrpn}'
                MFname = f'{x.SVOL}{x.SVOLNO}_MFACTOR'
                AFname = f'{x.SVOL}{x.SVOLNO}_AFACTR'
                data = f'{smemn}{smemsb}'

                if path in store:
                    if data in store[path]:
                        t = store[path][data].astype(float64).to_numpy()[0:steps]
                    else:
                        data = f'{smemn}'
                        if data in store[path]:
                            t = store[path][data].astype(float64).to_numpy()[0:steps]
                        else:
                            print('ERROR in FLOWS, cant resolve ', path)
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