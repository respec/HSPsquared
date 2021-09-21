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
from HSP2.utilities import transform, versions, get_timeseries, expand_timeseries_names
from HSP2.configuration import activities, noop, expand_masslinks

from typing import List

def main(hdfname, saveall=False, jupyterlab=True):
    """Runs main HSP2 program.

    Parameters
    ----------
    hdfname: str
        HDF5 (path) filename used for both input and output.
    saveall: Boolean
        [optional] Default is False.
        Saves all calculated data ignoring SAVE tables.
    """

    if not os.path.exists(hdfname):
        raise FileNotFoundError(f'{hdfname} HDF5 File Not Found')

    with HDFStore(hdfname, 'a') as store:
        msg = messages()
        msg(1, f'Processing started for file {hdfname}; saveall={saveall}')

        # read user control, parameters, states, and flags  from HDF5 file
        opseq, ddlinks, ddmasslinks, ddext_sources, ddgener, uci, siminfo = get_uci(store)
        start, stop = siminfo['start'], siminfo['stop']

        copy_instances = {}
        gener_instances = {}

        # main processing loop
        msg(1, f'Simulation Start: {start}, Stop: {stop}')
        for _, operation, segment, delt in opseq.itertuples():
            msg(2, f'{operation} {segment} DELT(minutes): {delt}')

            if operation == 'COPY':
                copy_instances[segment] = activities[operation](store, siminfo, ddext_sources[(operation,segment)]) 
            elif operation == 'GENER':
                try:
                    gener_instances[segment] = activities[operation](segment, copy_instances, gener_instances, ddlinks, ddgener) 
                except NotImplementedError as e:
                    print(f"GENER '{segment}' encountered unsupported feature during initialization and may not function correctly. Unsupported feature: '{e}'")
            else:
                siminfo['delt']      = delt
                siminfo['tindex']    = date_range(start, stop, freq=Minute(delt))[1:]
                siminfo['steps']     = len(siminfo['tindex'])

                # now conditionally execute all activity modules for the op, segment
                ts = get_timeseries(store,ddext_sources[(operation,segment)],siminfo)
                ts = get_gener_timeseries(ts, gener_instances, ddlinks[segment])
                flags = uci[(operation, 'GENERAL', segment)]['ACTIVITY']
                if operation == 'RCHRES':
                    get_flows(store, ts, flags, uci, segment, ddlinks, ddmasslinks, siminfo['steps'], msg)

                for activity, function in activities[operation].items():
                    if function == noop: #or not flags[activity]:
                        continue

                    if (activity in flags) and (not flags[activity]):
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
                            ui['PARAMETERS']['ROS']  = uci[(operation, 'HYDR', segment)]['PARAMETERS']['ROS'] 
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
                            ui['PARAMETERS']['AUX3FG'] = 0
                            if flags['HYDR']:
                                ui['PARAMETERS']['LEN'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LEN']
                                ui['PARAMETERS']['DELTH'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DELTH']
                                ui['PARAMETERS']['DB50'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DB50']
                                ui['PARAMETERS']['AUX3FG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['AUX3FG']
                        if activity == 'GQUAL':
                            ui['advectData'] = uci[(operation, 'ADCALC', segment)]['adcalcData']
                            ui['PARAMETERS']['HTFG'] = flags['HTRCH']
                            ui['PARAMETERS']['SEDFG'] = flags['SEDTRN']
                            # ui['PARAMETERS']['REAMFG'] = uci[(operation, 'OXRX', segment)]['PARAMETERS']['REAMFG']
                            ui['PARAMETERS']['HYDRFG'] = flags['HYDR']
                            if flags['HYDR']:
                                ui['PARAMETERS']['LKFG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LKFG']
                                ui['PARAMETERS']['AUX1FG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['AUX1FG']
                                ui['PARAMETERS']['AUX2FG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['AUX2FG']
                                ui['PARAMETERS']['LEN'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LEN']
                                ui['PARAMETERS']['DELTH'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DELTH']
                            if flags['OXRX']:
                                ui['PARAMETERS']['LKFG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LKFG']
                                ui['PARAMETERS']['CFOREA'] = uci[(operation, 'OXRX', segment)]['PARAMETERS']['CFOREA']
                            if flags['SEDTRN']:
                                ui['PARAMETERS']['SSED1'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED1']
                                ui['PARAMETERS']['SSED2'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED2']
                                ui['PARAMETERS']['SSED3'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED3']
                            if flags['HTRCH']:
                                ui['PARAMETERS']['CFSAEX'] = uci[(operation, 'HTRCH', segment)]['PARAMETERS']['CFSAEX']
                            elif flags['PLANK']:
                                if 'CFSAEX' in uci[(operation, 'PLANK', segment)]['PARAMETERS']:
                                    ui['PARAMETERS']['CFSAEX'] = uci[(operation, 'PLANK', segment)]['PARAMETERS']['CFSAEX']
                        
                        if activity == 'RQUAL':
                            # RQUAL inputs:
                            ui['advectData'] = uci[(operation, 'ADCALC', segment)]['adcalcData']
                            if flags['HYDR']:
                                ui['PARAMETERS']['LKFG'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LKFG']

                            ui['FLAGS']['HTFG'] = flags['HTRCH']
                            ui['FLAGS']['SEDFG'] = flags['SEDTRN']
                            ui['FLAGS']['GQFG'] = flags['GQUAL']
                            ui['FLAGS']['GQALFG4'] = uci[(operation, 'GQUAL', segment)]['GQUAL1']['QALFG4']
                            ui['FLAGS']['OXFG'] = flags['OXFG']
                            ui['FLAGS']['NUTFG'] = flags['NUTRX']
                            ui['FLAGS']['PLKFG'] = flags['PLANK']
                            ui['FLAGS']['PHFG'] = flags['PHCARB']

                            # OXRX module inputs:
                            ui_oxrx = uci[(operation, 'OXRX', segment)] 
                            
                            if flags['HYDR']:
                                ui_oxrx['PARAMETERS']['LEN'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['LEN']
                                ui_oxrx['PARAMETERS']['DELTH'] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['DELTH']
                            
                            if flags['HTRCH']:
                                ui_oxrx['PARAMETERS']['ELEV'] = uci[(operation, 'HTRCH', segment)]['PARAMETERS']['ELEV']

                            if flags['SEDTRN']:
                                ui['PARAMETERS']['SSED1'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED1']
                                ui['PARAMETERS']['SSED2'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED2']
                                ui['PARAMETERS']['SSED3'] = uci[(operation, 'SEDTRN', segment)]['STATES']['SSED3']

                            # NUTRX, PLANK, PHCARB module inputs:
                            ui_nutrx = uci[(operation, 'NUTRX', segment)] 
                            ui_plank = uci[(operation, 'PLANK', segment)] 
                            ui_phcarb = uci[(operation, 'PHCARB', segment)] 

                    ############ calls activity function like snow() ##############
                    if operation not in ['COPY','GENER']:
                        if (activity != 'RQUAL'):
                            errors, errmessages = function(store, siminfo, ui, ts)
                        else:                    
                            errors, errmessages = function(store, siminfo, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts)
                    ###############################################################

                    for errorcnt, errormsg in zip(errors, errmessages):
                        if errorcnt > 0:
                            msg(4, f'Error count {errorcnt}: {errormsg}')
                    if 'SAVE' in ui:
                        save_timeseries(store,ts,ui['SAVE'],siminfo,saveall,operation,segment,activity,jupyterlab)
       
                    if (activity == 'RQUAL'):
                        if 'SAVE' in ui_oxrx:   save_timeseries(store,ts,ui_oxrx['SAVE'],siminfo,saveall,operation,segment,'OXRX',jupyterlab)
                        if 'SAVE' in ui_nutrx:   save_timeseries(store,ts,ui_nutrx['SAVE'],siminfo,saveall,operation,segment,'NUTRX',jupyterlab)
                        if 'SAVE' in ui_plank:   save_timeseries(store,ts,ui_plank['SAVE'],siminfo,saveall,operation,segment,'PLANK',jupyterlab)
                        #if 'SAVE' in ui_phcarb:   save_timeseries(store,ts,ui_phcarb['SAVE'],siminfo,saveall,operation,segment,'PHCARB',jupyterlab)

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
    ddgener =defaultdict(dict)
    siminfo = {}
    opseq = 0

    for path in store.keys():   # finds ALL data sets into HDF5 file
        op, module, *other = path[1:].split(sep='/', maxsplit=3)
        s = '_'.join(other)
        if op == 'CONTROL':
            if module =='GLOBAL':
                temp = store[path].to_dict()['Info']
                siminfo['start'] = Timestamp(temp['Start'])
                siminfo['stop']  = Timestamp(temp['Stop'])
                siminfo['units'] = 1
                if 'Units' in temp:
                    if int(temp['Units']):
                        siminfo['units'] = int(temp['Units'])
            elif module == 'LINKS':
                for row in store[path].fillna('').itertuples():
                    if row.TVOLNO != '':
                        ddlinks[f'{row.TVOLNO}'].append(row)
                    else:
                        ddlinks[f'{row.TOPFST}'].append(row)

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
        elif op == 'GENER':
            for row in store[path].itertuples():
                start, stop = row.OPNID.split()
                for i in range(int(start), int(stop)): ddgener[module][f'G{i:03d}'] = row[2]
    return opseq, ddlinks, ddmasslinks, ddext_sources, ddgener, uci, siminfo

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
                if '_' + y in z:
                    df[z] = ts[z]
        df = df.astype(float32).sort_index(axis='columns')
    elif (operation == 'RCHRES' and (activity == 'CONS' or activity == 'GQUAL')):
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


def get_flows(store, ts, flags, uci, segment, ddlinks, ddmasslinks, steps, msg):
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
                rec['SMEMSB1'] = x.SMEMSB1
                rec['SMEMSB2'] = x.SMEMSB2
                rec['TMEMN'] = x.TMEMN
                rec['TMEMSB1'] = x.TMEMSB1
                rec['TMEMSB2'] = x.TMEMSB2
                rec['SVOL'] = x.SVOL
                recs.append(rec)
            else:  # Data from SCHEMATIC part of Links table
                if dat.SMEMN != '':
                    rec = {}
                    rec['MFACTOR'] = dat.MFACTOR
                    rec['SGRPN'] = dat.SGRPN
                    rec['SMEMN'] = dat.SMEMN
                    rec['SMEMSB1'] = dat.SMEMSB1
                    rec['SMEMSB2'] = dat.SMEMSB2
                    rec['TMEMN'] = dat.TMEMN
                    rec['TMEMSB1'] = dat.TMEMSB1
                    rec['TMEMSB2'] = dat.TMEMSB2
                    rec['SVOL'] = dat.SVOL
                    recs.append(rec)
                else:
                    # this is the kind that needs to be expanded
                    if dat.SGRPN == "ROFLOW" or dat.SGRPN == "OFLOW":
                        recs = expand_masslinks(flags,uci,dat,recs)

            for rec in recs:
                mfactor = rec['MFACTOR']
                sgrpn   = rec['SGRPN']
                smemn   = rec['SMEMN']
                smemsb1 = rec['SMEMSB1']
                smemsb2 = rec['SMEMSB2']
                tmemn   = rec['TMEMN']
                tmemsb1 = rec['TMEMSB1']
                tmemsb2 = rec['TMEMSB2']

                afactr = x.AFACTR
                factor = afactr * mfactor

                # KLUDGE until remaining HSP2 modules are available.
                if tmemn not in {'IVOL', 'ICON', 'IHEAT', 'ISED', 'ISED1', 'ISED2', 'ISED3', 
                                    'IDQAL', 'ISQAL1', 'ISQAL2', 'ISQAL3',
                                    'OXIF', 'NUIF1', 'NUIF2', 'PKIF'}:
                    continue
                if (sgrpn == 'OFLOW' and smemn == 'OVOL') or (sgrpn == 'ROFLOW' and smemn == 'ROVOL'):
                    sgrpn = 'HYDR'
                if (sgrpn == 'OFLOW' and smemn == 'OHEAT') or (sgrpn == 'ROFLOW' and smemn == 'ROHEAT'):
                    sgrpn = 'HTRCH'
                if (sgrpn == 'OFLOW' and smemn == 'OSED') or (sgrpn == 'ROFLOW' and smemn == 'ROSED'):
                    sgrpn = 'SEDTRN'
                if (sgrpn == 'OFLOW' and smemn == 'ODQAL') or (sgrpn == 'ROFLOW' and smemn == 'RODQAL'):
                    sgrpn = 'GQUAL'
                if (sgrpn == 'OFLOW' and smemn == 'OSQAL') or (sgrpn == 'ROFLOW' and smemn == 'ROSQAL'):
                    sgrpn = 'GQUAL'
                if (sgrpn == 'OFLOW' and smemn == 'OXCF2') or (sgrpn == 'ROFLOW' and smemn == 'OXCF1'):
                    sgrpn = 'OXRX'
                if (sgrpn == 'OFLOW' and (smemn == 'NUCF9' or smemn == 'OSNH4' or smemn == 'OSPO4')) or (sgrpn == 'ROFLOW' and (smemn == 'NUCF1' or smemn == 'NUFCF2')):
                    sgrpn = 'NUTRX'
                if (sgrpn == 'OFLOW' and smemn == 'PKCF2') or (sgrpn == 'ROFLOW' and smemn == 'PKCF1'):
                    sgrpn = 'PLANK'
                
                if tmemn == 'ISED' or tmemn == 'ISQAL':
                    tmemn = tmemn + tmemsb1    # need to add sand, silt, clay subscript

                smemn, tmemn = expand_timeseries_names(smemn, smemsb1, smemsb2, tmemn, tmemsb1, tmemsb2)

                path = f'RESULTS/{x.SVOL}_{x.SVOLNO}/{sgrpn}'
                MFname = f'{x.SVOL}{x.SVOLNO}_MFACTOR'
                AFname = f'{x.SVOL}{x.SVOLNO}_AFACTR'
                data = f'{smemn}{smemsb1}{smemsb2}'

                if path in store:
                    if data in store[path]:
                        t = store[path][data].astype(float64).to_numpy()[0:steps]
                    else:
                        data = f'{smemn}'
                        if data in store[path]:
                            t = store[path][data].astype(float64).to_numpy()[0:steps]
                        else:
                            print('ERROR in FLOWS, cant resolve ', path + ' ' + smemn)
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

                    # if poht to iheat, imprecision in hspf conversion factor requires a slight adjustment
                    if (smemn == 'POHT' or smemn == 'SOHT') and tmemn == 'IHEAT':
                       t *= 0.998553

                    # ??? ISSUE: can fetched data be at different frequency - don't know how to transform.
                    if tmemn in ts:
                        ts[tmemn] += t
                    else:
                        ts[tmemn] = t
                else:
                    print('ERROR in FLOWS for', path)
    return

def get_gener_timeseries(ts: Dict, gener_instances: Dict, ddlinks: List) -> Dict:
    """
    Uses links tables to load necessary TimeSeries from Gener class instances to TS dictionary
    """
    for link in ddlinks:
        if link.SVOL == 'GENER':
            gener = gener_instances[link.SVOLNO]
            series = gener.get_ts()
            ts[f'{link.TMEMN}{link.TMEMSB1} {link.TMEMSB2}'.rstrip()] = series
    return ts


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