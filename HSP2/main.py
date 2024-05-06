''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from re import S
from numpy import float64, float32
from pandas import DataFrame, date_range
from pandas.tseries.offsets import Minute
from datetime import datetime as dt
import os
from HSP2.utilities import versions, get_timeseries, expand_timeseries_names, save_timeseries, get_gener_timeseries
from HSP2.configuration import activities, noop, expand_masslinks
from HSP2.state import *
from HSP2.om import *
from HSP2.SPECL import *

from HSP2IO.io import IOManager, SupportsReadTS, Category

def main(io_manager:IOManager, saveall:bool=False, jupyterlab:bool=True) -> None:
    """Runs main HSP2 program.

    Parameters
    ----------
   
    saveall: Boolean - [optional] Default is False.
        Saves all calculated data ignoring SAVE tables.
    jupyterlab: Boolean - [optional] Default is True.
        Flag for specific output behavior for  jupyter lab.
    Return
    ------------
    None
    
    """

    hdfname = io_manager._input.file_path
    if not os.path.exists(hdfname):
        raise FileNotFoundError(f'{hdfname} HDF5 File Not Found')

    msg = messages()
    msg(1, f'Processing started for file {hdfname}; saveall={saveall}')

    # read user control, parameters, states, and flags uci and map to local variables
    uci_obj = io_manager.read_uci()
    opseq = uci_obj.opseq
    ddlinks = uci_obj.ddlinks
    ddmasslinks = uci_obj.ddmasslinks
    ddext_sources = uci_obj.ddext_sources
    ddgener = uci_obj.ddgener
    uci = uci_obj.uci
    siminfo = uci_obj.siminfo 
    ftables = uci_obj.ftables
    specactions = uci_obj.specactions
    monthdata = uci_obj.monthdata
    
    start, stop = siminfo['start'], siminfo['stop']

    copy_instances = {}
    gener_instances = {}

    #######################################################################################
    # initialize STATE dicts
    #######################################################################################
    # Set up Things in state that will be used in all modular activities like SPECL
    state = init_state_dicts()
    state_siminfo_hsp2(uci_obj, siminfo)
    # Add support for dynamic functions to operate on STATE
    # - Load any dynamic components if present, and store variables on objects 
    state_load_dynamics_hsp2(state, io_manager, siminfo)
    # Iterate through all segments and add crucial paths to state 
    # before loading dynamic components that may reference them
    state_init_hsp2(state, opseq, activities)
    # - finally stash specactions in state, not domain (segment) dependent so do it once
    state['specactions'] = specactions # stash the specaction dict in state
    state_initialize_om(state)
    state_load_dynamics_specl(state, io_manager, siminfo)   # traditional special actions
    state_load_dynamics_om(state, io_manager, siminfo)      # operational model for custom python
    # finalize all dynamically loaded components and prepare to run the model
    state_om_model_run_prep(state, io_manager, siminfo)
    #######################################################################################

    # main processing loop
    msg(1, f'Simulation Start: {start}, Stop: {stop}')
    tscat = {}
    for _, operation, segment, delt in opseq.itertuples():
        msg(2, f'{operation} {segment} DELT(minutes): {delt}')
        siminfo['delt'] = delt
        siminfo['tindex'] = date_range(start, stop, freq=Minute(delt))[1:]
        siminfo['steps'] = len(siminfo['tindex'])

        if operation == 'COPY':
            copy_instances[segment] = activities[operation](io_manager, siminfo, ddext_sources[(operation,segment)]) 
        elif operation == 'GENER':
            try:
                ts = get_timeseries(io_manager, ddext_sources[(operation, segment)], siminfo)
                ts = get_gener_timeseries(ts, gener_instances, ddlinks[segment], ddmasslinks)
                get_flows(io_manager, ts, {}, uci, segment, ddlinks, ddmasslinks, siminfo['steps'], msg)
                gener_instances[segment] = activities[operation](segment, siminfo, copy_instances, gener_instances, ddlinks, ddmasslinks, ts, ddgener)
            except NotImplementedError as e:
                print(f"GENER '{segment}' may not function correctly. '{e}'")
        else:

            # now conditionally execute all activity modules for the op, segment
            ts = get_timeseries(io_manager,ddext_sources[(operation,segment)],siminfo)
            ts = get_gener_timeseries(ts, gener_instances, ddlinks[segment],ddmasslinks)
            flags = uci[(operation, 'GENERAL', segment)]['ACTIVITY']
            if operation == 'RCHRES':
                # Add nutrient adsorption flags:
                if flags['NUTRX'] == 1:
                    flags['TAMFG'] = uci[(operation, 'NUTRX', segment)]['FLAGS']['NH3FG']
                    flags['ADNHFG'] = uci[(operation, 'NUTRX', segment)]['FLAGS']['ADNHFG']
                    flags['PO4FG'] = uci[(operation, 'NUTRX', segment)]['FLAGS']['PO4FG']
                    flags['ADPOFG'] = uci[(operation, 'NUTRX', segment)]['FLAGS']['ADPOFG']
                
                get_flows(io_manager, ts, flags, uci, segment, ddlinks, ddmasslinks, siminfo['steps'], msg)

            for activity, function in activities[operation].items():
                if function == noop: #or not flags[activity]:
                    continue

                if (activity in flags) and (not flags[activity]):
                    continue

                if (activity == 'RQUAL') and (not flags['OXRX']) and (not flags['NUTRX']) and (not flags['PLANK']) and (not flags['PHCARB']):
                    continue

                msg(3, f'{activity}')
                # Set context for dynamic executables and special actions
                state_context_hsp2(state, operation, segment, activity)
                
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
                        nexits = uci[(operation, 'HYDR', segment)]['PARAMETERS']['NEXITS']
                        for index in range(nexits):
                            ui['PARAMETERS']['OS' + str(index + 1)] = uci[(operation, 'HYDR', segment)]['PARAMETERS']['OS'+ str(index + 1)]
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
                        ui['FLAGS']['OXFG'] = flags['OXFG']
                        ui['FLAGS']['NUTFG'] = flags['NUTRX']
                        ui['FLAGS']['PLKFG'] = flags['PLANK']
                        ui['FLAGS']['PHFG'] = flags['PHCARB']
                        if flags['CONS']:
                            if 'PARAMETERS' in uci[(operation, 'CONS', segment)]:
                                if 'NCONS' in uci[(operation, 'CONS', segment)]['PARAMETERS']:
                                    ui['PARAMETERS']['NCONS'] = uci[(operation, 'CONS', segment)]['PARAMETERS']['NCONS']

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

                        # PLANK module inputs:
                        if flags['HTRCH']:
                            ui['PARAMETERS']['CFSAEX'] = uci[(operation, 'HTRCH', segment)]['PARAMETERS']['CFSAEX']

                        # NUTRX, PLANK, PHCARB module inputs:
                        ui_nutrx = uci[(operation, 'NUTRX', segment)] 
                        ui_plank = uci[(operation, 'PLANK', segment)] 
                        ui_phcarb = uci[(operation, 'PHCARB', segment)] 

                ############ calls activity function like snow() ##############
                if operation not in ['COPY','GENER']:
                    if (activity == 'HYDR'):
                        errors, errmessages = function(io_manager, siminfo, ui, ts, ftables, state)
                    elif (activity == 'SEDTRN'):
                        errors, errmessages = function(io_manager, siminfo, ui, ts, state)
                    elif (activity != 'RQUAL'):
                        errors, errmessages = function(io_manager, siminfo, ui, ts)
                    else:                    
                        errors, errmessages = function(io_manager, siminfo, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts, monthdata)
                ###############################################################

                for errorcnt, errormsg in zip(errors, errmessages):
                    if errorcnt > 0:
                        msg(4, f'Error count {errorcnt}: {errormsg}')

                # default to hourly output
                outstep = 2
                outstep_oxrx = 2
                outstep_nutrx = 2
                outstep_plank = 2
                outstep_phcarb = 2
                if 'BINOUT' in uci[(operation, 'GENERAL', segment)]:
                    if activity in uci[(operation, 'GENERAL', segment)]['BINOUT']:
                        outstep = uci[(operation, 'GENERAL', segment)]['BINOUT'][activity]
                    elif activity == 'RQUAL':
                        outstep_oxrx = uci[(operation, 'GENERAL', segment)]['BINOUT']['OXRX']
                        outstep_nutrx = uci[(operation, 'GENERAL', segment)]['BINOUT']['NUTRX']
                        outstep_plank = uci[(operation, 'GENERAL', segment)]['BINOUT']['PLANK']
                        outstep_phcarb = uci[(operation, 'GENERAL', segment)]['BINOUT']['PHCARB']

                if 'SAVE' in ui:
                    save_timeseries(io_manager,ts,ui['SAVE'],siminfo,saveall,operation,segment,activity,jupyterlab,outstep)
    
                if (activity == 'RQUAL'):
                    if 'SAVE' in ui_oxrx:   save_timeseries(io_manager,ts,ui_oxrx['SAVE'],siminfo,saveall,operation,segment,'OXRX',jupyterlab,outstep_oxrx)
                    if 'SAVE' in ui_nutrx and flags['NUTRX'] == 1:   save_timeseries(io_manager,ts,ui_nutrx['SAVE'],siminfo,saveall,operation,segment,'NUTRX',jupyterlab,outstep_nutrx)
                    if 'SAVE' in ui_plank and flags['PLANK'] == 1:  save_timeseries(io_manager,ts,ui_plank['SAVE'],siminfo,saveall,operation,segment,'PLANK',jupyterlab,outstep_plank)
                    if 'SAVE' in ui_phcarb and flags['PHCARB'] == 1:   save_timeseries(io_manager,ts,ui_phcarb['SAVE'],siminfo,saveall,operation,segment,'PHCARB',jupyterlab,outstep_phcarb)

    msglist = msg(1, 'Done', final=True)

    df = DataFrame(msglist, columns=['logfile'])
    io_manager.write_log(df)

    if jupyterlab:
        df = versions(['jupyterlab', 'notebook'])
        io_manager.write_versioning(df)
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

def get_flows(io_manager:SupportsReadTS, ts, flags, uci, segment, ddlinks, ddmasslinks, steps, msg):
    # get inflows to this operation
    for x in ddlinks[segment]:
        if x.SVOL != 'GENER':   # gener already handled in get_gener_timeseries
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
            else: # Data from SCHEMATIC part of Links table
                mldata = ddmasslinks[x.MLNO]
                for dat in mldata:
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

                if x.AFACTR != '':
                    afactr = x.AFACTR
                    factor = afactr * mfactor
                else:
                    factor = mfactor

                # KLUDGE until remaining HSP2 modules are available.
                if tmemn not in {'IVOL', 'ICON', 'IHEAT', 'ISED', 'ISED1', 'ISED2', 'ISED3',
                                    'IDQAL', 'ISQAL1', 'ISQAL2', 'ISQAL3',
                                    'OXIF', 'NUIF1', 'NUIF2', 'PKIF', 'PHIF',
                                    'ONE', 'TWO'}:
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
                if (sgrpn == 'OFLOW' and smemn == 'PHCF2') or (sgrpn == 'ROFLOW' and smemn == 'PHCF1'):
                    sgrpn = 'PHCARB'

                if tmemn == 'ISED' or tmemn == 'ISQAL':
                    tmemn = tmemn + tmemsb1    # need to add sand, silt, clay subscript
                if (sgrpn == 'HYDR' and smemn == 'OVOL') or (sgrpn == 'HTRCH' and smemn == 'OHEAT'):
                    smemsb2 = ''
                if sgrpn == 'GQUAL' and smemsb2 == '':
                    smemsb2 = '1'

                smemn, tmemn = expand_timeseries_names(sgrpn, smemn, smemsb1, smemsb2, tmemn, tmemsb1, tmemsb2)

                path = f'RESULTS/{x.SVOL}_{x.SVOLNO}/{sgrpn}'
                MFname = f'{x.SVOL}{x.SVOLNO}_MFACTOR'
                AFname = f'{x.SVOL}{x.SVOLNO}_AFACTR'
                data = f'{smemn}{smemsb1}{smemsb2}'

                data_frame = io_manager.read_ts(Category.RESULTS,x.SVOL,x.SVOLNO, sgrpn)
                try:
                    if data in data_frame.columns: t = data_frame[data].astype(float64).to_numpy()[0:steps]
                    else: t = data_frame[smemn].astype(float64).to_numpy()[0:steps]

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
                    if (smemn == 'PODOXM' or smemn == 'SODOXM') and tmemn == 'OXIF1':
                        t *= 1.000565

                    # ??? ISSUE: can fetched data be at different frequency - don't know how to transform.
                    if tmemn in ts:
                        ts[tmemn] += t
                    else:
                        ts[tmemn] = t

                except KeyError:
                    print('ERROR in FLOWS, cant resolve ', path + ' ' + smemn)

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
