''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.'''

"""
FUTURE:
    Allow timeseries to be called for MFACTOR and AFACTR
    Allow timeseries to have arbitrary math specified in MFACTOR and AFACTR
    Make sure midnight issue is fixed for Tim Cera's library
    GENERAL: have separate HSPF start& stop datetime from HSP2 datetimes
    Read LAPSE and SEASONS once?
    Add MEAN, INTREP to transform()
    eliminate H5py direct call using Pandas
"""


import os
import importlib
import h5py
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime as dt
from collections import defaultdict

import HSP2


def run_DoE(hdfname, simname, doe, saveall=False):
    ''' runs main HSP2 program
     simname is the directory to store simulation UCI and results directories
     doe is the dictionary of UCI data defining the design of experiment
     saveall - (optional) saves all calculated data, ignoring SAVE tables'''

    if not os.path.exists(hdfname):
        print (hdfname + ' HDF5 File Not Found, QUITTING')
        return

    logpath = os.path.join(os.path.dirname(hdfname), 'logfile.txt')
    with pd.HDFStore(hdfname) as store, open(logpath, 'w') as logfile:
        msg = messages(logfile)
        msg(1, 'HSP2 Started for file ' + hdfname)
        msg(2, 'Design of Experiment HDF5 directory is ' + simname)

        # ordered list of modules replacing function names with real functions
        sequence = defaultdict(list)
        for _,x in store['HSP2/CONFIGURATION'].sort_values(by=['Order']).iterrows():
            if x.Function and x.Function != 'noop':
                importlib.import_module(x.Module)
                x.Function = eval(x.Module + '.' + x.Function)
                sequence[x.Target].append(x)

        if 'TIMESERIES/LAPSE24'   not in store:
            store['TIMESERIES/LAPSE24']   = store['HSP2/LAPSE24']
        if 'TIMESERIES/SEASONS12' not in store:
            store['TIMESERIES/SEASONS12'] = store['HSP2/SEASONS12']
        if 'TIMESERIES/SaturatedVaporPressureTable' not in store:
            store['TIMESERIES/SaturatedVaporPressureTable'] = store['HSP2/SaturatedVaporPressureTable']

        # initial values for simulation wide data dictionary
        general = store['CONTROL/GLOBAL'].Data.to_dict()
        general['msg'] = msg
        msg(1, '  Start ' + general['sim_start'] + '    Stop ' + general['sim_end'])

        #create monthly tables; Example: monthlys['PERLND', 'P001']['CEPSCM']
        keys = store['HSP2/KEYS']
        monthlys = defaultdict(dict)
        for key in [key for key in keys if 'MONTHLY' in key]:
            tokens = key.split('/')     # target=tokens[1], variable=tokens[-1]
            for indx,row in store[key].iterrows():
                monthlys[(tokens[1], indx)][tokens[-1]] = tuple(row)

        doeuci = defaultdict(dict)
        for x in doe.itertuples():
            doeuci[x.Run, x.Operation, x.Segment][x.Parameter]= x.Value
        runlist = doe.Run.unique()

        doe.to_hdf(hdfname, simname + '/DoE', data_columns=True, format='table')

        ucs = get_ucs(keys, store, msg)     # read all default user control info
        tsdd = defaultdict(list)
        for row in store['/CONTROL/EXT_SOURCES'].itertuples():
            tsdd[row.TVOL,row.TVOLNO].append(row)      # get timeseries' info

        # get data for LINK (combined NETWORK & SCHEMATIC) and MASSLINK information
        linkdd = defaultdict(list)
        for _,row in store['CONTROL/LINKS'].iterrows():
            linkdd[row.TVOL, row.TVOLNO].append(row)
        mldd = defaultdict(list)
        for i,row in store['CONTROL/MASS_LINK'].iterrows():
            mldd[row.MLNO].append(row)
        xflow = store['/HSP2/FLOWEXPANSION']
        xflowdd = {}
        lookup = defaultdict(list)
        for _,row in xflow.iterrows():
            xflowdd[row.Flag, row.INFLOW] = row
            lookup[row.Flag].append(row.INFLOW)
        flowdata = {'linkdd':linkdd, 'mldd':mldd, 'xflowdd':xflowdd, 'lookup':lookup}
        msg(1, 'Finished setup')

        # embedded "SMART OP_SEQUENCE"
        sep = '_'
        master = nx.DiGraph()
        for x in store['CONTROL/LINKS'].itertuples():
            master.add_edge(x.SVOL + sep + x.SVOLNO, x.TVOL + sep + x.TVOLNO)
        changed = list(set([(x.Operation + sep + x.Segment) for x in doe.itertuples()]))
        for x in nx.topological_sort(master):
            if x not in changed:
                master.remove_node(x)
            else:
                changed.extend(master.successors(x))
        order = [tuple(x.split(sep)) for x in nx.topological_sort(master)]
        deltdict = {(x.TARGET,x.ID):x.DELT for x in store['CONTROL/OP_SEQUENCE'].itertuples()}
        newopseq = [(x[0], x[1], deltdict[x]) for x in order]
        newopseq = pd.DataFrame(newopseq, columns=['TARGET', 'ID', 'DELT'])

        for run in runlist:  # loop over all runs in the design of experiment
            msg(2, 'Starting RUN ' + run)
            # main program -- loop over OP_SEQUENCE table
            for opseq in newopseq.itertuples():
                msg(2, opseq.TARGET + ' ' + opseq.ID + '   DELT=' + opseq.DELT)

                tindex = pd.date_range(general['sim_start'],general['sim_end'],freq=opseq.DELT+'min')
                general['tindex']   = tindex
                general['sim_len']  = len(general['tindex'])
                general['sim_delt'] = float(opseq.DELT)

                ts = get_timeseries(tsdd[opseq.TARGET,opseq.ID], general['tindex'], store)

                # Loop over each Flag in the operation, do function available
                activity = ucs[opseq.TARGET, 'ACTIVITY', opseq.ID]
                for x in sequence[opseq.TARGET]:
                    if activity[x.Flag]:
                        uc = ucs[opseq.TARGET,x.Flag, opseq.ID].to_dict()   #??? isn't this already a dict???
                        if (opseq.TARGET,opseq.ID) in monthlys:
                            uc.update(dict(monthlys[opseq.TARGET, opseq.ID]))
                        uc.update(ucs[opseq.TARGET, 'GENERAL_INFO', opseq.ID])
                        if opseq.TARGET=='RCHRES' and x.Flag=='HYDRFG':
                            uc['rchtab'] = store['FTABLES/' + uc['FTBUCI']]  # get FTABLE

                        if (run, opseq.TARGET, opseq.ID) in doeuci:
                            uc.update(doeuci[run, opseq.TARGET, opseq.ID]) #???

                        if (opseq.TARGET, opseq.ID) in linkdd:
                            flows(ts,opseq.TARGET,opseq.ID,x.Flag,flowdata,tindex,store,simname,run)

                        ###########################################################
                        errs, errstrs = x.Function(store, general, uc, ts)   # calls core HSP2 functions
                        ###########################################################

                        # save computed timeseries (at computation DELT)
                        savetable = ucs[opseq.TARGET, x.Flag, 'SAVE', opseq.ID]
                        save = tuple(savetable.index) if saveall else tuple(savetable[savetable==True].index)
                        save = set(save) & set(ts)
                        df = pd.DataFrame(index=tindex)
                        for y in save:
                            if ts[y].ndim == 1:
                                df[y] = ts[y]
                            else:
                                for i in range(ts[y].shape[1]):
                                    df[y+str(i+1)] = ts[y][:,i]
                        df = df.sort_index(axis=1)
                        path = simname + '/RESULTS/RUN' + str(run) + '/' + opseq.TARGET + '_' + opseq.ID + '/' + x.Path.split('/')[1]
                        store.put(path, df.astype(np.float32))

                        for errorcnt, errormsg in zip(errs, errstrs):  # print returned error messages and counts
                            if errorcnt > 0:
                                msg(3, 'Message count ' + str(errorcnt) +  '   Message ' + errormsg)

        msg(1, 'Run completed')
        store.put('/RUN_INFO/VERSIONS', versions(), format='t', data_columns=True)

    # Copy runtime log file to the HDF5 file as record of run
    with h5py.File(hdfname, 'a') as hdf, open(logpath, 'r') as f:
        if '/RUN_INFO/log' in hdf:
            del hdf['/RUN_INFO/log']
        data = f.read()
        ds = hdf['/RUN_INFO/'].create_dataset('log', shape=(1,), dtype='S' + str(len(data)))
        ds[:] = data
    return      # RUN is DONE


def messages(fname):
    '''Closure routine; msg() prints messages to screen and run log'''
    def msg(indent, message):
        m = str(dt.now())[:22] + '   ' * indent + message
        print(m)
        fname.write(m)
    return msg


def get_ucs(keys, store, msg):
    ''' create the UCI dictionary: ucis['PERLND','PWATFG'].loc['P001',:].to_dict()'''
    ucs = {}
    for x in ['PERLND', 'IMPLND', 'RCHRES']:
        for indx, row in store[x + '/GENERAL_INFO'].iterrows():
            ucs[x, 'GENERAL_INFO', indx] = row.to_dict()
        for indx, row in store[x + '/ACTIVITY'].iterrows():
            ucs[x, 'ACTIVITY', indx] = row

    getflag = {}
    for x in store['HSP2/CONFIGURATION'].itertuples():
        getflag[x.Path[:-1] if x.Path.endswith('/') else x.Path] = x.Flag
    data = defaultdict(list)
    for key in keys:
        tokens = key.split('/')
        if (tokens[1] in ['PERLND', 'IMPLND', 'RCHRES'] and 'MONTHLY' not in key
         and 'ACTIVITY' not in key and 'SAVE' not in key and 'GENERAL_INFO' not in key):
            indx = tokens[1] + '/' + tokens[2]
            data[tokens[1], getflag[indx]].append(key)

    for x in data:
        temp = pd.concat([store[path] for path in data[x]], axis=1)
        tempnames = temp.columns
        if x[0]=='RCHRES' and x[1]=='HYDRFG':
            for var in ['COLIN', 'OUTDG']:
                names = [name for name in tempnames if var in name]
                temp[var] = temp.apply(lambda x: tuple([x[name] for name in names]),axis=1)
                for name in names:
                    del temp[name]
            for var in ['FUNCT', 'ODGTF', 'ODFVF']:
                names = [name for name in tempnames if var in name]
                for name in names:
                    temp[name] = temp[name].astype(int)
                temp[var] = temp.apply(lambda x: tuple([x[name] for name in names]),axis=1)
                for name in names:
                    del temp[name]
        for indx,row in temp.iterrows():
            ucs[x[0], x[1], indx] = row

        tokens = data[x][0].split('/')
        for indx, row in store[tokens[1] + '/' + tokens[2] + '/' + 'SAVE'].iterrows():
            ucs[x[0], x[1], 'SAVE', indx] = row
    return ucs


def get_timeseries(tsdd, tindex, store):
    ''' gets timeseries at the current timestep and trucated to the sim interval'''
    ts = {}
    if not tsdd:
        return ts

    for row in tsdd:
        path = 'TIMESERIES/' + row.SVOLNO
        temp = store[path] if row.SVOL == '*' else pd.read_hdf(row.SVOL, path)
        tran = 'SAME' if not row.TRAN else row.TRAN
        if type(row.MFACTOR) == str and row.MFACTOR[0] == '*':
            mfactor = store['TIMERSERIES/' + row.MFACTOR[1:]]
            mfactor = transform(mfactor, tindex, 'SAME')
        else:
            mfactor = float(row.MFACTOR)
        temp = transform(temp, tindex, tran) * mfactor

        if row.TMEMSB == '':
            if row.TMEMN in ts:
                ts[row.TMEMN] += temp.values.astype(float)
            else:
                ts[row.TMEMN]  = temp.values.astype(float)
        else:
            tmp = row.TMEMSB.split()
            if len(tmp) == 1:
                tmemsb = '' if int(tmp[0]) == 1 else str(int(tmp[0])-1)
                if row.TMEMN + tmemsb in ts:
                    ts[row.TMEMN + tmemsb] += temp.values.astype(float)
                else:
                    ts[row.TMEMN + tmemsb]  = temp.values.astype(float)
            else:
                for i in range(int(tmp[0])-1, int(tmp[1])):
                    tmemsb = '' if i==0 else str(i)
                    if row.TMEMN + tmemsb in ts:
                        ts[row.TMEMN + tmemsb] += temp.values.astype(float)
                    else:
                        ts[row.TMEMN + tmemsb]  = temp.values.astype(float)
    return ts


def get_trans(fname):
    fn = fname.to_upper()
    if   'MONTHLY12' in fn: return 'MONTHLY12'
    elif 'HOURLY24'  in fn: return 'HOURLY24'
    elif 'SPARSE'    in fn: return 'SPARSE'
    else:                   return 'SAME'   # default


def transform(ts, tindex, how):
    if len(ts)==len(tindex) and ts.index[0]==tindex[0] and ts.index.freq==tindex.freq:
        pass

    elif how == 'SAME' and 'M' in ts.index.freqstr:  # possible Pandas bug
        ts = ts.reindex(tindex, method='bfill')

    elif  how in ['SAME', 'LAST']:
        if ts.index.freq > tindex.freq:
            ts = ts.reindex(tindex, method='PAD')
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).last()

    elif  how in ['MEAN']:
        if ts.index.freq > tindex.freq:
            ts = ts.reindex(tindex, method='PAD')
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).mean()

    elif how in ['DIV', 'SUM']:
        if ts.index.freq > tindex.freq:
            ratio = float(tindex.freq.nanos) / float(ts.index.freq.nanos)
            ts = ts.reindex(tindex, method='PAD') * ratio
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).sum()

    elif how in ['MONTHLY12', 'DAYVAL'] and len(ts) == 12:
        start = pd.to_datetime('01/01/' + str(tindex[0].year-1))
        stop  = pd.to_datetime('12/31/' + str(tindex[-1].year+1))
        tempindex = pd.date_range(start, stop, freq='MS')
        tiled = np.tile(ts, len(tempindex)/12)
        if how == 'DAYVAL':  # HSPF "interpolation"  interp to day, pad fill the day
            daily = pd.Series(tiled,index=tempindex).resample('D')
            ts = daily.interpolate(method='time').resample(tindex.freqstr).pad()
        else:
            ts = pd.Series(tiled, index=tempindex).resample(tindex.freqstr).pad()

    elif how in ['HOURLY24', 'LAPSE'] and  len(ts) == 24:
        start = pd.to_datetime('01/01/' + str(tindex[0].year-1))
        stop  = pd.to_datetime('12/31/' + str(tindex[-1].year+1) + ' 23:59')
        tempindex = pd.date_range(start, stop, freq='H')
        tile = np.tile(ts, len(tempindex)/24)
        ts = pd.Series(tile, index=tempindex)
        if tindex.freq > tempindex.freq:
            ts = ts.resample(tindex.freqstr).mean()
        elif tindex.freq < tempindex.freq:
            ts = ts.reindex(ts, method='PAD')

    elif how == 'SPARSE':
        x = pd.Series(np.NaN, tindex)
        for indx,value in ts.iteritems():
            iloc = tindex.get_loc(indx, method='nearest')
            x[x.index[iloc]] = value
            ts = x.fillna(method='pad')

    else:
        print('UNKNOWN AGG/DISAGG METHOD: ' + how)

    ts = ts[tindex[0]: tindex[-1]] # truncate to simulation [start,stop]
    if len(ts) != len(tindex):  # shouldn't happen - debug
        print(' '.join(['LENGTH mismatch', len(ts), len(tindex)]))
    return ts


def flows(ts, tvol, tvolno, flag, flowdata, tindex, store, simname, run):
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
        path = simname + '/RESULTS/RUN' + str(run) + '/' + x['SVOL'] + '_' + x['SVOLNO'] + '/' + x['SGRPN']

        print('SIMPATH is ' + path)


        if path in store:
            t = store[path][x['SMEMN'] + x['SMEMSB']]
        else:
            path = '/RESULTS/' + x['SVOL'] + '_' + x['SVOLNO'] + '/' + x['SGRPN']

            print('NEED NON SIMPATH DATA ' + path)


            t = store[path][x['SMEMN'] + x['SMEMSB']]

        if type(x['AFACTR']) == str and x['AFACTR'][0] == '*':
            afactr = store['TIMESERIES/' + x['AFACTR'][1:]]
            afactr = transform(afactr, tindex, 'SAME')
        else:
            afactr = float(x['AFACTR'])

        if type(x['MFACTOR']) == str and x['MFACTOR'][0] == '*':
            mfactor = store['TIMERSERIES/' + x['MFACTOR'][1:]]
            mfactor = transform(mfactor, tindex, 'SAME')
        else:
            mfactor = float(x['MFACTOR'])

        t = transform(t, tindex, 'SAME') * mfactor * afactr
        if x['TMEMN'] in ts:
            ts[x['TMEMN']] += t.values
        else:
            ts[x['TMEMN']]  = t.values
    return


def versions():
    ''' Returns the versions of the Python and HSP2 library modules in a DataFrame'''
    import sys
    import platform    # not used, but reports on current CPU and operating system
    import numba       # not used by main but used by other HSP2 functions
    import tables      # used internally by Pandas

    packages = {'HSP2': HSP2.__version__, 'PYTHON': sys.version.replace('\n', ''),
     'NUMBA': numba.__version__, 'NUMPY': np.__version__, 'PANDAS': pd.__version__,
     'H5PY': h5py.__version__, 'PYTABLES': tables.__version__, 'NETWORKX': nx.__version__,
     'os': platform.platform(), 'processor': platform.processor()}

    cols = ['HSP2','PYTHON','NUMBA','NUMPY','PANDAS','H5PY','PYTABLES','NETWORKX', 'os','processor']
    return pd.DataFrame(packages, index=['Version'], columns=cols).T.astype(str)


def initm(general, ui, ts, flag, monthly, name):
    ''' initialize timeseries with HSPF interpolation of monthly array or with fixed value'''
    if ui[flag] and monthly in ui:
        ts[name] = transform(ui[monthly], general['tindex'], 'DAYVAL').values
    else:
        ts[name] = np.full(general['sim_len'], ui[name])
