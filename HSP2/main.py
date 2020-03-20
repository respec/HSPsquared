from numpy import zeros, float64
from pandas import HDFStore, Timestamp, Timedelta, read_hdf, DataFrame, date_range
from pandas.tseries.offsets import Minute
from numba import types
from numba.typed import Dict
from collections import defaultdict
from importlib import import_module
from datetime import datetime as dt
import os
from utilities import transform

def noop (store, siminfo, ui, ts):
    ERRMSGS = []
    errors = zeros(len(ERRMSGS), dtype=int)
    return errors, ERRMSGS


# Note: activities define execution order. Default preset to noop(). Users may modify this
activities = {
  'PERLND': {'ATEMP':noop, 'SNOW':noop, 'PWATER':noop, 'SEDMNT':noop,
     'PSTEMP':noop, 'PWTGAS':noop, 'PQUAL':noop, 'MSTLAY':noop, 'PEST':noop,
     'NITR':noop, 'PHOS':noop, 'TRACER':noop},
  'IMPLND': {'ATEMP':noop, 'SNOW':noop, 'IWATER':noop, 'SOLIDS':noop,
     'IWTGAS':noop, 'IQUAL':noop},
  'RCHRES': {'HYDR':noop, 'ADCALC':noop, 'CONS':noop, 'HTRCH':noop,
     'SEDTRN':noop, 'GQUAL':noop, 'OXRX':noop, 'NUTRX':noop, 'PLANK':noop,
     'PHCARB':noop}}


def main(hdfname, saveall=False):          # primary HSP2 highest level routine
    if not os.path.exists(hdfname):
        print(f'{hdfname} HDF5 File Not Found, QUITTING')
        return

    # Dynamically imports HSP2 functions.  Function name is lower case of
    # file name: snow() is main function of module SNOW.py, etc.
    s = set(os.listdir('.'))
    for op in ('PERLND', 'IMPLND', 'RCHRES'):
        for key in activities[op]:
            if f'{key}.py' in s:
                activities[op][key] = getattr(import_module(key), key.lower())

    uic = defaultdict(dict)
    siminfo = {}
    logpath = os.path.join(os.path.dirname(hdfname), 'logfile.txt')
    with HDFStore(hdfname) as store, open(logpath, 'w') as logfile:
        msg = messages(logfile)
        msg(1, f'Run Started for file {hdfname}')

        # read user control and user data from HDF5 file
        opseq, ddlinks, ddmasslinks, ext_sourcesdd = get_uc(store,uic,siminfo)

        start = siminfo['start']
        stop  = siminfo['stop']
        msg(1, f'Simulation Start {start},  Simulation Stop {stop}')

        # main processing loop
        for _, operation, segment, delt in opseq.itertuples():
            msg(2, f'{operation} {segment} DELT(minutes) = {delt}')

            siminfo['operation'] = operation
            siminfo['segment']   = segment
            siminfo['delt']      = delt
            siminfo['freq']      = Minute(int(delt))
            siminfo['steps']     = int((stop-start)/Timedelta('1H')) + 1

            # explicit creation of Numba dictionary with signatures
            ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
            get_timeseries(store, ts, ext_sourcesdd[(operation,segment)], siminfo)

            # now conditionally execute all activity modules for the op, segment
            flags = uic[(operation, 'GENERAL')]['ACTIVITY'][segment]
            for activity, function in activities[operation].items():
                if function == noop:
                    continue

                if flags[activity]:
                    msg(3, f'{activity}')

                    siminfo['activity'] = activity
                    ui = uic[(operation, activity)]
                    if operation == 'RCHRES':
                        get_flows(store,ts,activity,segment,ddlinks,ddmasslinks)

                    ###########################################################
                    # calls activity function, like iwater()
                    errors, errmessages = function(store, siminfo, ui, ts)
                    ###########################################################

                    for errorcnt, errormsg in zip(errors, errmessages):  # print returned error messages and counts
                        if errorcnt > 0:
                            msg(4, f'Error count {errorcnt}: {errormsg}')

                    save_timeseries(store,ts,ui['SAVE'][segment],siminfo,saveall)
        msg(1, 'Run completed')
    return


def messages(fname):
    '''Closure routine; msg() prints messages to screen and run log'''
    def msg(indent, message):
        m = str(dt.now())[:22] + '   ' * indent + message
        print(m)
        fname.write(m)
    return msg


def get_uc(store, uic, siminfo):
    # read user control and user data from HDF5 file
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
                ext_sourcesdd = defaultdict(list)
                for row in store[path].itertuples():
                    ext_sourcesdd[(row.TVOL, row.TVOLNO)].append(row)
            elif module == 'OP_SEQUENCE':
                opseq = store[path]
        elif op in {'PERLND', 'IMPLND', 'RCHRES'}:
            uic[(op, module)][s] = store[path].to_dict('index')
    return opseq, ddlinks, ddmasslinks, ext_sourcesdd


def get_timeseries(store, ts, ext_sourcesdd, siminfo):
    ''' makes timeseries for the current timestep and trucated to the sim interval'''
    if not ext_sourcesdd:
        return

    for row in ext_sourcesdd:
        path = f'TIMESERIES/{row.SVOLNO}'
        temp1 = store[path] if row.SVOL == '*' else read_hdf(row.SVOL, path)

        if row.MFACTOR != 1.0:
            temp1 *= row.MFACTOR

        temp = transform(temp1, row.TRAN, siminfo).to_numpy().astype(float)
        if len(temp) > siminfo['steps']:
            temp = temp[0:siminfo['steps']]

        if row.TMEMSB == '':
            if row.TMEMN in ts:
                ts[row.TMEMN] += temp
            else:
                ts[row.TMEMN]  = temp
        else:
            tmp = row.TMEMSB.split()
            if len(tmp) == 1:
                tmemsb = '' if int(tmp[0]) == 1 else str(int(tmp[0])-1)
                if row.TMEMN + tmemsb in ts:
                    ts[row.TMEMN + tmemsb] += temp
                else:
                    ts[row.TMEMN + tmemsb]  = temp
            else:
                for i in range(int(tmp[0])-1, int(tmp[1])):
                    tmemsb = '' if i==0 else str(i)
                    if row.TMEMN + tmemsb in ts:
                        ts[row.TMEMN + tmemsb] += temp
                    else:
                        ts[row.TMEMN + tmemsb]  = temp
    return


def save_timeseries(store, ts, savedict, siminfo, saveall):
    # save computed timeseries (at computation DELT)
    #save = set(savedict.keys()) if saveall else {k for k,v in savedict.items() if v}
    save = {k for k,v in savedict.items() if v or saveall}

    tindex = date_range(siminfo['start'],siminfo['stop'],freq=siminfo['freq'])
    df = DataFrame(index=tindex)
    for y in (save & set(ts.keys())):
        df[y] = ts[y]
    df = df.astype('float32').sort_index(axis='columns')

    path = f"/RESULTS/{siminfo['operation']}_{siminfo['segment']}/{siminfo['activity']}"
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