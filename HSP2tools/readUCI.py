'''
Copyright 2020 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
'''

from collections import defaultdict
from pandas import Series, DataFrame, concat, HDFStore, set_option, to_numeric
from pandas import Timestamp, Timedelta, read_hdf, read_csv
set_option('io.hdf.default_format', 'table')

import os.path
import HSP2tools

Lapse = Series ([0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0037,
 0.0040, 0.0041, 0.0043, 0.0046, 0.0047, 0.0048, 0.0049, 0.0050, 0.0050,
 0.0048, 0.0046, 0.0044, 0.0042, 0.0040, 0.0038, 0.0037, 0.0036])

Seasons = Series ([0,0,0, 1,1,1,1,1,1, 0,0,0]).astype(bool)

Svp = Series([1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005,
 1.005, 1.01, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.1, 1.29, 1.66,
 2.13, 2.74,3.49, 4.40, 5.55,6.87, 8.36, 10.1,12.2,14.6, 17.5, 20.9, 24.8,
 29.3, 34.6, 40.7, 47.7, 55.7, 64.9])


def reader(filename):
    # simple reader to return non blank, non comment and proper length lines
    with open(filename, 'r') as file:
        for line in file:
            if '***' in line or not line.strip():
                continue
            yield f'{line.rstrip(): <90}'        # prevent short line problems


def getlines(f):
    lines = []
    for line in f:
        if line[0:3] == 'END' :
            break
        lines.append(line)
    return lines


convert = {'C':str, 'I':int, 'R':float}
def parseD(line, parse):
    d = {}
    for name, type_, start, end, default in parse:
        field = line[start:end].strip()
        d[name] = convert[type_](field) if field else convert[type_](default)
    return d

def parseD2(line, parse, d):
    icnt = 0
    for name, type_, start, end, default in parse:
        field = line[start:end].strip()
        icnt += 1
        # don't do anything with the first 7 values
        if icnt > 8:
            d[name] = convert[type_](field) if field else convert[type_](default)
    return d

def parseD3(line, parse, d):
    icnt = 0
    for name, type_, start, end, default in parse:
        field = line[start:end].strip()
        icnt += 1
        # don't do anything with the first 14 values
        if icnt > 15:
            d[name] = convert[type_](field) if field else convert[type_](default)
    return d


def get_opnid(opnidstr, operation):
    first, *last = opnidstr.split()
    b = int(last[0]) if last else int(first)
    a = int(first)
    for x in range(a, b+1):
        yield f'{operation[0]}{x:03d}'


def fix_df(df, op, save, ddfaults, valid):
    '''fix NANs and excess ids, missing indicies, bad names'''
    if set(df.index) - valid:
        df = df.drop(index = set(df.index) - valid) # drop unnecessary ids
    for name1 in valid - set(df.index):
        df = df.append(Series(name=name1))        # add missing ids with NaNs
    if df.isna().any().any():                      # replace NaNs with defaults
        for col in df.columns:
            df[col] = df[col].fillna(ddfaults[op, save, col])
    cols = [c.replace('(','').replace(')','') for c in df.columns]
    df.columns = cols
    df = df.apply(to_numeric, errors='ignore')
    return df


# Ignore these tables during processing, not used by HSP2
skip = {
 ('PERLND', 'PRINT-INFO'),
 ('PERLND', 'BINARY-INFO'),
 ('IMPLND', 'PRINT-INFO'),
 ('IMPLND', 'BINARY-INFO'),
 ('RCHRES', 'PRINT-INFO'),
 ('RCHRES', 'BINARY-INFO')}


ops = {'PERLND','IMPLND','RCHRES'}
conlike = {'CONS':'NCONS', 'PQUAL':'NQUAL', 'IQUAL':'NQUAL', 'GQUAL':'NQUAL'}
def readUCI(uciname, hdfname):
    # create lookup dictionaries from 'ParseTable.csv' and 'rename.csv'
    parse = defaultdict(list)
    defaults = {}
    cat = {}
    path = {}
    datapath = os.path.join(HSP2tools.__path__[0], 'data', 'ParseTable.csv')
    for row in read_csv(datapath).itertuples():
        parse[row.OP,row.TABLE].append((row.NAME, row.TYPE, row.START, row.STOP, row.DEFAULT))

        defaults[row.OP, row.SAVE, row.NAME] = convert[row.TYPE](row.DEFAULT)

        cat[row.OP,row.TABLE]  = row.CAT
        path[row.OP,row.TABLE] = row.SAVE
    rename = {}
    extendlen = {}
    datapath = os.path.join(HSP2tools.__path__[0], 'data', 'rename.csv')
    for row in read_csv(datapath).itertuples():
        if row.LENGTH != 1:
            extendlen[row.OPERATION,row.TABLE] = row.LENGTH
        rename[row.OPERATION,row.TABLE] = row.RENAME

    net = None; sc = None
    with HDFStore(hdfname, mode = 'a') as store:
        info = (store, parse, path, defaults, cat, rename, extendlen)

        f = reader(uciname)
        for line in f:
            if line[0:6] == 'GLOBAL':       global_(info, getlines(f))
            if line[0:3] == 'OPN':              opn(info, getlines(f))
            if line[0:7] == 'NETWORK':  net=network(info, getlines(f))
            if line[0:9] == 'SCHEMATIC':sc=schematic(info,getlines(f))
            if line[0:9] == 'MASS-LINK':   masslink(info, getlines(f))
            if line[0:7] == 'FTABLES':      ftables(info, getlines(f))
            if line[0:3] == 'EXT':              ext(info, getlines(f))
            if line[0:6] == 'PERLND':     operation(info, getlines(f),'PERLND')
            if line[0:6] == 'IMPLND':     operation(info, getlines(f),'IMPLND')
            if line[0:6] == 'RCHRES':     operation(info, getlines(f),'RCHRES')

        colnames = ('AFACTR', 'MFACTOR', 'MLNO', 'SGRPN', 'SMEMN', 'SMEMSB',
         'SVOL', 'SVOLNO', 'TGRPN', 'TMEMN', 'TMEMSB', 'TRAN', 'TVOL',
         'TVOLNO', 'COMMENTS')
        linkage = concat((net, sc), ignore_index=True, sort=True)
        for cname in colnames:
            if cname not in linkage.columns:
                linkage[cname] = ''
        linkage = linkage.sort_values(by=['TVOLNO']).replace('na','')
        linkage.to_hdf(store, '/CONTROL/LINKS', data_columns=True)

        Lapse.to_hdf(store, 'TIMESERIES/LAPSE_Table')
        Seasons.to_hdf(store, 'TIMESERIES/SEASONS_Table')
        Svp.to_hdf(store, 'TIMESERIES/Saturated_Vapor_Pressure_Table')

        keys = set(store.keys())
        # rename needed for restart. NOTE issue with line 157 in PERLND SNOW HSPF
        # where PKSNOW = PKSNOW + PKICE at start - ONLY
        path = '/PERLND/SNOW/STATES'
        if path in keys:
            df = read_hdf(store, path)
            df=df.rename(columns={'PKSNOW':'PACKF','PKICE':'PACKI','PKWATR':'PACKW'})
            df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/SNOW/STATES'
        if path in keys:
            df = read_hdf(store, path)
            df=df.rename(columns={'PKSNOW':'PACKF','PKICE':'PACKI','PKWATR':'PACKW'})
            df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/SNOW/FLAGS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SNOPFG' not in df.columns:   # didn't read SNOW-FLAGS table
                df['SNOPFG']  = 0
                df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/SNOW/FLAGS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SNOPFG' not in df.columns:   # didn't read SNOW-FLAGS table
                df['SNOPFG']  = 0
                df.to_hdf(store, path, data_columns=True)

        # Need to fixup missing data
        path = '/IMPLND/IWATER/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'PETMIN' not in df.columns:   # didn't read IWAT-PARM2 table
                df['PETMIN'] = 0.35
                df['PETMAX'] = 40.0
                df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/IWTGAS/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SDLFAC' not in df.columns:   # didn't read LAT-FACTOR table
                df['SDLFAC'] = 0.0
                df['SLIFAC'] = 0.0
                df.to_hdf(store, path, data_columns=True)
            if 'SOTMP' not in df.columns:  # didn't read IWT-INIT table
                df['SOTMP'] = 60.0
                df['SODOX'] = 0.0
                df['SOCO2'] = 0.0
                df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/IQUAL/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SDLFAC' not in df.columns:   # didn't read LAT-FACTOR table
                df['SDLFAC'] = 0.0
                df['SLIFAC'] = 0.0
                df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/PWTGAS/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SDLFAC' not in df.columns:  # didn't read LAT-FACTOR table
                df['SDLFAC'] = 0.0
                df['SLIFAC'] = 0.0
                df['ILIFAC'] = 0.0
                df['ALIFAC'] = 0.0
                df.to_hdf(store, path, data_columns=True)
            if 'SOTMP' not in df.columns:  # didn't read PWT-TEMPS table
                df['SOTMP'] = 60.0
                df['IOTMP'] = 60.0
                df['AOTMP'] = 60.0
                df.to_hdf(store, path, data_columns=True)
            if 'SODOX' not in df.columns:  # didn't read PWT-GASES table
                df['SODOX'] = 0.0
                df['SOCO2'] = 0.0
                df['IODOX'] = 0.0
                df['IOCO2'] = 0.0
                df['AODOX'] = 0.0
                df['AOCO2'] = 0.0
                df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/PWATER/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'FZG' not in df.columns:   # didn't read PWAT-PARM5 table
                df['FZG']  = 1.0
                df['FZGL'] = 0.1
                df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/PQUAL/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SDLFAC' not in df.columns:  # didn't read LAT-FACTOR table
                df['SDLFAC'] = 0.0
                df['SLIFAC'] = 0.0
                df['ILIFAC'] = 0.0
                df['ALIFAC'] = 0.0
                df.to_hdf(store, path, data_columns=True)

        path = '/RCHRES/GENERAL/INFO'
        if path in keys:
            dfinfo = read_hdf(store, path)
            path = '/RCHRES/HYDR/PARAMETERS'
            if path in keys:
                df = read_hdf(store, path)
                df['NEXITS'] = dfinfo['NEXITS']
                df['LKFG']   = dfinfo['LKFG']
                if 'IREXIT' not in df.columns:   # didn't read HYDR-IRRIG table
                    df['IREXIT'] = 0
                    df['IRMINV'] = 0.0
                df['FTBUCI'] = df['FTBUCI'].map(lambda x: f'FT{int(x):03d}')
                df.to_hdf(store, path, data_columns=True)
            del dfinfo['NEXITS']
            del dfinfo['LKFG']
            dfinfo.to_hdf(store, 'RCHRES/GENERAL/INFO', data_columns=True)

        path = '/RCHRES/HTRCH/FLAGS'
        if path in keys:
            df = read_hdf(store, path)
            if 'BEDFLG' not in df.columns:  # didn't read HT-BED-FLAGS table
                df['BEDFLG'] = 0
                df['TGFLG']  = 2
                df['TSTOP']  = 55
                df.to_hdf(store, path, data_columns=True)

        path = '/RCHRES/HTRCH/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'ELEV' not in df.columns:  # didn't read HEAT-PARM table
                df['ELEV']  = 0.0
                df['ELDAT'] = 0.0
                df['CFSAEX']= 1.0
                df['KATRAD']= 9.37
                df['KCOND'] = 6.12
                df['KEVAP'] = 2.24
                df.to_hdf(store, path, data_columns=True)

        path = '/RCHRES/HTRCH/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'ELEV' not in df.columns:  # didn't read HT-BED-PARM table
                df['MUDDEP']= 0.33
                df['TGRND'] = 59.0
                df['KMUD']  = 50.0
                df['KGRND'] = 1.4
                df.to_hdf(store, path, data_columns=True)

        path = '/RCHRES/HTRCH/STATES'
        if path in keys:
            df = read_hdf(store, path)
            if 'ELEV' not in df.columns:  # didn't read HEAT-INIT table
                df['TW']    = 60.0
                df['AIRTMP']= 60.0
    return


def global_(info, lines):
    store, parse, path, *_ = info
    d = parseD(lines[1], parse['GLOBAL','START'])
    start = str(Timestamp(f"{d['SYR']}-{d['SMO']}-{d['SDA']}")
      + Timedelta(int(d['SHR']), 'h') + Timedelta(int(d['SMI']), 'T'))[0:16]
    stop  = str(Timestamp(f"{d['EYR']}-{d['EMO']}-{d['EDA']}")
      + Timedelta(int(d['EHR']), 'h') + Timedelta(int(d['EMI']), 'T'))[0:16]
    data = [lines[0].strip(), start, stop]
    dfglobal = DataFrame(data, index=['Comment','Start','Stop'],columns=['Info'])
    dfglobal.to_hdf(store, '/CONTROL/GLOBAL', data_columns=True)


def opn(info, lines):
    store, parse, path, *_ = info
    lst = []
    for line in lines:
        tokens = line.split()
        if tokens[0] == 'INGRP' and tokens[1] == 'INDELT':
            s = tokens[2].split(':')
            indelt = int(s[0]) if len(s) == 1 else 60 * int(s[0]) + int(s[1])
        elif tokens[0] in ops:
            s = f'{tokens[0][0]}{int(tokens[1]):03d}'
            lst.append((tokens[0], s, indelt))
    dfopn = DataFrame(lst, columns = ['OPERATION', 'SEGMENT', 'INDELT_minutes'])
    dfopn.to_hdf(store, '/CONTROL/OP_SEQUENCE', data_columns=True)


def network(info, lines):
    store, parse, path, *_ = info
    lst = []
    for line in lines:
        d = parseD(line, parse['NETWORK','na'])
        if d['SVOL'] in ops and d['TVOL'] in ops:
            d['SVOLNO'] = f"{d['SVOL'][0]}{int(d['SVOLNO']):03d}"
            if 'TVOLNO' in d:
                d['TVOLNO'] = f"{d['TVOL'][0]}{int(d['TVOLNO']):03d}"
            elif 'TOPFST' in d:
                d['TOPFST'] = f"{d['TVOL'][0]}{int(d['TOPFST']):03d}"
            lst.append(d)
    return DataFrame(lst, columns=d) if lst else DataFrame()


def schematic(info, lines):
    store, parse, path, *_ = info
    lst = []
    for line in lines:
        d = parseD(line, parse['SCHEMATIC','na'])
        if d['SVOL'] in ops and d['TVOL'] in ops:
            d['MLNO']   = f"ML{int(d['MLNO']):03d}"
            d['SVOLNO'] = f"{d['SVOL'][0]}{int(d['SVOLNO']):03d}"
            d['TVOLNO'] = f"{d['TVOL'][0]}{int(d['TVOLNO']):03d}"
            lst.append(d)
    return DataFrame(lst, columns=d) if lst else DataFrame()


def masslink(info, lines):
    store, parse, path, *_ = info
    lst = []
    for line in lines:
        if line[2:11] == 'MASS-LINK':
            name = line[12:].rstrip()
        elif line[2:5] != 'END':
            d = parseD(line, parse['MASS-LINK','na'])
            d['MLNO'] = f'ML{int(name):03d}'
            lst.append(d)
    if lst:
        dfmasslink = DataFrame(lst, columns=d).replace('na','')
        del dfmasslink['TGRPN']
        dfmasslink['COMMENTS'] = ''
        dfmasslink.to_hdf(store, '/CONTROL/MASS_LINKS', data_columns=True)


def ftables(info, llines):
    store, parse, path, *_ = info
    header=['Depth','Area','Volume','Disch1','Disch2','Disch3','Disch4','Disch5']
    lines = iter(llines)
    for line in lines:
        if line[2:8] == 'FTABLE':
            unit = int(line[8:])
            name = f'FT{unit:03d}'
            rows,cols = next(lines).split()
            lst = []
        elif line[2:5] == 'END':
            dfftable = DataFrame(lst, columns=header[0:int(cols)])
            dfftable.to_hdf(store, f'/FTABLES/{name}', data_columns=True)
        else:
            lst.append(parseD(line, parse['FTABLES','FTABLE']))


def ext(info, lines):
    store, parse, path, *_ = info
    lst = []
    lst_cols = {}
    for line in lines:
        d = parseD(line, parse['EXT SOURCES','na'])
        if d['TVOL'] in ops:
            d['SVOLNO'] = f"TS{int(d['SVOLNO']):03d}"
            d['SVOL'] = '*'
            if d['TGRPN'] == 'EXTNL':
                d['TGRPN'] = ''
            toplst = int(d['TOPFST']) if d['TOPLST'] == 'na' else int(d['TOPLST'])
            for i in range(int(d['TOPFST']), toplst + 1):
                d['TVOLNO'] = f"{d['TVOL'][0]}{i:03d}"
                lst.append(d.copy())
                lst_cols = d

    if lst:
        dfext = DataFrame(lst, columns = lst_cols).replace('na','')
        dfext['COMMENT'] = ''
        del dfext['TOPFST']
        del dfext['TOPLST']
        dfext = dfext.sort_values(by=['TVOLNO'])
        dfext.to_hdf(store, '/CONTROL/EXT_SOURCES', data_columns=True)


Months=('JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC')
def operation(info, llines, op):
    store, parse, dpath, ddfaults, dcat, rename, extendlen = info
    counter = set()

    history = defaultdict(list)
    lines = iter(llines)
    for line in lines:
        tokens = line.split()
        if len(tokens) == 1:
            table = tokens[0]
            if dcat[op,table] == 'EXTENDED':
                rows = {}
                extended_line = 0
                for line in lines:
                    extended_line += 1
                    if (op, table) not in parse or line[2:5] == 'END':
                        break
                    if extended_line == 1:
                        d = parseD(line, parse[op, table])
                    elif extended_line == 2:
                        d = parseD2(line, parse[op, table], d)
                    elif extended_line == 3:
                        d = parseD3(line, parse[op, table], d)
                        for opnid in get_opnid(d.pop('OPNID'), op):
                            rows[opnid] = d
            else:
                rows = {}
                for line in lines:
                    if (op,table) not in parse or line[2:5] == 'END':
                        break
                    d = parseD(line, parse[op,table])
                    for opnid in get_opnid(d.pop('OPNID'), op):
                        rows[opnid] = d
            df = DataFrame.from_dict(rows, orient='index')
            history[dpath[op,table],dcat[op,table]].append((table,df))

    if len(history['GENERAL','INFO']) > 0:
        (_,df) = history['GENERAL','INFO'][0]
        valid = set(df.index)
        for path,cat in history:
            counter.add(path)
            if cat == 'SKIP':
                continue
            if cat in {'PARAMETERS', 'STATES', 'FLAGS', 'ACTIVITY','INFO'}:
                df = concat([temp[1] for temp in history[path,cat]], axis='columns')
                df = fix_df(df, op, path, ddfaults, valid)
                if cat == 'ACTIVITY' and op == 'PERLND':
                    df = df.rename(columns = {'AIRTFG':'ATEMP', 'SNOWFG':'SNOW',
                      'PWATFG':'PWATER', 'SEDFG':'SEDMNT', 'PSTFG':'PSTEMP',
                      'PWGFG':'PWTGAS', 'PQALFG':'PQUAL','MSTLFG':'MSTLAY',
                      'PESTFG':'PEST', 'NITRFG':'NITR', 'PHOSFG':'PHOS',
                      'TRACFG':'TRACER'})
                if cat ==  'ACTIVITY' and op == 'IMPLND':
                    df = df.rename(columns = {'ATMPFG':'ATEMP', 'SNOWFG':'SNOW',
                      'IWATFG':'IWATER', 'SLDFG':'SOLIDS', 'IWGFG':'IWTGAS',
                      'IQALFG':'IQUAL'})
                if cat == 'ACTIVITY' and op == 'RCHRES':
                    df = df.rename(columns  = {'HYDRFG':'HYDR', 'ADFG':'ADCALC',
                      'CONSFG':'CONS', 'HTFG':'HTRCH', 'SEDFG':'SEDTRN',
                      'GQALFG':'GQUAL', 'OXFG':'OXRX', 'NUTFG':'NUTRX',
                      'PLKFG':'PLANK', 'PHFG':'PHCARB'})
                df.to_hdf(store, f'{op}/{path}/{cat}', data_columns=True)
            elif cat == 'MONTHLYS':
                for (table,df) in history[path,cat]:
                    df = fix_df(df, op, path, ddfaults, valid)
                    df.columns = Months
                    name = rename[(op, table)]
                    df.to_hdf(store, f'{op}/{path}/MONTHLY/{name}', data_columns=True)
            elif cat == 'EXTENDED':
                temp = defaultdict(list)
                for table,df in history[path,cat]:
                    temp[table].append(df)
                for table,lst in temp.items():
                    df = concat(lst, axis='columns')
                    length = extendlen[op,table]
                    name = rename[op,table]
                    df.columns = [name+str(i) for i in range(len(df.columns))]
                    df = df[df.columns[0:length]]
                    df = fix_df(df, op, path, ddfaults, valid)
                    df.to_hdf(store, f'{op}/{path}/EXTENDEDS/{name}', data_columns=True)
            elif cat == 'SILTCLAY':
                table,df = history[path,cat][0]
                df = fix_df(df, op, path, ddfaults, valid)
                df.to_hdf(store, f'{op}/{path}/SILT', data_columns=True)
                table,df = history[path,cat][1]
                df = fix_df(df, op, path, ddfaults, valid)
                df.to_hdf(store, f'{op}/{path}/CLAY', data_columns=True)
            elif cat == 'CONS':
                count = 0
                for table,df in history[path,cat]:
                    if table == 'NCONS':
                        temp_path = '/RCHRES/CONS/PARAMETERS'
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.to_hdf(store, temp_path, data_columns=True)
                    elif table == 'CONS-DATA':
                        count += 1
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.to_hdf(store, f'{op}/{path}/{cat}{count}', data_columns=True)
            elif cat == 'PQUAL' or cat == 'IQUAL':
                count = 0
                for table,df in history[path,cat]:
                    if table == 'NQUALS':
                        if cat == 'IQUAL':
                            temp_path = '/IMPLND/IQUAL/PARAMETERS'
                        else:
                            temp_path = '/PERLND/PQUAL/PARAMETERS'
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.to_hdf(store, temp_path, data_columns=True)
                    elif table.startswith('MON'):
                        name = rename[(op, table)]
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.columns = Months
                        df.to_hdf(store, f'{op}/{path}/{cat}{count}/MONTHLY/{name}', data_columns=True)
                    else:
                        if table == 'QUAL-PROPS':
                            count += 1
                            tag = 'FLAGS'
                        else:
                            tag = 'PARAMETERS'
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.to_hdf(store, f'{op}/{path}/{cat}{count}/{tag}', data_columns=True)
            elif cat == 'GQUAL':
                count = 0
                for table,df in history[path,cat]:
                    if table.startswith('MON'):
                        name = rename[(op, table)]
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.columns = Months
                        df.to_hdf(store, f'{op}/{path}/{cat}{count}/MONTHLY/{name}', data_columns=True)
                    else:
                        if table == 'GQ-QALDATA':
                            count += 1
                        df = concat([temp[1] for temp in history[path, cat]], axis='columns')
                        df = fix_df(df, op, path, ddfaults, valid)
                        df.to_hdf(store, f'{op}/{path}/{cat}{count}', data_columns=True)
            else:
                print('UCI TABLE is not understood (yet) by readUCI', op, cat)

    savetable = defaultdict(dict)
    datapath = os.path.join(HSP2tools.__path__[0], 'data', 'SaveTable.csv')
    for row in read_csv(datapath).itertuples():
        savetable[row.OPERATION, row.ACTIVITY][row.NAME] = row.VALUE
    for activity in counter - set(['GENERAL']):
        df = DataFrame(index=sorted(valid))
        for name,value in savetable[op,activity].items():
            df[name] = int(value)
        df.to_hdf(store, f'{op}/{activity}/SAVE', data_columns=True)
