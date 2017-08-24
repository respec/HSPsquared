''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


from collections import  defaultdict
import pandas as pd
import HSP2tools

cast = {'C':str, 'I':int, 'R':float}  # OLD HSPF values


def rd(filename):
    ''' filereader to return complete, non blank, non comment lines and tokens'''
    f = open(filename)
    def r():
        for line in f:
            if '***' in line or not line.strip():
                continue
            line2 = line.rstrip() + ' ' * (90 - len(line.rstrip())) # prevent short line problems
            tokens = line2.split()
            yield (line2, tokens, len(tokens))
        f.close()
    return r


dprefix = {'TS':'TS', 'P':'P', 'I':'I', 'R':'R', 'FT':'FT', 'C':'C'}

def readUCI(ucifile, hdffile, HSPF=False, metric=False, prefix=dprefix):
    ''' Reads UCI file and saves the data in an HSP2 HDF5 file
    CALL: readUCI(ucifile, hdffile, HSPF=False, metric=False)
       ucifile is the name of the HSPF UCI file
       hdffile is the name of the target HDF5 file, created if missing
       HSPF flag (if False imports only the data needed by HSP2, else imports all data)
       metric flag (if False imports only English unit data, ele imports metric data)'''

    h2name  = HSP2tools.__file__[:-12] + r'\HSP2Data\HSP2.h5'
    df = pd.read_hdf(h2name, 'FLOWEXPANSION')
    df.to_hdf(hdffile, '/HSP2/FLOWEXPANSION',  data_columns=True, format='t')

    df = pd.read_hdf(h2name, '/CONFIGURATION')
    df.to_hdf(hdffile, '/HSP2/CONFIGURATION', data_columns=True, format='t')

    ts = pd.read_hdf(h2name, '/LAPSE')
    ts.to_hdf(hdffile, '/HSP2/LAPSE24')

    ts = pd.read_hdf(h2name, '/SEASONS')
    ts.to_hdf(hdffile, '/HSP2/SEASONS12')

    svp = pd.read_hdf(h2name, '/SaturatedVaporPressureTable')
    svp.to_hdf(hdffile, '/HSP2/SaturatedVaporPressureTable')

    f = rd(ucifile)
    for line, tokens, numtokens in f():
        block = tokens[0]

        if   block == 'END' and tokens[1] == 'RUN':  break
        elif block == 'RUN':        continue
        elif block == 'GLOBAL':     read_global( f, hdffile, h2name)
        elif block == 'PERLND':     read_segment(f, hdffile, block, h2name, prefix, HSPF, metric)
        elif block == 'IMPLND':     read_segment(f, hdffile, block, h2name, prefix, HSPF, metric)
        elif block == 'RCHRES':     read_segment(f, hdffile, block, h2name, prefix, HSPF, metric)
        elif block == 'NETWORK':    read_table  (f, hdffile, block, '/CONTROL/NETWORK',   h2name, HSPF, prefix)
        elif block == 'SCHEMATIC':  read_table  (f, hdffile, block, '/CONTROL/SCHEMATIC', h2name, HSPF, prefix)
        elif block == 'EXT' and tokens[1] == 'SOURCES':
            read_table  (f, hdffile, 'EXT_SOURCES', '/CONTROL/EXT_SOURCES', h2name, HSPF, prefix)
        elif block == 'MASS-LINK':
            mllist = []
            for line, tokens, numtokens in f():
                if numtokens == 2 and tokens[0] == 'MASS-LINK':
                    df = read_table(f, hdffile, 'MASS_LINK', '', h2name, HSPF, prefix)
                    df['MLNO'] = tokens[1]
                    mllist.append(df)
                elif numtokens == 2 and tokens[0] == 'END':
                    df = pd.concat(mllist, ignore_index=True)
                    df['AUX_TS'] = ''
                    df['COMMENTS'] = ''
                    df.to_hdf(hdffile, '/CONTROL/MASS_LINK', data_columns=True, format='t')
                    break
        elif block == 'FTABLES':
            scan = pd.read_hdf(h2name, '/PARSEDATA/FTABLES').values
            for line, tokens, numtokens in f():
                pf = prefix['FT']
                if numtokens >= 2 and tokens[0] == 'END':
                    break
                elif numtokens == 2 and tokens[0] == 'FTABLE':
                    sid = '{:03n}'.format(int(tokens[1]))
                    read_ftable(f, hdffile, scan, '/FTABLES/' + pf + sid)
                elif tokens[0].startswith('FTABLE'):
                    sid = '{:03n}'.format(int(tokens[0][6:]))
                    read_ftable(f, hdffile, scan, '/FTABLES/' + pf + + sid)
        elif block == 'OPN':
            oplist = []
            for line, tokens, numtokens in f():
                if 'END' ==  tokens[0] and 'OPN' == tokens[1] and  'SEQUENCE'  == tokens[2]:
                    df = pd.DataFrame(oplist, columns=['TARGET', 'ID', 'DELT'])
                    df.to_hdf(hdffile, '/CONTROL/OP_SEQUENCE',  format='t', data_columns=True)
                    break
                if tokens[0] == 'INGRP' and tokens[1]=='INDELT':
                    s = [x.strip() for x in tokens[2].split(':')]
                    delt = int(s[0]) if len(s) == 1 else 60*int(s[0]) + int(s[1])
                elif not HSPF and tokens[0] in ['PERLND', 'RCHRES', 'IMPLND']:
                    if len(tokens) == 4:
                        s = [x.strip() for x in tokens[3].split(':')]
                        delt = int(s[0]) if len(s) == 1 else 60*int(s[0]) + int(s)
                    sid = prefix[tokens[0][0]] + '{:0>3s}'.format(tokens[1])
                    oplist.append([tokens[0], sid, str(delt)])
                elif HSPF:
                    if len(tokens) == 4:
                        s = [x.strip() for x in tokens[3].split(':')]
                        delt = int(s[0]) if len(s) == 1 else 60*int(s[0]) + int(s)
                    sid = tokens[0][0] + '{:0>3s}'.format(tokens[1])
                    oplist.append([tokens[0], sid, str(delt)])
        elif block == 'EXT':
            for line, tokens, numtokens in f():
                if tokens[0] == 'END':
                    break

    with pd.get_store(hdffile) as store:
        keys = store.keys()
    series = pd.Series(sorted(keys))
    series.to_hdf(hdffile, 'HSP2/KEYS')

    # SNOW has bad naming PKICE, PKSNOW, PKWATER rather than PACKI, PACKF, PACKW
    # this makes hot restart not work as expected.
    keyset = set(keys)
    key = '/PERLND/SNOW/STATE'
    if key in keyset:
        df = pd.read_hdf(hdffile, key)
        df.PKSNOW += df.PKICE   # line 157 in PERLND SNOW
        df = df.rename(columns={'PKICE':'PACKI', 'PKSNOW':'PACKF', 'PKWATR':'PACKW'})
        df.to_hdf(hdffile, key, format='t', data_columns=True)
    key = '/IMPLND/SNOW/STATE'
    if key in keyset:
        df = pd.read_hdf(hdffile, key)
        df.PKSNOW += df.PKICE   # line 157 in PERLND SNOW
        df = df.rename(columns={'PKICE':'PACKI', 'PKSNOW':'PACKF', 'PKWATR':'PACKW'})
        df.to_hdf(hdffile, key, format='t', data_columns=True)

    # Need to fixup missing data
    path = '/IMPLND/IWATER/PARAMETERS'
    if path in keys:
        df = pd.read_hdf(hdffile, path)
        if 'PETMIN' not in df.columns:   # didn't read IWAT-PARM2 table
            df['PETMIN'] = 0.35
            df['PETMAX'] = 40.0
            df.to_hdf(hdffile, path, format='t', data_columns=True)

    path = '/PERLND/PWATER/PARAMETERS'
    if path in keys:
        df = pd.read_hdf(hdffile,path)
        if 'FZG' not in df.columns:   # didn't read PWAT-PARM5 table
            df['FZG']  = 1.0
            df['FZGL'] = 0.1
            df.to_hdf(hdffile, path, format='t', data_columns=True)

    path = '/PERLND/SNOW/FLAGS'
    if path in keys:
        df = pd.read_hdf(hdffile, path)
        if 'SNOPFG' not in df.columns:   # didn't read IWAT-PARM2 table
            df['SNOPFG']  = 0
            df.to_hdf(hdffile, path, format='t', data_columns=True)

    path = '/IMPLND/SNOW/FLAGS'
    if path in keys:
        df = pd.read_hdf(hdffile, path)
        if 'SNOPFG' not in df.columns:   # didn't read IWAT-PARM2 table
            df['SNOPFG']  = 0
            df.to_hdf(hdffile, path, format='t', data_columns=True)

    path = '/RCHRES/HYDR/PARAMETERS'
    if path in keys:
        df = pd.read_hdf(hdffile, path)
        if 'IREXIT' not in df.columns:   # didn't read HYDR-IRRIG table
            df['IREXIT'] = 0
            df['IRMINV'] = 0.0
            df.to_hdf(hdffile, path, format='t', data_columns=True)

    # Need to combine Schematic and Network data into single array
    with pd.get_store(hdffile) as store:
        schematic = store['/CONTROL/SCHEMATIC'] if '/CONTROL/SCHEMATIC' in store else pd.DataFrame()
        network   = store['/CONTROL/NETWORK']   if '/CONTROL/NETWORK'   in store else pd.DataFrame()
        links = pd.concat([schematic, network], ignore_index=True)

        if '/CONTROL/SCHEMATIC' in store:
            del store['/CONTROL/SCHEMATIC']
        if '/CONTROL/NETWORK'   in store:
            del store['/CONTROL/NETWORK']

        cols = ['SVOL','SVOLNO','SMEMN','SMEMSB', 'SGRPN','MFACTOR','AFACTR',
                'TVOL','TVOLNO','TMEMN','TMEMSB', 'TGRPN', 'MLNO', 'AUX_TS', 'COMMENTS']
        for c in cols:
            if c not in links.columns:
                links[c] = ''

    links.to_hdf(hdffile, '/CONTROL/LINKS', format='t', data_columns=True)

    print('uciReader is Done')
    return


def read_global(f, hdfname, h2):
    ''' parses the UCI GLOBAL table data'''

    parse = pd.read_hdf(h2,'/PARSEDATA/GLOBAL')
    df = pd.DataFrame(columns=['Data'])
    for line, tokens, numtokens in f():
        if tokens[0] == 'END':
            df.to_hdf(hdfname, '/CONTROL/GLOBAL', data_columns=True, format='t')
            return
        elif tokens[0] == 'START':
            scan = parse[parse['type']=='START'].values
            d = {}
            for _, v,c,e,default in scan:
                value = line[c:e].strip()
                if value:
                    value = '{:0>2s}'.format(value)
                else:
                    value = default
                d[v] = value

            if d['EHR'] == '24':
                d['EHR'] = '23'
                d['EMI'] = '59'


            if d['SHR'] == '24':
                d['SHR'] = '23'
                d['SMI'] = '59'

            df.loc['sim_end',  'Data']=d['EYR']+'-'+d['EMO']+'-'+d['EDA']+ ' '+d['EHR']+':'+d['EMI']
            df.loc['sim_start','Data']=d['SYR']+'-'+d['SMO']+'-'+d['SDA']+ ' '+d['SHR']+':'+d['SMI']

        elif tokens[0] == 'RUN':
            continue
        elif tokens[0] == 'RESUME':
            scan = parse[parse['type']=='RESUME'].values
            for _, v,c,e,default in scan:
                value = line[c:e].strip()
                if v == 'EMFG':
                    df.loc['units', 'Data'] = value
        else:
            df.loc['info', 'Data'] = line.strip()
    return


def read_table(f, hdfname, name, group, h2, HSPF, prefix):
    ''' Parses NETWORK, SCHEMATIC, MASS_LINK UCI tables'''

    dfparse = pd.read_hdf(h2, '/PARSEDATA/' + name)
    indx = list(dfparse['Variable'].values)
    if (name=='EXT_SOURCES') or (name=='NETWORK'):
        indx = indx + ['TVOLNO']

    rowlist = []
    for line, tokens, numtokens in f():
        if tokens[0].strip() == 'END':
            df = pd.DataFrame(rowlist)
            if df.empty:
                return df

            if 'TOPFST' in df.columns:
                del df['TOPFST']
                del df['TOPLST']

            if name == 'EXT_SOURCES':
                df.SVOL = '*'       # means source is self
                df['AUX_TS'] = ''
                df['COMMENTS'] = ''
                df.MFACTOR = pd.to_numeric(df.MFACTOR)
                if not HSPF:
                    del df['TGRPN']
                    del df['SMEMN']
                    del df['SGAPST']
                df = df.sort_values('TVOLNO')
                df = df.reset_index(drop=True)

            if name == 'SCHEMATIC':
                 df.AFACTR = pd.to_numeric(df.AFACTR)
                 if not HSPF:
                    pat = 'PERLND|IMPLND|RCHRES'
                    boolean = df['SVOL'].str.contains(pat) & df['TVOL'].str.contains(pat)
                    df = df[boolean]
                    df = df.sort_values('TVOLNO')
                    df = df.dropna()
                    df = df.reset_index(drop=True)

            if name == 'NETWORK':
                df.MFACTOR = pd.to_numeric(df.MFACTOR)
                if not HSPF:
                    pat = 'PERLND|IMPLND|RCHRES'
                    boolean = df['SVOL'].str.contains(pat) & df['TVOL'].str.contains(pat)
                    df = df[boolean]
                    df = df.sort_values(['TVOLNO', 'TMEMN'])
                    df = df.dropna()
                    df = df.reset_index(drop=True)

            if not df.empty and group:
                df.to_hdf(hdfname, group, data_columns=True, format='t')
            return df
        else:
            row = pd.Series(index=indx)
            start = ''
            stop  = ''
            for v,c,e,default,t in dfparse.values:
                value = line[c:e].strip()
                value = value if  value else default     # if null, use the default

                if v=='SVOLNO' and name=='EXT_SOURCES':
                    value = prefix['TS'] + value
                elif v=='SVOLNO':
                    pf = prefix[row['SVOL'][0]] if row['SVOL'][0] in prefix else row['SVOL'][0]
                    value = pf + '{:0>3s}'.format(value)

                if v=='TVOLNO':
                    try:
                        if row['TVOL'][0] not in prefix:
                            print('PREFIX ERROR ' + str(row))
                            print()
                    except:
                        print('ILLEGAL PREFIX' + str(name))
                        print(row)
                        continue

                    value = prefix[row['TVOL'][0]] + '{:0>3s}'.format(value)
                if v=='TOPFST':
                    start = int(value)
                if v=='TOPLST':
                    stop = int(value) if value else start
                row[v] = value

            if start and stop:
                for i in range(start, stop+1):
                    pf = prefix[row['TVOL'][0]] if row['TVOL'][0] in prefix else row['TVOL'][0]
                    row['TVOLNO'] = pf + '{:03d}'.format(i)
                    rowlist.append(row.copy())
            else:
                rowlist.append(row)


def read_ftable(f, hdfname, scan, group):
    ''' Parses UCI FTABLES'''
    line,tokens,numtokens = f().next()
    ncol = int(tokens[1])
    cols = ['Depth','Area','Volume','Disch1','Disch2','Disch3','Disch4','Disch5'][:ncol]

    rowlist = []
    for line, tokens, numtokens in f():
        if tokens[0].startswith('END'):   # sometimes END is jammed upto FTABLE ID
            df = pd.DataFrame(rowlist)
            if group and not df.empty:
                df.to_hdf(hdfname, group, data_columns=True, format='t')
            return
        elif len(tokens) == ncol:  # free format read, not everyone puts into correct columns
            row = pd.Series(index=cols)
            for i,x in enumerate(line.split()):
                try:
                    row[cols[i]] = float(x)
                except:
                    print('Parsing error line: ' + str(line))
                    continue
            rowlist.append(row)
        else:
            row = pd.Series(index=cols)
            for v,c,e in scan[:ncol,:]:
                try:
                    row[v] = float(line[c:e].strip())
                except:
                    print('Parsing error line: ' + str(line))
            rowlist.append(row)


def read_segment(f, hdffile, target, h2, prefix, HSPF, metric=False):
    '''Parses UCI PERLND, IMPLND, and RCHRES tables'''

    df = pd.read_hdf(h2, '/CONFIGURATION')
    pathlookup = {row.Flag:row.Path for i, row in df.iterrows() if row.Target==target}
    flaglookup = {row.Path:row.Flag for i, row in df.iterrows() if row.Target==target}

    dfgroups = pd.read_hdf(h2, '/PARSEDATA/' + target + 'GROUPS')
    boolean = (dfgroups.HDFType=='') | (dfgroups.HDFType=='MONTHLY')
    dfgroups2 = dfgroups[boolean]
    lookup = {row.Table:pathlookup[row.Flag] + row.HDFGroup for i,row in dfgroups2.iterrows()}

    dfparse  = pd.read_hdf(h2, '/PARSEDATA/' + target)
    keep = 'last' if metric else 'first'
    dfparse = dfparse.drop_duplicates(subset=['Variable','Table'], keep=keep)
    for i, row in dfparse.iterrows():
        dfparse.loc[i, 'Variable'] = row.Variable.replace('-','_').replace('(','__').replace(')','')

    colslookup = defaultdict(list)
    for i,row in dfparse.iterrows():
        if row.Table in lookup:
            colslookup[lookup[row.Table]].append(row.Variable)
    for key in colslookup:
        if 'MONTHLY' in key:
            colslookup[key] =  colslookup[key][:13]
        elif 'ACTIVITY' not in key:
            colslookup[key] = (sorted(set(colslookup[key]) - set(['OPNID'])))
    frames = {key:pd.DataFrame(columns=colslookup[key]) for key in colslookup}

    scan = None
    idrange = None
    for line,tokens,numtokens in f():
        if numtokens==2 and tokens[0]=='END' and tokens[1]==target:
            break
        elif tokens[0]=='END':
            scan = []
            continue
        elif numtokens==1:
            tabletype = tokens[0]
            scan = dfparse[dfparse['Table']==tabletype].values
        elif len(scan) > 0:
            for _,_,v,c,default,t,r,g,e in scan:
                if v == 'OPNID':
                    ids = line[c:e].strip().split()
                    last = ids[1] if len(ids)==2 else ids[0]
                    idrange = range(int(ids[0]), 1 + int(last))
                else:
                    value = line[c:e].strip()
                    try:
                        value = cast[t](value if value else default)
                    except:
                        print('Parsing error line: ' +  str(line))
                        continue
                    if tabletype in lookup:
                        key =  lookup[tabletype]
                        df = frames[key]
                        for id in idrange:
                            idd = prefix[target[0]] + '{:0>3s}'.format(str(id))
                            df.loc[idd,v] = value

    savelist = []
    for key in frames.keys():
        df = frames[key]
        if not df.empty:
            if key == 'RCHRES/HYDR/PARAMETERS':
                df['FTBUCI'] = df['FTBUCI'].apply(lambda x: prefix['FT'] + '{:03n}'.format(int(x)))
            '''
            if key == 'RCHRES/HYDR/FLAGS':
                if 'VOL' in df[key]:
                    dff = df[['VOL']]
                    dff = dff.dropna()
                    dff = dff.convert_objects(convert_numeric=True)
                    dff.to_hdf(hdffile, 'RCHRES/HYDR/STATE', data_columns=True, format='t')
                    df = df.drop(['VOL'], axis=1)
            '''
            df = df.convert_objects(convert_numeric=True)
            if 'ICAT' in df.columns:
                df['ICAT'] = df['ICAT'].fillna('').astype(str)
            df = df.dropna(axis='columns', how='all')
            if 'GENERAL_INFO' in key:
                df['LANDUSE'] = ''
            df = df.sort_index()
            df = df.dropna()   #???
            df.to_hdf(hdffile, key, data_columns=True, format='t')

            tokens = key.split('/')
            if tokens[1] not in ['ACTIVITY', 'GENERAL_INFO']:
                spath = '/'.join(tokens[:2]) + '/'
                sflag = flaglookup[spath]
                savelist.append(sflag)

    indx = frames[target + '/ACTIVITY'].index
    for flag in set(savelist):
        saveinfo = pd.read_hdf(h2, '/SAVE/' + target)
        saveinfo = saveinfo[saveinfo.Flag==flag]
        df = pd.DataFrame(index=indx)
        for i,row in saveinfo.iterrows():
                df[row.Name] = row.Value
        df = df.astype(bool)
        df = df.sort_index()
        df = df.dropna()   #???
        df.to_hdf(hdffile, pathlookup[flag] + '/SAVE', format='t', data_columns=True)
