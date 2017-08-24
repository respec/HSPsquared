''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import h5py
import shutil
import os
import pandas as pd
import numpy as np
import struct
import datetime


def save_document(hdfname, doc):
    ''' Saves a generic document into an HDF5 file
    CALL: save_document(hdfname, doc)
       hdfname is the HSP2 HDF5 file to save into
       doc is the filename of the document to be stored'''

    docpath, docname = os.path.split(doc)
    okname = docname.replace(' ', '').replace('.','')    # make legal for HDF5
    temp = np.fromfile(doc, dtype=np.uint8, count=-1)
    with h5py.File(hdfname) as hdf:
        if 'Documents/' + okname in hdf:
            del hdf['Documents/' + okname]
        if 'Documents' not in hdf:
            g = hdf.create_group('Documents')
        else:
            g = hdf['Documents']
        ds = g.create_dataset(okname, data=temp)
        ds.attrs['filename'] = docname
        ds.attrs['filepath'] = docpath


def restore_document(hdfname, doc, path=None):
    ''' Takes a document out of an HSP2 HDF5 file and puts it back to its original directory
    CALL: (hdfname, doc, path=None)
        hdfname is the HSP2 HDF5 file name with the stored document
        doc is the original name of the stored document to be extracted into the directory'''

    docpath, docname = os.path.split(doc)
    okname = docname.replace(' ', '').replace('.','')    # make legal for HDF5
    with h5py.File(hdfname) as hdf:
        if 'Documents/' + okname not in hdf:
            print(' '.join(['Document', doc, 'NOT found in', hdfname]))
        else:
            ds = hdf['Documents/' + okname]
            p = path if  path else ds.attrs['filepath']
            ds[...].tofile(os.path.join(p, docname))


def reset_tutorial():
    ''' Deletes the Tutoral data and reloads it from the DataSource directory'''
    if os.path.exists('TutorialData'):
        shutil.rmtree('TutorialData')
    shutil.copytree('DataSource', 'TutorialData')


def checkHDF(hdfname):
    ''' utility to do HSPF data integrety checks on the HSP2 HDF5 file'''
    # Now do for all FTABLES (HYDR)
    keys = pd.read_hdf(hdfname, 'HSP2/KEYS')
    keyset = set(keys)

    for key in keyset:
        if key.startswith('/FTABLES'):
            df = pd.read_hdf(hdfname, key)
            boolean = (df.shape[0] > 0                # NROWS > 0
            and df.at[0, 'Volume'] == 0.0            # First row volume = 0.0
            and all(df >= 0.0)                       # No negative values allowed
            and not any (df['Depth'].diff() < 0.0)   # Depth must be monotonically increasing
            and not any (df['Volume'].diff() < 0.0)) # Volume must be monotonically increasing
            if not boolean:
                print('Problem with FTABLE' + str(key))

    # Now Need to Check if every REACH specifies a valid FTABLE
    skip = len('/FTABLES/FTABLE')
    keylist = []
    for key in keys:
        if key.startswith('/FTABLES'):
            keylist.append(key[skip:])
    print('DONE')


def readPLTGEN(filename):
    ''' Reads HSPF PLTGEN files and creates a DataFrame that can be plotted like HSP2'''

    foundcols = False
    cols = []
    lst = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i < 25 and 'LINTYP' in line:
                foundcols = True
            elif i < 25 and line[5:].startswith('Time series'):
                foundcols = False
            elif i < 25 and foundcols:
                header =  line[4:30].strip()
                if not header:
                    foundcols = False
                else:
                    cols.append(header)

            if i > 25:
                y, mm, d, h, m = line[4:22].split()
                if int(h) == 24:
                    d = [datetime.datetime(int(y), int(mm), int(d), tzinfo=None)]
                else:
                    d = [datetime.datetime(int(y), int(mm), int(d), int(h)-1, int(m), tzinfo=None)]
                data = [float(x) for x in line[23:].split()]
                lst.append(d + data)

    df = pd.DataFrame(lst)
    df.columns = ['Date'] + cols
    df = df.set_index(['Date'])
    return df


def dayval(start_date, end_date, monthly, method='interpolate'):
    ''' given value at start of month, interpolate to day values, pad to hours'''

    start = pd.to_datetime(start_date)
    stop  = pd.to_datetime(end_date.replace('24:', '23:'))

    strstart = '01/01/'  + str(start.year- 1)
    strstop  = '12/31/'  + str(stop.year + 1)

    tindex = pd.date_range(strstart, strstop, freq='MS')
    tiled = np.tile(monthly, len(tindex)/12)

    if method == 'interpolate':
        daily = pd.Series(tiled, index=tindex).resample('D').pad()
        hourly = daily.interpolate(method='time').resample('H').pad()
    else:
        hourly = pd.Series(tiled, index=tindex).resample('H').pad()

    return hourly[start_date:end_date].values


def get_HBNdata(binfilename, label='PERLND,11,PWATER,PERO'):
    '''Underlying function to read from a HBN binary file.

    This code was adapted from Tim Cera's hspfbintoolbox.py because of issues
    running it on Windows with Python 2.7. It is probably due to the unicode
    fixes for compatibility with Python 3.x

    hspfbintoolbox Copyright 2016 by Tim Cera, P.E.
    hspfbintoolbox has BSD license
    '''

    testem = {'PERLND': ['ATEMP', 'SNOW', 'PWATER', 'SEDMNT', 'PSTEMP', 'PWTGAS',
                         'PQUAL', 'MSTLAY','PEST', 'NITR', 'PHOS', 'TRACER', ''],
              'IMPLND': ['ATEMP', 'SNOW', 'IWATER', 'SOLIDS',
                         'IWTGAS', 'IQUAL', ''],
              'RCHRES': ['HYDR', 'CONS', 'HTRCH', 'SEDTRN', 'GQUAL', 'OXRX', 'NUTRX',
                         'PLANK', 'PHCARB', 'INFLOW', 'OFLOW', 'ROFLOW', '']
             }
    code2freqmap = {5: 'Y',
                4: 'M',
                3: 'D',
                2: None}

    results = {}

    # Fixup and test the labels - could be in it's own function
    operation, segmentid, operationtype, variable = label.split(',')

    operation = operation.upper()
    if operation not in testem.keys():
            print('OPERATION must be one of PERLND< IMPLND, or RCHRES')
    segmentid = int(segmentid)
    if not 0 < segmentid < 1000:
       print('Land use ID must be in range of 1...999')

    operationtype = operationtype.upper()
    if operationtype not in testem[operation]:
        print('Operationtype is wrong')

    with open(binfilename, 'rb') as fl:
        vnames = {}
        tindex = 0

        optype_list = ['PERLND', 'IMPLND', 'RCHRES']
        while 1:
            fl.seek(tindex)
            initial_search = fl.read(25)
            search_index = [initial_search.find(i) for i in optype_list]

            maxsindex = max(search_index)
            search_index_list = [maxsindex if i == -1 else i for i in search_index]
            search_index = min(search_index_list)
            if search_index == -1:
                break
            tindex = tindex + search_index - 4
            fl.seek(tindex)
            rectype, optype, lue, section = struct.unpack('I8sI8s', fl.read(24))
            optype = optype.strip()           # PERLND, IMPLND, RCHRES
            section = section.strip()         # like PWATER
            lue = int(lue)                    # land use segment id

            if rectype == 0:
                fl.seek(tindex - 4)
                reclen1, reclen2, reclen3, reclen = struct.unpack('4B', fl.read(4))
                reclen1 = int(reclen1/4)
                reclen2 = reclen2*64 + reclen1
                reclen3 = reclen3*16384 + reclen2
                reclen = reclen*4194304 + reclen3
                fl.seek(tindex + 24)
                slen = 0
                while slen < reclen - 28:
                    length = struct.unpack('I', fl.read(4))[0]
                    slen = slen + length + 4
                    variable_name = struct.unpack('{0}s'.format(length), fl.read(length))[0]
                    vnames.setdefault((lue, section), []).append(variable_name.strip())

            if rectype == 1:      # Data record
                numvals = len(vnames[(lue, section)])
                unit_flag, level, year, month, day, hour, minute = struct.unpack('7I', fl.read(28))
                freq = code2freqmap[level]
                vals = struct.unpack('{0}f'.format(numvals), fl.read(4*numvals))
                if hour == 24:
                    ndate = datetime.datetime(year, month, day) + datetime.timedelta(hours=24) + datetime.timedelta(minutes=minute)
                else:
                    ndate = datetime.datetime(year, month, day, hour, minute)

                for i, name in enumerate(vnames[(lue, section)]):
                    if lue == segmentid and section == operationtype and name==variable:
                        results.setdefault(freq,[]).append((ndate, vals[i]))

            tindex = fl.tell()

    r = {}
    for level in results.keys():
        times, values = zip(*results[level])
        r[level] = pd.Series(data=values, index=times)

    return r, 'English' if unit_flag==1 else 'Metric'
