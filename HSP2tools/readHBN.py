''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
and hspfbintoolbox.py by Tim Cera
License: LGPL2
'''

from struct import unpack
from numpy import fromfile
from pandas import DataFrame
from datetime import datetime, timedelta
from collections import defaultdict

tcodes = {1:'Minutely', 2:'Hourly', 3:'Daily', 4:'Monthly', 5:'Yearly'}

def readHBN(hbnfile, hdfname):
    '''
    Reads ALL data from hbnfile and saves to HDF5 hdfname file

    Parameters
    ----------
    hbnfile : str
        Name/path of HBN filled created by HSPF.
    hdfname : str
        Name/path of HDF5 file to store HBN data as Pandas DataFrames.

    Returns
    -------
    dfsummary : DataFrame
        Summary information of data found in HBN file (also saved to HDF5 file.)
    '''

    data = fromfile(hbnfile, 'B')
    if data[0] != 0xFD:
        print('BAD HBN FILE - must start with magic number 0xFD')
        return

    # Build layout maps of the file's contents
    mapn = defaultdict(list)
    mapd = defaultdict(list)
    index = 1                   # already used first byte (magic number)
    while index < len(data):
        rc1,rc2,rc3,rc, rectype,operation,id,activity = unpack('4BI8sI8s',data[index:index+28])
        rc1     = int(rc1 >> 2)
        rc2     = int(rc2) * 64      + rc1        # 2**6
        rc3     = int(rc3) * 16384   + rc2        # 2**14
        reclen  = int(rc)  * 4194304 + rc3 - 24   # 2**22

        operation = operation.decode('ascii').strip()  # Python3 converts to bytearray not string
        activity  = activity.decode('ascii').strip()

        if operation not in {'PERLND', 'IMPLND', 'RCHRES'}:
            print('ALIGNMENT ERROR', operation)

        if rectype==1:    # data record
            tcode = unpack('I', data[index+32 : index+36])[0]
            mapd[operation, id, activity, tcode].append((index,reclen))
        elif rectype == 0:  # data names record
            i = index + 28
            slen = 0
            while slen < reclen:
                ln = unpack('I', data[i+slen : i+slen+4])[0]
                n  = unpack(f'{ln}s', data[i+slen+4 : i+slen+4+ln])[0].decode('ascii').strip()
                mapn[operation, id, activity].append(n.replace('-',''))
                slen += 4+ln
        else:
            print('UNKNOW RECTYPE', rectype)
        if reclen < 36:
            index += reclen + 29                        # found by trial and error
        else:
            index += reclen + 30


    summary = []
    summarycols = ['Operation', 'Activity', 'segment', 'Frequency', 'Shape', 'Start', 'Stop']
    summaryindx = []
    for (operation, id, activity, tcode) in mapd:
        rows = []
        times = []
        nvals = len(mapn[operation,id,activity])
        for (index, reclen) in mapd[operation, id, activity, tcode]:
            yr,mo,dy,hr,mn = unpack('5I', data[index+36 : index+56 ])
            dt = datetime(yr,mo,dy,0,mn) + timedelta(hours=hr)
            times.append(dt)

            index += 56
            row = unpack(f'{nvals}f', data[index:index+(4*nvals)])
            rows.append(row)
        dfname = f'{operation}_{activity}_{id:03d}_{tcode}'
        df = DataFrame(rows, index=times, columns=mapn[operation,id,activity]).sort_index('index')
        df.to_hdf(hdfname, dfname, complib='blosc', complevel=9)

        summaryindx.append(dfname)
        summary.append((operation, activity, str(id), tcodes[tcode], str(df.shape), df.index[0], df.index[-1]))

    dfsummary = DataFrame(summary, columns=summarycols, index=summaryindx)
    dfsummary.to_hdf(hdfname, 'SUMMARY', data_columns=True, format='t')
    return dfsummary
