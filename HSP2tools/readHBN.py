''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
License: LGPL2
'''

from struct import unpack
from numpy import fromfile, reshape, datetime64, timedelta64, empty
from pandas import DataFrame
from datetime import datetime
from collections import defaultdict, Counter

tcodes = {1:'_Minutely', 2:'_Hourly', 3:'_Daily', 4:'_Monthly', 5:'_Yearly'}

def readHBN(hbnfile, hdffile):
    ''' Extracts all data in hbnfile and saves to HDF5 file'''
    data = fromfile(hbnfile, 'B')
    if data[0] != 0xFD:
        print('BAD HBN FILE - must start with magic number 0xFD')
        return

    # Build layout maps of the file's contents
    map = {}
    mapf = defaultdict(list)
    target = {'P', 'I', 'R'}    # (P)erland, (I)mplnd, and (R)chres
    index = 1                   # already used first byte (magic number)
    while index < len(data):
        if chr(data[index+8]) not in target:
            index -= 1   # works, but why???

        reclen, rectype, optype, lue, section = unpack('2I8sI8s', data[index : index+28])
        reclen  = reclen >> 2
        optype  = optype.decode().strip()  # Python3 converts to bytearray not string
        section = section.decode().strip()

        if rectype:
            level = unpack('I', data[index+32 : index+36])[0]  # 28 + 4
            mapf[(optype, lue, section, level)].append(index)
        else:
            map[(optype, lue, section)] = (index, reclen)
        index += reclen + 6

    # For each mapf key, use map to extract floating point values. Save to HDF5 file
    for optype, lue, section, level in mapf.keys():
        index, reclen = map[(optype, lue, section)]

        # field names are concatenation of length, string pairs.  Use trick to ignore lengths.
        fields = ''.join([' ' if chr(y) < 'A' else chr(y) for y in data[index+28:index+reclen+4]]).split()  # 4 ??? why
        nfields = len(fields)

        dindex = mapf[(optype, lue, section, level)]
        d = empty(len(dindex) * nfields)      # preallocate for speed, just like MATLAB
        t = empty(len(dindex)).astype('datetime64[ns]')

        findex = 0
        tindex = 0
        for di in dindex:
            yr, mo, dy, hr, mn = unpack('5I', data[di+36 : di+56])  # 28 + 8
            t[tindex] = todatetime64(yr, mo, dy, hr)
            tindex += 1

            d[findex:findex+nfields] = unpack(f'{nfields}f', data[di+56:di+56+nfields*4])
            findex += nfields

        # fix duplicate names in DataFrame (from nexits for example)
        cnts = Counter(fields)
        copy = cnts.copy()
        temp = []
        for x in fields[::-1]:
            count = cnts[x]
            if copy[x] == 1:
                temp.append(x)
            else:
                temp.append(x+str(count))
                cnts[x] -= 1

        d = reshape(d, (len(dindex), nfields))
        df = DataFrame(d, index=t, columns=reversed(temp)).astype('float32')
        df.to_hdf(hdffile, f'/RESULTS{tcodes[level]}/{optype}/{section}/{optype[0]}{lue:03d}')
        del d,t
    return


def todatetime64(yr=1900, mo=1, dy=1, hr=0):
    '''returns datatime64 with fix for midnight given yr,mo,dy,hr'''
    if hr == 24:
        return datetime64(datetime(yr,mo,dy,23), 'm') + timedelta64(1,'h')
    else:
        return datetime64(datetime(yr,mo,dy,hr), 'm')