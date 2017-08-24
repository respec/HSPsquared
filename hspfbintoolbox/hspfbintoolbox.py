'''
hspfbintoolbox to read HSPF binary files.
'''

from __future__ import print_function

import datetime
import warnings
import os
import sys
import struct

import mando
import pandas as pd

from tstoolbox import tsutils


code2intervalmap = {5: 'yearly',
                    4: 'monthly',
                    3: 'daily',
                    2: 'bivl'}

interval2codemap = {'yearly': 5,
                    'monthly': 4,
                    'daily': 3,
                    'bivl': 2}

code2freqmap = {5: 'Y',
                4: 'M',
                3: 'D',
                2: None}


def tupleMatch(a, b):
    '''Part of partial ordered matching.
    See http://stackoverflow.com/a/4559604
    '''
    return len(a) == len(b) and all(i is None or j is None or i == j
                                    for i, j in zip(a, b))


def tupleCombine(a, b):
    '''Part of partial ordered matching.
    See http://stackoverflow.com/a/4559604
    '''
    return tuple([i is None and j or i for i, j in zip(a, b)])


def tupleSearch(findme, haystack):
    '''Partial ordered matching with 'None' as wildcard
    See http://stackoverflow.com/a/4559604
    '''
    return [(i, tupleCombine(findme, h))
            for i, h in enumerate(haystack) if tupleMatch(findme, h)]


def _get_data(binfilename,
              interval='daily',
              labels=[',,,'],
              catalog_only=True):
    '''Underlying function to read from the binary file.  Used by
    'extract', 'catalog', and 'dump'.
    '''
    testem = {'PERLND': ['ATEMP', 'SNOW', 'PWATER', 'SEDMNT',
                         'PSTEMP', 'PWTGAS', 'PQUAL', 'MSTLAY',
                         'PEST', 'NITR', 'PHOS', 'TRACER', ''],
              'IMPLND': ['ATEMP', 'SNOW', 'IWATER', 'SOLIDS',
                         'IWTGAS', 'IQUAL', ''],
              'RCHRES': ['HYDR', 'CONS', 'HTRCH', 'SEDTRN',
                         'GQUAL', 'OXRX', 'NUTRX', 'PLANK',
                         'PHCARB', 'INFLOW', 'OFLOW', 'ROFLOW', ''],
              '': ['']}

    collect_dict = {}
    lablist = []

    # Normalize interval code
    try:
        intervalcode = interval2codemap[interval.lower()]
    except AttributeError:
        intervalcode = None

    # Fixup and test the labels - could be in it's own function
    for lindex, label in enumerate(labels):
        words = [lindex] + label.split(',')
        if len(words) != 5:
            raise ValueError('''
*
*   The label '{0}' has the wrong number of entries.
*
'''.format(label))

        if words[1]:
            words[1] = words[1].upper()
            if words[1] not in testem.keys():
                raise ValueError('''
*
*   Operation type must be one of 'PERLND', 'IMPLND', or 'RCHRES',
*   or missing (to get all) instead of {0}.
*
'''.format(words[1]))

        if words[2]:
            try:
                words[2] = int(words[2])
                if words[2] < 1 or words[2] > 999:
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError('''
*
*   The land use element must be an integer from 1 to 999 inclusive,
*   instead of {0}.
*
'''.format(words[2]))

        if words[3]:
            words[3] = words[3].upper()
            if words[3] not in testem[words[1]]:
                raise ValueError('''
*
*   The {0} operation type only allows the sections:
*   {1},
*   instead you gave {2}.
*
'''.format(words[1], testem[words[1]][:-1], words[3]))

        for index in list(range(len(words))):
            if words[index] is '':
                words[index] = None

        words.append(intervalcode)
        lablist.append(words)

    with open(binfilename, 'rb') as fl:

        mindate = datetime.datetime.max
        maxdate = datetime.datetime.min

        labeltest = {}
        vnames = {}
        ndates = {}
        rectype = 0
        fl.read(1)
        while 1:
            try:
                reclen1, reclen2, reclen3, reclen = struct.unpack('4B',
                                                                  fl.read(4))
            except struct.error:
                break
            reclen1 = int(reclen1/4)
            reclen2 = reclen2*64 + reclen1
            reclen3 = reclen3*16384 + reclen2
            reclen = reclen*4194304 + reclen3
            slen = 0
            rectype, optype, lue, section = struct.unpack('I8sI8s',
                                                          fl.read(24))

            rectype = int(rectype)
            lue = int(lue)
            optype = optype.strip()
            section = section.strip()

            if rectype == 0:
                while slen < reclen - 28:
                    length = struct.unpack('I', fl.read(4))[0]
                    slen = slen + length + 4
                    variable_name = struct.unpack('{0}s'.format(length),
                                                  fl.read(length))[0]
                    vnames.setdefault((lue, section), []).append(variable_name)

            if rectype == 1:
                # Data record
                numvals = len(vnames[(lue, section)])

                (_,
                 level,
                 year,
                 month,
                 day,
                 hour,
                 minute) = struct.unpack('7I', fl.read(28))

                vals = struct.unpack('{0}f'.format(numvals),
                                     fl.read(4*numvals))
                if hour == 24:
                    ndate = datetime.datetime(year, month, day) + \
                        datetime.timedelta(hours=24) + \
                        datetime.timedelta(minutes=minute)
                else:
                    ndate = datetime.datetime(year, month, day, hour, minute)

                for i, vname in enumerate(vnames[(lue, section)]):
                    tmpkey = (None,
                              optype.decode('ascii'),
                              int(lue),
                              section.decode('ascii'),
                              vname.decode('ascii'),
                              level)
                    if catalog_only is False:
                        res = tupleSearch(tmpkey, lablist)
                        if res:
                            nres = (res[0][0],) + res[0][1][1:]
                            labeltest[nres[0]] = 1
                            collect_dict.setdefault(nres, []).append(vals[i])
                            ndates.setdefault(level, {})[ndate] = 1
                    else:
                        mindate = min(mindate, ndate)
                        maxdate = max(maxdate, ndate)
                        collect_dict[tmpkey[1:]] = (mindate, maxdate)
            fl.read(2)

    if not collect_dict:
        raise ValueError('''
*
*   The label specifications matched no records in the binary file.
*
''')

    if catalog_only is False:
        not_in_file = []
        for loopcnt in list(range(len(lablist))):
            if loopcnt not in labeltest.keys():
                not_in_file.append(labels[loopcnt])
        if not_in_file:
            warnings.warn('''
*
*   The specification{0} {1}
*   matched no records in the binary file.
*
'''.format("s"[len(not_in_file) == 1:], not_in_file))

    return ndates, collect_dict


@mando.command
def extract(hbnfilename, interval, *labels, **kwds):
    '''
    Prints out data to the screen from a HSPF binary output file.

    :param hbnfilename: The HSPF binary output file
    :param interval: One of 'yearly', 'monthly', 'daily', or 'BIVL'.
        The 'BIVL' option is a sub-daily interval defined in the UCI file.
        Typically 'BIVL' is used for hourly output, but can be set to any
        value that evenly divides into a day.
    :param labels: The remaining arguments uniquely identify a time-series
        in the binary file.  The format is
        'OPERATIONTYPE,ID,SECTION,VARIABLE'.

        For example: 'PERLND,101,PWATER,UZS IMPLND,101,IWATER,RETS'

        Leaving a section without an entry will wildcard that
        specification.  To get all the PWATER variables for PERLND 101 the
        label would read:

        'PERLND,101,PWATER,'

        To get TAET for all PERLNDs:

        'PERLND,,,TAET'

        Note that there are spaces ONLY between label specifications.
    :param time_stamp: For the interval defines the location of the time
        stamp. If set to 'begin', the time stamp is at the begining of the
        interval.  If set to any other string, the reported time stamp will
        represent the end of the interval.  Default is 'begin'.  Place after
        ALL labels.
    :param sorted: Should ALL columns be sorted?
        Default is False.  Place after ALL labels.
    '''
    try:
        time_stamp = kwds.pop('time_stamp')
    except KeyError:
        time_stamp = 'begin'
    if time_stamp not in ['begin', 'end']:
        raise ValueError('''
*
*   The "time_stamp" optional keyword must be either
*   "begin" or "end".  You gave {0}.
*
'''.format(time_stamp))

    try:
        sortall = bool(kwds.pop('sorted'))
    except KeyError:
        sortall = False
    if not(sortall is True or sortall is False):
        raise ValueError('''
*
*   The "sorted" optional keyword must be either
*   True or False.  You gave {0}.
*
'''.format(sortall))

    if len(kwds) > 0:
        raise ValueError('''
*
*   The extract command only accepts optional keywords 'time_stamp' and
*   'sorted'.  You gave {0}.
*
'''.format(list(kwds.keys())))

    interval = interval.lower()
    if interval not in ['bivl', 'daily', 'monthly', 'yearly']:
        raise ValueError('''
*
*   The "interval" argument must be one of "bivl",
*   "daily", "monthly", or "yearly".  You supplied
*   "{0}".
*
'''.format(interval))

    index, data = _get_data(hbnfilename, interval, labels,
                            catalog_only=False)
    index = index[interval2codemap[interval]]
    index = list(index.keys())
    index.sort()
    skeys = list(data.keys())
    if sortall is True:
        skeys.sort(key=lambda tup: tup[1:])
    else:
        skeys.sort()

    result = pd.concat([pd.Series(data[i], index=index) for i in skeys],
                       axis=1, join_axes=[pd.Index(index)])

    columns = ['{0}_{1}_{2}_{3}'.format(i[1], i[2], i[4], i[5]) for i in skeys]
    result.columns = columns

    if time_stamp == 'begin':
        result = tsutils.asbestfreq(result)
        result = result.tshift(-1)

    return tsutils.printiso(result)


@mando.command
def catalog(hbnfilename):
    '''
    Prints out a catalog of data sets in the binary file.

    The first part of each line up to the first space is the label that can
    be used with the 'extract' command.

    :param hbnfilename: The HSPF binary output file
    '''
    catlog = _get_data(hbnfilename, None, [',,,'], catalog_only=True)[1]
    if tsutils.test_cli() is False:
        return catlog
    catkeys = list(catlog.keys())
    catkeys.sort()
    for cat in catkeys:
        print('{0},{1},{2},{3}  ,{5}, {6}, {7}'.format(
            *(cat + catlog[cat] +
              (code2intervalmap[cat[-1]],))))


@mando.command
def dump(hbnfilename, time_stamp='begin'):
    '''
    Prints out ALL data from a HSPF binary output file.

    :param hbnfilename: The HSPF binary output file
    :param time_stamp: For the interval defines the location of the time
        stamp. If set to 'begin', the time stamp is at the begining of the
        interval.  If set to any other string, the reported time stamp will
        represent the end of the interval.  Default is 'begin'.
    '''
    if time_stamp not in ['begin', 'end']:
        raise ValueError('''
*
*   The "time_stamp" optional keyword must be either
*   "begin" or "end".  You gave {0}.
*
'''.format(time_stamp))

    index, data = _get_data(hbnfilename, None, [',,,'], catalog_only=False)
    skeys = list(data.keys())
    skeys.sort()

    result = pd.concat([pd.Series(data[i], index=index) for i in skeys],
                       axis=1, join_axes=[pd.Index(index)])

    columns = ['{0}_{1}_{2}_{3}'.format(i[1], i[2], i[4], i[5]) for i in skeys]
    result.columns = columns

    if time_stamp == 'begin':
        result = tsutils.asbestfreq(result)
        result = result.tshift(-1)

    return tsutils.printiso(result)


@mando.command()
def about():
    """Display version number and system information.
    """
    tsutils.about(__name__)


@mando.command
def time_series(hbnfilename, interval, *labels, **kwds):
    ''' DEPRECATED: Use 'extract' instead.
    '''
    return extract(hbnfilename, interval, *labels, **kwds)


def main():
    if not os.path.exists('debug_hspfbintoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
