''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
License: LGPL2
'''

import numpy as np
import pandas as pd
from numba import jit, njit
import datetime

import warnings

# look up attributes NAME, data type (Integer; Real; String) and data length by attribute number
attrinfo = {1:('TSTYPE','S',4),     2:('STAID','S',16),    11:('DAREA','R',1),
           17:('TCODE','I',1),     27:('TSBYR','I',1),     28:('TSBMO','I',1),
           29:('TSBDY','I',1),     30:('TSBHR','I',1),     32:('TFILL', 'R',1),
           33:('TSSTEP','I',1),    34:('TGROUP','I',1),    45:('STNAM','S',48),
           83:('COMPFG','I',1),    84:('TSFORM','I',1),    85:('VBTIME','I',1),
          444:('DATMOD','S',12),  443:('DATCRE','S',12),   22:('DCODE','I',1),
           10:('DESCRP','S', 80),   7:('ELEV','R',1),       8:('LATDEG','R',1),
            9:('LNGDEG','R',1),   288:('SCENARIO','S',8), 289:('CONSTITUENT','S',8),
          290:('LOCATION','S',8)}

freq = {7:'100YS', 6:'YS', 5:'MS', 4:'D', 3:'h', 2:'min', 1:'S'}   # pandas date_range() frequency by TCODE, TGROUP


def readWDM(wdmfile, hdffile, compress_output=False):
    iarray = np.fromfile(wdmfile, dtype=np.int32)
    farray = np.fromfile(wdmfile, dtype=np.float32)

    date_epoch = np.datetime64(0,'Y')
    dt_year = np.timedelta64(1, 'Y')
    dt_month = np.timedelta64(1, 'M')
    dt_day = np.timedelta64(1, 'D')
    dt_hour = np.timedelta64(1, 'h')
    dt_minute = np.timedelta64(1, 'm')
    dt_second = np.timedelta64(1, 's')

    if iarray[0] != -998:
        raise ValueError ('Provided file does not match WDM format. First int32 should be -998.')
    nrecords    = iarray[28]    # first record is File Definition Record
    ntimeseries = iarray[31]

    dsnlist = []
    for index in range(512, nrecords * 512, 512):
        if not (iarray[index]==0 and iarray[index+1]==0 and iarray[index+2]==0 and iarray[index+3]==0) and iarray[index+5]==1:
            dsnlist.append(index)
    if len(dsnlist) != ntimeseries:
        raise RuntimeError (f'Wrong number of Time Series Records found expecting:{ntimeseries} found:{len(dsnlist)}')

    with pd.HDFStore(hdffile) as store:
        summary = []
        summaryindx = []

        # check to see which extra attributes are on each dsn
        columns_to_add = []
        search = ['STAID', 'STNAM', 'SCENARIO', 'CONSTITUENT', 'LOCATION']
        for att in search:
            found_in_all = True
            for index in dsnlist:
                dattr = {}
                psa = iarray[index + 9]
                if psa > 0:
                    sacnt = iarray[index + psa - 1]
                for i in range(psa + 1, psa + 1 + 2 * sacnt, 2):
                    id = iarray[index + i]
                    ptr = iarray[index + i + 1] - 1 + index
                    if id not in attrinfo:
                        continue
                    name, atype, length = attrinfo[id]
                    if atype == 'I':
                        dattr[name] = iarray[ptr]
                    elif atype == 'R':
                        dattr[name] = farray[ptr]
                    else:
                        dattr[name] = ''.join([_inttostr(iarray[k]) for k in range(ptr, ptr + length // 4)]).strip()
                if att not in dattr:
                    found_in_all = False
            if found_in_all:
                columns_to_add.append(att)

        for index in dsnlist:
            # get layout information for TimeSeries Dataset frame
            dsn   = iarray[index+4]
            psa   = iarray[index+9]
            if psa > 0:
                sacnt = iarray[index+psa-1]
            pdat  = iarray[index+10]
            pdatv = iarray[index+11]
            frepos = iarray[index+pdat]

            print(f'{dsn} reading from wdm')
            # get attributes
            dattr = {'TSBDY':1, 'TSBHR':1, 'TSBMO':1, 'TSBYR':1900, 'TFILL':-999.}   # preset defaults
            for i in range(psa+1, psa+1 + 2*sacnt, 2):
                id = iarray[index + i]
                ptr = iarray[index + i + 1] - 1 + index
                if id not in attrinfo:
                    # print('PROGRAM ERROR: ATTRIBUTE INDEX not found', id, 'Attribute pointer', iarray[index + i+1])
                    continue

                name, atype, length = attrinfo[id]
                if atype == 'I':
                    dattr[name] = iarray[ptr]
                elif atype == 'R':
                    dattr[name] = farray[ptr]
                else:
                    dattr[name] = ''.join([_inttostr(iarray[k]) for k in range(ptr, ptr + length//4)]).strip()

            # Get timeseries timebase data
            records = [] 
            offsets = []
            for i in range(pdat+1, pdatv-1):
                a = iarray[index+i]
                if a != 0:
                    record, offset = _splitposition(a)
                    records.append(record)
                    offsets.append(offset)
            if len(records) == 0:
                continue   

            # calculate number of data points in each group, tindex is final index for storage
            tgroup = dattr['TGROUP']
            tstep  = dattr['TSSTEP']
            tcode  = dattr['TCODE']

            records = np.asarray(records)
            offsets = np.asarray(offsets)

            dates, values, stop_datetime = _process_groups(iarray, farray, records, offsets, tgroup)
            stop_datetime = datetime.datetime(*_bits_to_date(stop_datetime))
            dates = np.array(dates)
            dates_converted = _date_convert(dates, date_epoch, dt_year, dt_month, dt_day, dt_hour, dt_minute, dt_second)
            series = pd.Series(values, index=dates_converted)
            try:
                series.index.freq = str(tstep) + freq[tcode] 
            except ValueError:
                series.index.freq = None

            dsname = f'TIMESERIES/TS{dsn:03d}'
            if compress_output:
                series.to_hdf(store, key=dsname, complib='blosc', complevel=9)  
            else:
                series.to_hdf(store, key=dsname, format='t', data_columns=True)

            data = [
                str(series.index[0]), str(stop_datetime), str(tstep) + freq[tcode],
                len(series),  dattr['TSTYPE'], dattr['TFILL']
                ]
            columns = ['Start', 'Stop', 'Freq','Length', 'TSTYPE', 'TFILL']
            for x in columns_to_add:
                if x in dattr:
                    data.append(dattr[x])
                    columns.append(x)

            summary.append(data)
            summaryindx.append(dsname[11:])

        dfsummary = pd.DataFrame(summary, index=summaryindx, columns=columns)
        store.put('TIMESERIES/SUMMARY',dfsummary, format='t', data_columns=True)

    return dfsummary

@njit 
def _splitdate(x):
    year = np.int64(x >> 14)
    month = np.int64(x >> 10 & 0xF)
    day = np.int64(x >> 5 & 0x1F)
    hour = np.int64(x & 0x1F)
    return _correct_date(year, month, day, hour, 0,0)

@njit 
def _splitcontrol(x):
    nval = x >> 16
    ltstep = x >> 10 & 0x3f 
    ltcode = x >> 7 & 0x7
    comp = x >> 5 & 0x3
    qual  = x & 0x1f
    return nval, ltstep, ltcode, comp, qual

@njit 
def _splitposition(x):
    return((x>>9) - 1, (x&0x1FF) - 1) #args: record, offset

@njit
def _inttostr(i):
    return chr(i & 0xFF) + chr(i>>8 & 0xFF) + chr(i>>16 & 0xFF) + chr(i>>24 & 0xFF)

@njit 
def _bits_to_date(x):
    year = x >> 26
    month = x >> 22 & 0xf
    day = x >> 17 & 0x1f
    hour = x >> 12 & 0x1f
    minute = x >> 6 & 0x3f
    second = x & 0x3f
    return year, month, day, hour, minute, second

@njit 
def _date_to_bits(year, month, day, hour, minute, second):
    x = year << 26 | month << 22 | day << 17 | hour << 12 | minute << 6 | second 
    return x

@njit 
def _increment_date(date, timecode, timestep):
    year, month, day, hour, minute, second = _bits_to_date(date)
    
    if timecode == 7: year += 100 * timestep
    elif timecode == 6 : year += timestep
    elif timecode == 5 : month += timestep
    elif timecode == 4 : day += timestep
    elif timecode == 3 : hour += timestep
    elif timecode == 2 : minute += timestep
    elif timecode == 1 : second += timestep

    return _correct_date(year, month, day, hour, minute, second)

@njit 
def _correct_date(year, month, day, hour, minute, second):
    while second >= 60:
        second -= 60
        minute += 1
    while minute >= 60:
        minute -= 60
        hour += 1
    while hour >= 24:
        hour -= 24
        day += 1
    while day > _days_in_month(year, month):
        day -= _days_in_month(year, month)
        month += 1
    while month > 12:
        month -= 12
        year += 1
    return _date_to_bits(year, month, day, hour, minute, second)
    
@njit 
def _days_in_month(year, month):
    if month > 12: month %= 12
    
    if month in (1,3,5,7,8,10,12):
        return 31
    elif month in (4,6,9,11):
        return 30
    elif month == 2:
        if _is_leapyear(year): return 29
        else: return 28

@njit 
def _is_leapyear(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    else:
        return False

@njit
def _date_convert(dates, date_epoch, dt_year, dt_month, dt_day, dt_hour, dt_minute, dt_second):
    converted_dates = []
    for x in dates:
        year, month, day, hour, minute, second = _bits_to_date(x)
        date = date_epoch
        date += (year - 1970) * dt_year
        date += (month - 1) * dt_month
        date += (day - 1) * dt_day
        date += hour * dt_hour
        date += minute * dt_minute
        date += second * dt_second
        converted_dates.append(date)
    return converted_dates

@njit
def _process_groups(iarray, farray, records, offsets, tgroup):
    date_array = [0] #need initialize with a type for numba
    value_array = [0.0]

    for i in range(0,len(records)):
        record = records[i]
        offset = offsets[i]
        index = record * 512 + offset
        pscfwr = iarray[record * 512 + 3] #should be 0 for last record in timeseries 
        current_date = _splitdate(iarray[index])
        group_enddate = _increment_date(current_date, tgroup, 1)
        offset +=1
        index +=1

        while current_date < group_enddate:
            nval, ltstep, ltcode, comp, qual = _splitcontrol(iarray[index])  
            #compressed - only has single value which applies to full range
            if comp == 1:
                for i in range(0, nval, 1):
                    date_array.append(current_date)
                    current_date = _increment_date(current_date, ltcode, ltstep) 
                    value_array.append(farray[index + 1])
                index += 2
                offset +=2
            else:
                for i in range(0, nval, 1):
                    date_array.append(current_date)
                    current_date = _increment_date(current_date, ltcode, ltstep) 
                    value_array.append(farray[index + 1 + i])
                index += 1 + nval
                offset +=1 + nval
            
            if offset >= 511:
                offset = 4
                index = (pscfwr - 1) * 512 + offset
                record = pscfwr
                pscfwr = iarray[(record - 1) * 512 + 3] #should be 0 for last record in timeseries

    date_array = date_array[1:]
    value_array = value_array[1:]

    return date_array, value_array, group_enddate

'''
Get single time series data from a WDM file
based on a collection of attributes (name-value pairs)
'''
def get_wdm_data_set(wdmfile, attributes):
    if attributes == None:
        return None

    search_loc = attributes['location']
    search_cons = attributes['constituent']
    search_dsn = attributes['dsn']

    iarray = np.fromfile(wdmfile, dtype=np.int32)
    farray = np.fromfile(wdmfile, dtype=np.float32)

    if iarray[0] != -998:
        print('Not a WDM file, magic number is not -990. Stopping!')
        return None
    nrecords    = iarray[28]    # first record is File Definition Record
    ntimeseries = iarray[31]

    dsnlist = []
    for index in range(512, nrecords * 512, 512):
        if not (iarray[index]==0 and iarray[index+1]==0 and iarray[index+2]==0 and iarray[index+3]==0) and iarray[index+5]==1:
            dsnlist.append(index)
    if len(dsnlist) != ntimeseries:
        print('PROGRAM ERROR, wrong number of DSN records found')

    summary = []
    summaryindx = []

    # check to see which extra attributes are on each dsn
    columns_to_add = []
    search = ['STAID', 'STNAM', 'SCENARIO', 'CONSTITUENT', 'LOCATION']
    '''
    for att in search:
        found_in_all = True
        for index in dsnlist:
            dattr = {}
            psa = iarray[index + 9]
            if psa > 0:
                sacnt = iarray[index + psa - 1]
            for i in range(psa + 1, psa + 1 + 2 * sacnt, 2):
                id = iarray[index + i]
                ptr = iarray[index + i + 1] - 1 + index
                if id not in attrinfo:
                    continue
                name, atype, length = attrinfo[id]
                if atype == 'I':
                    dattr[name] = iarray[ptr]
                elif atype == 'R':
                    dattr[name] = farray[ptr]
                else:
                    dattr[name] = ''.join([itostr(iarray[k]) for k in range(ptr, ptr + length // 4)]).strip()
            if att not in dattr:
                found_in_all = False
        if found_in_all:
            columns_to_add.append(att)
    '''

    for index in dsnlist:
        # get layout information for TimeSeries Dataset frame
        dsn = iarray[index+4]
        psa = iarray[index+9]
        if psa > 0:
            sacnt = iarray[index+psa-1]
        pdat = iarray[index+10]
        pdatv = iarray[index+11]
        frepos = iarray[index+pdat]

        print(f'{dsn} reading from wdm')

        # get attributes
        dattr = {'TSBDY':1, 'TSBHR':1, 'TSBMO':1, 'TSBYR':1900, 'TFILL':-999.}   # preset defaults
        for i in range(psa+1, psa+1 + 2*sacnt, 2):
            id = iarray[index + i]
            ptr = iarray[index + i + 1] - 1 + index
            if id not in attrinfo:
                # print('PROGRAM ERROR: ATTRIBUTE INDEX not found', id, 'Attribute pointer', iarray[index + i+1])
                continue

            name, atype, length = attrinfo[id]
            if atype == 'I':
                dattr[name] = iarray[ptr]
            elif atype == 'R':
                dattr[name] = farray[ptr]
            else:
                dattr[name] = ''.join([itostr(iarray[k]) for k in range(ptr, ptr + length//4)]).strip()

        if (search_dsn > 0 and search_dsn == dsn):
            pass
        else:
            # could do more attribute based filtering here such as constituent, location etc
            if (search_cons == dattr['TSTYPE']):
                pass
            else:
                continue

        # Get timeseries timebase data
        records = []
        for i in range(pdat+1, pdatv-1):
            a = iarray[index+i]
            if a != 0:
                records.append(splitposition(a))
        if len(records) == 0:
            continue   # WDM preallocation, but nothing saved here yet

        srec, soffset = records[0]
        start = splitdate(iarray[srec*512 + soffset])

        sprec, spoffset = splitposition(frepos)
        finalindex = sprec * 512 + spoffset

        # calculate number of data points in each group, tindex is final index for storage
        tgroup = dattr['TGROUP']
        tstep  = dattr['TSSTEP']
        tcode  = dattr['TCODE']
        cindex = pd.date_range(start=start, periods=len(records)+1, freq=freq[tgroup])
        tindex = pd.date_range(start=start, end=cindex[-1], freq=str(tstep) + freq[tcode])
        counts = np.diff(np.searchsorted(tindex, cindex))

        ## Get timeseries data
        floats = np.zeros(sum(counts),  dtype=np.float32)
        findex = 0
        for (rec,offset),count in zip(records, counts):
            findex = getfloats(iarray, farray, floats, findex, rec, offset, count, finalindex, tcode, tstep)

        ts = pd.Series(floats[:findex], index=tindex[:findex])
        df = pd.DataFrame({'ts': ts})
        return df

    return None

########################
### legacy functions ###
########################

def todatetime(yr=1900, mo=1, dy=1, hr=0):
    '''takes yr,mo,dy,hr information then returns its datetime64'''
    warnings.warn("use '_date_convert' instead; Removed for numba compatible datetime handling; reference commit 1b52a1736e45a497ccdf78cd6e2eab8c0b7a444f", DeprecationWarning)
    if hr == 24:
        return datetime.datetime(yr, mo, dy, 23) + pd.Timedelta(1,'h')
    else:
        return datetime.datetime(yr, mo, dy, hr)

def splitdate(x):
    '''splits WDM int32 DATWRD into year, month, day, hour -> then returns its datetime64'''
    warnings.warn("use '_splitdate' instead; naming updated to indicate internal function", DeprecationWarning)
    return todatetime(x >> 14, x >> 10 & 0xF, x >> 5 & 0x1F, x & 0x1F) # args: year, month, day, hour

def splitcontrol(x):
    ''' splits int32 into (qual, compcode, units, tstep, nvalues)'''
    warnings.warn("use '_splitcontrol' instead; naming updated to indicate internal function", DeprecationWarning)
    return(x & 0x1F, x >> 5 & 0x3, x >> 7 & 0x7, x >> 10 & 0x3F, x >> 16)

def splitposition(x):
    ''' splits int32 into (record, offset), converting to Pyton zero based indexing'''
    warnings.warn("use '_spiltposition' instead; naming updated to indicate internal function", DeprecationWarning)
    return((x>>9) - 1, (x&0x1FF) - 1)

def itostr(i):
    warnings.warn("use '_inttostr' instead; naming updated to indicate internal function; method also handles integer argments so updated name from 'i' to 'int' for additonal clarity", DeprecationWarning)
    return chr(i & 0xFF) + chr(i>>8 & 0xFF) + chr(i>>16 & 0xFF) + chr(i>>24 & 0xFF)

def getfloats(iarray, farray, floats, findex, rec, offset, count, finalindex, tcode, tstep):
    warnings.warn("discontinue use and replace with new 'process_groups' instead; Function replaced by incompatible group/block processing approach; reference commit c5b2a1cdd6a55eccc0db67d7840ec3eaf904dcec .",DeprecationWarning)
    index = rec * 512 + offset + 1
    stop = (rec + 1) * 512
    cntr = 0
    while cntr < count and findex < len(floats):
        if index == stop -1 :
            print ('Problem?',str(rec)) #perhaps not, block cannot start at word 512 of a record because not spot for values
        if index >= stop-1:
            rec = iarray[rec * 512 + 3] - 1  # 3 is forward data pointer, -1 is python indexing
            print ('Process record ',str(rec))
            index = rec * 512 + 4  # 4 is index of start of new data
            stop = (rec + 1) * 512

        x = iarray[index]  # block control word or maybe date word at start of group
        nval = x >> 16
        ltstep = x >> 10 & 0x3f
        ltcode = x >> 7 & 0x7
        comp = x >> 5 & 0x3
        qual  = x & 0x1f
        ldate = todatetime() # dummy
        if ltstep != tstep or ltcode != tcode:
            nval = adjustNval(ldate, ltstep, tstep, ltcode, tcode, comp, nval)
            if nval == -1:  #unable to convert block
                try:
                    ldate = splitdate(x)
                    if isinstance(ldate,datetime.date):
                       print('Problem resolved - date found ',ldate)
                       nval = 1
                    else:
                        print('BlockConversionFailure at ', str(rec + 1), ' ', str(index % 512))
                except:
                  print ('BlockConversionFailure at ',str(rec+ 1),' ',str(index % 512))
                # try next word
                comp = -1
        index += 1
        if comp == 0:
            for k in range(nval):
                if findex >= len(floats):
                    return findex
                floats[findex] = farray[index + k]
                findex += 1
            index += nval
        elif comp > 0:
            for k in range(nval):
                if findex >= len(floats):
                    return findex
                floats[findex] = farray[index]
                findex += 1
            index += 1
        cntr += nval
    return findex

def adjustNval(ldate, ltstep, tstep, ltcode, tcode, comp, nval):
    warnings.warn("supporting function for deprecated 'get_floats' function;", DeprecationWarning)
    lnval = nval
    if comp != 1:
        nval = -1  # only can adjust compressed data
    else:
        if tcode == 2:  # minutes
            if ltcode == 6:  # from years
                if leap_year(ldate.year):
                    ldays = 366
                else:
                    ldays = 365
                nval = ldays * lnval * 60 /ltstep
            elif ltcode == 5: # from months
                from calendar import monthrange
                ldateRange = monthrange(ldate.year, ldate.month)
                print ('month block ', ldateRange)
                nval = ldateRange[2] * lnval * 60/ ltstep
            elif ltcode == 4:  # from days
                nval = lnval * 60 / ltstep
            elif ltcode == 3:  # from hours
                nval = lnval * 1440 / ltstep
            else:  # dont know how to convert
                nval = -1
        elif tcode == 3:  # hours
            if ltcode == 6:  # from years
                nval = -1
            elif ltcode == 5: # from months
                nval = -1
            elif ltcode == 4:  # from days
                nval = lnval * 24 / ltstep
            else:  # dont know how to convert
                nval = -1
        else:
            nval = -1  # dont know how to convert

    nval = int(nval)
    if nval == -1:  # conversion problem
        print('Conversion problem (tcode ', str(tcode), ', ', str(ltcode), '), (tstep ', str(tstep), ',', str(ltstep), '), (comp ', str(comp), ')')
    else:
        print('Conversion complete (tcode ', str(tcode), ', ', str(ltcode), '), (tstep ', str(tstep), ',', str(ltstep), '), (nval ',
              str(nval) + ',', str(lnval), ')')

    return nval
