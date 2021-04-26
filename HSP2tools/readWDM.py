''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
License: LGPL2
'''

import numpy as np
import pandas as pd
from numba import jit, njit
import datetime
from dateutil.relativedelta import relativedelta
import timeit

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

freq = {7:'100YS', 6:'YS', 5:'MS', 4:'D', 3:'H', 2:'min', 1:'S'}   # pandas date_range() frequency by TCODE, TGROUP


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
            stop_datetime = datetime.datetime(*bits_to_date(stop_datetime))
            dates = np.array(dates)
            dates_converted = date_convert(dates, date_epoch, dt_year, dt_month, dt_day, dt_hour, dt_minute, dt_second)
            series = pd.Series(values, index=dates_converted)
            series.index.freq = str(tstep) + freq[tcode] 

            dsname = f'TIMESERIES/TS{dsn:03d}'
            if compress_output:
                series.to_hdf(store, dsname, complib='blosc', complevel=9)  
            else:
                series.to_hdf(store, dsname, format='t', data_columns=True)

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
    return correct_date(year, month, day, hour, 0,0)

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
def bits_to_date(x):
    year = x >> 26
    month = x >> 22 & 0xf
    day = x >> 17 & 0x1f
    hour = x >> 12 & 0x1f
    minute = x >> 6 & 0x3f
    second = x & 0x3f
    return year, month, day, hour, minute, second

@njit 
def date_to_bits(year, month, day, hour, minute, second):
    x = year << 26 | month << 22 | day << 17 | hour << 12 | minute << 6 | second 
    return x

@njit 
def increment_date(date, timecode, timestep):
    year, month, day, hour, minute, second = bits_to_date(date)
    
    if timecode == 7: year += 100 * timestep
    elif timecode == 6 : year += timestep
    elif timecode == 5 : month += timestep
    elif timecode == 4 : day += timestep
    elif timecode == 3 : hour += timestep
    elif timecode == 2 : minute += timestep
    elif timecode == 1 : second += timestep

    return correct_date(year, month, day, hour, minute, second)

@njit 
def correct_date(year, month, day, hour, minute, second):
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
    return date_to_bits(year, month, day, hour, minute, second)
    
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
def date_convert(dates, date_epoch, dt_year, dt_month, dt_day, dt_hour, dt_minute, dt_second):
    converted_dates = []
    for x in dates:
        year, month, day, hour, minute, second = bits_to_date(x)
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
        group_enddate = increment_date(current_date, tgroup, 1)
        offset +=1
        index +=1

        while current_date < group_enddate:
            nval, ltstep, ltcode, comp, qual = _splitcontrol(iarray[index])  
            #compressed - only has single value which applies to full range
            if comp == 1:
                for i in range(0, nval, 1):
                    date_array.append(current_date)
                    current_date = increment_date(current_date, ltcode, ltstep) 
                    value_array.append(farray[index + 1])
                index += 2
                offset +=2
            else:
                for i in range(0, nval, 1):
                    date_array.append(current_date)
                    current_date = increment_date(current_date, ltcode, ltstep) 
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
