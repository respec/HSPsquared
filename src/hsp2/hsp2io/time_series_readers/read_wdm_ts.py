"""
Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
License: LGPL2
"""

# Future imports
from __future__ import annotations

# Standard library imports
import datetime
import urllib.request
from pathlib import Path
from typing import Literal

# Third party imports
import numpy as np
import pandas as pd
from numba import njit

# First party imports
from hsp2.hsp2.utilities import pandas_offset_by_version
from hsp2.hsp2io.protocols import Category

# look up attributes NAME, data type (Integer; Real; String) and data length by attribute number
attrinfo = {
    1: ("TSTYPE", "S", 4),
    2: ("STAID", "S", 16),
    11: ("DAREA", "R", 1),
    17: ("TCODE", "I", 1),
    27: ("TSBYR", "I", 1),
    28: ("TSBMO", "I", 1),
    29: ("TSBDY", "I", 1),
    30: ("TSBHR", "I", 1),
    32: ("TFILL", "R", 1),
    33: ("TSSTEP", "I", 1),
    34: ("TGROUP", "I", 1),
    45: ("STNAM", "S", 48),
    83: ("COMPFG", "I", 1),
    84: ("TSFORM", "I", 1),
    85: ("VBTIME", "I", 1),
    444: ("DATMOD", "S", 12),
    443: ("DATCRE", "S", 12),
    22: ("DCODE", "I", 1),
    10: ("DESCRP", "S", 80),
    7: ("ELEV", "R", 1),
    8: ("LATDEG", "R", 1),
    9: ("LNGDEG", "R", 1),
    288: ("SCENARIO", "S", 8),
    289: ("CONSTITUENT", "S", 8),
    290: ("LOCATION", "S", 8),
}

freq = {
    7: f"100{pandas_offset_by_version('YS')}",
    6: f"{pandas_offset_by_version('YS')}",
    5: f"{pandas_offset_by_version('MS')}",
    4: "D",
    3: "h",
    2: "min",
    1: "S",
}  # pandas date_range() frequency by TCODE, TGROUP


def read_wdm_ts(
    wdmfile: str | Path,
    category: Category | Literal["RESULT", "INPUT"] = Category.INPUTS,
    operation: str | None = None,
    segment: str | None = None,
    activity: str | None = None,
    dsns=None,
    ts_number_shift=0,
):
    """
    Read time series data from a WDM file.

    Parameters
    ----------
    wdmfile : str
        Local path to the WDM file or http:// or https:// URI.  If a URI the
        WDM file will be downloaded to a temporary file and deleted after use.
    category : Category or str
        Category of the time series data to read. Currently for WDM files this
        can only be category.INPUTS or "INPUT".
    operation : str, optional
        Not used for WDM files, included to have identical function signature
        compatibility with other time series readers. Default is None.
    segment : str, optional
        The TSXXXXXX ID of the time series data to read. The XXXXXX is
        converted to a DSN number and used to find the correct series from the
        WDM file. Default is None, which then uses the `dsns` and
        `ts_number_shift` parameters to find and return the correct time
        series.  In the returned dictionary `segment` is used as the key for
        the time series data.  Default is None, which generates keys in the
        returned dictionary as "TS" followed by the `ts_number_shift` + `dsns`
        numbers.
    activity : str, optional
        Not used for WDM files, included to have identical function signature
        compatibility with other time series readers. Default is None.
    dsns : int or list of int, optional
        Data set numbers to read. If None, all time series data sets are read.
        Default is None.  Overridden by `segment` if provided, which is
        converted to a DSN number and `ts_offset_number` and is used to find
        the correct series from the WDM file.
    ts_number_shift : int, optional
        Shift to apply to the WDM DSN numbers when constructing dataset names.
        Default is 0.  For WDM/WDM1 files in the UCI file the offset is 100000,
        WDM2 is 200000, WDM3 is 300000, WDM4 is 400000.  This allows the
        dataset names to be consistent with the UCI file when multiple WDM
        files are used.

    Returns
    -------
    dict
        A dictionary where keys are dataset names (e.g., "TS100001") and values
        are pandas Series containing the time series data.
    """
    if category not in (Category.INPUTS, "INPUT"):
        raise ValueError(
            f"Unsupported category '{category}' for WDM files. Only 'INPUT' is supported."
        )

    if segment:
        if not isinstance(segment, str) or not segment.startswith("TS"):
            raise ValueError("segment must be a string starting with 'TS'")
        dsns = [int(segment[2:]) % 100000]
        ts_number_shift = int(segment[2]) * 100000
    wdmfile = str(wdmfile)

    delete_tmp_file = False
    if wdmfile.startswith(("http://", "https://", "ftp://", "ftps://", "s3://")):
        delete_tmp_file = True
        wdmfile, _ = urllib.request.urlretrieve(wdmfile)

    iarray = np.fromfile(wdmfile, dtype=np.int32)
    farray = np.fromfile(wdmfile, dtype=np.float32)

    date_epoch = np.datetime64(0, "Y")
    dt_year = np.timedelta64(1, "Y")
    dt_month = np.timedelta64(1, "M")
    dt_day = np.timedelta64(1, "D")
    dt_hour = np.timedelta64(1, "h")
    dt_minute = np.timedelta64(1, "m")
    dt_second = np.timedelta64(1, "s")

    if iarray[0] != -998:
        raise ValueError(
            "Provided file does not match WDM format. First int32 should be -998."
        )
    ntimeseries = iarray[31]

    if dsns is None:
        dsnlist: list[int] = []
        nrecords = iarray[28]  # first record is File Definition Record
        dsnlist.extend(
            index
            for index in range(512, nrecords * 512, 512)
            if (
                iarray[index] != 0
                or iarray[index + 1] != 0
                or iarray[index + 2] != 0
                or iarray[index + 3] != 0
            )
            and iarray[index + 5] == 1
        )
        if len(dsnlist) != ntimeseries:
            raise RuntimeError(
                f"Wrong number of Time Series Records found expecting:{ntimeseries} found:{len(dsnlist)}"
            )
    else:
        if isinstance(dsns, int):
            dsns = [dsns]
        elif not isinstance(dsns, (list, tuple)) or not all(
            isinstance(dsn, int) for dsn in dsns
        ):
            raise ValueError("dsns must be an int or list/tuple of ints")
        dsnlist = dsns

    # check to see which extra attributes are on each dsn
    columns_to_add = []
    search = ["STAID", "STNAM", "SCENARIO", "CONSTITUENT", "LOCATION"]
    for att in search:
        found_in_all = True
        for index in dsnlist:
            dattr = {}
            psa = iarray[index + 9]
            if psa > 0:
                sacnt = iarray[index + psa - 1]
            for i in range(psa + 1, psa + 1 + 2 * sacnt, 2):
                lid = iarray[index + i]
                ptr = iarray[index + i + 1] - 1 + index
                if lid not in attrinfo:
                    continue
                name, atype, length = attrinfo[lid]
                if atype == "I":
                    dattr[name] = iarray[ptr]
                elif atype == "R":
                    dattr[name] = farray[ptr]
                else:
                    dattr[name] = "".join(
                        [_inttostr(iarray[k]) for k in range(ptr, ptr + length // 4)]
                    ).strip()
            if att not in dattr:
                found_in_all = False
        if found_in_all:
            columns_to_add.append(att)

    summary = []
    summaryindx = []
    collect = {}
    for index in dsnlist:
        # get layout information for TimeSeries Dataset frame
        dsn = iarray[index + 4]
        psa = iarray[index + 9]
        if psa > 0:
            sacnt = iarray[index + psa - 1]
        pdat = iarray[index + 10]
        pdatv = iarray[index + 11]

        print(f"{dsn} reading from wdm")
        # get attributes
        dattr = {
            "TSBDY": 1,
            "TSBHR": 1,
            "TSBMO": 1,
            "TSBYR": 1900,
            "TFILL": -999.0,
        }  # preset defaults
        for i in range(psa + 1, psa + 1 + 2 * sacnt, 2):
            lid = iarray[index + i]
            ptr = iarray[index + i + 1] - 1 + index
            if lid not in attrinfo:
                # print('PROGRAM ERROR: ATTRIBUTE INDEX not found', id, 'Attribute pointer', iarray[index + i+1])
                continue

            name, atype, length = attrinfo[lid]
            if atype == "I":
                dattr[name] = iarray[ptr]
            elif atype == "R":
                dattr[name] = farray[ptr]
            else:
                dattr[name] = "".join(
                    [_inttostr(iarray[k]) for k in range(ptr, ptr + length // 4)]
                ).strip()

        # Get timeseries timebase data
        records = []
        offsets = []
        for i in range(pdat + 1, pdatv - 1):
            a = iarray[index + i]
            if a != 0:
                record, offset = _splitposition(a)
                records.append(record)
                offsets.append(offset)
        if not records:
            continue

        # calculate number of data points in each group, tindex is final index
        # for storage
        tgroup = dattr["TGROUP"]
        tstep = dattr["TSSTEP"]
        tcode = dattr["TCODE"]

        records = np.asarray(records)
        offsets = np.asarray(offsets)

        dates, values, stop_datetime = _process_groups(
            iarray, farray, records, offsets, tgroup
        )
        stop_datetime = datetime.datetime(*_bits_to_date(stop_datetime))
        dates = np.array(dates)
        dates_converted = _date_convert(
            dates,
            date_epoch,
            dt_year,
            dt_month,
            dt_day,
            dt_hour,
            dt_minute,
            dt_second,
        )
        series = pd.Series(values, index=dates_converted)
        try:
            series.index.freq = str(tstep) + freq[tcode]
        except ValueError:
            series.index.freq = None

        dsname = f"TS{dsn + ts_number_shift:03d}"
        collect[dsname] = series

        data = [
            str(series.index[0]),
            str(stop_datetime),
            str(tstep) + freq[tcode],
            len(series),
            dattr["TSTYPE"],
            dattr["TFILL"],
        ]
        columns = ["Start", "Stop", "Freq", "Length", "TSTYPE", "TFILL"]
        for x in columns_to_add:
            if x in dattr:
                data.append(dattr[x])
                columns.append(x)

        summary.append(data)
        summaryindx.append(dsname)

    collect["SUMMARY"] = pd.DataFrame(summary, index=summaryindx, columns=columns)

    if delete_tmp_file:
        Path(wdmfile).unlink(missing_ok=True)
    return collect


@njit
def _splitdate(x):
    year = np.int64(x >> 14)
    month = np.int64(x >> 10 & 0xF)
    day = np.int64(x >> 5 & 0x1F)
    hour = np.int64(x & 0x1F)
    return _correct_date(year, month, day, hour, 0, 0)


@njit
def _splitcontrol(x):
    nval = x >> 16
    ltstep = x >> 10 & 0x3F
    ltcode = x >> 7 & 0x7
    comp = x >> 5 & 0x3
    qual = x & 0x1F
    return nval, ltstep, ltcode, comp, qual


@njit
def _splitposition(x):
    return ((x >> 9) - 1, (x & 0x1FF) - 1)  # args: record, offset


@njit
def _inttostr(i):
    return (
        chr(i & 0xFF) + chr(i >> 8 & 0xFF) + chr(i >> 16 & 0xFF) + chr(i >> 24 & 0xFF)
    )


@njit
def _bits_to_date(x):
    year = x >> 26
    month = x >> 22 & 0xF
    day = x >> 17 & 0x1F
    hour = x >> 12 & 0x1F
    minute = x >> 6 & 0x3F
    second = x & 0x3F
    return year, month, day, hour, minute, second


@njit
def _date_to_bits(year, month, day, hour, minute, second):
    return year << 26 | month << 22 | day << 17 | hour << 12 | minute << 6 | second


@njit
def _increment_date(date, timecode, timestep):
    year, month, day, hour, minute, second = _bits_to_date(date)

    if timecode == 7:
        year += 100 * timestep
    elif timecode == 6:
        year += timestep
    elif timecode == 5:
        month += timestep
    elif timecode == 4:
        day += timestep
    elif timecode == 3:
        hour += timestep
    elif timecode == 2:
        minute += timestep
    elif timecode == 1:
        second += timestep

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
    if month > 12:
        month %= 12

    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if _is_leapyear(year) else 28


@njit
def _is_leapyear(year):
    if year % 400 == 0:
        return True
    return False if year % 100 == 0 else year % 4 == 0


@njit
def _date_convert(
    dates, date_epoch, dt_year, dt_month, dt_day, dt_hour, dt_minute, dt_second
):
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
    date_array = [0]  # need initialize with a type for numba
    value_array = [0.0]

    for i in range(len(records)):
        record = records[i]
        offset = offsets[i]
        index = record * 512 + offset
        pscfwr = iarray[record * 512 + 3]  # should be 0 for last record in timeseries
        current_date = _splitdate(iarray[index])
        group_enddate = _increment_date(current_date, tgroup, 1)
        offset += 1
        index += 1

        while current_date < group_enddate:
            nval, ltstep, ltcode, comp, _qual = _splitcontrol(iarray[index])
            # compressed - only has single value which applies to full range
            if comp == 1:
                for i in range(nval):
                    date_array.append(current_date)
                    current_date = _increment_date(current_date, ltcode, ltstep)
                    value_array.append(farray[index + 1])
                index += 2
                offset += 2
            else:
                for i in range(nval):
                    date_array.append(current_date)
                    current_date = _increment_date(current_date, ltcode, ltstep)
                    value_array.append(farray[index + 1 + i])
                index += 1 + nval
                offset += 1 + nval

            if offset >= 511:
                offset = 4
                index = (pscfwr - 1) * 512 + offset
                record = pscfwr
                pscfwr = iarray[
                    (record - 1) * 512 + 3
                ]  # should be 0 for last record in timeseries

    date_array = date_array[1:]
    value_array = value_array[1:]

    return date_array, value_array, group_enddate
