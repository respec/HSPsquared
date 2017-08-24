#!/usr/bin/env python2

from __future__ import print_function

'''
The WDM class supplies a series of utilities for working with WDM files
with Python.  The class uses f2py to wrap the minimally necessary WDM
routines.
'''

import datetime
import os
import os.path
import re
import sys

import pandas as pd

from . import wdm

# Load in WDM subroutines

# Mapping between WDM TCODE and pandas interval code
MAPTCODE = {
    1: 'S',
    2: 'T',
    3: 'H',
    4: 'D',
    5: 'M',
    6: 'A',
    }

MAPFREQ = {
    'S': 1,
    'T': 2,
    'H': 3,
    'D': 4,
    'M': 5,
    'A': 6,
    }


class WDMError(Exception):
    pass


class DSNDoesNotExist(Exception):
    def __init__(self, dsn):
        self.dsn = dsn

    def __str__(self):
        if self.dsn < 1 or self.dsn > 32000:
            return '''
*
*   The DSN number must be >= 1 and <= 32000.
*   You supplied {0}.
*
'''.format(self.dsn)

        return '''
*
*   The DSN {0} does not exist in the dataset.
*
'''.format(self.dsn)


class LibraryNotFoundError(Exception):
    pass


class WDMFileExists(Exception):
    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return '''
*
*   File {0} exists.
*
'''.format(self.filename)


class DSNExistsError(Exception):
    def __init__(self, dsn):
        self.dsn = dsn

    def __str__(self):
        return '''
*
*   DSN {0} exists.
*
'''.format(self.dsn)


class WDM():
    ''' Class to open and read from WDM files.
    '''
    def __init__(self):

        # timcvt: Convert times to account for 24 hour
        # timdif: Time difference
        # wdmopn: Open WDM file
        # wdbsac: Set string attribute
        # wdbsai: Set integer attribute
        # wdbsar: Set real attribute
        # wdbckt: Check if DSN exists
        # wdflcl: Close WDM file
        # wdlbax: Create label for new DSN
        # wdtget: Get time-series data
        # wdtput: Write time-series data
        # wddsrn: Renumber a DSN
        # wddsdl: Delete a DSN
        # wddscl: Copy a label

        self.timcvt = wdm.timcvt
        self.timdif = wdm.timdif
        self.wdbopn = wdm.wdbopn
        self.wdbsac = wdm.wdbsac
        self.wdbsai = wdm.wdbsai
        self.wdbsar = wdm.wdbsar
        self.wdbsgc = wdm.wdbsgc
        self.wdbsgi = wdm.wdbsgi
        self.wdbsgr = wdm.wdbsgr
        self.wdckdt = wdm.wdckdt
        self.wdflcl = wdm.wdflcl
        self.wdlbax = wdm.wdlbax
        self.wdtget = wdm.wdtget
        self.wdtput = wdm.wdtput
        self.wtfndt = wdm.wtfndt
        self.wddsrn = wdm.wddsrn
        self.wddsdl = wdm.wddsdl
        self.wddscl = wdm.wddscl

        self.openfiles = {}

    def wmsgop(self):
        # WMSGOP is a simple open of the message file
        afilename = os.path.join(sys.prefix,
                                 'share',
                                 'wdmtoolbox',
                                 'message.wdm')
        return self._open(afilename, 100, ronwfg=1)

    def dateconverter(self, datestr):
        words = re.findall(r'\d+', str(datestr))
        words = [int(i) for i in words]
        dtime = [1900, 1, 1, 0, 0, 0]
        dtime[:len(words)] = words
        return dtime

    def _open(self, wdname, wdmsfl, ronwfg=0):
        ''' Private method to open WDM file.
        '''
        if wdname not in self.openfiles:
            wdname = wdname.strip()
            if ronwfg == 1:
                if not os.path.exists(wdname):
                    raise ValueError('''
*
*   Trying to open
*   {0}
*   in read-only mode and it cannot be found.
*
    '''.format(wdname))
            retcode = self.wdbopn(wdmsfl,
                                  wdname,
                                  ronwfg)
            self._retcode_check(retcode, additional_info='wdbopn')
            self.openfiles[wdname] = wdmsfl
        return wdmsfl

    def _retcode_check(self, retcode, additional_info=' '):
        retcode_dict = {
            -1: 'non specific error on WDM file open',
            -4: 'copy/update failed due to data overlap problem - part of source needed',
            -5: 'copy/update failed due to data overlap problem',
            -6: 'no data present',
            -8: 'bad dates',
            -9: 'data present in current group',
            -10: 'no date in this group',
            -11: 'no non-missing data, data has not started yet',
            -14: 'data specified not within valid range for data set',
            -15: 'time units and time step must match label exactly with VBTIME = 1',
            -20: 'problem with on or more of GPGLG, DXX, NVAL, QUALVL, LTSTEP, LTUNIT',
            -21: 'data from WDM does not match expected date',
            -23: 'not a valid table',
            -24: 'not a valid associated table',
            -25: 'template already exists',
            -26: 'can not add another table',
            -27: 'no tables to return info about',
            -28: 'table does not exist yet',
            -30: 'more than whole table',
            -31: 'more than whole extension',
            -32: 'data header does not match',
            -33: 'problems with row/space specs',
            -36: 'missing needed following data for a get',
            -37: 'no data present',
            -38: 'missing part of time required',
            -39: 'missing data group',
            -40: 'no data available',
            -41: 'no data to read',
            -42: 'overlap in existing group',
            -43: 'can not add another space time group',
            -44: 'trying to get/put more data that in block',
            -45: 'types do not match',
            -46: 'bad space time group specification parameter',
            -47: 'bad direction flag',
            -48: 'conflicting spec of space time dim and number of timeseries data sets',
            -49: 'group does not exist',
            -50: 'requested attributes missing from this data set',
            -51: 'no space for another DLG',
            -61: 'old data set does not exist',
            -62: 'new data set already exists',
            -71: 'data set already exists',
            -72: 'old data set does not exist',
            -73: 'new data set already exists',
            -81: 'data set does not exists',
            -82: 'data set exists, but is wrong DSTYP',
            -83: 'WDM file already open, can not create it',
            -84: 'data set number out of valid range',
            -85: 'trying to write to a read-only data set',
            -87: 'can not remove message WDM file from buffer',
            -88: 'can not open another WDM file',
            -89: 'check digit on 1st record of WDM file is bad',
            -101: 'incorrect character value for attribute',
            -102: 'attribute already on label',
            -103: 'no room on label for attribute',
            -104: 'data present, can not update attribute',
            -105: 'attribute not allowed for this type data set',
            -106: 'can not delete attribute, it is required',
            -107: 'attribute not present on this data set',
            -108: 'incorrect integer value for attribute',
            -109: 'incorrect real value for attribute',
            -110: 'attributes not found on message file',
            -111: 'attribute name not found (no match)',
            -112: 'more attributes exists which match SAFNAM',
            -121: 'no space for another attribute',
            1: 'varies - generally more data/groups/table',
            2: 'no more data available for this DLG group'
            }

        if retcode in retcode_dict:
            raise WDMError('''
*
*   WDM library function returned error code {0}. {1}
*   WDM error: {2}
*
'''.format(retcode, additional_info, retcode_dict[retcode]))
        if retcode != 0:
            raise WDMError('''
*
*   WDM library function returned error code {0}. {1}
*
'''.format(retcode, additional_info))

    def renumber_dsn(self, wdmpath, odsn, ndsn):
        odsn = int(odsn)
        ndsn = int(ndsn)

        wdmfp = self._open(wdmpath, 101)
        retcode = self.wddsrn(
            wdmfp,
            odsn,
            ndsn)
        self._close(wdmpath)
        self._retcode_check(retcode, additional_info='wddsrn')

    def delete_dsn(self, wdmpath, dsn):
        dsn = int(dsn)

        wdmfp = self._open(wdmpath, 101)
        if self.wdckdt(wdmfp, dsn) != 0:
            retcode = self.wddsdl(
                wdmfp,
                dsn)
            self._retcode_check(retcode, additional_info='wddsdl')
        self._close(wdmpath)

    def copydsnlabel(self, inwdmpath, indsn, outwdmpath, outdsn):
        assert inwdmpath != outwdmpath
        indsn = int(indsn)
        outdsn = int(outdsn)
        dsntype = 0
        inwdmfp = self._open(inwdmpath, 101)
        outwdmfp = self._open(outwdmpath, 102)
        retcode = self.wddscl(
            inwdmfp,
            indsn,
            outwdmfp,
            outdsn,
            dsntype)
        self._close(inwdmpath)
        self._close(outwdmpath)
        self._retcode_check(retcode, additional_info='wddscl')

    #Added by RTH on 4/1/2017
    def exists_dsn(self, wdmpath, dsn):
        wdmfp = self._open(wdmpath, 101, ronwfg=1)
        value = self.wdckdt(wdmfp, dsn)
        self._close(wdmpath)
        return value


    def describe_dsn(self, wdmpath, dsn):
        wdmfp = self._open(wdmpath, 101, ronwfg=1)
        if self.wdckdt(wdmfp, dsn) == 0:
            raise DSNDoesNotExist(dsn)

        tdsfrc, llsdat, lledat, retcode = self.wtfndt(
            wdmfp,
            dsn,
            1)  # GPFLG  - get(1)/put(2) flag
        # Ignore retcode == -6 which means that the dsn doesn't have any data.
        # It it is a new dsn, of course it doesn't have any data.
        if retcode == -6:
            retcode = 0
        self._retcode_check(retcode, additional_info='wtfndt')

        tstep, retcode = self.wdbsgi(
            wdmfp,
            dsn,
            33,  # saind = 33 for time step
            1)   # salen
        self._retcode_check(retcode, additional_info='wdbsgi')

        tcode, retcode = self.wdbsgi(
            wdmfp,
            dsn,
            17,  # saind = 17 for time code
            1)   # salen
        self._retcode_check(retcode, additional_info='wdbsgi')

        tsfill, retcode = self.wdbsgr(
            wdmfp,
            dsn,
            32,  # saind = 32 for tsfill
            1)   # salen
        # retcode = -107 if attribute not present
        if retcode == -107:
            # Since I use tsfill if not found will set to default.
            tsfill = -999.0
            retcode = 0
        else:
            tsfill = tsfill[0]
        self._retcode_check(retcode, additional_info='wdbsgr')

        ostr, retcode = self.wdbsgc(
            wdmfp,
            dsn,
            290,    # saind = 290 for location
            8)      # salen
        if retcode == -107:
            ostr = 'NA'
            retcode = 0
        self._retcode_check(retcode, additional_info='wdbsgr')

        scen_ostr, retcode = self.wdbsgc(
            wdmfp,
            dsn,
            288,    # saind = 288 for scenario
            8)      # salen
        if retcode == -107:
            scen_ostr = 'NA'
            retcode = 0
        self._retcode_check(retcode, additional_info='wdbsgr')

        con_ostr, retcode = self.wdbsgc(
            wdmfp,
            dsn,
            289,    # saind = 289 for constitiuent
            8)      # salen
        if retcode == -107:
            con_ostr = 'NA'
            retcode = 0
        self._retcode_check(retcode, additional_info='wdbsgr')

        base_year, retcode = self.wdbsgi(
            wdmfp,
            dsn,
            27,  # saind = 27 for base_year
            1)   # salen
        self._retcode_check(retcode, additional_info='wdbsgi')

        self._close(wdmpath)

        self.timcvt(llsdat)
        self.timcvt(lledat)
        try:
            sdate = datetime.datetime(*llsdat).isoformat()
        except ValueError:
            sdate = None
        try:
            edate = datetime.datetime(*lledat).isoformat()
        except ValueError:
            edate = None

        tstep = tstep[0]
        tcode = tcode[0]
        base_year = base_year[0]

        try:
            ostr = str(ostr, "utf-8")
            scen_ostr = str(scen_ostr, "utf-8")
            con_ostr = str(con_ostr, "utf-8")
        except TypeError:
            ostr = ''.join(ostr)
            scen_ostr = ''.join(scen_ostr)
            con_ostr = ''.join(con_ostr)

        return {'dsn':         dsn,
                'start_date':  sdate,
                'end_date':    edate,
                'llsdat':      llsdat,
                'lledat':      lledat,
                'tstep':       tstep,
                'tcode':       tcode,
                'tcode_name':  MAPTCODE[tcode],
                'location':    ostr.strip(),
                'scenario':    scen_ostr.strip(),
                'constituent': con_ostr.strip(),
                'tsfill':      tsfill,
                'freq':        MAPTCODE[tcode],
                'base_year':   base_year}

    def create_new_wdm(self, wdmpath, overwrite=False):
        ''' Create a new WDM fileronwfg
        '''
        if overwrite and os.path.exists(wdmpath):
            os.remove(wdmpath)
        elif os.path.exists(wdmpath):
            raise WDMFileExists(wdmpath)
        ronwfg = 2
        wdmfp = self._open(wdmpath, 101, ronwfg=ronwfg)
        self._close(wdmpath)

    def create_new_dsn(self, wdmpath, dsn, tstype='', base_year=1900, tcode=4,
                       tsstep=1, statid=' ', scenario='', location='',
                       description='', constituent='', tsfill=-999.0):
        ''' Create self.wdmfp/dsn. '''
        wdmfp = self._open(wdmpath, 101)
        messfp = self.wmsgop()

        if self.wdckdt(wdmfp, dsn) == 1:
            raise DSNExistsError(dsn)

        # Parameters for wdlbax taken from ATCTSfile/clsTSerWDM.cls
        psa = self.wdlbax(
            wdmfp,
            dsn,
            1,    # DSTYPE - always 1 for time series
            10,   # NDN    - number of down pointers
            10,   # NUP    - number of up pointers
            30,   # NSA    - number of search attributes
            100,  # NSASP  - amount of search attribute space
            300)  # NDP    - number of data pointers
                  # PSA    - pointer to search attribute space

        for saind, salen, saval in [(34, 1, 6),  # tgroup
                                    (83, 1, 1),  # compfg
                                    (84, 1, 1),  # tsform
                                    (85, 1, 1),  # vbtime
                                    (17, 1, int(tcode)),  # tcode
                                    (33, 1, int(tsstep)),  # tsstep
                                    (27, 1, int(base_year)),  # tsbyr
                                    ]:
            retcode = self.wdbsai(
                wdmfp,
                dsn,
                messfp,
                saind,
                salen,
                saval)
            self._retcode_check(retcode, additional_info='wdbsai')

        for saind, salen, saval in [(32, 1, tsfill)]:  # tsfill
            retcode = self.wdbsar(
                wdmfp,
                dsn,
                messfp,
                saind,
                salen,
                saval)
            self._retcode_check(retcode, additional_info='wdbsar')

        for saind, salen, saval, error_name in [
            (2, 16, statid, 'Station ID'),
            (1, 4, tstype.upper(), 'Time series type - tstype'),
            (45, 48, description.upper(), 'Description'),
            (288, 8, scenario.upper(), 'Scenario'),
            (289, 8, constituent.upper(), 'Constituent'),
            (290, 8, location.upper(), 'Location'),
                ]:
            saval = saval.strip()
            if len(saval) > salen:
                raise ValueError('''
*
*   String "{0}" is too long for {1}.  Must
*   have a length equal or less than {2}.
*
'''.format(saval, error_name, salen))

            saval = '{0: <{1}}'.format(saval, salen)

            retcode = self.wdbsac(
                wdmfp,
                dsn,
                messfp,
                saind,
                salen,
                saval)
            self._retcode_check(retcode, additional_info='wdbsac')
        self._close(wdmpath)

    def _tcode_date(self, tcode, date):
        ''' Uses tcode to set the significant parts of the date tuple. '''
        rdate = [1, 1, 1, 0, 0, 0]
        if tcode <= 6:
            rdate[0] = date[0]
        if tcode <= 5:
            rdate[1] = date[1]
        if tcode <= 4:
            rdate[2] = date[2]
        if tcode <= 3:
            rdate[3] = date[3]
        if tcode <= 2:
            rdate[4] = date[4]
        if tcode <= 1:
            rdate[5] = date[5]
        return rdate

    def write_dsn(self, wdmpath, dsn, data, start_date):
        ''' Write to self.wdmfp/dsn the time-series data. '''
        dsn_desc = self.describe_dsn(wdmpath, dsn)
        tcode = dsn_desc['tcode']
        tstep = dsn_desc['tstep']

        dstart_date = start_date.timetuple()[:6]
        llsdat = self._tcode_date(tcode, dstart_date)
        if dsn_desc['base_year'] > llsdat[0]:
            raise ValueError('''
*
*   The base year for this DSN is {0}.  All data to insert must be after the
*   base year.  Instead the first year of the series is {1}.
*
'''.format(dsn_desc['base_year'], llsdat[0]))

        nval = len(data)

        wdmfp = self._open(wdmpath, 101)
        retcode = self.wdtput(
            wdmfp,
            dsn,
            tstep,
            llsdat,
            nval,
            1,
            0,
            tcode,
            data)
        self._close(wdmpath)
        self._retcode_check(retcode, additional_info='wdtput')

    def read_dsn(self, wdmpath, dsn, start_date=None, end_date=None):
        ''' Read from a DSN.
        '''

        # Call wdatim_ to get LLSDAT, LLEDAT, TSTEP, TCODE
        desc_dsn = self.describe_dsn(wdmpath, dsn)

        llsdat = desc_dsn['llsdat']
        lledat = desc_dsn['lledat']
        tcode = desc_dsn['tcode']
        tstep = desc_dsn['tstep']

        # These calls convert 24 to midnight of the next day
        self.timcvt(llsdat)
        self.timcvt(lledat)

        if start_date is not None:
            start_date = self.dateconverter(start_date)
            llsdat = start_date
        if end_date is not None:
            end_date = self.dateconverter(end_date)
            lledat = end_date

        # Determine the number of values (ITERM) from LLSDAT to LLEDAT
        iterm = self.timdif(
            llsdat,
            lledat,
            tcode,
            tstep)

        dtran = 0
        qualfg = 30
        # Get the data and put it into dictionary
        wdmfp = self._open(wdmpath, 101, ronwfg=1)
        dataout, retcode = self.wdtget(
            wdmfp,
            dsn,
            tstep,
            llsdat,
            iterm,
            dtran,
            qualfg,
            tcode)
        self._close(wdmpath)
        self._retcode_check(retcode, additional_info='wdtget')

        # Find the begining in datetime.datetime format
        tstart = datetime.datetime(llsdat[0],
                                   llsdat[1],
                                   llsdat[2],
                                   llsdat[3],
                                   llsdat[4],
                                   llsdat[5])

        # Convert time series to pandas DataFrame
        index = pd.date_range(
            tstart,
            periods=len(dataout),
            freq='{0:d}{1}'.format(tstep, MAPTCODE[tcode]))

        tmpval = pd.DataFrame(
            pd.Series(
                dataout,
                index=index,
                name='{0}_DSN_{1}'.format(
                    os.path.basename(wdmpath), dsn)))
        return tmpval

    def read_dsn_por(self, wdmpath, dsn):
        ''' Read the period of record for a DSN.
        '''
        return self.read_dsn(wdmpath, dsn, start_date=None, end_date=None)

    def _close(self, wdmpath):
        ''' Close the WDM file.
        '''
        if wdmpath in self.openfiles:
            retcode = self.wdflcl(self.openfiles[wdmpath])
            self._retcode_check(retcode, additional_info='wdflcl')
            toss = self.openfiles.pop(wdmpath)


if __name__ == '__main__':
    wdm_obj = WDM()
    fname = 'test.wdm'
    if os.name == 'nt':
        fname = r'c:\test.wdm'
    wdm_obj.create_new_wdm(fname, overwrite=True)
    listonumbers = [34.2, 35.0, 36.9, 38.2, 40.2, 20.1, 18.4, 23.6]
    wdm_obj.create_new_dsn(fname, 1003, tstype='EXAM', scenario='OBSERVED', tcode=4, location='EXAMPLE')
    wdm_obj.write_dsn(fname, 1003, listonumbers, datetime.datetime(2000, 1, 1))
    print(wdm_obj.read_dsn_por(fname, 1003))
