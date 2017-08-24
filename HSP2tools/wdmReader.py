''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

This module calls the Tim Cera's wdmtoolbox which has a BSD license.
wdmtoolbox Copyright 2016 by Tim Cera, P.E.

A modified version of the wdmtoolbox wdmuitl module is included in HSP2 due to
bug's reported to Tim and to problems installing in a Windows environment.'''


import pandas as pd
import os
from wdmtoolbox import wdmutil
import re


def ReadWDM(wdmfile, hdffile, **options):
    ''' reads timeseries data from a wdmfile and puts it into an HSP2 HDF5 file
    It uses a modified version of Tim Cera's wdmtoolbox
    CALL: ReadWDM(wdmfile, hdffile, options)
       wdmfile is the name of the HSPF WDM file to be read
       hdffile is the target HDF5 file to put the data, created if necessary
       OPTIONS:
           UCI - boolean flag. If true truncates timeseries to simulation start/stop
           StartDateTime and EndDateTime if both present and are good DateTime strings
             truncates timeseries data to that interval [StartDateTime,EndDateTime']'''

    print('Processing WDM file ' + str(wdmfile))

    hdfname = os.path.realpath(hdffile)
    wdmname = wdmfile

    ids = pd.read_hdf(hdfname, '/CONTROL/EXT_SOURCES')
    y   = ids[['SVOLNO', 'TRAN', 'SSYST']].drop_duplicates()

    general = pd.read_hdf(hdfname, '/CONTROL/GLOBAL')['Data'].to_dict()
    sim_start = general['sim_start']
    sim_end   = general['sim_end']

    WDM = wdmutil.WDM()
    for _,row in y.iterrows():
        pat = '([a-zA-Z]+)([0-9]+)'
        m = re.search(pat, row.SVOLNO)
        key = int(m.group(2))
        if not WDM.exists_dsn(wdmname, key):
            continue

        xdf   = WDM.read_dsn(wdmname, key)
        xdesc = WDM.describe_dsn(wdmname, key)
        xdesc['TRAN']  = row.TRAN
        xdesc['SSYST'] = row.SSYST

        wts = xdf[xdf.columns[0]]
        if 'UCI' in options and options['UCI'] == True:
            wts = wts.truncate(sim_start, sim_end)
        if 'StartDateTime' in options and 'EndDateTime' in options:
            wts = wts.truncate(options['StartDateTime'], options['EndDateTime'])

        with pd.get_store(hdfname) as store:
            dsname = '/TIMESERIES/' + row.SVOLNO
            store.put(dsname, wts)
            store.get_storer(dsname).attrs.start_date  = xdesc['start_date']
            store.get_storer(dsname).attrs.constituent = xdesc['constituent']
            store.get_storer(dsname).attrs.dsn         = xdesc['dsn']
            store.get_storer(dsname).attrs.end_date    = xdesc['end_date']
            store.get_storer(dsname).attrs.location    = xdesc['location']
            store.get_storer(dsname).attrs.scenario    = xdesc['scenario']
            store.get_storer(dsname).attrs.tcode       = xdesc['tcode']
            store.get_storer(dsname).attrs.tcode_name  = xdesc['tcode_name']
            store.get_storer(dsname).attrs.tsfill      = xdesc['tsfill']
            store.get_storer(dsname).attrs.tstep       = xdesc['tstep']

            store.get_storer(dsname).attrs.agg_method  = xdesc['TRAN']
            store.get_storer(dsname).attrs.wdm_units   = xdesc['SSYST']
            store.get_storer(dsname).attrs.units       = '???'

            print(' '.join([dsname, xdesc['tcode_name'], xdesc['start_date'],
             xdesc['end_date'], str(len(wts)), xdesc['constituent']]))
    print('Done with ' + str(wdmfile))
