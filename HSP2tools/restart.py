''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from pandas import date_range, HDFStore, Timestamp
from pandas.tseries.offsets import Minute

states = {
 ('PERLND','SNOW') : ['COVINX','DULL','PACKF','PACKI','PACKW','PAKTMP','RDENPF','SKYCLR','XLNMLT'],
 ('IMPLND','SNOW') : ['COVINX','DULL','PACKF','PACKI','PACKW','PAKTMP','RDENPF','SKYCLR','XLNMLT'],
 ('PERLND','PWATER') : ['AGWS', 'CEPS', 'GWVS', 'IFWS', 'LZS', 'SURS', 'UZS'],
 ('IMPLND','IWATER') : ['RETS', 'SURS'],
 ('RCHRES', 'HYDR')  : ['VOL']}


def restart(hdfname, newstart):
    '''
    Updates STATE values in HSP2 HDF file to start at later newstart date from
    computed values.  User can extend timeseries by predictive or historic data
    to continue simulation. In this case, the user must set a new stop date!

    Parameters
    ----------
    hdfname : str
        HSP2 HDF5 file.
    newstart : str (in Datatime format for Timestamp)
        DateTime for restarting the simulation.

    Returns
    -------
    None.
    '''

    with HDFStore(hdfname) as store:
        df = store['CONTROL/OP_SEQUENCE']
        delt = df.loc[0,'INDELT_minutes']

        df = store['CONTROL/GLOBAL']
        start = Timestamp(df.loc['Start', 'Info'])
        stop  = Timestamp(df.loc['Stop',  'Info'])
        dates = date_range(start, stop, freq=Minute(delt))

        # deterime new start date for restart; previous date if not exact match
        startindx = dates.get_loc(newstart, method='pad')
        startdate = dates[startindx]

        df.loc['Start', 'Info'] = str(startdate)
        df.to_hdf(hdfname, 'CONTROL/GLOBAL', format='table', data_columns=True)

        for path in [p[1:] for p in store.keys() if p.startswith('/RESULTS')]:
            _, x, activity = path.split('/')
            operation, segment = x.split('_')
            if (operation, activity) not in states:
                continue

            df = store[path] [states[operation, activity]]
            df = df.iloc[startindx, :].to_frame()
            df.columns = [segment]

            storepath = f'{operation}/{activity}/STATES'
            dff = store[storepath]
            dff.update(df.T)
            dff.to_hdf(store, storepath, format='table', data_columns=True)
    return
