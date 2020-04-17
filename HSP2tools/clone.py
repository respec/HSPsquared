''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from pandas import concat, HDFStore


def clone(hdfname, operation, fromID, toID):
    '''
    Add new segment ID to all HSP2 HDF5 tables for operation with values fromID
    NOTE: Does not add new segment to CONTROL/OP_SEQUENCE. User must update
    timeseries in CONTROL/EXT_SOURCES and values in CONTROL/LINKS & CONTROL/MASS_LINKS

    Parameters
    ----------
    hdfname : str
        Name of HSP2 HDF5 file
    operation : str
        One of PERLND, IMPLND or RCHRES
    fromID : str
        Segment name to copy values from
    toID : str
        New segment name

    Returns
    -------
    None.
    '''

    with HDFStore(hdfname) as store:
        paths = [key for key in store.keys() if key.startswith(f'/{operation}')]
        for path in paths:
            df = store[path]
            if fromID in df.index:
                df.loc[toID, :] = df.loc[fromID,:]
                df.to_hdf(store, path, format='table', data_columns=True)

        path = 'CONTROL/EXT_SOURCES'
        df = store[path]
        indx = df[(df.TVOL == operation) & (df.TVOLNO == fromID)].index
        newdf = df.loc[indx,:]
        newdf['TVOLNO'] = toID
        dff = concat([df, newdf], ignore_index=True)
        dff.to_hdf(store, path, format='table', data_columns=True)

        path = 'CONTROL/LINKS'
        df = store[path]
        indx = df[(df.SVOL == operation) & (df.SVOLNO == fromID)].index
        newdf = df.loc[indx,:]
        newdf['SVOLNO'] = toID
        dff = concat([df, newdf], ignore_index=True)
        dff.to_hdf(store, path, format='table', data_columns=True)
    return


def removeClone(hdfname, operation, ID):
    '''
    Removes segment ID from all HSP2 HDF5 tables for specified operation

    Parameters
    ----------
    hdfname : str
        Name of HSP2 HDF5 file
    operation : str
        One of PERLND, IMPLND or RCHRES
    ID : str
        Segment name to remove everywhere in HSP2 HDF5 file

    Returns
    -------
    None.
    '''

    with HDFStore(hdfname) as store:
        paths = [key for key in store.keys() if key.startswith(f'/{operation}')]
        for path in paths:
            df = store[path]
            if ID in df.index:
                df = df.drop(index=ID)
                df.to_hdf(store, path, format='table', data_columns=True)

        path = 'CONTROL/OP_SEQUENCE'
        df = store[path]
        indx = df[(df.OPERATION==operation) & (df.SEGMENT==ID)].index
        df = df.drop(index=indx)
        df.to_hdf(store, path, format='table', data_columns=True)

        path = 'CONTROL/EXT_SOURCES'
        df = store[path]
        indx = df[(df.TVOL == operation) & (df.TVOLNO == ID)].index
        df = df.drop(index=indx)
        df.to_hdf(store, path, format='table', data_columns=True)

        path = 'CONTROL/LINKS'
        df = store[path]
        indx = df[(df.SVOL == operation) & (df.SVOLNO == ID)].index
        df = df.drop(index=indx)
        df.to_hdf(store, path, format='table', data_columns=True)
    return
