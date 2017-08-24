''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''

import pandas as pd

def read(fname, indexcol=None):
    _,ext = fname.rsplit('.', 1)
    if ext.lower() == 'csv':
        return pd.read_csv(fname, index_col=indexcol)   if indexcol else pd.read_csv(fname)
    elif ext.lower() == 'xlsx':
        return pd.read_excel(fname, index_col=indexcol) if indexcol else pd.read_excel(fname)
    else:
        print 'Unknown file extension', ext
        return None


def csvReader(hdfname, csvname, operation, activity):
    ''' csvReader updates HDF5 file, hdfname, using data from csv or xlsx file, csvname.
    The operation is one of 'PERLND', 'IMPLND', 'RCHRES', 'FTABLES'
    The activity is the normal HSPF name such as PWATER'''

    if operation in ['PERLND', 'IMPLND', 'RCHRES'] and activity not in ['GENERAL_INFO', 'SAVE']:
        indf = read(csvname)
        cols = set(indf.columns)
        for subdir in ['PARAMETERS', 'STATE', 'FLAGS']:
            path = '/'.join([operation, activity, subdir])
            hdf = pd.read_hdf(hdfname, path)
            for data in cols & set(hdf.columns):
                for i,seg, p in indf[['SEGMENT', data]].itertuples():
                    hdf.loc[seg, data] = p
            if cols & set(hdf.columns):
                hdf.to_hdf(hdfname, path, format='t', data_columns=True)

    elif operation == 'FTABLES':
        path = '/'.join([operation, activity])
        hdf = pd.read_hdf(hdfname, path)
        for data in read(csvname, indexcol='Index').itertuples():
            hdf.loc[data.Index] = data[1:]
        hdf.to_hdf(hdfname, path, format='t', data_columns=True)
