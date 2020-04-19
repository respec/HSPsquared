''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from pandas import HDFStore, read_csv
from io import StringIO

def fetchtable(hdfname, path, names=[], usercol=None, usercolvalue=None, CSV=False):
    '''
    Fetch HSP2 table and after user modification, use it to update HDF5 file.
    If CSV=True is used, the return to fetch() must also be a CSV string!

    Parameters
    ----------
    hdfname : str
        HSP2 HDF5 filename
    path : str
        HDF5 path to desired dataset (table)
    names : list of strings, optional
        Columns in selected table to return
        The default is [] (all columns).
    usercol : str, optional
        Name of user defined column of strings in desired table
        The default is None.
    usercolvalue : str, optional
        Rows of table matching this value are to be returned
        The default is None. REQUIRED if usercol is specified
    CSV : bool
        Output table is CSV string
        The default is False

    Returns
    -------
    table : Pandas DataFrame
        Table at path (with possible subsetting by names and usercol) for user
        to modify.
    replace : (closure) function
        Call this function with modified table to update table in HDF5 file
    '''

    with HDFStore(hdfname) as store:
        dforiginal = store[path]

        df = dforiginal.copy()
        if usercol:  # subset rows using user defined table
            df = df[df[usercol] == usercolvalue]
        if names:    # subset columns by names in list
            df = df[names]
        df = df.copy()
        if CSV:
            df = df.to_csv()

        def replace(dff):
            nonlocal dforiginal
            if CSV:
                dff = read_csv(StringIO(dff), index_col=0)
                print(dff)
            dforiginal.update(dff)
            dforiginal.to_hdf(store, path, format='table', data_columns=True)
    return df, replace
