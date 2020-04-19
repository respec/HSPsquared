''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from pandas import read_csv, read_excel, HDFStore


def readCSV(csvname, hdfname, datapath):
    '''
    Updates or creates DataFrame in HDF5 file from CSV file.
    Does not append new rows to existing HDF5 table

    Parameters
    ----------
    csvname : str
        name of CSV,TSV, XLSX file to update/create HDF5
    hdfname : str
        Name of HDF5 file to be updated
    datapath : str
        Pathname in HDF5 file to be update/created

    Returns
    -------
    None.
    '''

    with HDFStore(hdfname) as store:
        if datapath in store:
            df = store[datapath]
            df.update(read(csvname, indexcol=0))
            df.to_hdf(store, datapath, data_columns=True, format='t')
        else:
            df = read(csvname, indexcol=0)
            df.to_hdf(store, datapath, data_columns=True, format='t')
    return


def read(csvname, indexcol=None):
    if   csvname.lower().endswith('.csv'): return read_csv(csvname, index_col=0)
    elif csvname.lower().endswith('.tsv'): return read_csv(csvname, index_col=0, delimiter='\t')
    elif csvname.lower().endswith('xlsx'): return read_excel(csvname, index_col=0)
    else:
        print('CSV file has unknown file extension')
        return None