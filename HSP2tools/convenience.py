''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
'''


import pandas as pd
from numpy import full as full
from collections import namedtuple as namedtuple


def fetch(hdfname, operation, activity, subtype=None, landuse=None):
    ''' returns an optionally filtered dataframe from the HDF5 file for the specified operation & activity, optionally filtered
    CALL: fetch(hdffilename, operation, activity, subtype, landuse)
    returns replaceinfo, PandasDataframe
        operation is one of 'PERLND', 'IMPLND', or 'RCHRES'
        activity is one of items like 'PWATER', 'SNOW', 'HYDR'
        subtype (optional) is one or more of 'INITIALIZATIONS', 'FLAGS', 'PARAMETERS', 'MONTHLY'
            default is  ['INITIALIZATIONS', 'FLAGS', 'PARAMETERS']
        LANDUSE is a USER DEFINED column in the operation's GENERAL_INFO table
        The landuse argument (optional) is one or more of the items in LANDUSE
        default is not to filter, but to return all data.

        replaceinfo is used by replace() to save changes back to the HDF5 file'''

    keys = pd.read_hdf(hdfname, '/HSP2/KEYS')
    keys =  [key for key in keys if operation in key and activity in key and 'SAVE' not in key]
    temp = []
    if type(subtype) is str:
        subt = (subtype,)
    elif type(subtype) in [list, tuple]:
        subt = tuple(subtype)
    else:
        subt = ('PARAMETERS', 'INITIALIZATIONS', 'FLAGS')
    for s in subt:
        temp.extend([k for k in keys if s in k])
    dfs = [pd.read_hdf(hdfname, key) for key in temp]
    df = pd.concat(dfs, axis=1)

    if landuse is not None:
        if type(landuse) is str:
            lu = (landuse,)
        elif type(landuse) in [list,tuple]:
            lu = tuple(landuse)

        use = pd.read_hdf(hdfname, operation + '/GENERAL_INFO').LANDUSE
        boolean = full(len(df), False).astype(bool)
        for x in lu:
            boolean |= (use==x)
        df = df[boolean]
    return (hdfname, operation, activity), df


replacedata = namedtuple('replacedata', 'hdfname operation name')
def replace(restore, df):
    ''' saves the modified DataFrame from fetch() back to the HDF5 file
    CALL: replace(replaceinfo, DataFrame); no return value
        replaceinfo is the tuple returned by fetch() needed to save the data back to HDF5
        DataFrame is a modified dataframe from the fetch()'''

    rd = replacedata(*restore)
    keys = pd.read_hdf(rd.hdfname, '/HSP2/KEYS')
    keys =  [key for key in keys if rd.operation in key and rd.name in key and 'SAVE' not in key]

    dfset = set(df.columns)
    for k in keys:
        temp = pd.read_hdf(rd.hdfname, k)
        if len(set(temp.columns) & dfset) == 0:
            continue
        for indx, row in df.iterrows():
            temp.loc[indx] = row
        temp.to_hdf(rd.hdfname, k, data_columns=True, format='t')
    return


def clone_segment(hdfname, target, clonefrom, cloneto):
    ''' clones all data in the "clonefrom" segment to the new "cloneto" segment
    CALL: clone_segment(hdffilename, operation, clonefrom, cloneto); no return value
        operation is one of PERLND, IMPLND, or RCHRES
        clonefrom and cloneto are segment ids   '''

    hdfkeys = set(pd.read_hdf(hdfname, '/HSP2/KEYS'))
    for k in ['/CONTROL/EXT_SOURCES', '/CONTROL/NETWORK', '/CONTROL/SCHEMATIC']:
        if k in hdfkeys:
            tlist = []
            df = pd.read_hdf(hdfname, k)
            for indx, row in df.iterrows():
                if row.TVOL==target and row.TVOLNO==clonefrom:
                    row.TVOLNO = cloneto
                    tlist.append(row.copy())
            dflist = pd.DataFrame(tlist)
            dff = pd.concat([df, dflist], ignore_index=True)
            dff.to_hdf(hdfname, k, data_columns=True, format='table')

    for k in [k for k in hdfkeys if target in k]:
        df = pd.read_hdf(hdfname, k)
        df.loc[cloneto] = df.loc[clonefrom]
        if 'LSID' in df:
            df.LSID = df.LSID.astype(str)
        df.to_hdf(hdfname, k, data_columns=True, format='table')

    df = pd.read_hdf(hdfname, '/CONTROL/GLOBAL')
    df.loc['DirtyKeys'] = 'True'
    df.to_hdf(hdfname, '/CONTROL/GLOBAL', data_columns=True, format='table')
    return


def remove_segment(hdfname, target, segment):
    '''remove all the data for the named segment, generally for cloned segments,
    CALL: remove_segment(hdffilename, operation, segmentID); no return value
        operation is one of PERLND, IMPLND, or RCHRES
        segmentid is the pid, iid or rid segment ID to be removed'''

    hdfkeys = set(pd.read_hdf(hdfname, '/HSP2/KEYS'))
    for k in ['/CONTROL/EXT_SOURCES', '/CONTROL/NETWORK', '/CONTROL/SCHEMATIC']:
        if k in hdfkeys:
            df = pd.read_hdf(hdfname, k)
            drops = [i for i,row in df.iterrows() if row.TVOL==target and row.TVOLNO==segment]
            df = df.drop(drops)
            df.to_hdf(hdfname, k, data_columns=True, format='table')

    for k in [key for key in hdfkeys if target in key]:
        df = pd.read_hdf(hdfname, k)
        df = df.drop([segment])
        df.to_hdf(hdfname, k, data_columns=True, format='table')

    df = pd.read_hdf(hdfname, '/CONTROL/GLOBAL')
    df.loc['DirtyKeys'] = 'True'
    df.to_hdf(hdfname, '/CONTROL/GLOBAL', data_columns=True, format='table')
    return
