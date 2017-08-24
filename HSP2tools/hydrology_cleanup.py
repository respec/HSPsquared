''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import pandas as pd

def cleanup(hdfname):
    with pd.get_store(hdfname) as store:
        id1 = store['/CONTROL/OP_SEQUENCE'].ID.values
        df = store['/CONTROL/LINKS']
        idset = set(df.SVOLNO.values) | set(df.TVOLNO.values) | set(id1)

        keys = store.keys()
        for key in keys:
            tokens = key.split('/')
            if tokens[1] in ['PERLND', 'IMPLND']:
                if tokens[2] in ['ACTIVITY', 'GENERAL_INFO', 'ATEMP', 'SNOW', 'PWATER', 'IWATER']:
                    df = store[key]
                    bad = set(df.index.values) - idset
                    df = df.drop(bad)
                    store.put(key, df, format='t', data_columns=True)
                else:
                    if key in store:
                        del store[key]

            if tokens[1] == 'RCHRES':
                if tokens[2] in ['ACTIVITY', 'GENERAL_INFO', 'HYDR']:
                    df = store[key]
                    bad = set(df.index.values) - idset
                    df = df.drop(bad)
                    store.put(key, df, format='t', data_columns=True)
                else:
                    if key in store:
                        del store[key]

            if key == '/CONTROL/EXT_SOURCES':
                df = store[key]
                lst = [i for i,row in df.iterrows() if row.TVOLNO not in idset]
                df = df.drop(lst)
                store.put(key, df, format='t', data_columns=True)

    with pd.get_store(hdfname) as store:
        store.put('/HSP2/KEYS', pd.Series(store.keys()))
    print('Cleanup done')
