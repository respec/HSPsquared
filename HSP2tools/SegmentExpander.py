from datetime import datetime
from pandas import HDFStore
from pandas import read_hdf

start = datetime.now()
h5name = 'C://dev//HSPsquared//tests//Watonwan//HSP2results//Watonwan.h5'
newh5name = 'C://dev//HSPsquared//tests//Watonwan//HSP2results//WatonwanExpanded.h5'

d = {}
with HDFStore(h5name, mode = 'a') as store:
    keys = set(store.keys())

    # first establish what the new ids will be, update schematic block
    path = '/CONTROL/LINKS'
    if path in keys:
        df = read_hdf(store, path)
        for i, dfdata in df.iterrows():
            # build a dict to store the conversion table from old to new perlnd ids
            if dfdata['SVOL'] == 'PERLND' or dfdata['SVOL'] == 'IMPLND':
                if dfdata['TVOL'] == 'RCHRES':
                    oldid = dfdata['SVOLNO']
                    newid = dfdata['SVOLNO'] + dfdata['TVOLNO']
                    if oldid in d:
                        l = d[oldid]
                        l.append(newid)
                        d[oldid] = l
                    else:
                        l = [newid]
                        d[oldid] = l
                    df.loc[i, 'SVOLNO'] = newid
        for i, dfdata in df.iterrows():
            # now update any other instance of the old ids in links
            if dfdata['SVOL'] == 'PERLND' or dfdata['SVOL'] == 'IMPLND':
                if dfdata['SVOLNO'] in d:
                    oldid = dfdata['SVOLNO']
                    l = d[oldid]
                    ind = 0
                    for newid in l:
                        ind += 1
                        if ind == 1:
                            df.loc[i, 'SVOLNO'] = newid
                        else:
                            # need to add record to df
                            newind = i + (ind/100)
                            df.loc[newind] = dfdata
                            df.loc[newind, 'SVOLNO'] = newid
        df = df.sort_index().reset_index(drop=True)

with HDFStore(newh5name, mode='a') as newstore:
    df.to_hdf(newstore, path, data_columns=True, append=True)

print(f'runtime: {datetime.now() - start}')