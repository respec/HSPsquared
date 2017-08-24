''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import pandas as pd

def newstate(hdfname, start, optype, opmodule):
    try:
        oldstate = pd.read_hdf(hdfname, optype + '/' + opmodule + '/STATE')
    except:
        return pd.DataFrame()

    cols = oldstate.columns
    newstate = pd.DataFrame(columns=cols)
    for seg in oldstate.index:
        try:
            temp = pd.read_hdf(hdfname, 'RESULTS/' + optype + '_' + seg + '/' + opmodule)
            newstate.loc[seg] = temp.loc[start, cols]
        except:
            continue
    return newstate


def update_state(hdfname, start):
    df = pd.read_hdf(hdfname, 'CONTROL/GLOBAL')
    dates = pd.date_range(df.loc['sim_start', 'Data'], df.loc['sim_end', 'Data'], freq='H')
    startindx = dates.get_loc(start, method='nearest')
    start = dates[startindx]
    df.loc['sim_start', 'Data'] = str(start)
    df.to_hdf(hdfname, 'CONTROL/GLOBAL', data_columns=True, format='t')

    ops = [('PERLND', 'ATEMP'), ('PERLND', 'SNOW'), ('PERLND', 'PWATER'),
           ('IMPLND', 'ATEMP'), ('IMPLND', 'SNOW'), ('IMPLND', 'IWATER'),
           ('RCHRES', 'HYDR')]
    for optype, opmodule in ops:
        nstate = newstate(hdfname, start, optype, opmodule)
        if not nstate.empty:
            nstate.to_hdf(hdfname, optype + '/' + opmodule + '/STATE', data_columns=True, format='t')
