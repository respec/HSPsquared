''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D. '''


import pandas as pd


def snow_balance(hdfname, segmentType, segmentID='P001'):
    ''' Computes the snow water balance
        snow_balance(hdfname, segmentType, segmentID)
            hdfname is name (with path as necessary)
            segmentType is 'PERLND' or 'IMPLND'
            segmentID is the PID or IID for the desired segment'''

    if segmentType not in ['PERLND', 'IMPLND']:
        print('bad type')
        return pd.DataFrame()

    path = '/RESULTS/' + segmentType + '_'  + segmentID + '/SNOW'
    data = pd.read_hdf(hdfname, path)

    sv = data[['PACKF','PACKW']].copy()
    sv['PACK'] = sv['PACKF'] + sv['PACKW']
    initial = sv.loc[sv.index[0], 'PACK']
    sv = sv.resample('M').last().copy()

    sv['ShiftedPACK'] = sv['PACK'].shift()
    sv.loc[sv.index[0], 'ShiftedPACK'] = initial

    fluxes = ['SNOWF', 'PRAIN', 'SNOWE',  'WYIELD']
    flx = data[fluxes].resample('M').sum().copy()

    cat = pd.concat([sv, flx], axis=1)

    numerator = (cat['PACK'] - cat['ShiftedPACK']) - (cat['SNOWF'] + cat['PRAIN']- cat['SNOWE'] - cat['WYIELD'])
    denominator = cat['ShiftedPACK'] + cat['SNOWF'] + cat['PRAIN']
    cat['BALANCE_PCT'] = 100.0 * numerator / denominator
    cat['REFVAL'] = cat['ShiftedPACK'] + cat['SNOWF'] + cat['PRAIN']

    cat.fillna(0.0, inplace=True)
    return cat[['BALANCE_PCT', 'REFVAL']]


def pwater_balance(hdfname, segmentID='P001'):
    ''' Computes the PERLND water balance
        pwater_balance(hdfname, segmentType, segmentID)
            hdfname is name (with path as necessary)
            segmentID is the PID (name) for the desired segment'''

    path = '/RESULTS/PERLND_' + segmentID + '/PWATER'
    data = pd.read_hdf(hdfname, path)

    sv = data[['PERS']]
    initial = sv['PERS'][0]
    sv = sv.resample('M').last().copy()

    sv['ShiftedPERS'] = sv['PERS'].shift()
    sv.loc[sv.index[0], 'ShiftedPERS'] = initial

    f = ['SUPY', 'SURLI', 'UZLI', 'IFWLI', 'LZLI', 'AGWLI', 'PERO', 'IFWO', 'AGWO']
    ff = [f for f in f if f in data.columns]
    flx = data[ff].resample('M').sum().copy()

    cat = pd.concat([sv, flx], axis=1)

    surli = cat['SURLI'] if 'SURLI' in cat else 0.0
    uzli  = cat['UZLI']  if 'UZLI'  in cat else 0.0
    agwli = cat['AGWLI'] if 'AGWLI' in cat else 0.0
    ifwli = cat['IFWLI'] if 'IFWLI' in cat else 0.0

    watin = cat['SUPY'] + surli + uzli + agwli + ifwli
    watout = cat['PERO'] + cat['IFWO'] + cat['AGWO']

    numerator = (cat['PERS'] - cat['ShiftedPERS']) - (watin - watout)
    denominator = cat['ShiftedPERS'] + watin

    cat['BALANCE_PCT'] = 100.0 * numerator / denominator
    cat['ERROR'] = numerator
    cat['REFVAL'] = cat['ShiftedPERS'] + cat['SUPY']

    cat.fillna(0.0, inplace=True)
    return cat[['BALANCE_PCT', 'REFVAL']]


def iwater_balance(hdfname, segmentID='I001'):
    ''' Computes the IMPLND water balance
        iwater_balance(hdfname, segmentType, segmentID)
            hdfname is name (with path as necessary)
            segmentID is the IID (name) for the desired segment'''

    path = '/RESULTS/IMPLND_' +  segmentID + '/IWATER'
    data = pd.read_hdf(hdfname, path)

    sv = data[['RETS', 'SURS']]
    initrets = sv.loc[sv.index[0], 'RETS']
    initsurs = sv.loc[sv.index[0], 'SURS']
    sv = sv.resample('M').last().copy()

    sv['ShiftedRETS'] = sv['RETS'].shift()
    sv.loc[sv.index[0], 'ShiftedRETS'] = initrets
    sv['ShiftedSURS'] = sv['SURS'].shift()
    sv.loc[sv.index[0], 'ShiftedSURS'] = initsurs

    fluxes = ['SUPY', 'SURI', 'SURO', 'IMPEV']
    flx = data[fluxes].resample('M').sum().copy()

    cat = pd.concat([sv, flx], axis=1)

    surli = cat['SURLI'] if 'SURLI' in cat else 0.0
    watin = cat['SUPY'] + surli
    watout = cat['SURO'] + cat['IMPEV']

    numerator = (cat['RETS'] - cat['ShiftedRETS'] + cat['SURS'] - cat['ShiftedSURS']) - (watin - watout)
    denominator = cat['ShiftedRETS'] + cat['ShiftedSURS'] + watin

    cat['BALANCE_PCT'] = 100.0 * numerator / denominator
    cat['ERROR'] = numerator
    cat['REFVAL'] = cat['ShiftedRETS'] + cat['ShiftedSURS'] + cat['SUPY']

    cat.fillna(0.0, inplace=True)
    return cat[['BALANCE_PCT', 'REFVAL']]
