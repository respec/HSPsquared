''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
'''


import pandas as pd
import os
import os.path
import HSP2tools


def makeH5():
    '''creates 'hidden' HDF5 file with UCI parsing information + other useful data '''
    sitepath = HSP2tools.__file__[:-12] + '\\HSP2Data\\'
    h2name = sitepath + 'HSP2.h5'

    lapse = pd.Series ([ 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0037,
         0.0040, 0.0041, 0.0043, 0.0046, 0.0047, 0.0048, 0.0049, 0.0050, 0.0050,
         0.0048, 0.0046, 0.0044, 0.0042, 0.0040, 0.0038, 0.0037, 0.0036])
    lapse.to_hdf(h2name, '/LAPSE')

    seasons = pd.Series ([0,0,0, 1,1,1,1,1,1, 0,0,0]).astype(bool)
    seasons.to_hdf(h2name, '/SEASONS')

    svp = pd.Series([1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005,
     1.005, 1.01, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.1, 1.29, 1.66,
     2.13, 2.74,3.49, 4.40, 5.55,6.87, 8.36, 10.1,12.2,14.6, 17.5, 20.9, 24.8,
     29.3, 34.6, 40.7, 47.7, 55.7, 64.9])
    svp.to_hdf(h2name, '/SaturatedVaporPressureTable')

    # define execution order, python functions to call, and HDF5 internal path
    d = [
     ['PERLND',   -1,  'ACTIVITY',     '',               '',       'PERLND/ACTIVITY'],
     ['PERLND',   -1,  'GENERAL_INFO', '',               '',       'PERLND/GENERAL_INFO'],
     ['PERLND',  100,  'AIRTFG',       'HSP2.hperair',   'atemp',  'PERLND/ATEMP/'],
     ['PERLND',  200,  'SNOWFG',       'HSP2.hpersno',   'snow',   'PERLND/SNOW/'],
     ['PERLND',  300,  'PWATFG',       'HSP2.hperwat',   'pwater', 'PERLND/PWATER/'],
     ['PERLND',  400,  'SEDFG',        'noop',           'noop',   'PERLND/SEDMNT/'],
     ['PERLND',  500,  'PSTFG',        'noop',           'noop',   'PERLND/PSTEMP/'],
     ['PERLND',  600,  'PWGFG',        'noop',           'noop',   'PERLND/PWTGAS/'],
     ['PERLND',  700,  'PQALFG',       'noop',           'noop',   'PERLND/PQUAL/'],
     ['PERLND',  800,  'MSTLFG',       'noop',           'noop',   'PERLND/MSTLAY/'],
     ['PERLND',  900,  'PESTFG',       'noop',           'noop',   'PERLND/PEST/'],
     ['PERLND', 1000,  'NITRFG',       'noop',           'noop',   'PERLND/NITR/'],
     ['PERLND', 1100,  'PHOSFG',       'noop',           'noop',   'PERLND/PHOS/'],
     ['PERLND', 1200,  'TRACFG',       'noop',           'noop',   'PERLND/TRACER/'],

     ['IMPLND',   -1,  'ACTIVITY',     '',               '',       'IMPLND/ACTIVITY'],
     ['IMPLND',   -1,  'GENERAL_INFO', '',               '',       'IMPLND/GENERAL_INFO'],
     ['IMPLND',  100,  'ATMPFG',       'HSP2.hperair',   'atemp',  'IMPLND/ATEMP/'],
     ['IMPLND',  200,  'SNOWFG',       'HSP2.hpersno',   'snow',   'IMPLND/SNOW/'],
     ['IMPLND',  300,  'IWATFG',       'HSP2.himpwat',   'iwater', 'IMPLND/IWATER/'],
     ['IMPLND',  400,  'SLDFG',        'noop',           'noop',   'IMPLND/SOLIDS/'],
     ['IMPLND',  500,  'IWGFG',        'noop',           'noop',   'IMPLND/IWTGAS/'],
     ['IMPLND',  600,  'IQALFG',       'noop',           'noop',   'IMPLND/IQUAL/'],

     ['RCHRES',   -1,  'ACTIVITY',     '',               '',       'RCHRES/ACTIVITY'],
     ['RCHRES',   -1,  'GENERAL_INFO', '',               '',       'RCHRES/GENERAL_INFO'],
     ['RCHRES',  100,  'HYDRFG',       'HSP2.hrchhyd',   'hydr',   'RCHRES/HYDR/'],
     ['RCHRES',  200,  'ADFG',         'noop',           'noop',   'RCHRES/ADCALC/'],
     ['RCHRES',  300,  'CONSFG',       'noop',           'noop',   'RCHRES/CONS/'],
     ['RCHRES',  400,  'HTFG',         'noop',           'noop',   'RCHRES/HTRCH/'],
     ['RCHRES',  500,  'SEDFG',        'noop',           'noop',   'RCHRES/SEDTRN/'],
     ['RCHRES',  600,  'GQALFG',       'noop',           'noop',   'RCHRES/GQUAL/'],
     ['RCHRES',  700,  'OXFG',         'noop',           'noop',   'RCHRES/OXRX/'],
     ['RCHRES',  800,  'NUTFG',        'noop',           'noop',   'RCHRES/NUTRX/'],
     ['RCHRES',  900,  'PLKFG',        'noop',           'noop',   'RCHRES/PLANK/'],
     ['RCHRES', 1000,  'PHFG',         'noop',           'noop',   'RCHRES/PHCARB/'],
     ]
    df = pd.DataFrame(d, columns=['Target','Order','Flag','Module','Function','Path'])
    df.to_hdf(h2name, '/CONFIGURATION', format='t', data_columns=True)


    # This table defines the expansion to INFLOW, ROFLOW, OFLOW for RCHRES networks
    d = [
        ['IVOL',  'ROVOL',  'OVOL',  'HYDRFG', 'HYDR'],
        ['ICON',  'ROCON',  'OCON',  'CONSFG', 'CONS'],
        ['IHEAT', 'ROHEAT', 'OHEAT', 'HTFG',   'HTRCH'],
        ['ISED',  'ROSED',  'OSED',  'SEDFG',  'SEDTRN'],
        ['IDQAL', 'RODQAL', 'ODQAL', 'GQALFG', 'GQUAL'],
        ['ISQAL', 'ROSQAL', 'OSQAL', 'GQALFG', 'GQUAL'],
        ['OXIF',  'OXCF1',  'OXCF2', 'OXFG',   'OXRX'],
        ['NUIF1', 'NUCF1',  'NUCF1', 'NUTFG',  'NUTRX'],
        ['NUIF2', 'NUCF2',  'NUCF9', 'NUTFG',  'NUTRX'],
        ['PKIF',  'PKCF1',  'PKCH2', 'PLKFG',  'PLANK'],
        ['PHIF',  'PHCF1',  'PHCF2', 'PHFG',   'PHCARB']]
    df = pd.DataFrame(d, columns=['INFLOW', 'ROFLOW', 'OFLOW', 'Flag', 'Name'])
    df.to_hdf(h2name, '/FLOWEXPANSION', format='t', data_columns=True)


    # The next 3 DataFrames initialize the SAVE variables
    d = [
     ['AIRTFG', 'AIRTMP',    1], # state

     ['SNOWFG', 'ALBEDO',    0], # state ???
     ['SNOWFG', 'COVINX',    1], # state
     ['SNOWFG', 'DEWTMP',    0], # state ???
     ['SNOWFG', 'DULL',      1], # state
     ['SNOWFG', 'MELT',      0], # flux
     ['SNOWFG', 'NEGHTS',    0], # state ???
     ['SNOWFG', 'PACKF',     1], # state
     ['SNOWFG', 'PACKI',     1], # state
     ['SNOWFG', 'PACKW',     1], # state
     ['SNOWFG', 'PAKTMP',    1], # state
     ['SNOWFG', 'PDEPTH',    0], # state   ???
     ['SNOWFG', 'PRAIN',     0], # flux
     ['SNOWFG', 'RAINF',     1], # flux
     ['SNOWFG', 'RDENPF',    1], # state
     ['SNOWFG', 'SKYCLR',    1], # state
     ['SNOWFG', 'SNOCOV',    1], # state  ??? needed by PWATER
     ['SNOWFG', 'SNOTMP',    0], # state ???
     ['SNOWFG', 'SNOWE',     0], # flux
     ['SNOWFG', 'SNOWF',     0], # flux
     ['SNOWFG', 'WYIELD',    1], # flux
     ['SNOWFG', 'XLNMLT',    1], # state

     ['PWATFG', 'AGWET',     0],
     ['PWATFG', 'AGWI',      0],
     ['PWATFG', 'AGWO',      1], # flux
     ['PWATFG', 'AGWS',      1], # state
     ['PWATFG', 'BASET',     0],
     ['PWATFG', 'CEPE',      0],
     ['PWATFG', 'CEPS',      1], # state
     ['PWATFG', 'GWEL',      0],
     ['PWATFG', 'GWVS',      1], # state
     ['PWATFG', 'IFWI',      0],
     ['PWATFG', 'IFWO',      1], # flux
     ['PWATFG', 'IFWS',      1], # state
     ['PWATFG', 'IGWI',      0],
     ['PWATFG', 'INFFAC',    0], # state ???
     ['PWATFG', 'INFIL',     0],
     ['PWATFG', 'IRDRAW',    0],
     ['PWATFG', 'IRRAPP',    0],
     ['PWATFG', 'IRRDEM',    0],
     ['PWATFG', 'IRSHRT',    0],
     ['PWATFG', 'LZET',      0],
     ['PWATFG', 'LZI',       0],
     ['PWATFG', 'LZS',       1], # state
     ['PWATFG', 'RPARM',     0],
     ['PWATFG', 'PERC',      0],
     ['PWATFG', 'PERO',      1], # flux
     ['PWATFG', 'PERS',      0], # state ???
     ['PWATFG', 'PET',       0],
     ['PWATFG', 'PETADJ',    0], # state ???
     ['PWATFG', 'RPARM',     0],
     ['PWATFG', 'RZWS',      0],
     ['PWATFG', 'SUPY',      0],
     ['PWATFG', 'SURET',     0],
     ['PWATFG', 'SURI',      0],
     ['PWATFG', 'SURO',      1], # flux
     ['PWATFG', 'SURS',      1], # state
     ['PWATFG', 'TAET',      0],
     ['PWATFG', 'TGWS',      0],
     ['PWATFG', 'UZET',      0],
     ['PWATFG', 'UZI',       0],
     ['PWATFG', 'UZS',       1], # state

     ['SEDFG',  'COVER',     0],
     ['SEDFG',  'DET',       0],
     ['SEDFG',  'DETS',      0],
     ['SEDFG',  'NVSI',      0],
     ['SEDFG',  'SCRSD',     0],
     ['SEDFG',  'SOSED',     0],
     ['SEDFG',  'STCAP',     0],
     ['SEDFG',  'WSSD',      0],
     ['PSTFG',  'AIRTC',     0],
     ['PSTFG',  'LGTMP',     0],
     ['PSTFG',  'SLTMP',     0],
     ['PSTFG',  'ULTMP',     0],
     ['PWGFG',  'AOCO2',     0],
     ['PWGFG',  'AOCO2M',    0],
     ['PWGFG',  'AODOX',     0],
     ['PWGFG',  'AODOXM',    0],
     ['PWGFG',  'AOHT',      0],
     ['PWGFG',  'AOTMP',     0],
     ['PWGFG',  'IOCO2',     0],
     ['PWGFG',  'IOCO2M',    0],
     ['PWGFG',  'IODOX',     0],
     ['PWGFG',  'IODOXM',    0],
     ['PWGFG',  'IOHT',      0],
     ['PWGFG',  'IOTMP',     0],
     ['PWGFG',  'POCO2M',    0],
     ['PWGFG',  'PODOXM',    0],
     ['PWGFG',  'POHT',      0],
     ['PWGFG',  'SOCO2',     0],
     ['PWGFG',  'SOCO2M',    0],
     ['PWGFG',  'SODOX',     0],
     ['PWGFG',  'SODOXM',    0],
     ['PWGFG',  'SOHT',      0],
     ['PWGFG',  'SOTMP',     0],
     ['PQALFG', 'AOQC',      0],
     ['PQALFG', 'AOQUAL',    0],
     ['PQALFG', 'IOQC',      0],
     ['PQALFG', 'IOQUAL',    0],
     ['PQALFG', 'ISQOAL',    0],
     ['PQALFG', 'POQC',      0],
     ['PQALFG', 'POQUAL',    0],
     ['PQALFG', 'PQADDR',    0],
     ['PQALFG', 'PQADEP',    0],
     ['PQALFG', 'PQADWT',    0],
     ['PQALFG', 'SCRQS',     0],
     ['PQALFG', 'SOQC',      0],
     ['PQALFG', 'SOQO',      0],
     ['PQALFG', 'SOQOC',     0],
     ['PQALFG', 'SOQS',      0],
     ['PQALFG', 'SOQSP',     0],
     ['PQALFG', 'SOQUAL',    0],
     ['PQALFG', 'SQO',       0],
     ['PQALFG', 'WASHQS',    0],
     ['MSTLFG', 'MST',       0],
     ['MSTLFG', 'FRAC',      0],
     ['PESTFG', 'ADEGPS',    0],
     ['PESTFG', 'APS',       0],
     ['PESTFG', 'IPS',       0],
     ['PESTFG', 'LDEGPS',    0],
     ['PESTFG', 'LPS',       0],
     ['PESTFG', 'PEADDR',    0],
     ['PESTFG', 'PEADEP',    0],
     ['PESTFG', 'PEADWT',    0],
     ['PESTFG', 'POPST',     0],
     ['PESTFG', 'SDEGPS',    0],
     ['PESTFG', 'SDPS',      0],
     ['PESTFG', 'SOSDPS',    0],
     ['PESTFG', 'SPS',       0],
     ['PESTFG', 'SSPSS',     0],
     ['PESTFG', 'TDEGPS',    0],
     ['PESTFG', 'TOPST',     0],
     ['PESTFG', 'TOTPST',    0],
     ['PESTFG', 'TPS',       0],
     ['PESTFG', 'TSPSS',     0],
     ['PESTFG', 'UDEGPS',    0],
     ['PESTFG', 'UPS',       0],
     ['NITRFG', 'AGPLTN',    0],
     ['NITRFG', 'AMIMB',     0],
     ['NITRFG', 'AMNIT',     0],
     ['NITRFG', 'AMUPA',     0],
     ['NITRFG', 'AMUPB',     0],
     ['NITRFG', 'AMVOL',     0],
     ['NITRFG', 'AN',        0],
     ['NITRFG', 'DENIF',     0],
     ['NITRFG', 'IN',        0],
     ['NITRFG', 'LITTRN',    0],
     ['NITRFG', 'LN',        0],
     ['NITRFG', 'NDFCT',     0],
     ['NITRFG', 'NFIXFX',    0],
     ['NITRFG', 'NIADDR',    0],
     ['NITRFG', 'NIADEP',    0],
     ['NITRFG', 'NIADWT',    0],
     ['NITRFG', 'NIIMB',     0],
     ['NITRFG', 'NITIF',     0],
     ['NITRFG', 'NIUPA',     0],
     ['NITRFG', 'NIUPB',     0],
     ['NITRFG', 'NUPTG',     0],
     ['NITRFG', 'ORNMN',     0],
     ['NITRFG', 'PONH4',     0],
     ['NITRFG', 'PONITR',    0],
     ['NITRFG', 'PONO3',     0],
     ['NITRFG', 'POORN',     0],
     ['NITRFG', 'REFRON',    0],
     ['NITRFG', 'RETAGN',    0],
     ['NITRFG', 'RTLBN',     0],
     ['NITRFG', 'RTLLN',     0],
     ['NITRFG', 'RTRBN',     0],
     ['NITRFG', 'RTRLN',     0],
     ['NITRFG', 'SN',        0],
     ['NITRFG', 'SOSEDN',    0],
     ['NITRFG', 'SSAMS',     0],
     ['NITRFG', 'SSNO3',     0],
     ['NITRFG', 'SSSLN',     0],
     ['NITRFG', 'SSSRN',     0],
     ['NITRFG', 'TDENIF',    0],
     ['NITRFG', 'TN',        0],
     ['NITRFG', 'TNIT',      0],
     ['NITRFG', 'TOTNIT',    0],
     ['NITRFG', 'TSAMS',     0],
     ['NITRFG', 'TSNO3',     0],
     ['NITRFG', 'TSSLN',     0],
     ['NITRFG', 'TSSRN',     0],
     ['NITRFG', 'UN',        0],
     ['PHOSFG', 'AP',        0],
     ['PHOSFG', 'IP',        0],
     ['PHOSFG', 'LP',        0],
     ['PHOSFG', 'ORPMN',     0],
     ['PHOSFG', 'P4IMB',     0],
     ['PHOSFG', 'PDFCT',     0],
     ['PHOSFG', 'PHADDR',    0],
     ['PHOSFG', 'PHADEP',    0],
     ['PHOSFG', 'PHADWT',    0],
     ['PHOSFG', 'PHOIF',     0],
     ['PHOSFG', 'POPHOS',    0],
     ['PHOSFG', 'PUPTG',     0],
     ['PHOSFG', 'SEDP',      0],
     ['PHOSFG', 'SOSEDP',    0],
     ['PHOSFG', 'SP',        0],
     ['PHOSFG', 'SSP4S',     0],
     ['PHOSFG', 'TOTPHO',    0],
     ['PHOSFG', 'TP',        0],
     ['PHOSFG', 'TPHO',      0],
     ['PHOSFG', 'TSP4S',     0],
     ['PHOSFG', 'UP',        0],
     ['TRACFG', 'ATRSU',     0],
     ['TRACFG', 'ITRSU',     0],
     ['TRACFG', 'LTRSU',     0],
     ['TRACFG', 'POTRS',     0],
     ['TRACFG', 'SSTRS',     0],
     ['TRACFG', 'STRSU',     0],
     ['TRACFG', 'TRADDR',    0],
     ['TRACFG', 'TRADEP',    0],
     ['TRACFG', 'TRADWT',    0],
     ['TRACFG', 'TRSU',      0],
     ['TRACFG', 'TSTRS',     0],
     ['TRACFG', 'UTRSU',     0],
     ]
    df = pd.DataFrame(d, columns=['Flag', 'Name', 'Value'])
    df.to_hdf(h2name, '/SAVE/PERLND', format='t', data_columns=True)

    d = [
     ['ATMPFG', 'AIRTMP', 1],

     ['SNOWFG', 'ALBEDO',    0],
     ['SNOWFG', 'COVINX',    1], # state
     ['SNOWFG', 'DEWTMP',    0],
     ['SNOWFG', 'DULL',      1], # state
     ['SNOWFG', 'MELT',      0],
     ['SNOWFG', 'NEGHTS',    0],
     ['SNOWFG', 'PACKF',     1], # state
     ['SNOWFG', 'PACKI',     1], # state
     ['SNOWFG', 'PACKW',     1], # state
     ['SNOWFG', 'PAKTMP',    1], # state
     ['SNOWFG', 'PDEPTH',    0],
     ['SNOWFG', 'PRAIN',     0],
     ['SNOWFG', 'RAINF',     1], # flux
     ['SNOWFG', 'RDENPF',    1], # state
     ['SNOWFG', 'SKYCLR',    1], # state
     ['SNOWFG', 'SNOCOV',    1], # needed by PWATER
     ['SNOWFG', 'SNOTMP',    0],
     ['SNOWFG', 'SNOWE',     0],
     ['SNOWFG', 'SNOWF',     0],
     ['SNOWFG', 'WYIELD',    1], # flux
     ['SNOWFG', 'XLNMLT',    1], # state

     ['IWATFG', 'IMPEV',  0], # flux
     ['IWATFG', 'PET',    0], # flux
     ['IWATFG', 'PETADJ', 0], # state ???
     ['IWATFG', 'RETS',   1], # state
     ['IWATFG', 'SUPY',   0], # flux
     ['IWATFG', 'SURI',   0],
     ['IWATFG', 'SURO',   1], # flux
     ['IWATFG', 'SURS',   1], # state

     ['SLDFG',  'SLDS',   0],
     ['SLDFG',  'SOSLD',  0],
     ['IWGFG',  'SOTMP',  0],
     ['IWGFG',  'SODOX',  0],
     ['IWGFG',  'SOCO2',  0],
     ['IWGFG',  'SOHT',   0],
     ['IWGFG',  'SODOXM', 0],
     ['IWGFG',  'SOCO2M', 0],
     ['IQALFG', 'SQO',    0],
     ['IQALFG', 'SOQSP',  0],
     ['IQALFG', 'SOQS',   0],
     ['IQALFG', 'SOQO',   0],
     ['IQALFG', 'SOQOC',  0],
     ['IQALFG', 'SOQUAL', 0],
     ['IQALFG', 'SOQC',   0],
     ['IQALFG', 'IQADDR', 0],
     ['IQALFG', 'IQADWT', 0],
     ['IQALFG', 'IQADEP', 0],
     ]
    df = pd.DataFrame(d, columns=['Flag', 'Name', 'Value'])
    df.to_hdf(h2name, '/SAVE/IMPLND', format='t', data_columns=True)

    d = [
     ['HYDRFG', 'CDFVOL', 0],
     ['HYDRFG', 'CIVOL',  0],
     ['HYDRFG', 'CO',     0],
     ['HYDRFG', 'COVOL',  0],
     ['HYDRFG', 'CRO',    0],
     ['HYDRFG', 'CROVOL', 0],
     ['HYDRFG', 'CVOL',   0],
                              # COLIND needed  - computed ???
     ['HYDRFG', 'DEP',    0], # state ???
     ['HYDRFG', 'IVOL',   0], # flux
     ['HYDRFG', 'O',      0], # state ???
     ['HYDRFG', 'OVOL',   1], # flux
     ['HYDRFG', 'PRSUPY', 0], # flux
     ['HYDRFG', 'RIRDEM', 0],
     ['HYDRFG', 'RIRSHT', 0],
     ['HYDRFG', 'RO',     0], # state ???
     ['HYDRFG', 'ROVOL',  1], # flux
     ['HYDRFG', 'SAREA',  0], # state ???
     ['HYDRFG', 'TAU',    0],
     ['HYDRFG', 'USTAR',  0],
     ['HYDRFG', 'VOL',    1], # state
     ['HYDRFG', 'VOLEV',  0], # flux

     ['CONSFG', 'COADDR', 0],
     ['CONSFG', 'COADEP', 0],
     ['CONSFG', 'COADWT', 0],
     ['CONSFG', 'CON',    0],
     ['CONSFG', 'ICON',   0],
     ['CONSFG', 'OCON',   0],
     ['CONSFG', 'ROCON',  0],
     ['HTFG',   'AIRTMP', 0],
     ['HTFG',   'HTCF4',  0],
     ['HTFG',   'HTEXCH', 0],
     ['HTFG',   'IHEAT',  0],
     ['HTFG',   'OHEAT',  0],
     ['HTFG',   'ROHEAT', 0],
     ['HTFG',   'SHDFAC', 0],
     ['HTFG',   'TW',     0],
     ['SEDFG',  'BEDDEP', 0],
     ['SEDFG',  'DEPSCR', 0],
     ['SEDFG',  'ISED',   0],
     ['SEDFG',  'OSED',   0],
     ['SEDFG',  'ROSED',  0],
     ['SEDFG',  'RSED',   0],
     ['SEDFG',  'SSED',   0],
     ['SEDFG',  'TSED',   0],
     ['GQALFG', 'ADQAL',  0],
     ['GQALFG', 'DDQAL',  0],
     ['GQALFG', 'DQAL',   0],
     ['GQALFG', 'DSQAL',  0],
     ['GQALFG', 'GQADDR', 0],
     ['GQALFG', 'GQADEP', 0],
     ['GQALFG', 'GQADWT', 0],
     ['GQALFG', 'IDQAL',  0],
     ['GQALFG', 'ISQAL',  0],
     ['GQALFG', 'ODQAL',  0],
     ['GQALFG', 'OSQAL',  0],
     ['GQALFG', 'PDQAL',  0],
     ['GQALFG', 'RDQAL',  0],
     ['GQALFG', 'RODQAL', 0],
     ['GQALFG', 'ROSQAL', 0],
     ['GQALFG', 'RRQAL',  0],
     ['GQALFG', 'RSQAL',  0],
     ['GQALFG', 'SQAL',   0],
     ['GQALFG', 'SQDEC',  0],
     ['GQALFG', 'TIQAL',  0],
     ['GQALFG', 'TOSQAL', 0],
     ['GQALFG', 'TROQAL', 0],
     ['OXFG',   'BOD',    0],
     ['OXFG',   'DOX',    0],
     ['OXFG',   'OXCF1',  0],
     ['OXFG',   'OXCF2',  0],
     ['OXFG',   'OXCF3',  0],
     ['OXFG',   'OXCF4',  0],
     ['OXFG',   'OXIF',   0],
     ['OXFG',   'SATDO',  0],
     ['NUTFG',  'DNUST',  0],
     ['NUTFG',  'DNUST2', 0],
     ['NUTFG',  'NUADDR', 0],
     ['NUTFG',  'NUADEP', 0],
     ['NUTFG',  'NUADWT', 0],
     ['NUTFG',  'NUCF1',  0],
     ['NUTFG',  'NUCF2',  0],
     ['NUTFG',  'NUCF3',  0],
     ['NUTFG',  'NUCF4',  0],
     ['NUTFG',  'NUCF5',  0],
     ['NUTFG',  'NUCF6',  0],
     ['NUTFG',  'NUCF7',  0],
     ['NUTFG',  'NUCF8',  0],
     ['NUTFG',  'NUCF9',  0],
     ['NUTFG',  'NUIF1',  0],
     ['NUTFG',  'NUIF2',  0],
     ['NUTFG',  'NUST',   0],
     ['NUTFG',  'OSNH4',  0],
     ['NUTFG',  'OSPO4',  0],
     ['NUTFG',  'RSNH4',  0],
     ['NUTFG',  'RSPO4',  0],
     ['NUTFG',  'SNH4',   0],
     ['NUTFG',  'SPO4',   0],
     ['NUTFG',  'TNUCF1', 0],
     ['NUTFG',  'TNUCF2', 0],
     ['NUTFG',  'TNUIF',  0],
     ['PLKFG',  'PHYTO',  0],
     ['PLKFG',  'ZOO',    0],
     ['PLKFG',  'BENAL',  0],
     ['PLKFG',  'TBENAL', 0],
     ['PLKFG',  'PHYCLA', 0],
     ['PLKFG',  'BALCLA', 0],
     ['PLKFG',  'PKST3',  0],
     ['PLKFG',  'PKST4',  0],
     ['PLKFG',  'PKIF',   0],
     ['PLKFG',  'TPKIF',  0],
     ['PLKFG',  'PKCF1',  0],
     ['PLKFG',  'TPKCF1', 0],
     ['PLKFG',  'PKCF2',  0],
     ['PLKFG',  'TPKCF2', 0],
     ['PLKFG',  'PLADDR', 0],
     ['PLKFG',  'PLADWT', 0],
     ['PLKFG',  'PLADEP', 0],
     ['PLKFG',  'PKCF5',  0],
     ['PLKFG',  'PKCF6',  0],
     ['PLKFG',  'PKCF7',  0],
     ['PLKFG',  'TPKCF7', 0],
     ['PLKFG',  'PKCF8',  0],
     ['PLKFG',  'PKCF9',  0],
     ['PLKFG',  'PKCF10', 0],
     ['PHFG',   'PHCF1',  0],
     ['PHFG',   'PHCF2',  0],
     ['PHFG',   'PHCF3',  0],
     ['PHFG',   'PHIF',   0],
     ['PHFG',   'PHST',   0],
     ['PHFG',   'SATCO2', 0]]
    df = pd.DataFrame(d, columns=['Flag', 'Name', 'Value'])
    df.to_hdf(h2name, '/SAVE/RCHRES', format='t', data_columns=True)


    '''
    Groups control how the HDF5 file is organized and defines the processing
    for each UCI table.
    Column 1 designates the OP SEQUENCE flag associated with the data

    Column 2 is the UCI Table name and should match HSPF docs

    Column 3 (HDFGroup) defines where that table's data is stored in the HDF5 file
        #QUALID is replaced by the last qualid value.
	#PESTID is replaced by the last pestid value.
	#CONID is replaced by the last conid value.
	#GQUALID is replaced by the last gqualid value.
	#LAYER4 is replaced by one of ['Surface', 'Upper', 'Lower', 'Ground']  per layerid
	#+ is replace by the number of the line in the multiline table
    Column 4 (HDFType) defines the processing needed for that table:
	*  Not Implimented
        '' indicates normal processing and storage in HDF5
        MONTHLY indicates that the line has 12 monthly values for a variable. In
            this case, Column 5 (Data) has the expected Fortran name to associate
            with the data when it is stored. The names in the sequence file and table
            may be very different! For example, the table MON-MANNING has fields named
            MANXXX (XXX is month like JAN), but the code gets the name NSURM.
            The data is stored in the HDF5 location, HDFGroup + name.
	MONTHLY4 - same as MONTHLY, but table repeated for Surface, Upper, Lower, Ground
	LAYER4, Line repeats for Surface, Upper, Lower, Ground
	SC  indicates the table is read twice, first for Salt, then for Clay
	LAYER4, Line repeats for Surface, Upper, Lower, Ground
        +   indicates that the table's data is spread on consecutive lines which
            should be combined before storing in the HDF5 file. Column 5 (Data) has
            3 values: the name of the data for the HDF5 file, the (maximum) number items per  line,
	    and the total number of expected data items.

    Column 5 is data specific to the HDFType.
    '''

    perlndgroups = [
      ['ACTIVITY','ACTIVITY',      '',                           '',        ''],
      ['GENERAL_INFO','GEN-INFO',  '',                           '',        ''],
      ['AIRTFG', 'ATEMP-DAT',      'PARAMETERS',                 '',        ''],
      ['SNOWFG', 'ICE-FLAG',       'FLAGS',                      '',        ''],
      ['SNOWFG', 'SNOW-FLAGS',     'FLAGS',                      '',        ''],
      ['SNOWFG', 'SNOW-INIT1',     'STATE',                      '',        ''],
      ['SNOWFG', 'SNOW-INIT2',     'STATE',                      '',        ''],
      ['SNOWFG', 'SNOW-PARM1',     'PARAMETERS',                 '',        ''],
      ['SNOWFG', 'SNOW-PARM2',     'PARAMETERS',                 '',        ''],
      ['SNOWFG', 'MON-MELT-FAC',   'MONTHLY/KMELTM',             'MONTHLY', ''],
      ['PWATFG', 'PWAT-PARM1',     'FLAGS',                      '',        ''],
      ['PWATFG', 'PWAT-STATE1',    'STATE',                      '',        ''],
      ['PWATFG', 'PWAT-PARM2',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'PWAT-PARM3',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'PWAT-PARM4',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'PWAT-PARM5',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'PWAT-PARM6',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'PWAT-PARM7',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'MON-INTERCEP',   'MONTHLY/CEPSCM',             'MONTHLY', ''],
      ['PWATFG', 'MON-UZSN',       'MONTHLY/UZSNM',              'MONTHLY', ''],
      ['PWATFG', 'MON-MANNING',    'MONTHLY/NSURM',              'MONTHLY', ''],
      ['PWATFG', 'MON-INTERFLW',   'MONTHLY/INTFWM',             'MONTHLY', ''],
      ['PWATFG', 'MON-LZETPARM',   'MONTHLY/LZETPM',             'MONTHLY', ''],
      ['PWATFG', 'MON-IRC',        'MONTHLY/IRCM',               'MONTHLY', ''],
      ['PWATFG', 'IRRIG-PARM1',    'PARAMETERS',                 '',        ''],
      ['PWATFG', 'IRRIG-PARM2',    'PARAMETERS',                 '',        ''],
      ['PWATFG', 'CROP-DATES',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'CROP-STAGES',    'PARAMETERS',                 '',        ''],   # duplicated for each crop above??? Assume NO
      ['PWATFG', 'CROP-SEASPM',    'PARAMETERS',                 '',        ''],  # duplicated for each crop above??? Aasume NO
      ['PWATFG', 'SOIL-DATA',      'PARAMETERS',                 '',        ''],
      ['PWATFG', 'SOIL-DATA2',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'SOIL-DATA3',     'PARAMETERS',                 '',        ''],
      ['PWATFG', 'MON-IRR-CRDP',   'MONTHLY/CRDEPM',             'MONTHLY', ''],
      ['PWATFG', 'MON-IRR-AWD',    'MONTHLY/IRAWDM',             'MONTHLY', ''],
      ['PWATFG', 'IRRIG-SCHED',    'PARAMETERS',                 '',        ''],  # repeated NSKED times, 2 per line for each PERLND, names unique on ONE line
      ['PWATFG', 'IRRIG-SOURCE',   'PARAMETERS',                 '',        ''],
      ['PWATFG', 'IRRIG-TARGET',   'PARAMETERS',                 '',        ''],
      ['SEDFG',  'SED-PARM1',      'FLAGS',                      '',        ''],
      ['SEDFG',  'SED-PARM2',      'PARAMETERS',                 '',        ''],
      ['SEDFG',  'SED-PARM3',      'PARAMETERS',                 '',        ''],
      ['SEDFG',  'MON-COVER',      'MONTHLY/COVERM',             'MONTHLY', ''],
      ['SEDFG',  'MON-NVSI',       'MONTHLY/NVSIM',              'MONTHLY', ''],
      ['SEDFG',  'SED-STOR',       'PARAMETERS',                 '',        ''],
      ['PSTFG',  'PSTEMP-PARM1',   'FLAGS',                      '',        ''],
      ['PSTFG',  'PSTEMP-PARM2',   'PARAMETERS',                 '',        ''],
      ['PSTFG',  'PSTEMP-TEMPS',   'STATE',                      '',        ''],
      ['PSTFG',  'MON-ASLT',       'MONTHLY/ASLTM',              'MONTHLY', ''],
      ['PSTFG',  'MON-BSLT',       'MONTHLY/BSLTM',              'MONTHLY', ''],
      ['PSTFG',  'MON-ULTP1',      'MONTHLY/ULTP1M',             'MONTHLY', ''],
      ['PSTFG',  'MON-ULTP2',      'MONTHLY/ULTP2M',             'MONTHLY', ''],
      ['PSTFG',  'MON-LGTP1',      'MONTHLY/LGTP1M',             'MONTHLY', ''],
      ['PSTFG',  'MON-LGTP2',      'MONTHLY/LGTP2M',             'MONTHLY', ''],
      ['PWGFG',  'PWT-PARM1',      'FLAGS',                      '',        ''],
      ['PWGFG',  'PWT-PARM2',      'PARAMETERS',                 '',        ''],
      ['PWGFG',  'LAT-FACTOR',     'PARAMETERS',                 '',        ''],
      ['PWGFG',  'MON-IFWDOX',     'MONTHLY/IDOXPM',             'MONTHLY', ''],
      ['PWGFG',  'MON-IFWCO2',     'MONTHLY/ICO2PM',             'MONTHLY', ''],
      ['PWGFG',  'MON-GRNDDOX',    'MONTHLY/ADOXPM',             'MONTHLY', ''],
      ['PWGFG',  'MON-GRNDCO2',    'MONTHLY/ACO2PM',             'MONTHLY', ''],
      ['PWGFG',  'PWT-TEMPS',      'STATE',                      '',        ''],
      ['PWGFG',  'PWT-GASES',      'STATE',                      '',        ''],
      ['MSTLFG', 'VUZFG',          'FLAGS',                      '',        ''],
      ['MSTLFG', 'UZSN-LZSN',      'STATE',                      '',        ''],
      ['MSTLFG', 'MST-PARM',       'PARAMETERS',                 '',        ''],
      ['MSTLFG', 'MST-TOPSTOR',    'STATE',                      '',        ''],
      ['MSTLFG', 'MST-TOPFLX',     'STATE',                      '',        ''],
      ['MSTLFG', 'MST-SUBSTOR',    'STATE',                      '',        ''],
      ['MSTLFG', 'MST-SUBFLX',     'STATE',                      '',        ''],
      ['PQALFG', 'NQUALS',         'PARAMETERS',                 '',        ''],
      ['PQALFG', 'PQL-AD-FLAGS',   'FLAGS',                      '',        ''],
      ['PQALFG', 'QUAL-PROPS',     '#QUALID/FLAGS',              '#',        ''],  # repeated for eqch qual
      ['PQALFG', 'QUAL-INPUT',     '#QUALID/PARAMETERS',         '#',        ''],
      ['PQALFG', 'MON-POTFW',      '#QUALID/MONTHLY/POTFWM',     'QMONTHLY', ''],
      ['PQALFG', 'MON-POTFS',      '#QUALID/MONTHLY/POTFSM',     'QMONTHLY', ''],
      ['PQALFG', 'MON-ACCUM',      '#QUALID/MONTHLY/ACQOPM',     'QMONTHLY', ''],
      ['PQALFG', 'MON-SQOLIM',     '#QUALID/MONTHLY/SQOLIM',     'QMONTHLY', ''],
      ['PQALFG', 'MON-IFLW-CONC',  '#QUALID/MONTHLY/IOQCM',      'QMONTHLY', ''],
      ['PQALFG', 'MON-GRND-CONC',  '#QUALID/MONTHLY/AOQCM',      'QMONTHLY', ''],
      ['PESTFG', 'PEST-FLAGS',     'FLAGS',                      '',        ''],
      ['PESTFG', 'PEST-AD-FLAGS',  'FLAGS',                      '',        ''],
      ['PESTFG', 'PEST-ID',        '#PESTID/FLAGS',              '#',       ''],  # The following tables repeated for each pesticide (max 3)
      ['PESTFG', 'PEST-THETA',     '#PESTID/PARAMETERS',         '#',       ''],
      ['PESTFG', 'PEST-FIRSTPM',   '#PESTID/PARAMETERS',         '#',       ''],
      ['PESTFG', 'PEST-CMAX',      '#PESTID/PARAMETERS',         '#',       ''],
      ['PESTFG', 'PEST-SVALPM',    '#PESTID/#LAYER4/PARAMETERS', 'LAYER4',  ''],
      ['PESTFG', 'PEST-NONSVPM',   '#PESTID/PARAMETERS',         '#',       ''],
      ['PESTFG', 'PEST-DEGRAD',    '#PESTID/PARAMETERS',         '#',       ''],
      ['PESTFG', 'PEST-STOR1',     '#PESTID/#LAYER4/STATE',      'LAYER4',  ''],
      ['PESTFG', 'PEST-STOR2',     '#PESTID/STATE',              '#',       ''],
      ['NITRFG', 'NIT-FLAGS',      'FLAGS',                      '',        ''],
      ['NITRFG', 'NIT-AD-FLAGS',   'FLAGS',                      '',        ''],
      ['NITRFG', 'NIT-FSTGEN',     'PARAMETERS',                 '',        ''],
      ['NITRFG', 'NNIT-FSTPM',     '#LAYER4/PARAMETERS',         'LAYER4',  ''],
      ['NITRFG', 'NNIT-ORGPM',     '#LAYER4/PARAMETERS',         'LAYER4',  ''],
      ['NITRFG', 'NIT-AMVOLAT',    'PARAMETERS',                 '',        ''],
      ['NITRFG', 'NIT-CMAX',       'PARAMETERS',                 '',        ''],
      ['NITRFG', 'NIT-SVALPM',     'PARAMETERS',                 '',        ''],
      ['NITRFG', 'NIT-UPTAKE',     'PARAMETERS',                 '',        ''],
      ['NITRFG', 'MON-NITUPT',     '#LAYER4/MONTHLY/',           'MONTHLY4','KPLNM'],
      ['NITRFG', 'NIT-YIELD',      'PARAMETERS',                 '',        ''],
      ['NITRFG', 'MON-NUPT-FR1',   'MONTHLY/KPLNM',              'MONTHLY', ''],
      ['NITRFG', 'MON-NUPT-FR2',   '#LAYER4/MONTHLY/',           'MONTHLY4','SNUPTM'],
      ['NITRFG', 'NIT-UPIMCSAT',   '#LAYER4/PARAMETERS',         'LAYER4',  ''],
      ['NITRFG', 'NIT-UPIMKMAX',   '#LAYER4PARAMETERS',          'LAYER4',  ''],
      ['NITRFG', 'MON-NITUPNI',    '#LAYER4/MONTHLY/',           'MONTHLY4','KUNIM'],
      ['NITRFG', 'MON-NITUPAM',    '#LAYER4/MONTHLY/',           'MONTHLY4','KUAMM'],
      ['NITRFG', 'MON-NITIMNI',    '#LAYER4/MONTHLY/',           'MONTHLY4','KINIM'],
      ['NITRFG', 'MON-NITIMAM',    '#LAYER4/MONTHLY/',           'MONTHLY4','KIAMM'],
      ['NITRFG', 'NIT-BGPLRET',    'PARAMETERS',                 '',        ''],
      ['NITRFG', 'MON-NPRETBG',    '#LAYER4/MONTHLY/',           'MONTHLY4','KRBNM'],
      ['NITRFG', 'MON-NPRETFBG',   'MONTHLY/BNPRFM',             '',        ''],
      ['NITRFG', 'NIT-AGUTF',      'PARAMETERS',                 '',        ''],
      ['NITRFG', 'MON-NITAGUTF',   '#LAYER4/MONTHLY/',           'MONTHLY4','ANUFM'],
      ['NITRFG', 'NIT-AGPLRET',    'PARAMETERS',                 '',        ''],
      ['NITRFG', 'MON-NPRETAG',    'MONTHLY/KRANM',              'MONTHLY', ''],
      ['NITRFG', 'MON-NPRETLI',    '#LAYER4/MONTHLY/',           'MONTHLY4','KRLNM'],   # only 2, but should break out OK
      ['NITRFG', 'MON-NPRETFLI',   'MONTHLY/LNPRFM',             'MONTHLY', ''],
      ['NITRFG', 'NIT-STOR1',      '#LAYER4/STATE',              'LAYER4',  ''],  # not really sure repeated
      ['NITRFG', 'NIT-STOR2',      'STATE',                      '',        ''],
      ['PHOSFG', 'PHOS-FLAGS',     'FLAGS',                      '',        ''],
      ['PHOSFG', 'PHOS-AD-FLAGS',  'FLAGS',                      '',        ''],
      ['PHOSFG', 'PPHOS-FSTGEN',   'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'PHOS-FSTPM',     'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'PHOS-CMAX',      'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'PHOS-SVALPM',    'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'PHOS-UPTAKE',    'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'MON-PHOSUPT',    '#LAYER4/MONTHLY/',           'MONTHLY4','KPLPM'],
      ['PHOSFG', 'PHOS-YIELD',     'PARAMETERS',                 '',        ''],
      ['PHOSFG', 'MON-PUPT-FR1',   'MONTHLY/PUPTFM',             'MONTHLY', ''],
      ['PHOSFG', 'MON-PUPT-FR2',   '#LAYER4/MONTHLY/',           'MONTHLY4','SPUPTM UPUPTM LPUPTM APUPTM'],
      ['PHOSFG', 'PHOS-STOR1',     '#LAYER4/STATE',              'LAYER4',  ''],  # ???? repeated for SZ, UZ, LZ, GW
      ['PHOSFG', 'PHOS-STOR2',     'STATE',                      '',        ''],
      ['TRACFG', 'TRAC-ID',        'PARAMETERS',                 '',        ''],
      ['TRACFG', 'TRAC-AD-FLAGS',  'FLAGS',                      '',        ''],
      ['TRACFG', 'TRAC-TOPSTOR',   'STATE',                      '',        ''],
      ['TRACFG', 'PTRAC-SUBSTOR',  'STATE',                      '',        '']]
    df = pd.DataFrame(perlndgroups, columns=['Flag','Table','HDFGroup','HDFType','Data'])
    df.to_hdf(h2name, '/PARSEDATA/PERLNDGROUPS', format='t', data_columns=True)

    implndgroups = [
      ['ACTIVITY','ACTIVITY',     '',                  '',        ''],
      ['GENERAL_INFO','GEN-INFO', '',                  '',        ''],
      ['ATMPFG', 'ATEMP-DAT',    'PARAMETERS',         '',        ''],
      ['SNOWFG', 'ICE-FLAG',     'FLAGS',              '',        ''],
      ['SNOWFG', 'SNOW-FLAGS',   'FLAGS',              '',        ''],
      ['SNOWFG', 'SNOW-INIT1',   'STATE',              '',        ''],
      ['SNOWFG', 'SNOW-INIT2',   'STATE',              '',        ''],
      ['SNOWFG', 'SNOW-PARM1',   'PARAMETERS',         '',        ''],
      ['SNOWFG', 'SNOW-PARM2',   'PARAMETERS',         '',        ''],
      ['SNOWFG', 'MON-MELT-FAC', 'MONTHLY/KMELTM',     'MONTHLY', ''],
      ['IWATFG', 'IWAT-STATE1',  'STATE',              '',        ''],
      ['IWATFG', 'IWAT-PARM1',   'FLAGS',              '',        ''],
      ['IWATFG', 'IWAT-PARM2',   'PARAMETERS',         '',        ''],
      ['IWATFG', 'IWAT-PARM3',   'PARAMETERS',         '',        ''],
      ['IWATFG', 'MON-RETN',     'MONTHLY/RETSCM',     'MONTHLY', ''],
      ['IWATFG', 'MON-MANNING',  'MONTHLY/NSURM',      'MONTHLY', ''],
      ['SLDFG',  'SLD-PARM1',    'FLAGS',              '',        ''],
      ['SLDFG',  'SLD-PARM2',    'PARAMETERS',         '',        ''],
      ['SLDFG',  'SLD-STOR',     'STATE',              '',        ''],
      ['SLDFG',  'MON-SACCUM',   'MONTHLY/ACCSDM',     'MONTHLY', ''],
      ['SLDFG',  'MON-REMOVG',   'MONTHLY/REMSDM',     'MONTHLY', ''],
      ['IWGFG',  'IWT-PARM1',    'PARAMETERS',         '',        ''],
      ['IWGFG',  'IWT-PARM2',    'PARAMETERS',         '',        ''],
      ['IWGFG',  'LAT-FACTOR',   'PARAMETERS',         '',        ''],
      ['IWGFG',  'MON-AWTF',     'MONTHLY/AWTFM',      'MONTHLY', ''],
      ['IWGFG',  'MON-BWTF',     'MONTHLY/BWTFM',      'MONTHLY', ''],
      ['IWGFG',  'IWT-INIT',     'STATE',              '',        ''],
      ['IQALFG', 'NQUALS',       'PARAMETERS',         '',        ''],
      ['IQALFG', 'IQL-AD-FLAGS', 'FLAGS',              '',        ''], # OK - not repeated
      ['IQALFG', 'QUAL-PROPS',   '#QUALID/FLAGS',      '#',       ''],
      ['IQALFG', 'QUAL-INPUT',   '#QUALID/PARAMETERS', '#',       ''],
      ['IQALFG', 'MON-POTFW',    '#QUALID/MONTHLY/',   '#MONTHLY', 'POTFWM'],
      ['IQALFG', 'MON-ACCUM',    '#QUALID/MONTHLY/',   '#MONTHLY', 'ACQOPM'],
      ['IQALFG', 'MON-SQOLIM',   '#QUALID/MONTHLY/',   '#MONTHLY', 'SQOLIM']]
    df = pd.DataFrame(implndgroups, columns=['Flag','Table','HDFGroup','HDFType','Data'])
    df.to_hdf(h2name, '/PARSEDATA/IMPLNDGROUPS', format='t', data_columns=True)

    rchresgroups = [
      ['ACTIVITY','ACTIVITY',      '',                             '',      ''],
      ['GENERAL_INFO','GEN-INFO',  '',                             '',      ''],
      ['HYDRFG',  'HYDR-PARM1',    'FLAGS',                        '',      ''],
      ['HYDRFG',  'HYDR-INIT',     'FLAGS',                        '',      ''],
      ['HYDRFG',  'HYDR-PARM2',    'PARAMETERS',                   '',      ''],
      ['HYDRFG',  'HYDR-IRRIG',    'PARAMETERS',                   '',      ''],
      ['HYDRFG',  'MON-CONVF',     'MONTHLY/CONVFM',               'MONTHLY',''],
      ['HYDRFG',  'HYDR-CATEGORY1','FLAGS',                        '*',     ''],  # ??? repeated  until
      ['HYDRFG',  'HYDR-CINIT',    'STATE',                        '*',     ''],  # ??? repeated (7 per line) until
      ['HYDRFG',  'HYDR-CPREC',    'PARAMETERS',                   '*',     ''],  # ??? repeated (7 per line) until
      ['HYDRFG',  'HYDR-CEVAP',    'PARAMETERS',                   '*',     ''],  # ??? repeated (5 per line) until
      ['HYDRFG',  'HYDR-CFVOL',    'PARAMETERS',                   '*',     ''],
      ['HYDRFG',  'HYDR-CDEMAND',  'PARAMETERS',                   '*',     ''],  # ??? repeated  until
      ['ADFG',    'ADCALC-DATA',   'PARAMETERS',                   '',      ''],
      ['HTFG',    'HT-BED-FLAGS',  'FLAGS',                        '',      ''],
      ['HTFG',    'HEAT-PARM',     'PARAMETERS',                   '',      ''],
      ['HTFG',    'HT-BED-PARM',   'PARAMETERS',                   '',      ''],
      ['HTFG',    'MON-HT-TGRND',  'MONTHLY/TGRNDM',               'MONTHLY',''],
      ['HTFG',    'HT-BED-DELH',   'PARAMETERS/DELH#+',            '+',     'DELH'],   #'DELH 7 100'],
      ['HTFG',    'HT-BED-DELTT',  'STATE/DELTT#+',                '+',     'DELTT'],  # 'DELTT 7 100'],
      ['HTFG',    'HEAT-INIT',     'STATE',                        '',      ''],
      ['HTFG',    'SHADE-PARM',    'PARAMETERS',                   '',      ''],
      ['SEDFG',   'SANDFG',        'FLAGS',                        '',      ''],
      ['SEDFG',   'SED-GENPARM',   'PARAMETERS',                   '',      ''],
      ['SEDFG',   'SED-HYDPARM',   'PARAMETERS',                   '',      ''],
      ['SEDFG',   'SAND-PM',       'SAND/PARAMETERS',              '',      ''],
      ['SEDFG',   'SILT-CLAY-PM',  'PARAMETERS/SC#+',              '+',     'SILT'],
      ['SEDFG',   'SSED-INIT',     'STATE',                        '',      ''],
      ['SEDFG',   'BED-INIT',      'STATE',                        '',      ''],
      ['GQALFG',  'GQ-GENDATA',    'FLAGS',                        '',      ''],
      ['GQALFG',  'GQ-AD-FLAGS',   'FLAGS',                        '',      ''],
      ['GQALFG',  'GQ-QALDATA',    '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-QALFG',      '#GQUALID/FLAGS',               '#',     ''],
      ['GQALFG',  'GQ-FLG2',       '#GQUALID/FLAGS',               '#',     ''],
      ['GQALFG',  'GQ-HYDPM',      '#GQUALID/PARAMETERS',          '#',     ''],  # ???  repeated
      ['GQALFG',  'GQ-ROXPM',      '#GQUALID/PARAMETERS',          '#',     ''],  # ??? repeated
      ['GQALFG',  'GQ-PHOTPM',     '#GQUALID/PARAMETERS/PHOTPM#+', '+',     ''],   #'PHOTPM 7 20'],  # need to fix for PHI, THETA
      ['GQALFG',  'GQ-CFGAS',      '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-BIOPM',      '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'MON-BIO',       '#GQUALID/MONTHLY/',            '#MONTHLY','BIOM'], # probably repeated per qual
      ['GQALFG',  'GQ-GENDECAY',   '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-SEDDECAY',   '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-KD',         '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-ADRATE',     '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-ADTHETA',    '#GQUALID/PARAMETERS',          '#',     ''],
      ['GQALFG',  'GQ-SEDCONC',    '#GQUALID/STATE',               '#',     ''],
      ['GQALFG',  'GQ-VALUES',     '#GQUALID/STATE',               '#',     ''],
      ['GQALFG',  'MON-WATEMP',    'MONTHLY/TEMPM',                'MONTHLY',''],
      ['GQALFG',  'MON-PHVAL',     'MONTHLY/PHVALM',               'MONTHLY',''],
      ['GQALFG',  'MON-ROXYGEN',   'MONTHLY/ROCM',                 'MONTHLY',''],
      ['GQALFG',  'GQ-ALPHA',      'PARAMETERS/ALPHA#+',           '+',      'ALPH'],   #'ALPH 7 18'],
      ['GQALFG',  'GQ-GAMMA',      'PARAMETERS/GAMMA#+',           '+',      'GAMM'],   #'GAMM 7 18'],
      ['GQALFG',  'GQ-DELTA',      'PARAMETERS/DELTA#+',           '+',      'DEL_'],   # 'DEL_  7 18'],
      ['GQALFG',  'GQ-CLDFACT',    'PARAMETERS/CLDFACT/#+',        '+',      'KCLD'],   # 'KCLD 7 18'],
      ['GQALFG',  'MON-CLOUD',     'MONTHLY/CLDM',                 'MONTHLY',''],
      ['GQALFG',  'MON-SEDCONC',   'MONTHLY/SDCNCM',               'MONTHLY',''],
      ['GQALFG',  'MON-PHYTO',     'MONTHLY/PHYM',                 'MONTHLY',''],
      ['GQALFG',  'GQ-DAUGHTER',   'PARAMETERS/#+',                '+',    'C'],         #'C 49 7'],  # repeated a lot!
      ['GQALFG',  'BENTH-FLAG',    'FLAGS',                        '',      ''],
      ['GQALFG',  'SCOUR-PARMS',   'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-FLAGS',      'FLAGS',                        '',      ''],
      ['OXFG',    'OX-GENPARM',    'PARAMETERS',                   '',      ''],
      ['OXFG',    'ELEV',          'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-BENPARM',    'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-CFOREA',     'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-TSIVOGLOUA', 'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-LEN-DELTH',  'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-TCGINV',     'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-REAPARM',    'PARAMETERS',                   '',      ''],
      ['OXFG',    'OX-INIT',       'STATE',                        '',      ''],
      ['NUTFG',   'NUT-FLAGS',     'FLAGS',                        '',      ''],
      ['NUTFG',   'NUT-AD-FLAGS',  'FLAGS',                        '',      ''],
      ['NUTFG',   'CONV-VAL1S',    'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-BENPARM',   'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-NITDENIT',  'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-NH3VOLAT',  'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-BEDCONC',   'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-ADSPARM',   'PARAMETERS',                   '',      ''],
      ['NUTFG',   'NUT-DINIT',     'STATE',                        '',      ''],
      ['NUTFG',   'NUT-ADSINIT',   'STATE',                        '',      ''],
      ['PLKFG',   'PLNK-FLAGS',    'FLAGS',                        '',      ''],
      ['PLKFG',   'BENAL-FLAGS',   'FLAGS',                        '',      ''],
      ['PLKFG',   'PLNK-AD-FLAGS', 'FLAGS',                        '',      ''],
      ['PLKFG',   'SURF-EXPOSED',  'PARAMETERS',                   '',      ''],
      ['PLKFG',   'PLNK-PARM1',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'PLNK-PARM2',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'PLNK-PARM3',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'PLNK-PARM4',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'PHYTO-PARM',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'ZOO-PARM1',     'PARAMETERS',                   '',      ''],
      ['PLKFG',   'ZOO-PARM2',     'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-PARM',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-GROW',    'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-RESSCR',  'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-GRAZE',   'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-LIGHT',   'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-RIFF1',   'PARAMETERS',                   '',      ''],
      ['PLKFG',   'BENAL-RIFF2',   'PARAMETERS',                   '',      ''],
      ['PLKFG',   'MON-BINV',      'MONTHLY/BINVM',                'MONTHLY',''],
      ['PLKFG',   'PLNK-INIT',     'STATE',                        '',      ''],
      ['PLKFG',   'BENAL-INIT',    'STATE',                        '',      ''],
      ['PHFG',    'PH-PARM1',      'PARAMETERS',                   '',      ''],
      ['PHFG',    'PH-PARM2',      'PARAMETERS',                   '',      ''],
      ['PHFG',    'PH-INIT',       'STATE',                        '',      ''],
      ['CONSFG',  'NCONS',         'PARAMETERS',                   '',      ''],
      ['CONSFG',  'CONS-DATA',     '#CONID/PARAMETERS',            '#',     ''],  # repeated NQUAL times
      ['CONSFG',  'CONS-AD-FLAGS', 'FLAGS',                        '',      ''],
      ]
    df = pd.DataFrame(rchresgroups, columns=['Flag','Table','HDFGroup','HDFType','Data'])
    df.to_hdf(h2name, '/PARSEDATA/RCHRESGROUPS', format='t', data_columns=True)


    # The next set of DataFrames provide parsing data for various other UCI tables
    d = [['Depth',   0, 10, 10],
         ['Area',   10, 20, 10],
         ['Volume', 20, 30, 10],
         ['Disch1', 30, 40, 10],
         ['Disch2', 40, 50, 10],
         ['Disch3', 50, 60, 10],
         ['Disch4', 60, 70, 10],
         ['Disch5', 70, 80, 10]]
    df = pd.DataFrame(d, columns=['Variable', 'Column', 'End', 'Width'])
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/FTABLES', format='t', data_columns=True)

    cols = ['type', 'Variable', 'Column', 'End', 'Width', 'Default']
    d = [['START', 'SYR', 14, 18, 4, ''],
         ['START', 'SMO', 19, 21, 2, '01'],
         ['START', 'SDA', 22, 24, 2, '01'],
         ['START', 'SHR', 25, 27, 2, '00'],
         ['START', 'SMI', 28, 30, 2, '00'],
         ['START', 'EYR', 39, 43, 4, ''],
         ['START', 'EMO', 44, 46, 2, '12'],
         ['START', 'EDA', 47, 49, 2, '31'],
         ['START', 'EHR', 50, 52, 2, '23'],
         ['START', 'EMI', 53, 55, 2, '59'],
         ['INFO',  'RUNINFO', 2, 80, 78, ''],
         ['RESUME','EMFG',55, 60, 5, '1'],
         ['RESUME','IHMFG',65,70, 5, '0']
         ]
    df = pd.DataFrame(d, columns=cols)
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/GLOBAL', format='t', data_columns=True)

    # The code for the Type column is C->string, I->Integer, F->Float
    cols = ['Variable', 'Column', 'End', 'Width', 'Default', 'Type']
    d = [['SVOL',     0,   6,  6,  '',     'C'],
         ['SVOLNO',   6,  11,  5,  '',     'I'],
         ['SMEMN',   11,  17,  6,  '',     'C'],  # also SFCLAS same position
         ['SMEMSB',  17,  20,  3,  '31',   'C'],  # also SFNO same position
         ['SSYST',   20,  24,  4,  'ENGL', 'I'],
         ['SGAPST',  24,  28,  4,  'UNDF', 'C'],
         ['MFACTOR', 28,  38, 10,  '1.0',  'F'],
         ['TRAN',    38,  43,  5,  '',     'C'],
         ['TVOL',    43,  50,  7,  '',     'C'],
         ['TOPFST',  50,  54,  4,  '',     'I'],
         ['TOPLST',  54,  58,  4,  '',     'I'],
         ['TGRPN',   58,  65,  7,  '',     'C'],
         ['TMEMN',   65,  71,  6,  '',     'C'],
         ['TMEMSB',  71,  75,  4,  '',     'C']]
    df = pd.DataFrame(d, columns=cols)
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/EXT_SOURCES', format='t', data_columns=True)

    d = [['SVOL',     0,   6,  6,  '',     'C'],
         ['SVOLNO',   6,  10,  4,  '',     'I'],
         ['SGRPN',   11,  17,  6,  '',     'C'],
         ['SMEMN',   18,  24,  6,  '',     'C'],
         ['SMEMSB',  24,  28,  4,  '',     'C'],
         ['MFACTOR', 28,  38, 10,  '1.0',  'F'],
         ['TRAN',    38,  42,  4,  '',     'C'],
         ['TVOL',    43,  49,  6,  '',     'C'],
         ['TOPFST',  50,  53,  3,  '',     'I'],
         ['TOPLST',  54,  57,  3,  '',     'I'],
         ['TGRPN',   58,  64,  6,  '',     'C'],
         ['TMEMN',   65,  71,  6,  '',     'C'],
         ['TMEMSB',  71,  75,  4,  '',     'C']]
    df = pd.DataFrame(d, columns=cols)
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/NETWORK', format='t', data_columns=True)

    d = [['SVOL',    0,   6,  6,  '',    'C'],
         ['SVOLNO',  6,  10,  4,  '',    'I'],
         ['AFACTR', 28,  38, 10,  '1.0', 'F'],
         ['TVOL',   43,  49,  6,  '',    'C'], #documenation: col 43,etc
         ['TVOLNO', 49,  53,  4,  '',    'I'],
         ['MLNO',   56,  60,  4,  '',    'I'],
         ['TMEMSB', 71,  75,  4,  '',    'C']]
    df = pd.DataFrame(d, columns=cols)
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/SCHEMATIC', format='t', data_columns=True)

    d = [['SVOL',     0,   6,  6,  '',    'C'],
         ['SGRPN',   11,  17,  6,  '',    'C'],
         ['SMEMN',   18,  24,  6,  '',    'C'],
         ['SMEMSB',  24,  28,  4,  '',    'C'],
         ['MFACTOR', 28,  38, 10,  '1.0', 'F'],
         ['TVOL',    43,  49,  6,  '',    'C'],
         ['TGRPN',   58,  64,  6,  '',    'C'],
         ['TMEMN',   65,  71,  6,  '',    'C'],
         ['TMEMSB',  71,  75,  4,  '',    'C']]
    df = pd.DataFrame(d, columns=cols)
    del df['Width']
    df.to_hdf(h2name, '/PARSEDATA/MASS_LINK', format='t', data_columns=True)

    df = read_seq(sitepath, 'PERLND.SEQ')
    df.to_hdf(h2name, '/PARSEDATA/PERLND', format='t', data_columns=True)

    df = read_seq(sitepath, 'Implnd.seq')
    df.to_hdf(h2name, '/PARSEDATA/IMPLND', format='t', data_columns=True)

    df = read_seq(sitepath, 'RCHRES.SEQ')
    df['Source'] = 'RCHRES'
    df.to_hdf(h2name, '/PARSEDATA/RCHRES', format='t', data_columns=True)

    print('DONE')


def read_seq(path, seqfile):
    ''' reads specified sequence file and returns DataFrame with HSPF table information'''

    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    monthset = set(months)

    seq = os.path.join(path, seqfile)

    cols = ['Source','Table','Variable','Column','Width','Default','Type','Range','Group']
    rows = []
    with open(seq) as f:
        row = pd.Series(index=cols)
        for i, line in enumerate(f):
            #if '_HELP' in line:
            #    help = []
            #    for line in f:
            #        if line.startswith(('_', '$', '#')):
            #            break
            #        help.append(line)

            tokens = line.split()
            if len(tokens) > 1:
                keyword = tokens[0]
                value = tokens[1]

                if   keyword == '$TNAME':    table           = value
                elif keyword == '_COLUMN':   row['Column']   = int(value)-1
                elif keyword == '_WIDTH':    row['Width']    = int(value)
                elif keyword == '_DEFAULT':  row['Default']  = value
                elif keyword == '_TYPE':     row['Type']     = value[:1]
                elif keyword == '_RANGE':    row['Range']    = value
                elif keyword == '#GROUP':    group           = int(value)
                elif keyword == '_PNAME':
                    if 'MON-' in table and value != 'OPNID':
                        if value[-3:] in monthset: row['Variable'] = value[-3:]
                        elif value.endswith('12'): row['Variable'] = 'DEC'
                        elif value.endswith('11'): row['Variable'] = 'NOV'
                        elif value.endswith('10'): row['Variable'] = 'OCT'
                        elif value[-1].isdigit():  row['Variable'] = months[int(value[-1])-1]
                        else:
                            print(' '.join(['ERROR', table, value]))
                    else:
                        row['Variable'] =  value
                    row['Source']   = os.path.basename(seq)
                    row['Table']    = table
                    row['Group']    = group
                    rows.append(row)
                    row = pd.Series(index=cols)
    df = pd.DataFrame(rows)

    # apply fix for 12.2 seq file problems wiht OPNID length 8
    cond = (df['Variable']=='OPNID') & (df['Width']==8)
    for x in df[cond]['Table']:
        index = df[df['Table'] == x].index
        for indx in index[1:]:
            df.loc[indx,'Column'] = 2 + df.loc[indx,'Column']
        df.loc[index[0],'Width'] = 10       # fix OPNID without changing start

    # apply fix for 12.2 seq file problem, SANDFG field too short for test10 UCI
    cond = df['Table'] == 'SANDFG'
    df.loc[cond, 'Width'] = 10

    # finish and save
    df['Group']  = df['Group']. astype(int)
    df['Column'] = df['Column'].astype(int)
    df['Width']  = df['Width']. astype(int)
    df['End']    = df['Column'] + df['Width']
    del df['Width']
    return df
