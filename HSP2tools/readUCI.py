'''
Copyright 2020 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
'''

from collections import defaultdict
from pandas import Series, DataFrame, concat, HDFStore, set_option, to_numeric
from pandas import Timestamp, Timedelta, read_hdf
set_option('io.hdf.default_format', 'table')

import os.path
import HSP2tools


 # USERS may modify HSPF DataSets here
Lapse = Series ([0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0035, 0.0037,
 0.0040, 0.0041, 0.0043, 0.0046, 0.0047, 0.0048, 0.0049, 0.0050, 0.0050,
 0.0048, 0.0046, 0.0044, 0.0042, 0.0040, 0.0038, 0.0037, 0.0036])

Seasons = Series ([0,0,0, 1,1,1,1,1,1, 0,0,0]).astype(bool)

Svp = Series([1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005, 1.005,
 1.005, 1.01, 1.01, 1.015, 1.02, 1.03, 1.04, 1.06, 1.08, 1.1, 1.29, 1.66,
 2.13, 2.74,3.49, 4.40, 5.55,6.87, 8.36, 10.1,12.2,14.6, 17.5, 20.9, 24.8,
 29.3, 34.6, 40.7, 47.7, 55.7, 64.9])


# Users change default saving to HDF5 file for any computed time series here
Savedict = {
    'PERLND' : {
        'SNOW':   {'ALBEDO':0, 'COVINX':1, 'DEWTMP':0, 'DULL':1, 'MELT':0,
                   'NEGHTS':0, 'PACKF':1, 'PACKI':1, 'PACKW':1, 'PAKTMP':1,
                   'PDEPTH':0, 'PRAIN':0, 'RAINF':1, 'RDENPF':1, 'SKYCLR':1,
                   'SNOCOV':1, 'SNOTMP':0, 'SNOWE':0, 'SNOWF':0, 'WYIELD':1,
                   'XLNMLT':1},
        'PWATER': {'AGWET':0, 'AGWI':0, 'AGWO':1, 'AGWS':1, 'BASET':0,
                   'CEPE':0, 'CEPS':1, 'GWEL':0, 'GWVS':1, 'IFWI':0, 'IFWO':1,
                   'IFWS':1, 'IGWI':0, 'INFFAC':0, 'INFIL':0, 'IRDRAW':0,
                   'IRRAPP':0, 'IRRDEM':0, 'IRSHRT':0, 'LZET':0, 'LZI':0,
                   'LZS':1, 'RPARM':0, 'PERC':0, 'PERO':1, 'PERS':0, 'PET': 0,
                   'PETADJ':0, 'RPARM':0, 'RZWS':0, 'SUPY':0, 'SURET':0,
                   'SURI':0, 'SURO':1, 'SURS':1, 'TAET':0, 'TGWS':0, 'UZET':0,
                   'UZI':0, 'UZS':1},
        'SEDMNT': {'COVER':0, 'DET':0, 'DETS':0, 'NVSI':0, 'SCRSD':0,
                   'SOSED':0, 'STCAP':0, 'WSSD':0},
        'PSTEMP': {'AIRTC':0, 'LGTMP':0, 'SLTMP':0, 'ULTMP':0},
        'PWTGAS': {'AOCO2':0, 'AOCO2M':0, 'AODOX':0, 'AODOXM':0, 'AOHT':0,
                   'AOTMP':0, 'IOCO2':0, 'IOCO2M':0, 'IODOX':0, 'IODOXM':0,
                   'IOHT':0, 'IOTMP':0, 'POCO2M':0, 'PODOXM':0, 'POHT':0,
                   'SOCO2':0, 'SOCO2M':0, 'SODOX':0, 'SODOXM':0, 'SOHT':0,
                   'SOTMP':0},
        'PQUAL':  {'AOQC':0, 'AOQUAL':0, 'IOQC':0, 'IOQUAL':0, 'ISQOAL':0,
                   'POQC':0, 'POQUAL':0, 'PQADDR':0, 'PQADEP':0, 'PQADWT':0,
                   'SCRQS':0, 'SOQC':0, 'SOQO':0, 'SOQOC':0, 'SOQS':0,
                   'SOQSP':0, 'SOQUAL':0, 'SQO':0, 'WASHQS':0},
        'MSTLAY': {'MST':0, 'FRAC':0},
        'PEST':   {'ADEGPS':0, 'APS':0, 'IPS':0, 'LDEGPS':0, 'LPS':0,
                   'PEADDR':0, 'PEADEP':0, 'PEADWT':0, 'POPST':0, 'SDEGPS':0,
                   'SDPS':0, 'SOSDPS':0, 'SPS':0, 'SSPSS':0, 'TDEGPS':0,
                   'TOPST':0, 'TOTPST':0, 'TPS':0, 'TSPSS':0, 'UDEGPS':0,
                   'UPS':0},
        'NITR':   {'AGPLTN':0, 'AMIMB':0, 'AMNIT':0, 'AMUPA':0, 'AMUPB':0,
                   'AMVOL':0, 'AN':0, 'DENIF':0, 'IN':0, 'LITTRN':0, 'LN':0,
                   'NDFCT':0, 'NFIXFX':0, 'NIADDR':0, 'NIADEP':0, 'NIADWT':0,
                   'NIIMB':0, 'NITIF':0, 'NIUPA':0, 'NIUPB':0, 'NUPTG':0,
                   'ORNMN':0, 'PONH4':0, 'PONITR':0, 'PONO3':0, 'POORN':0,
                   'REFRON':0, 'RETAGN':0, 'RTLBN':0, 'RTLLN':0, 'RTRBN':0,
                   'RTRLN':0, 'SN':0, 'SOSEDN':0, 'SSAMS':0, 'SSNO3':0,
                   'SSSLN':0, 'SSSRN':0, 'TDENIF':0, 'TN':0, 'TNIT':0,
                   'TOTNIT':0, 'TSAMS':0, 'TSNO3':0, 'TSSLN':0, 'TSSRN':0,
                   'UN':0},
        'PHOS':   {'AP':0, 'IP':0, 'LP':0, 'ORPMN':0, 'P4IMB':0, 'PDFCT':0,
                   'PHADDR':0, 'PHADEP':0, 'PHADWT':0, 'PHOIF':0, 'POPHOS':0,
                   'PUPTG':0, 'SEDP':0, 'SOSEDP':0, 'SP':0, 'SSP4S':0,
                   'TOTPHO':0, 'TP':0, 'TPHO':0, 'TSP4S':0, 'UP':0},
        'TRACER': {'ATRSU':0, 'ITRSU':0, 'LTRSU':0, 'POTRS':0, 'SSTRS':0,
                   'STRSU':0, 'TRADDR':0, 'TRADEP':0, 'TRADWT':0, 'TRSU':0,
                   'TSTRS':0, 'UTRSU':0}},
    'IMPLND' : {
        'SNOW':   {'ALBEDO':0, 'COVINX':1, 'DEWTMP':0, 'DULL':1, 'MELT':0,
                   'NEGHTS':0, 'PACKF':1, 'PACKI':1, 'PACKW':1, 'PAKTMP':1,
                   'PDEPTH':0, 'PRAIN':0, 'RAINF':1, 'RDENPF':1, 'SKYCLR':1,
                   'SNOCOV':1, 'SNOTMP':0, 'SNOWE':0, 'SNOWF':0, 'WYIELD':1,
                   'XLNMLT':1},
        'IWATER': {'IMPEV':0, 'PET':0, 'PETADJ':0, 'RETS':1,  'SUPY':0,
                   'SURI':0, 'SURO':1, 'SURS':1},
        'SOLIDS': {'SLDS':0, 'SOSLD':0},
        'IWTGAS': {'SOTMP':0, 'SODOX':0, 'SOCO2':0, 'SOHT':0, 'SODOXM':0,
                   'SOCO2M':0},
        'IQUAL':  {'SQO':0, 'SOQSP':0, 'SOQS':0, 'SOQO':0, 'SOQOC':0,
                   'SOQUAL':0, 'SOQC':0, 'IQADDR':0, 'IQADWT':0, 'IQADEP':0}},
    'RCHRES' : {
        'HYDR':   {'CDFVOL':0, 'CIVOL':0, 'CO':0, 'COVOL':0, 'CRO':0,
                   'CROVOL':0, 'CVOL':0, 'DEP':0, 'IVOL':0, 'O':0, 'OVOL':1,
                   'PRSUPY':0, 'RIRDEM':0, 'RIRSHT':0, 'RO':0, 'ROVOL':1,
                   'SAREA':0, 'TAU':0, 'USTAR':0, 'VOL':1, 'VOLEV':0},
        'ADCALC': {},
        'CONS':   {'COADDR':0, 'COADEP':0, 'COADWT':0, 'CON':0, 'ICON':0,
                   'OCON':0, 'ROCON':0},
        'HTRCH':  {'AIRTMP':0, 'HTCF4':0, 'HTEXCH':0, 'IHEAT':0, 'OHEAT':0,
                  'ROHEAT':0, 'SHDFAC':0, 'TW':0},
        'SEDTRN': {'BEDDEP':0, 'DEPSCR':0, 'ISED':0, 'OSED':0, 'ROSED':0,
                   'RSED':0, 'SSED':0, 'TSED':0},
        'GQUAL':  {'ADQAL':0, 'DDQAL':0, 'DQAL':0, 'DSQAL':0, 'GQADDR':0,
                   'GQADEP':0, 'GQADWT':0, 'IDQAL':0, 'ISQAL':0, 'ODQAL':0,
                   'OSQAL':0, 'PDQAL':0, 'RDQAL':0, 'RODQAL':0, 'ROSQAL':0,
                   'RRQAL':0, 'RSQAL':0, 'SQAL':0, 'SQDEC':0, 'TIQAL':0,
                   'TOSQAL':0, 'TROQAL':0},
        'OXRX':   {'BOD':0, 'DOX':0, 'OXCF1':0, 'OXCF2':0, 'OXCF3':0,
                   'OXCF4':0, 'OXIF':0, 'SATDO':0},
        'NUTRX':  {'DNUST':0, 'DNUST2':0, 'NUADDR':0, 'NUADEP':0, 'NUADWT':0,
                   'NUCF1':0, 'NUCF2':0, 'NUCF3':0, 'NUCF4':0, 'NUCF5':0,
                   'NUCF6':0, 'NUCF7':0, 'NUCF8':0, 'NUCF9':0, 'NUIF1':0,
                   'NUIF2':0, 'NUST':0, 'OSNH4':0, 'OSPO4':0, 'RSNH4':0,
                   'RSPO4':0, 'SNH4':0, 'SPO4':0, 'TNUCF1':0, 'TNUCF2':0,
                   'TNUIF':0},
        'PLANK':  {'PHYTO':0, 'ZOO':0, 'BENAL':0, 'TBENAL':0, 'PHYCLA':0,
                   'BALCLA':0, 'PKST3':0, 'PKST4':0, 'PKIF':0, 'TPKIF':0,
                   'PKCF1':0, 'TPKCF1':0, 'PKCF2':0, 'TPKCF2':0, 'PLADDR':0,
                   'PLADWT':0, 'PLADEP':0, 'PKCF5':0, 'PKCF6':0, 'PKCF7':0,
                   'TPKCF7':0, 'PKCF8':0, 'PKCF9':0, 'PKCF10':0},
        'PHCARB': {'PHCF1':0, 'PHCF2':0, 'PHCF3':0, 'PHIF':0, 'PHST':0,
                   'SATCO2':0}}}

dactivities = {'PERLND':{'1':'GENERAL', '2':'ATEMP', '3':'SNOW', '4':'PWATER',
  '5':'SEDMNT', '6':'PSTEMP', '7':'PWTGAS', '8':'PQUAL', '9':'MSTLAY',
  '10':'PEST', '11':'NITR', '12':'PHOS', '13':'TRACER', '81':'PQUAL'},

  'IMPLND':{'1':'GENERAL', '2':'ATEMP', '3':'SNOW', '4':'IWATER', '5':'SOLIDS',
  '6':'IWTGAS', '7':'IQUAL', '71':'IQUAL'},

  'RCHRES':{'1':'GENERAL', '2':'HYDR', '3':'ADCALC', '4':'CONS', '5':'HTRCH',
  '6':'SEDTRN', '7':'GQUAL', '8':'RQUAL', '9':'OXRX', '10':'NUTRX',
  '11':'PLANK', '12':'PHCARB', '13':'ACIDPH', '71':'GQUAL'}}


def get_parsedata(EnglishUnits=True):
    parse = defaultdict(list)
    dsave = {}
    doneit = set()
    for filename in ('PERLND.SEQ', 'implnd.seq', 'RCHRES.SEQ'):
        start = None
        width = None
        _type = None
        default = None

        # iterate over file lines, but skip blank lines if any
        operation = filename.split('.')[0].upper()
        datapath = os.path.join(HSP2tools.__path__[0], 'data', filename)
        with open(datapath, 'r') as file:
            for line in file:
                if not line.strip():  # Skip blank lines
                    continue

                keyword, *tokens = line.split()
                if keyword == '#GROUP' and tokens[0] == '1':
                    next(file)   # skip $TEXT line
                    for line in file:
                        if '$' == line[0]:
                            break    #  read as far as needed, move on
                        tbl = line[8:25].strip()
                        sav = line[58:60].strip()
                        if sav:
                            dsave[(operation,tbl)]=dactivities[operation][sav]
                elif keyword == '$TNAME':
                    tname = tokens[0]
                elif keyword == '_PNAME':
                    # all information is now available for table, pname
                    pname = tokens[0]
                    s = (pname, _type, start, start + width, default)
                    if not EnglishUnits:
                        if pname == 'OPNID':
                            parse[(operation, tname)] = [] # removes existing English unit info
                        parse[(operation, tname)].append(s)
                    elif EnglishUnits and (operation,tname,pname) not in doneit:
                        parse[(operation, tname)].append(s)
                    doneit.add((operation, tname, pname))
                elif keyword == '_COLUMN':
                    start = int(tokens[0]) - 1
                elif keyword == '_TYPE':
                    _type = tokens[0][0]     # first letter is desired value
                elif keyword == '_WIDTH':
                    width = int(tokens[0])
                elif keyword == '_DEFAULT':
                    default = tokens[0]

    #############  These tables not in sequence files #######################
    parse[('FTABLES', 'FTABLE')] = ( # name, datatype, start col, stop col, default
     ('Depth',  'R',  0, 10, 0),
     ('Area',   'R', 10, 20, 0),
     ('Volume', 'R', 20, 30, 0),
     ('Disch1', 'R', 30, 40, 0),
     ('Disch2', 'R', 40, 50, 0),
     ('Disch3', 'R', 50, 60, 0),
     ('Disch4', 'R', 60, 70, 0),
     ('Disch5', 'R', 70, 80, 0))

    parse[('GLOBAL','START')] = (
     ('SYR', 'C', 14, 19, '1900'),
     ('SMO', 'C', 19, 22, '01'),
     ('SDA', 'C', 22, 25, '01'),
     ('SHR', 'C', 25, 28, '00'),
     ('SMI', 'C', 28, 31, '00'),
     ('EYR', 'C', 39, 44, '1900'),
     ('EMO', 'C', 44, 47, '12'),
     ('EDA', 'C', 47, 50, '31'),
     ('EHR', 'C', 50, 53, '24'),
     ('EMI', 'C', 53, 56, '00'))

    parse[('EXT SOURCES',)] = (
     ('SVOL',   'C',  0, 6, ''),
     ('SVOLNO', 'C',  6, 11, ''),
     ('SMEMN',  'C', 11, 17, ''),
     ('SMEMSB', 'C', 17, 20, '31'),
     ('SSYST',  'C', 20, 24, 'ENGL'),
     ('SGAPST', 'C', 24, 28, ''),
     ('MFACTOR','F', 28, 38, '1.0'),
     ('TRAN',   'C', 38, 43, ''),
     ('TVOL',   'C', 43, 50, ''),
     ('TOPFST', 'C', 50, 54, ''),
     ('TOPLST', 'C', 54, 58, ''),
     ('TGRPN',  'C', 58, 65, ''),
     ('TMEMN',  'C', 65, 71, ''),
     ('TMEMSB', 'C', 71, 75, ''))

    parse[('NETWORK',)] = (
     ('SVOL',   'C', 0, 6, ''),
     ('SVOLNO', 'C', 6, 11, ''),
     ('SGRPN',  'C', 11, 18, ''),
     ('SMEMN',  'C', 18, 24, ''),
     ('SMEMSB', 'C', 24, 28, ''),
     ('MFACTOR','F', 28, 38, '1.0'),
     ('TRAN',   'C', 38, 43, ''),
     ('TVOL',   'C', 43, 50, ''),
     ('TOPFST', 'C', 50, 54, ''),
     ('TOPLST', 'C', 54, 58, ''),
     ('TGRPN',  'C', 58, 65, ''),
     ('TMEMN',  'C', 65, 71, ''),
     ('TMEMSB', 'C', 71, 75, ''))

    parse[('SCHEMATIC',)] = (
     ('SVOL',   'C',  0,   6,  ''),
     ('SVOLNO', 'I',  6,  10,  ''),
     ('AFACTR', 'F', 28,  38, '1.0'),
     ('TVOL',   'C', 43,  49,  ''),
     ('TVOLNO', 'I', 49,  53,  ''),
     ('MLNO',   'I', 56,  60,  ''),
     ('TMEMSB', 'C', 71,  75,  ''))

    parse[('MASS-LINK',)] = (
     ('SVOL',   'C', 0, 11, ''),
     ('SGRPN',  'C', 11, 18, ''),
     ('SMEMN',  'C', 18, 24, ''),
     ('SMEMSB', 'C', 24, 28, ''),
     ('MFACTOR','F', 28, 43, '1.0'),
     ('TVOL',   'C', 43, 58, ''),
     ('TGRPN',  'C', 58, 65, ''),
     ('TMEMN',  'C', 65, 71, ''),
     ('TMEMSB', 'C', 71, 75, ''))

    ########################## CORRECTIONS #################################
    parse[('IMPLND','SNOW-PARM1')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('LAT',    'R', 12, 22, '40.'),
     ('MELEV',  'R', 22, 32, '0.'),
     ('SHADE',  'R', 32, 42, '0.'),
     ('SNOWCF', 'R', 42, 52, '-999.'),
     ('COVIND', 'R', 52, 62, '-999.'),
     ('KMELT',  'R', 62, 72, '0.0'),
     ('TBASE',  'R', 72, 82, '32.'))

    parse[('IMPLND','IQL-AD-FLAGS')] = (
     ('OPNID',      'C',  0, 11, ''),
     ('IQADFG(1)',  'I', 11, 14, '0'),
     ('IQADFG(2)',  'I', 14, 18, '0'),
     ('IQADFG(3)',  'I', 18, 21, '0'),
     ('IQADFG(4)',  'I', 21, 25, '0'),
     ('IQADFG(5)',  'I', 25, 28, '0'),
     ('IQADFG(6)',  'I', 28, 32, '0'),
     ('IQADFG(7)',  'I', 32, 35, '0'),
     ('IQADFG(8)',  'I', 35, 39, '0'),
     ('IQADFG(9)',  'I', 39, 42, '0'),
     ('IQADFG(10)', 'I', 42, 46, '0'),
     ('IQADFG(11)', 'I', 46, 49, '0'),
     ('IQADFG(12)', 'I', 49, 53, '0'),
     ('IQADFG(13)', 'I', 53, 56, '0'),
     ('IQADFG(14)', 'I', 56, 60, '0'),
     ('IQADFG(15)', 'I', 60, 63, '0'),
     ('IQADFG(16)', 'I', 63, 67, '0'),
     ('IQADFG(17)', 'I', 67, 70, '0'),
     ('IQADFG(18)', 'I', 70, 74, '0'),
     ('IQADFG(19)', 'I', 74, 77, '0'),
     ('IQADFG(20)', 'I', 77, 80, '0'))

    parse[('PERLND','SNOW-PARM1')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('LAT',    'R', 12, 22, '40.'),
     ('MELEV',  'R', 22, 32, '0.'),
     ('SHADE',  'R', 32, 42, '0.'),
     ('SNOWCF', 'R', 42, 52, '-999.'),
     ('COVIND', 'R', 52, 62, '-999.'),
     ('KMELT',  'R', 62, 72, '0.0'),
     ('TBASE',  'R', 72, 82, '32.'))

    parse[('PERLND','PWAT-PARM2')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('FOREST', 'R', 12, 22, '0.'),
     ('LZSN',   'R', 22, 32, '-999.'),
     ('INFILT', 'R', 32, 42, '-999.'),
     ('LSUR',   'R', 42, 52, '-999.'),
     ('SLSUR',  'R', 52, 62, '-999.'),
     ('KVARY',  'R', 62, 72, '0.'),
     ('AGWRC',  'R', 72, 82, '-999.'))

    parse[('PERLND','PWAT-PARM3')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('PETMAX', 'R', 12, 22, '40.'),
     ('PETMIN', 'R', 22, 32, '35.'),
     ('INFEXP', 'R', 32, 42, '2.'),
     ('INFILD', 'R', 42, 52, '2.'),
     ('DEEPFR', 'R', 52, 62, '0.'),
     ('BASETP', 'R', 62, 72, '0.'),
     ('AGWETP', 'R', 72, 82, '0.'))

    parse[('PERLND','PWAT-PARM7')] : (
     ('OPNID',  'C',  0, 12, ''),
     ('STABNO', 'R', 12, 22, '0.0'),
     ('SRRC',   'R', 22, 32, '-999.0'),
     ('SREXP',  'R', 32, 42, '1.0'),
     ('IFWSC',  'R', 42, 52, '-999.0'),
     ('DELTA',  'R', 52, 62, '0.001'),
     ('UELFAC', 'R', 62, 72, '4.0'),
     ('LELFAC', 'R', 72, 82, '2.5'))

    parse[('PERLND','PWAT-STATE1')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('CEPS',   'R', 12, 22, '0.'),
     ('SURS',   'R', 22, 32, '0.'),
     ('UZS',    'R', 32, 42, '0.001'),
     ('IFWS',   'R', 42, 52, '0.'),
     ('LZS',    'R', 52, 62, '0.001'),
     ('AGWS',   'R', 62, 72, '0.'),
     ('GWVS',   'R', 72, 82, '0.'))

    parse[('PERLND','IRRIG-SOURCE')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('XPRIOR', 'R', 10, 20, '0.'),
     ('XFRAC',  'R', 20, 30, '1.0'),
     ('GPRIOR', 'R', 30, 40, '0.'),
     ('GFRAC',  'R', 40, 50, '1.0'),
     ('RPRIOR', 'R', 50, 60, '0.'),
     ('RFRAC',  'R', 60, 70, '1.0'),
     ('IRCHNO', 'R', 70, 80, '0.'))

    parse[('PERLND','IRRIG-SCHED')] = (
     ('OPNID',  'C',  0, 13, ''),
     ('IRYR1',  'R', 13, 18, '0.'),
     ('IRMO1',  'R', 18, 21, '1.'),
     ('IRDY1',  'R', 21, 24, '1.'),
     ('IRHR1',  'R', 24, 27, '0.'),
     ('IRMI1',  'R', 27, 29, '0.'),
     ('IRDUR1', 'R', 29, 34, '0.'),
     ('IRRAT1', 'R', 34, 50, '0.'),
     ('IRYR2',  'R', 50, 55, '0.'),
     ('IRMO2',  'R', 55, 58, '1.'),
     ('IRDY2',  'R', 58, 61, '1.'),
     ('IRHR2',  'R', 61, 64, '0.'),
     ('IRMI2',  'R', 64, 66, '0.'),
     ('IRDUR2', 'R', 66, 71, '0.'),
     ('IRRAT2', 'R', 71, 81, '0.'))

    parse[('PERLND','PQL-AD-FLAGS')] = (
     ('OPNID',      'C',  0, 11, ''),
     ('PQADFG(1)',  'I', 11, 14, '0'),
     ('PQADFG(2)',  'I', 14, 18, '0'),
     ('PQADFG(3)',  'I', 18, 21, '0'),
     ('PQADFG(4)',  'I', 21, 25, '0'),
     ('PQADFG(5)',  'I', 25, 28, '0'),
     ('PQADFG(6)',  'I', 28, 32, '0'),
     ('PQADFG(7)',  'I', 32, 35, '0'),
     ('PQADFG(8)',  'I', 35, 39, '0'),
     ('PQADFG(9)',  'I', 39, 42, '0'),
     ('PQADFG(10)', 'I', 42, 46, '0'),
     ('PQADFG(11)', 'I', 46, 49, '0'),
     ('PQADFG(12)', 'I', 49, 53, '0'),
     ('PQADFG(13)', 'I', 53, 56, '0'),
     ('PQADFG(14)', 'I', 56, 60, '0'),
     ('PQADFG(15)', 'I', 60, 63, '0'),
     ('PQADFG(16)', 'I', 63, 67, '0'),
     ('PQADFG(17)', 'I', 67, 70, '0'),
     ('PQADFG(18)', 'I', 70, 74, '0'),
     ('PQADFG(19)', 'I', 74, 77, '0'),
     ('PQADFG(20)', 'I', 77, 80, '0'))

    parse[('PERLND','NIT-FSTPM')] = (
     ('OPNID', 'C',  0, 10, ''),
     ('KDSAM', 'R', 10, 20, '0.'),
     ('KADAM', 'R', 20, 30, '0.'),
     ('KIMNI', 'R', 30, 40, '0.'),
     ('KAM',   'R', 40, 50, '0.'),
     ('KDNI',  'R', 50, 60, '0.'),
     ('KNI',   'R', 60, 70, '0.'),
     ('KIMAM', 'R', 70, 80, '0.'))

    parse[('RCHRES','HYDR-INIT')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('VOL',    'R', 12, 23, '0.'),
     ('ICAT',   'R', 23, 27, '-1.'),
     ('COLIN1', 'R', 27, 32, '4.'),
     ('COLIN2', 'R', 32, 37, '4.'),
     ('COLIN3', 'R', 37, 42, '4.'),
     ('COLIN4', 'R', 42, 47, '4.'),
     ('COLIN5', 'R', 47, 57, '4.'),
     ('OUTDG1', 'R', 57, 62, '0.'),
     ('OUTDG2', 'R', 62, 67, '0.'),
     ('OUTDG3', 'R', 67, 72, '0.'),
     ('OUTDG4', 'R', 72, 77, '0.'),
     ('OUTDG5', 'R', 77, 82, '0.'))

    parse[('RCHRES','HYDR-PARM1')] = (
     ('OPNID',  'C',  0, 11, ''),
     ('VCONFG', 'I', 11, 14, '0'),
     ('AUX1FG', 'I', 14, 17, '0'),
     ('AUX2FG', 'I', 17, 20, '0'),
     ('AUX3FG', 'I', 20, 25, '0'),
     ('ODFVF1', 'I', 25, 28, '0'),
     ('ODFVF2', 'I', 28, 31, '0'),
     ('ODFVF3', 'I', 31, 34, '0'),
     ('ODFVF4', 'I', 34, 37, '0'),
     ('ODFVF5', 'I', 37, 40, '0'),
     ('ODGTF1', 'I', 45, 48, '0'),
     ('ODGTF2', 'I', 48, 51, '0'),
     ('ODGTF3', 'I', 51, 54, '0'),
     ('ODGTF4', 'I', 54, 57, '0'),
     ('ODGTF5', 'I', 57, 65, '0'),
     ('FUNCT1', 'I', 65, 68, '1'),
     ('FUNCT2', 'I', 68, 71, '1'),
     ('FUNCT3', 'I', 71, 74, '1'),
     ('FUNCT4', 'I', 74, 77, '1'),
     ('FUNCT5', 'I', 77, 80, '1'))

    parse[('RCHRES','CONS-AD-FLAGS')] = (
     ('OPNID',      'C',  0, 11, ''),
     ('COADFG(1)',  'I', 11, 14, '0'),
     ('COADFG(2)',  'I', 14, 18, '0'),
     ('COADFG(3)',  'I', 18, 21, '0'),
     ('COADFG(4)',  'I', 21, 25, '0'),
     ('COADFG(5)',  'I', 25, 28, '0'),
     ('COADFG(6)',  'I', 28, 32, '0'),
     ('COADFG(7)',  'I', 32, 35, '0'),
     ('COADFG(8)',  'I', 35, 39, '0'),
     ('COADFG(9)',  'I', 39, 42, '0'),
     ('COADFG(10)', 'I', 42, 46, '0'),
     ('COADFG(11)', 'I', 46, 49, '0'),
     ('COADFG(12)', 'I', 49, 53, '0'),
     ('COADFG(14)', 'I', 56, 60, '0'),
     ('COADFG(15)', 'I', 60, 63, '0'),
     ('COADFG(16)', 'I', 63, 67, '0'),
     ('COADFG(17)', 'I', 67, 70, '0'),
     ('COADFG(18)', 'I', 70, 74, '0'),
     ('COADFG(19)', 'I', 74, 77, '0'),
     ('COADFG(20)', 'I', 77, 80, '0'))

    parse[('RCHRES','GQ-PHOTPM')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('PHOTP1', 'R', 10, 20, '0.'),
     ('PHOTP2', 'R', 20, 30, '0.'),
     ('PHOTP3', 'R', 30, 40, '0.'),
     ('PHOTP4', 'R', 40, 50, '0.'),
     ('PHOTP5', 'R', 50, 60, '0.'),
     ('PHOTP6', 'R', 60, 70, '0.'),
     ('PHOTP7', 'R', 70, 80, '0.'))

    parse[('RCHRES','GQ-ALPHA')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('ALPH1',  'R', 10, 20, '-999.'),
     ('ALPH2',  'R', 20, 30, '-999.'),
     ('ALPH3',  'R', 30, 40, '-999.'),
     ('ALPH4',  'R', 40, 50, '-999.'),
     ('ALPH5',  'R', 50, 60, '-999.'),
     ('ALPH6',  'R', 60, 70, '-999.'),
     ('ALPH7',  'R', 70, 80, '-999.'))

    parse[('RCHRES','GQ-GAMMA')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('GAMM1',  'R', 10, 20, '0.'),
     ('GAMM2',  'R', 20, 30, '0.'),
     ('GAMM3',  'R', 30, 40, '0.'),
     ('GAMM4',  'R', 40, 50, '0.'),
     ('GAMM5',  'R', 50, 60, '0.'),
     ('GAMM6',  'R', 60, 70, '0.'),
     ('GAMM7',  'R', 70, 80, '0.'))

    parse[('RCHRES','GQ-DELTA')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('DEL1',   'R', 10, 20, '0.'),
     ('DEL2',   'R', 20, 30, '0.'),
     ('DEL3',   'R', 30, 40, '0.'),
     ('DEL4',   'R', 40, 50, '0.'),
     ('DEL5',   'R', 50, 60, '0.'),
     ('DEL6',   'R', 60, 70, '0.'),
     ('DEL7',   'R', 70, 80, '0.'))

    parse[('RCHRES','GQ-CLDFACT')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('KCLD1',  'R', 10, 20, '0.'),
     ('KCLD2',  'R', 20, 30, '0.'),
     ('KCLD3',  'R', 30, 40, '0.'),
     ('KCLD4',  'R', 40, 50, '0.'),
     ('KCLD5',  'R', 50, 60, '0.'),
     ('KCLD6',  'R', 60, 70, '0.'),
     ('KCLD7',  'R', 70, 80, '0.'))

    parse[('RCHRES','GQ-DAUGHTER')] = (
     ('OPNID', 'C',  0,  10, ''),  # end column was 8
     ('ZERO',  'R', 10, 18, '0.'),
     ('C2F1',  'R', 20, 28, '0.'),
     ('C3F1',  'R', 30, 38, '0.'),
     ('C4F1',  'R', 40, 48, '0.'),
     ('C5F1',  'R', 50, 58, '0.'),
     ('C6F1',  'R', 60, 68, '0.'),
     ('C7F1',  'R', 70, 78, '0.'))

    parse[('RCHRES','PLNK-PARM1')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('RATCLP', 'R', 12, 22, '0.6'),
     ('NONREF', 'R', 22, 32, '0.5'),
     ('LITSED', 'R', 32, 42, '0.'),
     ('ALNPR',  'R', 42, 52, '1.'),
     ('EXTB',   'R', 52, 62, '-999.'),
     ('MALGR',  'R', 62, 72, '0.3'),
     ('PARADF', 'R', 72, 82, '1.0'))

    parse[('RCHRES','PLNK-PARM2')] = (
     ('OPNID',  'C',  0, 12, ''),
     ('CMMLT',  'R', 12, 22, '0.033'),
     ('CMMN',   'R', 22, 32, '0.045'),
     ('CMMNP',  'R', 32, 42, '0.0284'),
     ('CMMP',   'R', 42, 52, '0.015'),
     ('TALGRH', 'R', 52, 62, '95.'),
     ('TALGRL', 'R', 62, 72, '43.'),
     ('TALGRM', 'R', 72, 82, '77.'))

    parse[('RCHRES','BENAL-PARM')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('MBAL',   'R', 10, 20, '600.'),
     ('CFBALR', 'R', 20, 30, '1.'),
     ('CFBALG', 'R', 30, 40, '1.'),
     ('MINBAL', 'R', 40, 50, '0.0001'),
     ('CAMPR',  'R', 50, 60, '0.001'),
     ('FRAVL',  'R', 60, 70, '0.'),
     ('NMAXFX', 'R', 70, 80, '10.'))

    parse[('RCHRES','ACID-FLAGS')] = (
     ('OPNID',      'C',  0, 10, ''),
     ('ACFLAG(1)',  'I', 10, 15, '3'),
     ('ACFLAG(2)',  'I', 15, 20, '4'),
     ('ACFLAG(3)',  'I', 20, 25, '1'),
     ('ACFLAG(4)',  'I', 25, 30, '0'),
     ('ACFLAG(5)',  'I', 30, 35, '0'),
     ('ACFLAG(6)',  'I', 35, 40, '0'),
     ('ACFLAG(7)',  'I', 40, 45, '0'),
     ('ACFLAG(8)',  'I', 45, 50, '0'),
     ('ACFLAG(9)',  'I', 50, 55, '0'),
     ('ACFLAG(10)', 'I', 55, 60, '0'),
     ('ACFLAG(11)', 'I', 60, 65, '0'),
     ('ACFLAG(12)', 'I', 65, 70, '0'),
     ('ACFLAG(13)', 'I', 70, 75, '0'),
     ('ACFLAG(14)', 'I', 75, 80, '0'))

    parse[('RCHRES','ACID-PARMS')] = (
     ('OPNID',     'C',  0, 10, ''),
     ('ACPARM(1)', 'R', 10, 20, '1.0'),
     ('ACPARM(2)', 'R', 20, 30, '6.5E-32'),
     ('ACPARM(3)', 'R', 30, 40, '-1.0'),
     ('ACPARM(4)', 'R', 40, 50, '0.95'),
     ('ACPARM(5)', 'R', 50, 60, '2.50'),
     ('ACPARM(6)', 'R', 60, 70, '0.000'),
     ('ACPARM(7)', 'R', 70, 80, '0.000'))

    parse[('RCHRES','ACID-INIT')] = (
     ('OPNID',     'C',  0, 10, ''),
     ('ACCONC(1)', 'R', 10, 20, '0.000'),
     ('ACCONC(2)', 'R', 20, 30, '0.000'),
     ('ACCONC(3)', 'R', 30, 40, '0.000'),
     ('ACCONC(4)', 'R', 40, 50, '0.000'),
     ('ACCONC(5)', 'R', 50, 60, '0.000'),
     ('ACCONC(6)', 'R', 60, 70, '0.000'),
     ('ACCONC(7)', 'R', 70, 80, '0.000'))

    parse[('RCHRES','BENAL-GROW')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('MBALGR', 'R', 10, 20, '0.3'),
     ('TCBALG', 'R', 20, 30, '1.07'),
     ('CMMNB',  'R', 30, 40, '0.045'),
     ('CMMPB',  'R', 40, 50, '0.015'),
     ('CMMD1',  'R', 50, 60, '0.1'),
     ('CMMD2',  'R', 60, 70, '100.0'),
     ('CSLIT',  'R', 70, 80, '250.'))

    parse[('RCHRES','SANDFG')] = (
     ('OPNID',  'C',  0, 10, ''),
     ('SANDFG', 'I', 10, 17, '3'))

    return parse, dsave


def reader(filename):
    # simple reader to return non blank, non comment and proper length lines
    with open(filename, 'r') as file:
        for line in file:
            if '***' in line or not line.strip():
                continue
            yield f'{line.rstrip(): <90}'        # prevent short line problems


def getlines(f):
    lines = []
    for line in f:
        if line[0:3] == 'END' :
            break
        lines.append(line)
    return lines


convert = {'C':str, 'I':int, 'R':float, 'F':float}
def parseD(line, parse, tableid):
    d = {}
    for name, type_, start, end, default in parse[tableid]:
        field = line[start:end].strip()
        d[name] = convert[type_](field) if field else convert[type_](default)
    return d


def get_opnid(opnidstr, operation):
    first, *last = opnidstr.split()
    b = int(last[0]) if last else int(first)
    a = int(first)
    for x in range(a, b+1):
        yield f'{operation[0]}{x:03d}'


def fix_df(df, op, save, ddfaults, valid, sortflag=False):
    '''fix NANs and excess rids, missing indicies'''
    if set(df.index) - valid[op]:
        df.drop(index = set(df.index) - valid[op]) # drop unnecessary rids
    for name1 in valid[op] - set(df.index):
        df = df.append(Series(name=name1))         # add missing rids with NaNs
    if df.isna().any().any():                      # replace NaNs with defaults
        for col in df.columns:
            df[col] = df[col].fillna(ddfaults[op, save, col])
    if sortflag:
        df = df.sort_index(axis=1)
    return df


def get_cat(op, table, save):
    if   (op,table) in crop_stages:  cat = 'CROP_STAGES'
    elif (op,table) in pqual:        cat = 'PQUAL'
    elif (op,table) in pest:         cat = 'PEST'
    elif (op,table) in nitr:         cat = 'NITR'
    elif (op,table) in phos:         cat = 'PHOS'
    elif (op,table) in iqual:        cat = 'IQUAL'
    elif (op,table) in extended:     cat = 'EXTENDED'
    elif (op,table) in cons:         cat = 'CONS'
    elif (op,table) in gqual:        cat = 'GQUAL'
    elif (op,table) in skip:         cat = 'SKIP'

    elif 'STATE'    in table:        cat = 'STATES'
    elif 'INIT'     in table:        cat = 'STATES'
    elif 'ACTIVITY' in table:        cat = 'ACTIVITY';
    elif 'FLAG'     in table:        cat = 'FLAGS'
    elif 'GEN-INFO' in table:        cat = 'INFO'
    elif 'MON-'     in table:        cat = 'MONTHLYS'
    elif table[-2:] == 'FG':         cat = 'FLAGS'
    else:                            cat = 'PARAMETERS'
    return cat


#### PERLND tables ####
crop_stages = {   #section PWATER
 ('PERLND','CROP-STAGES'),  # FIRST, SECOND, THIRD
 ('PERLND','CROP-SEASPM')}  # FIRST, SECOND, THIRD
pqual = {
 ('PERLND', 'NQUALS'),
 ('PERLND', 'QUAL-PROPS'),
 ('PERLND', 'QUAL-INPUT'),
 ('PERLND', 'MON-POTFW'),
 ('PERLND', 'MON-POTFS'),
 ('PERLND', 'MON-ACCUM'),
 ('PERLND', 'MON-SQOLIM'),
 ('PERLND', 'MON-IFLW-CONC'),
 ('PERLND', 'MON-GRND-CONC')}
pest = {
 ('PERLND','PEST-FLAGS'),
 ('PERLND','SOIL-DATA'),  # also appears in section nitr! ???
 ('PERLND','PEST-ID'),
 ('PERLND','PEST-THETA'),
 ('PERLND','PEST-FIRSTPM'), # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND','PEST-CMAX'),
 ('PERLND','PEST-SVALPM'),  # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND','PEST-NONSVPM'), # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND','PEST-DEGRADE'),
 ('PERLND','PEST-STOR1'),   # SURFACE, UPPER
 ('PERLND','PEST-STOR2'),   # UPPER
 ('PERLND','PEST-STOR1')    # LOWER, GROUNDWATER
 }
nitr = {
 ('PERLND', 'SOIL-DATA'),
 ('PERLND', 'NIT-FLAGS'),
 ('PERLND', 'NIT-UPTAKE'),
 ('PERLND', 'MON-NITUPT'),  # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'NIT-FSTGEN'),
 ('PERLND', 'NIT-FSTPM'),   # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'NIT-CMAX'),
 ('PERLND', 'NIT-SVALPM'), # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'NIT-STOR1'),  # SURFACE, UPPER
 ('PERLND', 'NIT_STOR2'),  # UPPER
 ('PERLND', 'NIT-STOR1'),  # LOWER, GROUNDWATER
 }
phos = {
 ('PERLND', 'MON-PHOSUPT'),    # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'PHOS-FSTPM'),     # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'PHOS-SVALPM'),    # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'MON-PUPT-FR2'),   # SURFACE, UPPER, LOWER, GROUNDWATER
 ('PERLND', 'PHOS-STOR1'),     # SURFACE, UPPER, LOWER, GROUNDWATER
}

#### IMPLND tables ####
iqual = {
('IMPLND', 'NQUALS'),
('IMPLND', 'QUAL-PROPS'),
('IMPLND', 'QUAL-INPUT'),
('IMPLND', 'MON-POTFW'),
('IMPLND', 'MON-POTFS'),
('IMPLND', 'MON-ACCUM'),
('IMPLND', 'MON-SQOLIM')}

#### RCHRES tables ####
extended = {     # data spread into multiple reads of same table
 ('RCHRES','HT-BED-DELH'):  (100, 'DELH'),
 ('RCHRES','HT-BED-DELTT'): (100, 'DELTT'),
 ('RCHRES','GQ-PHOTPM'):     (20, 'PHOTPM'),  # 18 then PHI, THETA
 ('RCHRES','GQ-ALPHA'):      (18, 'ALPH'),
 ('RCHRES','GQ-GAMMA'):      (18, 'GAMM'),
 ('RCHRES','GQ-DELTA'):      (18, 'DEL'),
 ('RCHRES','GQ-CLDFACT'):    (18, 'KCLD'),
 ('RCHRES','GQ-DAUGHTER'):   (49, 'CF'),      # upto 7 x 7 array, zero filled, name incorrect
 ('RCHRES','ACID-FLAGS'):    (14, 'AC')}
cons = {
 ('RCHRES','NCONS'),
 ('RCHRES','CONS-DATA')}
gqual = {
 ('RCHRES','GQ-QALDATA'),
 ('RCHRES','GQ-QALFG'),
 ('RCHRES','GQ-FLG2'),
 ('RCHRES','GQ-HYDPM'),
 ('RCHRES','GQ-ROXPM'),
 ('RCHRES','GQ-PHOTPM'),
 ('RCHRES','GQ-CFGAS'),
 ('RCHRES','GQ-BIOPM'),
 ('RCHRES','MON-BIO'),
 ('RCHRES','GQ-GENDECAY'),
 ('RCHRES','GQ-SEDDECAY'),
 ('RCHRES','GQ-KD'),
 ('RCHRES','GQ-ADRATE'),
 ('RCHRES','GQ-ADTHETA'),
 ('RCHRES','GQ-SEDCONC'),
 ('RCHRES','GQ-DAUGHTER'),   # HYDROLYSIS,OXIDATION,PHOTOLYSIS,[DUMMY],BIODEGRADATION,FIRST-ORDER DECAY
 }

#### MONTHLY tables ####
rename = {
 ('PERLND', 'MON-MELT-FAC'): 'KMELT',
 ('PERLND', 'MON-INTERCEP'): 'CEPSC',
 ('PERLND', 'MON-UZSN'):     'UZSN',
 ('PERLND', 'MON-MANNING'):  'NSUR',
 ('PERLND', 'MON-INTERFLW'): 'INTFW',
 ('PERLND', 'MON-IRC'):      'IRC',
 ('PERLND', 'MON-LZETPARM'): 'LZETP',
 ('PERLND', 'MON-IRR-CRDP'): 'CRDEP',
 ('PERLND', 'MON-IRR-AWD'):  'IRAWD',
 ('PERLND', 'MON-COVER'):    'COVER',
 ('PERLND', 'MON-NVSI'):     'NVSI',
 ('PERLND', 'MON-ASLT'):     'ASLT',
 ('PERLND', 'MON-BSLT'):     'BSLT',
 ('PERLND', 'MON-ULTP1'):    'ULTP1',
 ('PERLND', 'MON-ULTP2'):    'ULTP2',
 ('PERLND', 'MON-LGTP1'):    'LGTP1',
 ('PERLND', 'MON-LGTP2'):    'LGTP2',
 ('PERLND', 'MON-IFWDOX'):   'IDOXP',
 ('PERLND', 'MON-IFWCO2'):   'ICO2P',
 ('PERLND', 'MON-GRNDDOX'):  'ADOXP',
 ('PERLND', 'MON-GRNDCO2'):  'ACO2P',
 ('PERLND', 'MON-POTFW'):    'POTFW',    # PQUALS
 ('PERLND', 'MON-POTFS'):    'POTFS',    # PQUALS
 ('PERLND', 'MON-ACCUM'):    'ACQOP',    # PQUALS
 ('PERLND', 'MON-SQOLIM'):   'SQOLIM',   # PQUALS
 ('PERLND', 'MON-IFLW-CONC'):'IOQC',     # PQUALS
 ('PERLND', 'MON-GRND-CONC'):'AOQC',     # PQUALS
 ('PERLND', 'MON-NITUPT'):   'KPLN',
 ('PERLND', 'MON-NITUPNI'):  'KUPNI',
 ('PERLND', 'MON-NITUPAM'):  'KUPAM',
 ('PERLND', 'MON-NITIMNI'):  'KIMNI',
 ('PERLND', 'MON-NITIMAM'):  'KIMAM',
 ('PERLND', 'MON-NPRETBG'):  'KPRBN',
 ('PERLND', 'MON-NPRETAG'):  'AGKPRN',
 ('PERLND', 'MON-NPRETLI'):  'KPRLN',
 ('PERLND', 'MON-NPRETFBG'): 'BGNPRF',
 ('PERLND', 'MON-NPRETFLI'): 'LINPRF',
 ('PERLND', 'MON-NITAGUTF'): 'ANUTF',
 ('PERLND', 'MON-PHOSUPT'):  'KPLP',
 ('IMPLND', 'MON-MELT-FAC'): 'KMELT',
 ('IMPLND', 'MON-RETN'):     'RETSC',
 ('IMPLND', 'MON-MANNING'):  'NSUR',
 ('IMPLND', 'MON-SACCUM'):   'ACCSDP',
 ('IMPLND', 'MON-REMOV'):    'REMSDP',
 ('IMPLND', 'MON-AWTF'):     'AWTF',
 ('IMPLND', 'MON-BWTF'):     'BWTF',
 ('IMPLND', 'MON-POTFW'):    'POTFW',    # IQUALS
 ('IMPLND', 'MON-ACCUM'):    'ACQOP',    # IQUALS
 ('IMPLND', 'MON-SQOLIM'):   'SQOLIM',   # IQUALS
 ('PERLND', 'MON-NUPT-FR1'): 'TNU',
 ('PERLND', 'MON-NUPT-FR2'): 'LNU',
 ('PERLND', 'MON-PUPT-FR1'): 'TUP',
 ('PERLND', 'MON-PUPT-FR2'): 'LPU',
 ('RCHRES', 'MON-BINV'):     'BINVM',
 ('RCHRES', 'MON-BIO'):      'BIOM',     # GQID
 ('RCHRES', 'MON-CLOUD'):    'CLDM',
 ('RCHRES', 'MON-CONVF'):     'VOL',
 ('RCHRES', 'MON-HT-TGRND'): 'TGRND',
 ('RCHRES', 'MON-PHVAL'):    'PHVA',
 ('RCHRES', 'MON-PHYTO'):    'PHYM',
 ('RCHRES', 'MON-ROXYGEN'):  'ROCM',
 ('RCHRES', 'MON-SEDCONC'):  'SDCN',
 ('RCHRES', 'MON-WATEMP'):   'TEMP'
}

# Ignore these tables during processing, not used by HSP2
skip = {
 ('PERLND', 'PRINT-INFO'),
 ('PERLND', 'BINARY-INFO'),
 ('IMPLND', 'PRINT-INFO'),
 ('IMPLND', 'BINARY-INFO'),
 ('RCHRES', 'PRINT-INFO'),
 ('RCHRES', 'BINARY-INFO')}


ops = {'PERLND','IMPLND','RCHRES'}
conlike = {'CONS':'NCONS', 'PQUAL':'NQUAL', 'IQUAL':'NQUAL'}
def readUCI(uciname, hdfname):
    parse, dsave = get_parsedata()

    # create defaults by op, table,item dict
    ddfaults = {}
    for x in parse:
        if len(x) == 1:
            continue
        op, table = x
        for item in parse[op,table]:
            if (op,table) in dsave:
                ddfaults[op, dsave[(op,table)], item[0]] = item[4]

    with HDFStore(hdfname, mode = 'a') as store:
        info = (store, parse, dsave, ddfaults)
        f = reader(uciname)
        for line in f:
            if line[0:6] == 'GLOBAL':       global_(info, getlines(f))
            if line[0:3] == 'OPN':              opn(info, getlines(f))
            if line[0:7] == 'NETWORK':  net=network(info, getlines(f))
            if line[0:9] == 'SCHEMATIC':sc=schematic(info,getlines(f))
            if line[0:9] == 'MASS-LINK':   masslink(info, getlines(f))
            if line[0:7] == 'FTABLES':      ftables(info, getlines(f))
            if line[0:3] == 'EXT':              ext(info, getlines(f))
            if line[0:6] == 'PERLND':     operation(info, getlines(f),'PERLND')
            if line[0:6] == 'IMPLND':     operation(info, getlines(f),'IMPLND')
            if line[0:6] == 'RCHRES':     operation(info, getlines(f),'RCHRES')

        colnames = ('AFACTR', 'MFACTOR', 'MLNO', 'SGRPN', 'SMEMN', 'SMEMSB',
         'SVOL', 'SVOLNO', 'TGRPN', 'TMEMN', 'TMEMSB', 'TRAN', 'TVOL',
         'TVOLNO', 'COMMENTS')
        linkage = concat((net, sc), ignore_index=True, sort=True)
        for cname in colnames:
            if cname not in linkage.columns:
                linkage[cname]=''
        linkage = linkage.sort_values('TVOLNO')
        linkage.to_hdf(store, '/CONTROL/LINKS', data_columns=True)

        Lapse.to_hdf(store, 'TIMESERIES/LAPSE_Table')
        Seasons.to_hdf(store, 'TIMESERIES/SEASONS_Table')
        Svp.to_hdf(store, 'TIMESERIES/Saturated_Vapor_Pressure_Table')

        keys = set(store.keys())
        # rename needed for restart. NOTE issue with line 157 in PERLND SNOW HSPF
        # where PKSNOW = PKSNOW + PKICE at start - ONLY
        path = '/PERLND/SNOW/STATES'
        if path in keys:
            df = read_hdf(store, path)
            df=df.rename(columns={'PKSNOW':'PACKF','PKICE':'PACKI','PKWATR':'PACKW'})
            df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/SNOW/STATES'
        if path in keys:
            df = read_hdf(store, path)
            df=df.rename(columns={'PKSNOW':'PACKF','PKICE':'PACKI','PKWATR':'PACKW'})
            df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/SNOW/FLAGS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SNOPFG' not in df.columns:   # didn't read IWAT-PARM2 table
                df['SNOPFG']  = 0
                df.to_hdf(store, path, data_columns=True)

        path = '/IMPLND/SNOW/FLAGS'
        if path in keys:
            df = read_hdf(store, path)
            if 'SNOPFG' not in df.columns:   # didn't read IWAT-PARM2 table
                df['SNOPFG']  = 0
                df.to_hdf(store, path, data_columns=True)

        # Need to fixup missing data
        path = '/IMPLND/IWATER/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'PETMIN' not in df.columns:   # didn't read IWAT-PARM2 table
                df['PETMIN'] = 0.35
                df['PETMAX'] = 40.0
                df.to_hdf(store, path, data_columns=True)

        path = '/PERLND/PWATER/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            if 'FZG' not in df.columns:   # didn't read PWAT-PARM5 table
                df['FZG']  = 1.0
                df['FZGL'] = 0.1
                df.to_hdf(store, path, data_columns=True)

        dfinfo = read_hdf(store, 'RCHRES/GENERAL/INFO')
        path = '/RCHRES/HYDR/PARAMETERS'
        if path in keys:
            df = read_hdf(store, path)
            df['NEXITS'] = dfinfo['NEXITS']
            df['LKFG']   = dfinfo['LKFG']
            if 'IREXIT' not in df.columns:   # didn't read HYDR-IRRIG table
                df['IREXIT'] = 0
                df['IRMINV'] = 0.0
            df['FTBUCI'] = df['FTBUCI'].map(lambda x: f'FT{int(x):03d}')
            df.to_hdf(store, path, data_columns=True)
        del dfinfo['NEXITS']
        del dfinfo['LKFG']
        dfinfo.to_hdf(store, 'RCHRES/GENERAL/INFO', data_columns=True)

    return


def global_(info, lines):
    store, parse, dsave, _ = info
    d = parseD(lines[1], parse, ('GLOBAL','START'))
    start = str(Timestamp(f"{d['SYR']}-{d['SMO']}-{d['SDA']}")
      + Timedelta(int(d['SHR']), 'h') + Timedelta(int(d['SMI']), 'T'))[0:16]
    stop  = str(Timestamp(f"{d['EYR']}-{d['EMO']}-{d['EDA']}")
      + Timedelta(int(d['EHR']), 'h') + Timedelta(int(d['EMI']), 'T'))[0:16]
    data = [lines[0].strip(), start, stop]
    dfglobal = DataFrame(data, index=['Comment','Start','Stop'],columns=['Info'])
    dfglobal.to_hdf(store, '/CONTROL/GLOBAL', data_columns=True)


def opn(info, lines):
    store, parse, dsave, _ = info
    lst = []
    for line in lines:
        tokens = line.split()
        if tokens[0] == 'INGRP' and tokens[1] == 'INDELT':
            s = tokens[2].split(':')
            indelt = int(s[0]) if len(s) == 1 else 60 * int(s[0]) + int(s[1])
        elif tokens[0] in ops:
            s = f'{tokens[0][0]}{int(tokens[1]):03d}'
            lst.append((tokens[0], s, indelt))
    dfopn = DataFrame(lst, columns = ['OPERATION', 'SEGMENT', 'INDELT_minutes'])
    dfopn.to_hdf(store, '/CONTROL/OP_SEQUENCE', data_columns=True)


def network(info, lines):
    store, parse, dsave, _ = info
    lst = []
    for line in lines:
        d = parseD(line, parse, ('NETWORK',))
        if d['SVOL'] in ops and d['TVOL'] in ops:
            d['SVOLNO'] = f"{d['SVOL'][0]}{int(d['SVOLNO']):03d}"
            d['TVOLNO'] = f"{d['TVOL'][0]}{int(d['TVOLNO']):03d}"
            lst.append(d)
    return DataFrame(lst, columns=d) if lst else DataFrame()


def schematic(info, lines):
    store, parse, dsave, _ = info
    lst = []
    for line in lines:
        d = parseD(line, parse, ('SCHEMATIC',))
        if d['SVOL'] in ops and d['TVOL'] in ops:
            d['MLNO']   = f"ML{int(d['MLNO']):03d}"
            d['SVOLNO'] = f"{d['SVOL'][0]}{int(d['SVOLNO']):03d}"
            d['TVOLNO'] = f"{d['TVOL'][0]}{int(d['TVOLNO']):03d}"
            lst.append(d)
    return DataFrame(lst, columns=d) if lst else DataFrame()


def masslink(info, lines):
    store, parse, dsave, _ = info
    lst = []
    for line in lines:
        if line[2:11] == 'MASS-LINK':
            name = line[12:].rstrip()
        elif line[2:5] != 'END':
            d = parseD(line, parse, ('MASS-LINK',))
            d['MLNO'] = f'ML{int(name):03d}'
            lst.append(d)
    if lst:
        dfmasslink = DataFrame(lst, columns=d)
        del dfmasslink['TGRPN']
        dfmasslink['COMMENTS'] = ''
        dfmasslink.to_hdf(store, '/CONTROL/MASS_LINKS', data_columns=True)


def ftables(info, llines):
    store, parse, dsave, _ = info
    header=['Depth','Area','Volume','Disch1','Disch2','Disch3','Disch4','Disch5']
    lines = iter(llines)
    for line in lines:
        if line[2:8] == 'FTABLE':
            unit = int(line[8:])
            name = f'FT{unit:03d}'
            rows,cols = next(lines).split()
            lst = []
        elif line[2:5] == 'END':
            dfftable = DataFrame(lst, columns=header[0:int(cols)])
            dfftable.to_hdf(store, f'/FTABLES/{name}', data_columns=True)
        else:
            lst.append(parseD(line, parse, ('FTABLES','FTABLE')))


def ext(info, lines):
    store, parse, dsave, _ = info
    lst = []
    for line in lines:
        d = parseD(line, parse, ('EXT SOURCES',))
        if d['TVOL'] in ops:
            d['SVOLNO'] = f"TS{int(d['SVOLNO']):03d}"
            d['SVOL'] = '*'
            if d['TGRPN'] == 'EXTNL':
                d['TGRPN'] = ''
            toplst = int(d['TOPFST']) if d['TOPLST'] == '' else int(d['TOPLST'])
            for i in range(int(d['TOPFST']), toplst + 1):
                d['TVOLNO'] = f"{d['TVOL'][0]}{i:03d}"
                lst.append(d.copy())
    dfext = DataFrame(lst, columns = d)
    del dfext['TOPFST']
    del dfext['TOPLST']
    del dfext['SMEMSB']
    dfext = dfext.sort_values('TVOLNO')
    dfext.to_hdf(store, '/CONTROL/EXT_SOURCES', data_columns=True)


Months=('JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC')
def operation(info, llines, op):
    store, parse, dsave, ddfaults = info
    valid = {}
    parsed = {}
    nonempty = []
    lines = iter(llines)
    for line in lines:
        tokens = line.split()
        if len(tokens) == 1:
            table = tokens[0]
            save = dsave[(op, table)]
            cat = get_cat(op, table, save)

            if (save,cat) not in parsed:
                parsed[(save,cat)] = {}
            if table not in parsed[(save,cat)]:
                parsed[(save,cat)][table] = {}
            for line in lines:
                if (op, table) not in parse or line[2:5] == 'END':
                    break
                d = parseD(line, parse, (op, table))
                for opnid in get_opnid(d.pop('OPNID'), op):
                    if opnid not in parsed[(save,cat)][table]:
                        parsed[(save,cat)][table][opnid] = []
                    parsed[(save,cat)] [table][opnid].append(d)

    for save,cat in parsed.keys():
        if cat == 'SKIPPED':
            continue
        elif cat in {'PARAMETERS', 'STATES', 'FLAGS', 'ACTIVITY', 'INFO'}:
            lst = []
            for tablekey in parsed[(save,cat)].keys():
                d = parsed[(save,cat)][tablekey]
                x = concat([DataFrame(d[key][0], index=[key]) for key in d.keys()])

                if tablekey =='SILT-CLAY-PM':
                    x.columns = ['SILT_' + z for z in x.columns]
                    lst.append(x)
                    x = concat([DataFrame(d[key][1], index=[key]) for key in d.keys()])
                    x.columns =  ['CLAY_' + z for z in x.columns]
                lst.append(x)
            df = concat(lst, axis=1, sort=False)
            if cat == 'ACTIVITY':
                valid[op] = set(df.index)
            flag = True if cat not in {'ACTIVITY', 'INFO'} else False
            df = fix_df(df, op, save, ddfaults, valid, sortflag=flag)
            df = df.apply(to_numeric, errors='ignore')
            if op == 'PERLND' and cat == 'ACTIVITY':
                df = df.rename(columns = {'AIRTFG':'ATEMP', 'SNOWFG':'SNOW',
                 'PWATFG':'PWATER', 'SEDFG':'SEDMNT', 'PSTFG':'PSTEMP',
                 'PWGFG':'PWTGAS', 'PQALFG':'PQUAL','MSTLFG':'MSTLAY',
                 'PESTFG':'PEST', 'NITRFG':'NITR', 'PHOSFG':'PHOS',
                 'TRACFG':'TRACER'})
            if op == 'IMPLND' and cat == 'ACTIVITY':
                df = df.rename(columns = {'ATMPFG':'ATEMP', 'SNOWFG':'SNOW',
                 'IWATFG':'IWATER', 'SLDFG':'SOLIDS', 'IWGFG':'IWTGAS',
                 'IQALFG':'IQUAL'})
            if op == 'RCHRES' and cat == 'ACTIVITY':
                df = df.rename(columns  = {'HYDRFG':'HYDR', 'ADFG':'ADCALC',
                 'CONSFG':'CONS', 'HTFG':'HTRCH', 'SEDFG':'SEDTRN',
                 'GQALFG':'GQUAL', 'OXFG':'OXRX', 'NUTFG':'NUTRX',
                 'PLKFG':'PLANK', 'PHFG':'PHCARB'})
            df.columns = [s.replace('(','_').replace(')', '') for s in df.columns]
            df.to_hdf(store, f'{op}/{save}/{cat}', data_columns=True)
            nonempty.append(save)
        elif cat == 'MONTHLYS':
            for tablekey in parsed[(save,cat)].keys():
                name = rename[(op, tablekey)]
                d = parsed[(save,cat)][tablekey]
                x = [DataFrame(d[key][0], index=[key]) for key in d.keys()]
                df = concat(x)
                df.columns = Months
                df = fix_df(df, op, save, ddfaults, valid)
                df.to_hdf(store, f'{op}/{save}/MONTHLY/{name}', data_columns=True)
        elif cat == 'EXTENDED':
            for tablekey in parsed[(save,cat)].keys():
                length, name = extended[(op, tablekey)]
                d = parsed[(save,cat)][tablekey]
                data = []
                for key in d.keys():
                    lst = []
                    for dd in d[key]:
                        lst.extend(dd.values())
                    data.append(lst[0:length])
                cnames = [name+str(i) for i in range(length)]
                df = DataFrame(data, index=d.keys(), columns=cnames )
                df = fix_df(df, op, save, ddfaults, valid)
                df.to_hdf(store, f'{op}/{save}/EXTENDEDS/{name}', data_columns=True)
                nonempty.append(save)
        elif cat in {'CONS', 'PQUAL', 'IQUAL'}:
            d = parsed[(save,cat)]
            firsttable, *tablekeys = d.keys()
            firstid = list(d[firsttable].keys())[0]

            count = d[firsttable][firstid][0][conlike[cat]]
            for i in range(count):
                lst = []
                for tablekey in tablekeys:
                    d = parsed[(save,cat)][tablekey]
                    x = [DataFrame(d[key][i], index=[key]) for key in d.keys()]
                    df = concat(x)
                    lst.append(df)
                savename = save + str(i)
                df = concat(lst, axis=1, sort=False)
                df = fix_df(df, op, save, ddfaults, valid)
                df.to_hdf(store, f'{op}/{save}/{savename}', data_columns=True)
                nonempty.append(save)

    sortedids = sorted(valid[op])
    nonempty = set(nonempty)
    for key in Savedict[op].keys():
        if key in nonempty:
            df = DataFrame(index=sortedids)
            for name,value in Savedict[op][key].items():
                df[name] = int(value)
            df.to_hdf(store, f'{op}/{key}/SAVE', data_columns=True)
