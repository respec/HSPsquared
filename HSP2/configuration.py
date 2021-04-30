''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros


# new activity modules must be added here and in *activites* below
from HSP2.ATEMP  import atemp
from HSP2.SNOW   import snow
from HSP2.PWATER import pwater
from HSP2.SEDMNT import sedmnt
from HSP2.PSTEMP import pstemp
from HSP2.PWTGAS import pwtgas
from HSP2.PQUAL  import pqual

from HSP2.IWATER import iwater
from HSP2.SOLIDS import solids
from HSP2.IWTGAS import iwtgas
from HSP2.IQUAL  import iqual

from HSP2.HYDR   import hydr, expand_HYDR_masslinks
from HSP2.ADCALC import adcalc
from HSP2.HTRCH import htrch, expand_HTRCH_masslinks
from HSP2.SEDTRN import sedtrn, expand_SEDTRN_masslinks
from HSP2.CONS import cons, expand_CONS_masslinks
from HSP2.GQUAL import gqual, expand_GQUAL_masslinks

def noop (store, siminfo, ui, ts):
    ERRMSGS = []
    errors = zeros(len(ERRMSGS), dtype=int)
    return errors, ERRMSGS

# Note: This is the ONLY place in HSP2 that defines activity execution order
activities = {
  'PERLND': {'ATEMP':atemp, 'SNOW':snow, 'PWATER':pwater, 'SEDMNT':sedmnt,
     'PSTEMP':pstemp, 'PWTGAS':pwtgas, 'PQUAL':pqual, 'MSTLAY':noop, 'PEST':noop,
     'NITR':noop, 'PHOS':noop, 'TRACER':noop},
  'IMPLND': {'ATEMP':atemp, 'SNOW':snow, 'IWATER':iwater, 'SOLIDS':solids,
     'IWTGAS':iwtgas, 'IQUAL':iqual},
  'RCHRES': {'HYDR':hydr, 'ADCALC':adcalc, 'CONS':cons, 'HTRCH':htrch,
     'SEDTRN':sedtrn, 'GQUAL':gqual, 'OXRX':noop, 'NUTRX':noop, 'PLANK':noop,
     'PHCARB':noop}}

def expand_masslinks(flags, uci, dat, recs):
    recs = expand_HYDR_masslinks(flags, uci, dat, recs)
    recs = expand_HTRCH_masslinks(flags, uci, dat, recs)
    recs = expand_CONS_masslinks(flags, uci, dat, recs)
    recs = expand_SEDTRN_masslinks(flags, uci, dat, recs)
    recs = expand_GQUAL_masslinks(flags, uci, dat, recs)
    return recs

# NOTE: the flowtype (Python set) at the top of utilities.py may need to be
# updated for new types of flows in new or modified HSP2 modules.