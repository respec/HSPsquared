''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros


# new activity modules must be added here and in *activites* below
from hsp2.hsp2.ATEMP  import atemp
from hsp2.hsp2.SNOW   import snow
from hsp2.hsp2.PWATER import pwater
from hsp2.hsp2.SEDMNT import sedmnt
from hsp2.hsp2.PSTEMP import pstemp
from hsp2.hsp2.PWTGAS import pwtgas
from hsp2.hsp2.PQUAL  import pqual

from hsp2.hsp2.IWATER import iwater
from hsp2.hsp2.SOLIDS import solids
from hsp2.hsp2.IWTGAS import iwtgas
from hsp2.hsp2.IQUAL  import iqual

from hsp2.hsp2.HYDR   import hydr, expand_HYDR_masslinks
from hsp2.hsp2.ADCALC import adcalc
from hsp2.hsp2.HTRCH import htrch, expand_HTRCH_masslinks
from hsp2.hsp2.SEDTRN import sedtrn, expand_SEDTRN_masslinks
from hsp2.hsp2.CONS import cons, expand_CONS_masslinks
from hsp2.hsp2.GQUAL import gqual, expand_GQUAL_masslinks
from hsp2.hsp2.RQUAL import rqual
from hsp2.hsp2.RQUAL import expand_OXRX_masslinks
from hsp2.hsp2.RQUAL import expand_NUTRX_masslinks
from hsp2.hsp2.RQUAL import expand_PLANK_masslinks
from hsp2.hsp2.RQUAL import expand_PHCARB_masslinks

#from hsp2.hsp2.GENER import gener
from hsp2.hsp2.COPY import Copy
from hsp2.hsp2.GENER import Gener

def noop (store, siminfo, ui, ts):
    ERRMSGS = []
    errors = zeros(len(ERRMSGS), dtype=int)
    return errors, ERRMSGS

# Note: This is the ONLY place in HSP2 that defines activity execution order
activities = {
  'COPY' : Copy,
  'GENER' : Gener,
  'PERLND': {'ATEMP':atemp, 'SNOW':snow, 'PWATER':pwater, 'SEDMNT':sedmnt,
     'PSTEMP':pstemp, 'PWTGAS':pwtgas, 'PQUAL':pqual, 'MSTLAY':noop, 'PEST':noop,
     'NITR':noop, 'PHOS':noop, 'TRACER':noop},
  'IMPLND': {'ATEMP':atemp, 'SNOW':snow, 'IWATER':iwater, 'SOLIDS':solids,
     'IWTGAS':iwtgas, 'IQUAL':iqual},
  'RCHRES': {'HYDR':hydr, 'ADCALC':adcalc, 'CONS':cons, 'HTRCH':htrch,
     'SEDTRN':sedtrn, 'RQUAL':rqual, 'GQUAL':gqual, 'OXRX':noop, 'NUTRX':noop, 'PLANK':noop,
     'PHCARB':noop}}

def expand_masslinks(flags, uci, dat, recs):
    recs = expand_HYDR_masslinks(flags, uci, dat, recs)
    recs = expand_HTRCH_masslinks(flags, uci, dat, recs)
    recs = expand_CONS_masslinks(flags, uci, dat, recs)
    recs = expand_SEDTRN_masslinks(flags, uci, dat, recs)
    recs = expand_GQUAL_masslinks(flags, uci, dat, recs)
    recs = expand_OXRX_masslinks(flags, uci, dat, recs)
    recs = expand_NUTRX_masslinks(flags, uci, dat, recs)
    recs = expand_PLANK_masslinks(flags, uci, dat, recs)
    recs = expand_PHCARB_masslinks(flags, uci, dat, recs)

    return recs

# NOTE: the flowtype (Python set) at the top of utilities.py may need to be
# updated for new types of flows in new or modified HSP2 modules.