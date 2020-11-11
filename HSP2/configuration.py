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
from HSP2.IWATER import iwater
from HSP2.SOLIDS import solids
from HSP2.HYDR   import hydr

def noop (store, siminfo, ui, ts):
    ERRMSGS = []
    errors = zeros(len(ERRMSGS), dtype=int)
    return errors, ERRMSGS

# Note: This is the ONLY place in HSP2 that defines activity execution order
activities = {
  'PERLND': {'ATEMP':atemp, 'SNOW':snow, 'PWATER':pwater, 'SEDMNT':sedmnt,
     'PSTEMP':pstemp, 'PWTGAS':noop, 'PQUAL':noop, 'MSTLAY':noop, 'PEST':noop,
     'NITR':noop, 'PHOS':noop, 'TRACER':noop},
  'IMPLND': {'ATEMP':atemp, 'SNOW':snow, 'IWATER':iwater, 'SOLIDS':solids,
     'IWTGAS':noop, 'IQUAL':noop},
  'RCHRES': {'HYDR':hydr, 'ADCALC':noop, 'CONS':noop, 'HTRCH':noop,
     'SEDTRN':noop, 'GQUAL':noop, 'OXRX':noop, 'NUTRX':noop, 'PLANK':noop,
     'PHCARB':noop}}


# NOTE: the flowtype (Python set) at the top of utilities.py may need to be
# updated for new types of flows in new or modified HSP2 modules.