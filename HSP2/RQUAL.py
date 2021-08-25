''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

import logging
import numpy as np
from numpy import where, zeros, array, float64
from numba import types
from numba.typed import Dict

from HSP2.utilities  import make_numba_dict
from HSP2.RQUAL_Class import RQUAL_Class

ERRSMGS = ('Placeholder')

def rqual(store, siminfo, uci, uci_oxrx, uci_nutrx, uci_plank, uci_phcarb, ts):
	''' Simulate constituents involved in biochemical transformations'''

	# errors (TO-DO! - needs implementation)
	ERRMSGS =('')
	errors = zeros(len(ERRMSGS), dtype=np.int32)	

	# simulation information:
	delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	delts  = siminfo['delt'] * 60
	uunits = siminfo['units']

	siminfo_ = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	for key in set(siminfo.keys()):
		value = siminfo[key]

		if type(value) in {int, float}:
			siminfo_[key] = float(value)
	
	# module flags:
	ui = make_numba_dict(uci)

	NUTFG = int(ui['NUTFG'])
	PLKFG = int(ui['PLKFG'])
	PHFG  = int(ui['PHFG'])

	# create numba dictionaries (empty if not simulated):
	ui_oxrx = make_numba_dict(uci_oxrx)

	ui_nutrx = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if NUTFG == 1:
		ui_nutrx = make_numba_dict(uci_nutrx)

	ui_plank = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if PLKFG == 1:
		ui_plank = make_numba_dict(uci_plank)

	ui_phcarb = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if PHFG == 1:
		ui_phcarb = make_numba_dict(uci_phcarb)

	# hydraulic results:
	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData

	ui['nexits'] = nexits
	ui['vol'] = vol

	ts['VOL'] = VOL
	ts['SROVOL'] = SROVOL
	ts['EROVOL'] = EROVOL

	for i in range(nexits):
		ts['SOVOL' + str(i + 1)] = SOVOL[:, i]
		ts['EOVOL' + str(i + 1)] = EOVOL[:, i]

	# initialize WQ simulation:
	RQUAL = RQUAL_Class(siminfo_, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts)

	# run WQ simulation:
	RQUAL.simulate(ts)

	# SAVE time series results (TO-DO! - needs implementation for outflow series)

	if NUTFG == 1:
		pass

		if PLKFG == 1:
			pass

			if PHFG == 1:
				pass

	return errors, ERRMSGS


#-------------------------------------------------------------------
# mass links:
#-------------------------------------------------------------------

def expand_OXRX_masslinks(flags, uci, dat, recs):
	if flags['OXRX']:
		for i in range(1,3):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'OXRX'

			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'OXCF1'
				rec['SMEMSB1'] = str(i)		# species index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'OXCF2'
				rec['SMEMSB1'] = dat.SMEMSB1  # first sub is exit number
				rec['SMEMSB2'] = str(i)		# species index	
					
			rec['TMEMN'] = 'OXIF'
			rec['TMEMSB1'] = str(i)		# species index
			rec['TMEMSB2'] = '1'
			rec['SVOL'] = dat.SVOL

			recs.append(rec)

	return

def expand_NUTRX_masslinks(flags, uci, dat, recs):
	
	if flags['NUTRX']:
		# dissolved species:
		for i in range(1,5):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'NUTRX'

			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'NUCF1'
				rec['SMEMSB1'] = str(i)   # species index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'NUCF9'
				rec['SMEMSB2'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = str(i)       # species index

			rec['TMEMN'] = 'NUIF1'
			rec['TMEMSB1'] = str(i)		# species index
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

		# particulate species (NH4, PO4):
		for j in range(1,5):		# sediment type
			
			# adsorbed NH4:
			if flags['TAMFG'] and flags['ADNHFG']:
				rec = {}
				rec['MFACTOR'] = dat.MFACTOR
				rec['SGRPN'] = 'NUTRX'

				if dat.SGRPN == "ROFLOW":
					rec['SMEMN'] = 'NUCF2'
					rec['SMEMSB1'] = str(j)   	# sediment type
					rec['SMEMSB2'] = '1'		# NH4 index
				else:
					rec['SMEMN'] = 'OSNH4'
					rec['SMEMSB2'] = dat.SMEMSB1  # exit number
					rec['SMEMSB2'] = str(j)       # sediment type

				rec['TMEMN'] = 'NUIF2'
				rec['TMEMSB1'] = str(j)		# sediment type
				rec['TMEMSB2'] = '1'		# NH4 index
				rec['SVOL'] = dat.SVOL
				recs.append(rec)

			# adsorbed PO4:
			if flags['PO4FG'] and flags['ADPOFG']:
				rec = {}
				rec['MFACTOR'] = dat.MFACTOR
				rec['SGRPN'] = 'NUTRX'

				if dat.SGRPN == "ROFLOW":
					rec['SMEMN'] = 'NUCF2'
					rec['SMEMSB1'] = str(j)   	# sediment type
					rec['SMEMSB2'] = '2'		# PO4 index
				else:
					rec['SMEMN'] = 'OSPO4'
					rec['SMEMSB2'] = dat.SMEMSB1  # exit number
					rec['SMEMSB2'] = str(j)       # sediment type

				rec['TMEMN'] = 'NUIF2'
				rec['TMEMSB1'] = str(j)			# sediment type
				rec['TMEMSB2'] = '2'
				rec['SVOL'] = dat.SVOL
				recs.append(rec)

	return

def expand_PLANK_masslinks(flags, uci, dat, recs):
	if flags['PLANK']:
		
		for i in range(1,6):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'PLANK'

			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'PKCF1'
				rec['SMEMSB1'] = str(i)   # species index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'TPKCF2'
				rec['SMEMSB2'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = str(i)       # species index

			rec['TMEMN'] = 'PKIF'
			rec['TMEMSB1'] = str(i)		#dat.TMEMSB1
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

	return

def expand_PHCARB_masslinks(flags, uci, dat, recs):
	
	if flags['PHCARB']:
		
		for i in range(1,3):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'PHCARB'

			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'PHCF1'
				rec['SMEMSB1'] = str(i)   # species index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'PHCF2'
				rec['SMEMSB2'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = str(i)       # species index

			rec['TMEMN'] = 'PHIF'
			rec['TMEMSB1'] = str(i)			# species index
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

	return