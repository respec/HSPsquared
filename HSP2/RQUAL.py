''' Copyright (c) 2020 by RESPEC, INC.
Authors: Robert Heaphy, Ph.D. and Paul Duda
License: LGPL2
'''

import logging
import numpy as np
from numpy import where, zeros, array, float64, full
from numba import types, njit
from numba.typed import Dict

from HSP2.utilities  import make_numba_dict, initm
from HSP2.RQUAL_Class import RQUAL_Class

ERRMSGS_oxrx = ('OXRX: Warning -- SATDO is less than zero. This usually occurs when water temperature is very high (above ~66 deg. C). This usually indicates an error in input GATMP (or TW, if HTRCH is not being simulated).',)
ERRMSGS_nutrx = ('NUTRX: Error -- Inconsistent flags for NH4; TAM is not being simulated, but NH3 volatilization and/or NH4 adsorption are being simulated.',
					'NUTRX: Error -- Inconsistent flags for PO4; PO4 is not being simulated, but PO4 adsorption is being simualted.',
					'NUTRX: Error -- Sediment-associated NH4 and/or PO4 is being simulated, but sediment is not being simulated in module SEDTRN.',
					'NUTRX: Error -- Inorganic nutrient mass stored in or leaving the reach non-zero, but is expected to be non-zero due to lack of suspended sediment mass.',
					'NUTRX: Error -- Inorganic nutrient mass in bed is expected to be zero (due to the lack of bed sediments).')
ERRMSGS_plank = ('PLANK: Error -- Zooplankton cannot be simulated without phytoplankton.',
					'PLANK: Error -- Ammonia cannot be included in the N supply if it is not being simulated.',
					'PLANK: Error -- Phosphate must be simulated if plankton are being simulated.')
ERRMSGS_phcarb = ('PHCARB: Error -- Invalid CONS index specified for ALKCON (i.e., ALKCON > NCONS).',
					'PHCARB: Error -- A satisfactory solution for pH has not been reached.')

def rqual(store, siminfo, uci, uci_oxrx, uci_nutrx, uci_plank, uci_phcarb, ts):
	''' Simulate constituents involved in biochemical transformations'''

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
	ui_oxrx['errlen'] = len(ERRMSGS_oxrx)

	ui_nutrx = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if NUTFG == 1:
		ui_nutrx = make_numba_dict(uci_nutrx)
		ui_nutrx['errlen'] = len(ERRMSGS_nutrx)

	ui_plank = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if PLKFG == 1:
		ui_plank = make_numba_dict(uci_plank)
		ui_plank['errlen'] = len(ERRMSGS_plank)

	ui_phcarb = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	if PHFG == 1:
		ui_phcarb = make_numba_dict(uci_phcarb)
		ui_phcarb['errlen'] = len(ERRMSGS_phcarb)

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

	phval_init = 7.
	tamfg = 0
	phflag = 2
	if 'NH3FG' in ui_nutrx:
		tamfg = ui_nutrx['NH3FG']
	if 'PHFLAG' in ui_nutrx:
		phflag = ui_nutrx['PHFLAG']
	if tamfg == 1:
		if 'PHVAL' in ui_nutrx:
			phval_init = ui_nutrx['PHVAL']
	if 'PHVAL' not in ts:
		ts['PHVAL'] = full(simlen, phval_init)
	if phflag == 3:
		ts['PHVAL'] = initm(siminfo, ui_nutrx, phflag, 'MONTHLY/PHVAL', phval_init)

	#---------------------------------------------------------------------
	#	input time series processing (atm. deposition, benthic inverts, etc.)
	#---------------------------------------------------------------------
	# NUTRX atmospheric deposition - initialize time series:
	if NUTFG == 1:
		for j in range(1, 4):
			n = (2 * j) - 1

			# dry deposition:
			nuadfg_dd = int(ui_nutrx['NUADFG' + str(n)])
			NUADFX = zeros(simlen)

			if nuadfg_dd > 0:
				NUADFX = initm(siminfo, ui_nutrx, nuadfg_dd, 'NUADFX' + str(j) + '_MONTHLY/NUADFX' + str(j), 0.0)
			elif nuadfg_dd == -1:
				if 'NUADFX' + str(j) in ts:
					NUADFX = ts['NUADFX' + str(j)]
				elif 'NUADFX' + str(j) + ' 1' in ts:
					NUADFX = ts['NUADFX' + str(j) + ' 1']
				else:
					pass		#ERRMSG?
			ts['NUADFX' + str(j)] = NUADFX

			# wet deposition:
			nuadfg_wd = int(ui_nutrx['NUADFG' + str(n+1)])
			NUADCN = zeros(simlen)

			if nuadfg_wd > 0:
				NUADCN = initm(siminfo, ui_nutrx, nuadfg_wd, 'NUADCN' + str(j) + '_MONTHLY/NUADCN' + str(j), 0.0)
			elif nuadfg_wd == -1:
				if 'NUADCN' + str(j) in ts:
					NUADCN = ts['NUADCN' + str(j)]
				elif 'NUADCN' + str(j) + ' 1' in ts:
					NUADCN = ts['NUADCN' + str(j) + ' 1']
				else:
					pass		#ERRMSG?
			ts['NUADCN' + str(j)] = NUADCN

			# convert units to internal
			if uunits == 1:  # convert from lb/ac.day to mg.ft3/l.ft2.ivl
				if 'NUADFX' + str(j) in ts:
					ts['NUADFX' + str(j)] *= 0.3677 * delt60 / 24.0
			else:  # convert from kg/ha.day to mg.m3/l.m2.ivl
				if 'NUADFX' + str(j) in ts:
					ts['NUADFX' + str(j)] *= 0.1 * delt60 / 24.0

	if PLKFG == 1:		
		# PLANK atmospheric deposition - initialize time series:
		for j in range(1, 4):
			n = (2 * j) - 1

			# dry deposition:
			PLADFX = zeros(simlen)
			pladfg_dd = int(ui_plank['PLADFG' + str(j)])

			if pladfg_dd > 0:
				PLADFX = initm(siminfo, ui_plank, pladfg_dd, 'PLADFX' + str(j) + '_MONTHLY/PLADFX' + str(j), 0.0)
			elif pladfg_dd == -1:
				if 'PLADFX' + str(j) in ts:
					PLADFX = ts['PLADFX' + str(j)]
				elif 'PLADFX' + str(j) + ' 1' in ts:
					PLADFX = ts['PLADFX' + str(j) + ' 1']
				else:
					pass		#ERRMSG?
			ts['PLADFX' + str(j)] = PLADFX

			# wet deposition:
			PLADCN = zeros(simlen)
			pladfg_wd = int(ui_plank['PLADFG' + str(n+1)])

			if pladfg_wd > 0:
				PLADCN = initm(siminfo, ui_plank, pladfg_wd, 'PLADCN' + str(j) + '_MONTHLY/PLADCN' + str(j), 0.0)
			elif pladfg_wd == -1:
				if 'PLADCN' + str(j) in ts:
					PLADCN = ts['PLADCN' + str(j)]
				elif 'PLADCN' + str(j) + ' 1' in ts:
					PLADCN = ts['PLADCN' + str(j) + ' 1']
				else:
					pass		#ERRMSG?
			ts['PLADCN' + str(j)] = PLADCN

			# convert units to internal
			if uunits == 1:  # convert from lb/ac.day to mg.ft3/l.ft2.ivl
				if 'PLADFX' + str(j) in ts:
					ts['PLADFX' + str(j)] *= 0.3677 * delt60 / 24.0
			else:  # convert from kg/ha.day to mg.m3/l.m2.ivl
				if 'PLADFX' + str(j) in ts:
					ts['PLADFX' + str(j)] *= 0.1 * delt60 / 24.0

		# PLANK - benthic invertebrates:
		balfg = 0
		binv_init = 0.0
		binvfg = 2
		if 'BALFG' in ui_plank:
			balfg = ui_plank['BALFG']
		if balfg == 2:  # user has selected multiple species with more complex kinetics
			if 'BINV' in ui_plank:
				binv_init = ui_plank['BINV']
			if 'BINVFG' in ui_plank:
				binvfg = ui_plank['BINVFG']
		if 'BINV' not in ts:
			ts['BINV'] = full(simlen, binv_init)
		if balfg == 2 and binvfg == 3:
			ts['BINV'] = initm(siminfo, ui_plank, binvfg, 'MONTHLY/BINV', binv_init)

	#---------------------------------------------------------------------
	# initialize & run integerated WQ simulation:
	#---------------------------------------------------------------------

	(err_oxrx, err_nutrx, err_plank, err_phcarb) \
		= _rqual_run(siminfo_, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts)

	#---------------------------------------------------------------------
	# compile errors & return:
	#---------------------------------------------------------------------

	(errors, ERRMSGS) = _compile_errors(NUTFG, PLKFG, PHFG, err_oxrx, err_nutrx, err_plank, err_phcarb)

	# for multiple exits, modify save table as needed
	if nexits > 1:
		u = uci_oxrx['SAVE']
		for i in range(nexits):
			u[f'OXCF2_{i + 1}1'] = u['OXCF2_11']
			u[f'OXCF2_{i + 1}2'] = u['OXCF2_12']

		u = uci_nutrx['SAVE']
		for i in range(nexits):
			u[f'NUCF9_{i + 1}1'] = u['NUCF9_11']
			u[f'NUCF9_{i + 1}2'] = u['NUCF9_12']
			u[f'NUCF9_{i + 1}3'] = u['NUCF9_13']
			u[f'NUCF9_{i + 1}4'] = u['NUCF9_14']
			u[f'OSNH4_{i + 1}1'] = u['OSNH4_11']
			u[f'OSNH4_{i + 1}2'] = u['OSNH4_12']
			u[f'OSNH4_{i + 1}3'] = u['OSNH4_13']
			u[f'OSNH4_{i + 1}4'] = u['OSNH4_14']
			u[f'OSPO4_{i + 1}1'] = u['OSPO4_11']
			u[f'OSPO4_{i + 1}2'] = u['OSPO4_12']
			u[f'OSPO4_{i + 1}3'] = u['OSPO4_13']
			u[f'OSPO4_{i + 1}4'] = u['OSPO4_14']

		u = uci_plank['SAVE']
		for i in range(nexits):
			u[f'PKCF2_{i + 1}1'] = u['PKCF2_11']
			u[f'PKCF2_{i + 1}2'] = u['PKCF2_12']
			u[f'PKCF2_{i + 1}3'] = u['PKCF2_13']
			u[f'PKCF2_{i + 1}4'] = u['PKCF2_14']
			u[f'PKCF2_{i + 1}5'] = u['PKCF2_15']
			u[f'TPKCF2_{i + 1}1'] = u['TPKCF2_11']
			u[f'TPKCF2_{i + 1}2'] = u['TPKCF2_12']
			u[f'TPKCF2_{i + 1}3'] = u['TPKCF2_13']
			u[f'TPKCF2_{i + 1}4'] = u['TPKCF2_14']
			u[f'TPKCF2_{i + 1}5'] = u['TPKCF2_15']

		u = uci_phcarb['SAVE']
		for i in range(nexits):
			u[f'OTIC{i + 1}'] = u['OTIC1']
			u[f'OCO2{i + 1}'] = u['OCO21']

	return errors, ERRMSGS


@njit(cache=True)
def _rqual_run(siminfo_, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts):

	nutrx_errors = zeros((0), dtype=np.int64)
	plank_errors = zeros((0), dtype=np.int64)
	phcarb_errors = zeros((0), dtype=np.int64)

	# initialize WQ simulation:
	RQUAL = RQUAL_Class(siminfo_, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts)

	# run WQ simulation:
	RQUAL.simulate(ts)

	# return error data:
	oxrx_errors = RQUAL.OXRX.errors
	if RQUAL.NUTFG == 1:
		nutrx_errors = RQUAL.NUTRX.errors
	if RQUAL.PLKFG == 1:
		plank_errors = RQUAL.PLANK.errors
	if RQUAL.PHFG == 1:
		phcarb_errors = RQUAL.PHCARB.errors

	return oxrx_errors, nutrx_errors, plank_errors, phcarb_errors


def _compile_errors(NUTFG, PLKFG, PHFG, err_oxrx, err_nutrx, err_plank, err_phcarb):

	errlen_oxr = len(err_oxrx)
	errlen_ntr = 0;	errlen_plk = 0;	errlen_phc = 0

	if NUTFG == 1: 
		errlen_ntr = len(err_nutrx)
		if PLKFG == 1:
			errlen_plk += len(err_plank)
			if PHFG == 1:
				errlen_phc += len(err_phcarb)

	errlen = errlen_oxr + errlen_ntr + errlen_plk + errlen_phc

	errors = zeros(errlen, dtype=np.int64)
	ERRMSGS = ()

	ierr = -1
	for i in range(errlen_oxr):
		ierr += 1
		errors[ierr] = err_oxrx[i]
		ERRMSGS += (ERRMSGS_oxrx[i],)

	for i in range(errlen_ntr):
		ierr += 1
		errors[ierr] = err_nutrx[i]
		ERRMSGS += (ERRMSGS_nutrx[i],)

	for i in range(errlen_plk):
		ierr += 1
		errors[ierr] = err_plank[i]
		ERRMSGS += (ERRMSGS_plank[i],)

	for i in range(errlen_phc):
		ierr += 1
		errors[ierr] = err_phcarb[i]
		ERRMSGS += (ERRMSGS_phcarb[i],)

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

	return recs

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
				rec['SMEMSB1'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = str(i)       # species index

			rec['TMEMN'] = 'NUIF1'
			rec['TMEMSB1'] = str(i)		# species index
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

		# particulate species (NH4, PO4):
		for j in range(1,4):		# sediment type
			
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
					rec['SMEMSB1'] = dat.SMEMSB1  # exit number
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
					rec['SMEMSB1'] = dat.SMEMSB1  # exit number
					rec['SMEMSB2'] = str(j)       # sediment type

				rec['TMEMN'] = 'NUIF2'
				rec['TMEMSB1'] = str(j)			# sediment type
				rec['TMEMSB2'] = '2'
				rec['SVOL'] = dat.SVOL
				recs.append(rec)

	return recs

def expand_PLANK_masslinks(flags, uci, dat, recs):
	if flags['PLANK']:
		
		for i in range(1,6):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'PLANK'

			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'PKCF1_'
				rec['SMEMSB1'] = str(i)   # species index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'PKCF2_'
				rec['SMEMSB1'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = str(i)       # species index

			rec['TMEMN'] = 'PKIF'
			rec['TMEMSB1'] = str(i)		#dat.TMEMSB1
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

	return recs

def expand_PHCARB_masslinks(flags, uci, dat, recs):
	
	if flags['PHCARB']:
		
		for i in range(1,3):
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'PHCARB'

			if dat.SGRPN == "ROFLOW":
				if i == 1:
					rec['SMEMN'] = 'ROTIC'
				elif i == 2:
					rec['SMEMN'] = 'ROCO2'
				rec['SMEMSB1'] = ''
				rec['SMEMSB2'] = ''
			else:
				if i == 1:
					rec['SMEMN'] = 'OTIC'
				elif i == 2:
					rec['SMEMN'] = 'OCO2'
				rec['SMEMSB1'] = dat.SMEMSB1  # exit number
				rec['SMEMSB2'] = ''       # species index

			rec['TMEMN'] = 'PHIF'
			rec['TMEMSB1'] = str(i)			# species index
			rec['TMEMSB2'] = ''
			rec['SVOL'] = dat.SVOL
			recs.append(rec)

	return recs