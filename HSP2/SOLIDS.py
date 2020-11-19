''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HIMPSLD.FOR module into Python''' 

from numpy import zeros, where
from numba import njit
from HSP2.utilities  import initm, make_numba_dict, hourflag


MFACTA = 1.0  # english units

ERRMSG = []


def solids(store, siminfo, uci, ts):
	'''Accumulate and remove solids from the impervious land segment'''
	
	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = siminfo['delt'] / 60     # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	tindex = siminfo['tindex']

	ui = make_numba_dict(uci)  # Note: all values converted to float automatically
	SDOPFG = ui['SDOPFG']
	keim   = ui['KEIM']
	jeim   = ui['JEIM']
	slds   = ui['SLDS']
	
	for name in ['SURO', 'SURS', 'PREC', 'SLSLD']:
		if name not in ts:
			ts[name] = zeros(simlen)
	SURO  = ts['SURO']
	SURS  = ts['SURS']	
	PREC  = ts['PREC']
	SLSLD = ts['SLSLD']  # lateral input of solids is considered
	
	# preallocate output arrays
	SOSLD = ts['SOSLD'] = zeros(simlen)
	SLDS  = ts['SLDS']  = zeros(simlen)

	drydfg = 1  # assume day is dry
	DAYFG = hourflag(siminfo, 0, dofirst=True).astype(bool)

	u = uci['PARAMETERS']
	# process optional monthly arrays to return interpolated data or constant array
	ts['ACCSDP'] = initm(siminfo, uci, u['VASDFG'], 'ACCSDM', u['ACCSDP'])
	ts['REMSDP'] = initm(siminfo, uci, u['VRSDFG'], 'REMSDM', u['REMSDP'])
	ACCSDP = ts['ACCSDP']
	REMSDP = ts['REMSDP']

	slds = ui['SLDS']
	for loop in range(simlen):
		suro   = SURO[loop]
		surs   = SURS[loop]
		prec   = PREC[loop]
		slsld  = SLSLD[loop]
		accsdp = ACCSDP[loop]
		remsdp = REMSDP[loop]
		dayfg  = DAYFG[loop]
		
		# washoff solids from the impervious segment
		if SDOPFG == 1:       # use method 1
			if suro > 0.0:    #impervious surface runoff occurs, so solids may be removed
				# calculate capacity for removing solids - units are tons/acre-ivl
				arg = surs + suro   	# get argument used in transport equations
				stcap = delt60 * keim * (arg / delt60)**jeim
				if stcap > slds:	# insufficient solids storage, base removal on that available; sosld is in tons/acre-ivl
					sosld = slds * suro / arg
				else:				# sufficient solids storage, base removal on the calculated capacity'''
					sosld = stcap * suro / arg
					slds  = slds - sosld
			else:
				sosld = 0.0  # no runoff occurs, so no removal by runoff
		else:              # using method 2
			# Warning: this method of computing solids washoff has not been tested. but it is dimensionally homogeneous
			if suro > 0.0:     # impervious surface runoff occurs, so solids may be removed
				stcap = delt60 * keim * (suro / delt60)**jeim  # calculate capacity for removing solids - units are tons/acre-ivl
				if  stcap > slds:   # insufficient solids storage, base removal on that available; sosld is in tons/acre-ivl
					sosld = slds
					slds  = 0.0
				else:               # sufficient solids storage, base removal on the calculated capacity
					sosld = stcap
					slds  = slds - sosld
			else:
				sosld = 0.0  # no runoff occurs, so no removal by runoff

		'''Accumulate and remove solids independent of runoff. The calculation is done at the start of each day, if the previous day was dry.'''
		if dayfg == 0:			# it is not the first interval of a new day
			if prec > 0.0:			# it is not a dry day
				drydfg = 0
		else:                     # it is the first interval of a new day
			if drydfg == 1:       # precipitation did not occur during the previous day
				# update storage due to accumulation and removal which  occurs independent of runoff - units are lbs/acre
				slds = accsdp + slds * (1.0 - remsdp)

			if prec > 0.0:
				# there is precipitation on the first interval of the new day
				drydfg = 0
			else:
				drydfg = 1

		SOSLD[loop] = sosld  # * MFACTA
		SLDS[loop]  = slds   # * MFACTA
	return errorsV, ERRMSG
