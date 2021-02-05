''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HIMPQUA.FOR module into Python''' 

from math import exp
from numpy import zeros, where, full
from numba import jit
from HSP2.utilities import initm, make_numba_dict, hourflag


''' DESIGN NOTES
Each constituent will be in its own subdirectory in the HDF5 file.
IQUAL high level will contain list of constituents.

NEED to fix units conversions

UNDEFINED: sliqsp
'''

ERRMSG = []

def iqual(store, siminfo, uci, ts):
	''' Simulate washoff of quality constituents (other than solids, Heat, dox, and co2)
	using simple relationships with solids And/or water yield'''

	nquals = 1
	if 'PARAMETERS' in uci:
		if 'NQUAL' in uci['PARAMETERS']:
			nquals = uci['PARAMETERS']['NQUAL']
	constituents = []
	for index in range(nquals):
		iqual = str(index + 1)
		flags = uci['IQUAL' + iqual + '_FLAGS']
		constituents.append(flags['QUALID'])

	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = siminfo['delt'] / 60     # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	tindex = siminfo['tindex']

	SURO  = ts['SURO']
	SOSLD = ts['SOSLD']
	PREC  = ts['PREC']
	
	for name in ['SLIQSX', 'SLIQO', 'SLIQSP']:
		if name not in ts:
			ts[name] = full(simlen, -1.0E30)
	SLIQSX = ts['SLIQSX']
	SLIQO  = ts['SLIQO']
	SLIQSP = ts['SLIQSP']

	ui = make_numba_dict(uci)
	# constituents = ui['CONSTITUENTS']   # (short) names of constituents
	slifac = ui['SLIFAC']
	
	DAYFG = hourflag(siminfo, 0, dofirst=True).astype(bool)
	# DAYFG[0] = 1

	index = 0
	for constituent in constituents:     # simulate constituent
		index +=1
		# update UI values for this constituent here!
		ui_flags = uci['IQUAL' + str(index) + '_FLAGS']
		ui_parms = uci['IQUAL' + str(index) + '_PARAMETERS']
		name = 'IQUAL' + str(index)  # arbitrary identification

		qualid = ui_flags['QUALID']
		qtyid  = ui_flags['QTYID']
		QSDFG  = ui_flags['QSDFG']
		QSOFG  = ui_flags['QSOFG']
		VQOFG  = ui_flags['VQOFG']
		
		sqo    = ui_parms['SQO']
		wsqop  = ui_parms['WSQOP']
		wsfac = 2.30 / wsqop

		# preallocate output arrays (always needed)
		SOQUAL = ts[name + '_SOQUAL'] = zeros(simlen)
		SOQC   = ts[name + '_SOQC']   = zeros(simlen)
		SOQO   = ts[name + '_SOQO'] = zeros(simlen)
		
		# preallocate output arrays (QUALOF)
		SQO    = ts[name + '_SQO']    = zeros(simlen)
		SOQOC  = ts[name + '_SOQOC']  = zeros(simlen)

		# preallocate output arrays (QUALSD)
		SOQS   = ts[name + '_SOQS']   = zeros(simlen)
		SOQSP  = ts[name + '_SOQSP'] = zeros(simlen)

		# preallocate output arrays for atmospheric deposition
		IQADDR = ts[name + '_IQADDR']   = zeros(simlen)
		IQADWT = ts[name + '_IQADWT'] = zeros(simlen)
		IQADEP = ts[name + '_IQADEP'] = zeros(simlen)

		SLIQO  = ts[name + '_SLIQO'] = zeros(simlen)   # lateral inflow
		INFLOW = ts[name + '_INFLOW'] = zeros(simlen)  # total inflow

		
		# handle monthly tables

		ts['POTFW'] = initm(siminfo, uci, ui_flags['VPFWFG'], 'IQUAL' + str(index) + '_MONTHLY/POTFW', ui_parms['POTFW'])
		ts['ACQOP'] = initm(siminfo, uci, ui_flags['VQOFG'], 'IQUAL' + str(index) + '_MONTHLY/ACQOP', ui_parms['ACQOP'])
		ts['SQOLIM'] = initm(siminfo, uci, ui_flags['VQOFG'], 'IQUAL' + str(index) + '_MONTHLY/SQOLIM', ui_parms['SQOLIM'])

		if 'FLAGS' in uci:
			u = uci['FLAGS']
			# get atmos dep timeseries
			iqadfgf = u['IQADFG' + str((index * 2) - 1)]
			if iqadfgf > 0:
				ts['IQADFX'] = initm(siminfo, uci, iqadfgf, 'IQUAL' + str(index) + '_MONTHLY/IQADFX', 0.0)
			elif iqadfgf == -1:
				ts['IQADFX'] = ts['IQADFX' + str(index) + ' 1']
			iqadfgc = u['IQADFG' + str(index * 2)]
			if iqadfgc > 0:
				ts['IQADCN'] = initm(siminfo, uci, iqadfgc, 'IQUAL' + str(index) + '_MONTHLY/IQADCN', 0.0)
			elif iqadfgc == -1:
				ts['IQADCN'] = ts['IQADCN' + str(index) + ' 1']

		if 'IQADFX' not in ts:
			ts['IQADFX'] = zeros(simlen)
		if 'IQADCN' not in ts:
			ts['IQADCN'] = zeros(simlen)

		POTFW  = ts['POTFW']
		ACQOP  = ts['ACQOP']
		SQOLIM = ts['SQOLIM']
		IQADFX = ts['IQADFX']
		IQADCN = ts['IQADCN']

		soqo = 0.0
		remqop = 0.0
		soqs = 0.0
		soqoc = 0.0
		for loop in range(simlen):
			suro   = SURO[loop]
			sosld  = SOSLD[loop]
			dayfg  = DAYFG[loop]
			sliqsx = SLIQSX[loop]
			sliqo  = SLIQO[loop]
			sliqsp = SLIQSP[loop]
			potfw  = POTFW[loop]
			acqop  = ACQOP[loop]
				
			# simulate by association with solids
			suroqs = 0.0
			soqsp  = 0.0
			if QSDFG:
				# washsd ()
				if dayfg == 1:      # it is the first interval of the day
					potfw = POTFW[loop]

				# associate with washoff of solids - units are qty/acre-ivl
				if sosld == 0.0:
					soqs = 0.0
				else:
					if sliqsp >= 0.0:     # lateral inflow has an effect on washoff potency factor
						soqsp = sliqsp * slifac + potfw * (1.0 - slifac)
						soqs  = sosld * soqsp
					else:                 # no effect of lateral inflow
						soqsp = potfw
						soqs  = sosld * potfw
				# end washsd()
				
				suroqs = soqs

			# simulate by association with overland flow
			suroqo = 0.0
			adtot  = 0.0
			adfxfx = 0.0
			adcnfx = 0.0
			if QSOFG != 0:  #  constituent n is simulated by association with overland flow; the value of qofp refers to the set of overland flow associated parameters to use
				if QSOFG >= 1:   # standard qualof simulation
					# washof ()
					''' Simulate accumulation of a quality constituent on the land surface and its removal using a constant unit rate and by direct washoff by overland flow'''
					if dayfg == 1:
						remqop = acqop / SQOLIM[loop]
						if QSOFG == 1 :   #update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
							sqo = acqop + sqo * (1.0 - remqop)

					# handle atmospheric deposition
					adfxfx = IQADFX[loop]  		                    # dry deposition
					adcnfx = IQADCN[loop] * PREC[loop] * 3630.0 	# wet deposition

					adtot = adfxfx + adcnfx  # total atmospheric deposition

					if QSOFG == 2:  # update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
						dummy = remqop + (adtot + sliqo) / (acqop / remqop)
						if dummy > 1.0:
							dummy = 1.0
						sqo = acqop * (delt60 / 24.0) + sqo * (1.0 - dummy)**(delt60 / 24.0)

					sqo = sqo + sliqo + adtot   # update storage

					# simulate washoff by overland flow - units are qty/acre-ivl
					soqo = 0.0
					if suro > 0.0 and sqo > 0.0:   # there is some quality constituent (no. qofp) in storage; washoff can occur
						soqo  = sqo * (1.0 - exp (-suro * wsfac))
						sqo   = sqo - soqo  # update storage of constituent - units are in qty/acre

					# compute and output concentration - units are qty/acre-in.
					soqoc = soqo / suro if suro > 0.0 else -1.0e30					
					# end washof()
					
				elif QSOFG == -1:
					''' special case for ches bay - constant conc of qualof input value of acqop = mg/l and soqo = lb/ac
					note - this assumes that qty = lb
					note - acqop is converted to (lb/ac/in) in the run interpeter
					the computed concs (soqoc and soqc) are reported in qty/ft3; the internal units are lb/ac/in and external units are lb/ft3
					the storage (sqo) is reported as zero'''
					acqop = acqop * 0.2266
					soqo  = suro * acqop
					soqoc = acqop
					sqo   = 0.0
				suroqo = soqo

			# sum outflows of constituent n from the land surface
			SOQUAL[loop] = soqual = suroqs + suroqo
			SOQC[loop]   = (soqual / suro / 3630.0) if suro > 0.0 else -1.0e30
			SQO[loop]    = sqo
			SOQS[loop]   = soqs   
			SOQOC[loop]  = soqoc / 3630.0     # 3630 converts from ft3 to ac-in

			SOQO[loop]   = soqo
			SOQSP[loop]  = soqsp

			IQADWT[loop] = adcnfx
			IQADDR[loop] = adfxfx
			IQADEP[loop] = adtot
			
	return errorsV, ERRMSG

