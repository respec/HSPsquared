''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HIMPQUA.FOR module into Python''' 

from math import exp
from numpy import zeros, where, full, float64, int64
from numba import njit
from HSP2.utilities import initm, make_numba_dict, hourflag, initmdiv


''' DESIGN NOTES
Each constituent will be in its own subdirectory in the HDF5 file.
IQUAL high level will contain list of constituents.

NEED to fix units conversions

UNDEFINED: sliqsp
'''

ERRMSGS =('IQUAL: A constituent must be associated with overland flow in order to receive atmospheric deposition inputs','')     #ERRMSG0

def iqual(io_manager, siminfo, uci, ts):
	''' Simulate washoff of quality constituents (other than solids, Heat, dox, and co2)
	using simple relationships with solids And/or water yield'''

	simlen = siminfo['steps']

	nquals = 1
	if 'PARAMETERS' in uci:
		if 'NQUAL' in uci['PARAMETERS']:
			nquals = int(uci['PARAMETERS']['NQUAL'])
	constituents = []
	for index in range(nquals):
		iqual = str(index + 1)
		flags = uci['IQUAL' + iqual + '_FLAGS']
		constituents.append(flags['QUALID'])

	ui = make_numba_dict(uci)
	ui['simlen'] = siminfo['steps']
	ui['delt60'] = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	ui['nquals'] = nquals
	ui['errlen'] = len(ERRMSGS)
	# constituents = ui['CONSTITUENTS']   # (short) names of constituents
	if 'FLAGS' in uci:
		u = uci['FLAGS']

	index = 0
	for constituent in constituents:  # simulate constituent
		index += 1
		# update UI values for this constituent here!
		ui_flags = uci['IQUAL' + str(index) + '_FLAGS']
		ui_parms = uci['IQUAL' + str(index) + '_PARAMETERS']
		qualid = ui_flags['QUALID']
		qtyid  = ui_flags['QTYID']
		QSDFG  = ui_flags['QSDFG']
		QSOFG  = ui_flags['QSOFG']
		VQOFG  = ui_flags['VQOFG']
		sqo    = ui_parms['SQO']
		wsqop  = ui_parms['WSQOP']
		ui['QSDFG' + str(index)] = QSDFG
		ui['QSOFG' + str(index)] = QSOFG
		ui['VQOFG' + str(index)] = VQOFG
		ui['sqo' + str(index)] = sqo
		ui['wsqop' + str(index)] = wsqop

		# handle monthly tables
		ts['POTFW' + str(index)] = initm(siminfo, uci, ui_flags['VPFWFG'], 'IQUAL' + str(index) + '_MONTHLY/POTFW', ui_parms['POTFW'])
		ts['ACQOP' + str(index)] = initm(siminfo, uci, ui_flags['VQOFG'], 'IQUAL' + str(index) + '_MONTHLY/ACQOP', ui_parms['ACQOP'])
		ts['SQOLIM' + str(index)] = initm(siminfo, uci, ui_flags['VQOFG'], 'IQUAL' + str(index) + '_MONTHLY/SQOLIM', ui_parms['SQOLIM'])

		ts['REMQOP' + str(index)] = initmdiv(siminfo, uci, ui_flags['VQOFG'], 'IQUAL' + str(index) + '_MONTHLY/ACQOP',
											 'IQUAL' + str(index) + '_MONTHLY/SQOLIM', ui_parms['ACQOP'],
											 ui_parms['SQOLIM'])

		iqadfgf = 0
		iqadfgc = 0
		ts['IQADFX' + str(index)] = zeros(simlen)
		ts['IQADCN' + str(index)] = zeros(simlen)
		if 'FLAGS' in uci:
			# get atmos dep timeseries
			iqadfgf = u['IQADFG' + str((index * 2) - 1)]
			if iqadfgf > 0:
				ts['IQADFX' + str(index)] = initm(siminfo, uci, iqadfgf, 'IQUAL' + str(index) + '_MONTHLY/IQADFX', 0.0)
			elif iqadfgf == -1:
				ts['IQADFX' + str(index)] = ts['IQADFX' + str(index) + ' 1']
			iqadfgc = u['IQADFG' + str(index * 2)]
			if iqadfgc > 0:
				ts['IQADCN' + str(index)] = initm(siminfo, uci, iqadfgc, 'IQUAL' + str(index) + '_MONTHLY/IQADCN', 0.0)
			elif iqadfgc == -1:
				ts['IQADCN' + str(index)] = ts['IQADCN' + str(index) + ' 1']
		ui['iqadfgf' + str(index)] = iqadfgf
		ui['iqadfgc' + str(index)] = iqadfgc

	for name in ['SLIQSX', 'SLIQO', 'SLIQSP']:
		if name not in ts:
			ts[name] = full(simlen, -1.0E30)

	ts['DAYFG'] = hourflag(siminfo, 0, dofirst=True).astype(float64)

	############################################################################
	errors = _iqual_(ui, ts)  # run IQUAL simulation code
	############################################################################

	return errors, ERRMSGS

@njit(cache=True)
def _iqual_(ui, ts):
	''' Simulate washoff of quality constituents (other than solids, Heat, dox, and co2)
	using simple relationships with solids And/or water yield'''

	errorsV = zeros(int(ui['errlen'])).astype(int64)

	simlen = int(ui['simlen'])
	delt60 = ui['delt60']
	nquals = int(ui['nquals'])

	SURO = ts['SURO']
	SOSLD = ts['SOSLD']
	PREC = ts['PREC']

	SLIQSX = ts['SLIQSX']
	SLIQO  = ts['SLIQO']
	SLIQSP = ts['SLIQSP']

	slifac = ui['SLIFAC']

	DAYFG = ts['DAYFG'].astype(int64)
	# DAYFG[0] = 1

	for i in range(nquals):     # simulate constituent
		index = i + 1
		# update UI values for this constituent here!
		#ui_flags = ui['ui_flags' + str(index)]
		#ui_parms = ui['ui_parms' + str(index)]
		name = 'IQUAL' + str(index)  # arbitrary identification

		QSDFG  = ui['QSDFG' + str(index)]
		QSOFG  = ui['QSOFG' + str(index)]
		VQOFG  = ui['VQOFG' + str(index)]

		iqadfgf = ui['iqadfgf' + str(index)]
		iqadfgc = ui['iqadfgc' + str(index)]
		if QSOFG == 0 and (iqadfgf != 0 or iqadfgc != 0):
			errorsV[0] += 1  # error - non-qualof cannot have atmospheric deposition

		sqo    = ui['sqo' + str(index)]
		wsqop  = ui['wsqop' + str(index)]
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

		POTFW  = ts['POTFW'  + str(index)]
		ACQOP  = ts['ACQOP'  + str(index)]
		SQOLIM = ts['SQOLIM' + str(index)]
		REMQOP = ts['REMQOP' + str(index)]
		IQADFX = ts['IQADFX' + str(index)]
		IQADCN = ts['IQADCN' + str(index)]

		soqo = 0.0
		remqop = 0.0
		soqs = 0.0
		soqoc = 0.0
		soqsp = 0.0
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
						remqop = REMQOP[loop]
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
					if suro > 0.0:
						soqoc = soqo / suro / 3630.0  # 3630 converts from ft3 to ac-in
					else:
						soqoc = -1.0e30
					# end washof()
					
				elif QSOFG == -1:
					''' special case for ches bay - constant conc of qualof input value of acqop = mg/l and soqo = lb/ac
					note - this assumes that qty = lb
					note - acqop is converted to (lb/ac/in) in the run interpeter
					the computed concs (soqoc and soqc) are reported in qty/ft3; the internal units are lb/ac/in and external units are lb/ft3
					the storage (sqo) is reported as zero'''
					acqop = acqop * 0.2266
					soqo  = suro * acqop
					soqoc = acqop / 3630.0  # 3630 converts from ft3 to ac-in
					sqo   = 0.0
				suroqo = soqo

			# sum outflows of constituent n from the land surface
			SOQUAL[loop] = soqual = suroqs + suroqo
			SOQC[loop]   = (soqual / suro / 3630.0) if suro > 0.0 else -1.0e30
			SQO[loop]    = sqo
			SOQS[loop]   = soqs   
			SOQOC[loop]  = soqoc

			SOQO[loop]   = soqo
			SOQSP[loop]  = soqsp

			IQADWT[loop] = adcnfx
			IQADDR[loop] = adfxfx
			IQADEP[loop] = adtot
			
	return errorsV

