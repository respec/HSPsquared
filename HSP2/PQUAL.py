''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERQUA.FOR module into Python''' 

from math import exp
from numpy import zeros, where, full
from numba import jit
from HSP2.utilities import initm, make_numba_dict, hourflag

''' DESIGN NOTES
Each constituent will be in its own subdirectory in the HDF5 file.
PQUAL high level will contain list of constituents.

NEED to check all units conversions
'''

ERRMSG = []

# english system
FACTA  = 1.0
CFACTA = 2.7548E-04
PFACTA = 1.0


def pqual(store, siminfo, uci, ts):
	''' Simulate quality constituents (other than sediment, heat, dox, and co2)
	using simple relationships with sediment and water yield'''

	nquals = 1
	if 'PARAMETERS' in uci:
		if 'NQUAL' in uci['PARAMETERS']:
			nquals = uci['PARAMETERS']['NQUAL']
	constituents = []
	for index in range(nquals):
		pqual = str(index + 1)
		flags = uci['PQUAL' + pqual + '_FLAGS']
		constituents.append(flags['QUALID'])

	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	tindex = siminfo['tindex']

	for name in ['SURO', 'IFWO', 'AGWO', 'PERO', 'WSSD', 'SCRSD']:
		if name not in ts:
			ts[name] = zeros(simlen)
	SURO  = ts['SURO']
	IFWO  = ts['IFWO']
	AGWO  = ts['AGWO']
	PERO  = ts['PERO']
	WSSD  = ts['WSSD']
	SCRSD = ts['SCRSD']
	PREC  = ts['PREC']

	ui = make_numba_dict(uci)
	sdlfac = ui['SDLFAC']
	slifac = ui['SLIFAC']
	ilifac = ui['ILIFAC']
	alifac = ui['ALIFAC']

	DAYFG = hourflag(siminfo, 0, dofirst=True).astype(bool)

	index = 0
	for constituent in constituents:     # simulate constituent
		index += 1
		# update UI values for this constituent here!
		ui_flags = uci['PQUAL' + str(index) + '_FLAGS']
		ui_parms = uci['PQUAL' + str(index) + '_PARAMETERS']
		name = 'IQUAL' + str(index)  # arbitrary identification

		qualid = ui_flags['QUALID']
		qtyid  = ui_flags['QTYID']
		QSDFG  = ui_flags['QSDFG']
		QSOFG  = ui_flags['QSOFG']
		QIFWFG = ui_flags['QIFWFG']
		QAGWFG = ui_flags['QAGWFG']
		sqo    = ui_parms['SQO']
		wsqop  = ui_parms['WSQOP']
		wsfac = 2.30 / wsqop

		# preallocate output arrays (always needed)
		SQO    = ts[name + '_SQO']    = zeros(simlen)

		# preallocate output arrays (QUALSD)
		SOQSP = ts[name + '_SOQSP'] = zeros(simlen)

		# preallocate output arrays (QUALOF)
		SOQOC = ts[name + '_SOQOC'] = zeros(simlen)
		SOQC = ts[name + '_SOQC'] = zeros(simlen)
		IOQC = ts[name + '_IOQC'] = zeros(simlen)
		AOQC = ts[name + '_AOQC'] = zeros(simlen)
		POQC = ts[name + '_POQC'] = zeros(simlen)

		WASHQS = ts[name + '_WASHQS'] = zeros(simlen)
		SCRQS  = ts[name + '_SCRQS'] = zeros(simlen)
		SOQS   = ts[name + '_SOQS'] = zeros(simlen)
		SOQO   = ts[name + '_SOQO'] = zeros(simlen)
		SOQS   = ts[name + '_SOQS']   = zeros(simlen)
		SOQUAL = ts[name + '_SOQUAL'] = zeros(simlen)
		IOQUAL = ts[name + '_IOQUAL'] = zeros(simlen)
		AOQUAL = ts[name + '_AOQUAL'] = zeros(simlen)
		POQUAL = ts[name + '_POQUAL'] = zeros(simlen)

		# preallocate output arrays for atmospheric deposition
		PQADDR = ts[name + '_PQADDR'] = zeros(simlen)
		PQADWT = ts[name + '_PQADWT'] = zeros(simlen)
		PQADEP = ts[name + '_PQADEP'] = zeros(simlen)

		SLIQO = ts[name + '_SLIQO'] = zeros(simlen)  # lateral inflow
		INFLOW = ts[name + '_INFLOW'] = zeros(simlen)  # total inflow

		for name in ['SLIQSP', 'ILIQC', 'ALIQC']:
			if name not in ts:
				ts[name] = zeros(simlen)
		SLIQSP = ts['SLIQSP']
		ILIQC  = ts['ILIQC']
		ALIQC  = ts['ALIQC']

		ts['POTFW'] = initm(siminfo, uci, ui_flags['VPFWFG'], 'PQUAL' + str(index) + '_MONTHLY/POTFW', ui_parms['POTFW'])
		ts['POTFS'] = initm(siminfo, uci, ui_flags['VPFSFG'], 'PQUAL' + str(index) + '_MONTHLY/POTFS', ui_parms['POTFS'])
		ts['ACQOP'] = initm(siminfo, uci, ui_flags['VQOFG'], 'PQUAL' + str(index) + '_MONTHLY/ACQOP', ui_parms['ACQOP'])
		ts['SQOLIM'] = initm(siminfo, uci, ui_flags['VQOFG'], 'PQUAL' + str(index) + '_MONTHLY/SQOLIM', ui_parms['SQOLIM'])
		ts['IOQCP'] = initm(siminfo, uci, ui_flags['VIQCFG'], 'PQUAL' + str(index) + '_MONTHLY/IOQC', ui_parms['IOQC'])
		ts['AOQCP'] = initm(siminfo, uci, ui_flags['VAQCFG'], 'PQUAL' + str(index) + '_MONTHLY/AOQC', ui_parms['AOQC'])
		if 'FLAGS' in uci:
			u = uci['FLAGS']
			# get atmos dep timeseries
			pqadfgf = u['PQADFG' + str((index * 2) - 1)]
			if pqadfgf > 0:
				ts['PQADFX'] = initm(siminfo, uci, pqadfgf, 'PQUAL' + str(index) + '_MONTHLY/PQADFX', 0.0)
			elif pqadfgf == -1:
				ts['PQADFX'] = ts['PQADFX' + str(index) + ' 1']
			pqadfgc = u['PQADFG' + str(index * 2)]
			if pqadfgc > 0:
				ts['PQADCN'] = initm(siminfo, uci, pqadfgc, 'PQUAL' + str(index) + '_MONTHLY/PQADCN', 0.0)
			elif pqadfgc == -1:
				ts['PQADCN'] = ts['PQADCN' + str(index) + ' 1']

		if 'PQADFX' not in ts:
			ts['PQADFX'] = zeros(simlen)
		if 'PQADCN' not in ts:
			ts['PQADCN'] = zeros(simlen)

		POTFW  = ts['POTFW']
		POTFS  = ts['POTFS']
		ACQOP  = ts['ACQOP']
		SQOLIM = ts['SQOLIM']
		IOQCP  = ts['IOQCP']
		AOQCP  = ts['AOQCP']
		PQADFX = ts['PQADFX']
		PQADCN = ts['PQADCN']

		soqo = 0.0
		remqop = 0.0
		soqs = 0.0
		for loop in range(simlen):
			dayfg  = DAYFG[loop]
			suro   = SURO[loop]
			ifwo   = IFWO[loop]
			agwo   = AGWO[loop]
			pero   = PERO[loop]
			wssd   = WSSD[loop]
			scrsd  = SCRSD[loop]
			sliqsp = SLIQSP[loop]   # undefined name: SLIQSP ???
			sliqo  = SLIQO[loop]    #  undefined name: SLIQO ???
			potfw  = POTFW[loop]
			potfs  = POTFS[loop]
			acqop  = ACQOP[loop]
			sqolim = SQOLIM[loop]
			ioqc   = IOQCP[loop]
			aoqc   = AOQCP[loop]
			iliqc  = ILIQC[loop]
			aliqc  = ALIQC[loop]
			
			# simulate by association with sediment
			suroqs = 0.0
			soqsp  = -1.0e30
			scrqs  = 0.0
			washqs = 0.0
			if QSDFG:
				# qualsd()
				''' Simulate removal of a quality constituent from the land surface by association with sediment'''
				if dayfg:     # it is the first interval of the day
					potfw = POTFW[loop]
					potfs = POTFS[loop]

				# associate with washoff of detached sediment - units are qty/acre-ivl
				if wssd == 0.0:
					washqs = 0.0  #  no washoff of sediment
				elif sliqsp >= 0.0:
					washqs = wssd * (sliqsp * slifac + potfw * (1.0 - slifac))  # lateral inflow has an effect on washoff potency factor
				else:
					washqs = wssd * potfw   # no effect of lateral inflow

				# associate with scouring of soil matrix - units are qty/acre-ivl
				scrqs = 0.0  if scrsd == 0.0 else scrsd * potfs
				soqs = washqs + scrqs  # sum removals

				# calculate effective outflow potency factor
				lsosed = wssd + scrsd
				soqsp = soqs / lsosed  if lsosed > 0.0 else -1.0e30

				suroqs = soqs
				# end of qualsd()

			# simulate by association with overland flow
			suroqo = 0.0
			adtot  = 0.0
			adfxfx = 0.0
			adcnfx = 0.0
			if QSOFG:   #constituent n is simulated by association with overland flow;
				# qualof()
				''' Simulate accumulation of a quality constituent on the land surface and its removal by a constant unit rate and by overland flow'''	
				if dayfg:
					remqop = acqop / sqolim

					if QSOFG == 1:
						# update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
						sqo = acqop + sqo * (1.0 - remqop)

				# handle atmospheric deposition
				adfxfx = PQADFX[loop]  # dry deposition
				adcnfx = PQADCN[loop] * PREC[loop] * 3630.0 # wet deposition

				adtot = adfxfx + adcnfx  # total atmospheric deposition
				intot = adtot + sliqo             	# add lateral inflow

				if QSOFG == 2:  # update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
					dummy = remqop + intot / (acqop / remqop)
					if dummy > 1.0:
						dummy = 1.0
					sqo = acqop * (delt60 / 24.0) + sqo * (1.0 - dummy)**(delt60 / 24.0)

				sqo = sqo + intot    	# update storage

				# simulate washoff by overland flow - units are qty/acre-ivl
				soqo = 0.0
				if suro > 0.0 and sqo > 0.0:  # there is overland flow # there is some quality constituent (no. qofp) in storage, washoff can occur
					dummy = suro * wsfac
					if dummy < 1.0e-5:
						soqo = 0.0  # washoff too small for stable calculation - set to zero
					else:           # calculate washoff
						dummy = 1.0 - exp(-dummy)
						soqo  = sqo * dummy

						# update storage of constituent - units are in qty/acre
						sqo = sqo - soqo

				# compute and output concentration - units are qty/acre-inch
				soqoc = soqo / suro  if suro > 0.0 else -1.0e30  # soqoc not used ???
				# end qualof()
				
				suroqo = soqo

			# sum outflows of constituent n from the land surface
			soqual = suroqs + suroqo
			poqual = soqual
			ioqual = 0.0
			aoqual = 0.0

			# compute the concentration - units are qty/acre-inch
			soqc = soqual / suro  if suro > 0.0 else -1.0e30

			# simulate quality constituent in interflow
			if QIFWFG != 0:
				# qualif()
				'''Simulate quality constituents by fixed concentration in interflow'''
				ioqc = IOQCP[loop] * 3630.0
				if ui_flags['VIQCFG'] == 3 or ui_flags['VIQCFG'] == 4:
					ioqc = ioqc * 6.238e-5

				# simulate constituents carried by interflow - units are qty/acre-ivl
				if ifwo > 0.0:      # there is interflow
					ioqce  = iliqc * ilifac + ioqc * (1.0 - ilifac)  if iliqc >= 0.0 else ioqc   # lifac not defined, iliqc not defined
					ioqual = ioqce * ifwo
				else:   # no interflow
					ioqce  = -1.0e30
					ioqual = 0.0				
				# qualif()

				poqual = poqual + ioqual   # cumulate outflow
				
			# simulate quality constituent in active groundwater outflow
			if QAGWFG:   #	constituent n is present in groundwater
				# qualgw()
				''' Simulate quality constituents by fixed concentration in groundwater flow'''
				aoqc = AOQCP[loop] * 3630.0
				if ui_flags['VAQCFG'] == 3 or ui_flags['VAQCFG'] == 4:
					aoqc = aoqc * 6.238e-5
					
				# simulate constituents carried by groundwater flow - units are qty/acre-ivl
				if agwo > 0.0:      # there is baseflow
					aoqce  = aliqc * alifac + aoqc * (1.0- alifac)  if aliqc >= 0.0 else aoqc   # kufac bit definedn aliqc bit defubed
					aoqual = aoqce * agwo
				else:             # no baseflow
					aoqce  = -1.0e30
					aoqual = 0.0				
				# end of qualgw()

				poqual = poqual + aoqual   # cumulate outflow
			# compute the concentration of constituent n in the total outflow
			poqc = poqual / pero  if pero > 0.0 else -1.0e30

			# end of constituent computations, save
			SOQUAL[loop] = soqual
			IOQUAL[loop] = ioqual
			AOQUAL[loop] = aoqual
			POQUAL[loop] = poqual

			SQO[loop]    = sqo
			SOQSP[loop]  = soqsp
			if soqoc > -1:
				SOQOC[loop] = soqoc / 3630.0  # 3630 converts from ft3 to ac-in
			else:
				SOQOC[loop] = soqoc
			SOQC[loop]   = soqc / 3630.0
			IOQC[loop]   = (ioqual / ifwo / 3630.0) if ifwo > 0.0 else -1.0e30
			AOQC[loop]   = (aoqual / agwo / 3630.0) if agwo > 0.0 else -1.0e30
			POQC[loop]   = poqc / 3630.0 if pero > 0.0 else -1.0e30

			WASHQS[loop] = washqs
			SCRQS[loop]  = scrqs
			SOQS[loop]   = soqs
			SOQO [loop]  = soqo

			PQADWT[loop] = adcnfx
			PQADDR[loop] = adfxfx
			PQADEP[loop] = adtot
	
	return errorsV, ERRMSG

