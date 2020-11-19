''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERQUA.FOR module into Python''' 

from math import exp
from numpy import zeros, where
from numba import jit
from HSP2  import initmc, initmcm

''' DESIGN NOTES
Each constituent will be in its own subdirectory in the HDF5 file.
PQUAL high level will contain list of constituents.

NEED to check all units conversions

NEED to add initmc(), initmcm() to HSP2
'''

ERRMSG = []

# english system
FACTA  = 1.0
CFACTA = 2.7548E-04
PFACTA = 1.0


def pqual(store, general, ui, ts):
	''' Simulate quality constituents (other than sediment, heat, dox, and co2)
	using simple relationships with sediment and water yield'''
	
	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = general['sim_delt'] / 60    # delt60 - simulation time interval in hours
	simlen = general['sim_len']
	tindex = general['tindex']

	constituents = ui['CONSTITUENTS']   # (short) names of constituents
	slifac = ui['SLIFAC']

	for name in ['SURO', 'IFWO', 'AGWO', 'PERO', 'WSSD', 'SCRSD']:
		if name not in ts:
			ts[name] = zeros(simlen)
	SURO  = ts['SURO']
	IFWO  = ts['IFWO']
	AGWO  = ts['AGWO']
	PERO  = ts['PERO']
	WSSD  = ts['WSSD']
	SCRSD = ts['SCRSD']
	
	DAYFG = where(tindex.hour==1, True, False)   # ??? need to check if minute == 0 or 1???
	DAYFG[0] = 1
	  
	for constituent in constituents:     # simulate constituent
		# update UI values for this constituent here!
		qualid = ui[constituent + '/QUALID']
		qtyid  = ui[constituent + '/QTYID']
		QSDFG  = ui[constituent + '/QSDFG']
		QSOFG  = ui[constituent + '/QSOFG']
		QIFWFG = ui[constituent + '/QIFWFG']
		QAGWVG = ui[constituent + '/QAGWVG']  # never  used???
		SQO    = ui[constituent + '/SQO']
		WSQOP  = ui[constituent + '/WSQOP']

		POTFW  = initmcm(general, ui, ts, constituent, 'QSDFG', 'VPFWFG', 'POTFWM', 'POTFW')
		POTFS  = initmcm(general, ui, ts, constituent, 'QSDFG', 'VPFSFG', 'POTFSM', 'POTFS')
		ACQOP  = initmcm(general, ui, ts, constituent, 'QSOFG', 'VQOFG', 'ACQOPM', 'ACQOP')
		SQOLIM = initmcm(general, ui, ts, constituent, 'QSOFG',  'VQOFG', 'SQOLIM', 'SQOLIM')
		IOQC   = initmcm(general, ui, ts, constituent, 'QIFWFG', 'VIQCFG', 'IOQCM', 'IOQC')
		AOQC   = initmcm(general, ui, ts, constituent, 'QAGWFG', 'VAQCFG', 'AOQCM', 'AOQC')		
		
		# preallocate storage for output
		SQO    = ts[constituent + 'SQO']    = zeros(simlen)
		WASHQS = ts[constituent + 'WASHQS'] = zeros(simlen)
		SCRQS  = ts[constituent + 'SCRQS']  = zeros(simlen)
		SOQS   = ts[constituent + 'SOQS']   = zeros(simlen)
		SOQO   = ts[constituent + 'SOQO']   = zeros(simlen)
		SOQUAL = ts[constituent + 'SOQUAL'] = zeros(simlen)
		IOQUAL = ts[constituent + 'IOQUAL'] = zeros(simlen)
		AOQUAL = ts[constituent + 'AOQUAL'] = zeros(simlen)
		POQUAL = ts[constituent + 'POQUAL'] = zeros(simlen)
		SOWOC  = ts[constituent + 'SOWOC']  = zeros(simlen)
		POQC   = ts[constituent + 'POQC']   = zeros(simlen)	
		
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
			sqolim = SQOLIM[loop]   # sqolim not used ???
			ioqc   = IOQC[loop]
			aoqc   = AOQC[loop]	

			wsfac = 2.30 / WSQOP			
			
			# simulate by association with sediment
			suroqs = 0.0
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
				soqsp = soqs / lsosed  if lsosed > 0.0 else -1.0e30			# soqsp not used ???	
				# end of qualsd()

			# simulate by association with overland flow
			suroqo = 0.0
			if QSOFG:   #constituent n is simulated by association with overland flow;
				# qualof()
				''' Simulate accumulation of a quality constituent on the land surface and its removal by a constant unit rate and by overland flow'''	
				if dayfg:
					acqop  = ACQOP[loop]
					remqop = REMQOP[loop]   # REMQOP not defined
					
					if QSOFG:    # update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
						sqo = acqop + sqo * (1.0 - remqop)  # sqo undefined (first time error???)

				# handle atmospheric deposition
				atdpcn = ATDPCN[loop]  		            # dry deposition   # ATDPCN not defined, atdpcn not used
				adcnfx = ADCNFX[loop] * PREC[loop]  	# wet deposition	

				adtot = adfxfx + adcnfx  # adfxfx not defined - should it be atdpcn????
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

			# compute the concentration - units are qty/acre-inch
			soqc = soqual / suro  if suro > 0.0 else -1.0e30      # soqc not used

			# simulate quality constituent in interflow
			if QIFWFG != 0:
				# qualif()
				'''Simulate quality constituents by fixed concentration in interflow'''
				if dayfg:      # it is the first interval of the day
					ioqc = IOQC[loop]

				# simulate constituents carried by interflow - units are qty/acre-ivl
				if ifwo > 0.0:      # there is interflow
					ioqce  = iliqc * lifac + ioqc * (1.0 - lifac)  if iliqc >= 0.0 else ioqc   # lifac not defined, iliqc not defined
					ioqual = ioqce * ifwo
				else:   # no interflow
					ioqce  = -1.0e30
					ioqual = 0.0				
				# qualif()

				poqual = poqual + ioqual   # cumulate outflow
				
			# simulate quality constituent in active groundwater outflow
			if QAGWFG:   #	constituent n is present in groundwater                    # QAGWFG not defined
				# qualgw()
				''' Simulate quality constituents by fixed concentration in groundwater flow'''
				if dayfg:        # it is the first interval of the day
					aoqc = AOQC[loop]
					
				# simulate constituents carried by groundwater flow - units are qty/acre-ivl
				if agwo > 0.0:      # there is baseflow
					aoqce  = aliqc * lifac + aoqc * (1.0- lifac)  if aliqc >= 0.0 else aoqc   # kufac bit definedn aliqc bit defubed
					aoqual = aoqce * agwo
				else:             # no baseflow
					aoqce  = -1.0e30
					aoqual = 0.0				
				# end of qualgw()

				poqual = poqual + aoqual   # cumulate outflow
			# compute the concentration of constituent n in the total outflow
			poqc = poqual / pero  if pero > 0.0 else -1.0e30

		# end of constituent computations, save
		SQO[loop]    = sqo
		WASHQS[loop] = washqs
		SCRQS[loop]  = scrqs
		SOQS[loop]   = soqs
		SOQO [loop]  = soqo
		SOQUAL[loop] = soqual
		IOQUAL[loop] = ioqual
		AOQUAL[loop] = aoqual
		POQUAL[loop] = poqual
		SOWOC[loop]  = sowoc   # sowoc not defined
		POQC[loop]   = poqc
	
	return errorsV, ERRMSG

	
'''
C       english units - conversion from mg/l to lb/ft3
        CVT= 6.238E-5

IF (UUNITS .EQ. 1) THEN
C           convert from qty/ft3 to qty/ac.in
            DO 30 I= 1, 12
              PQACNM(I,J)= PQACNM(I,J)*3630.0
'''