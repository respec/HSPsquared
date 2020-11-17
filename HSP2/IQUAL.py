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

	nquals = int((len(uci) - 2) / 2)
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

		qualid = ui_flags['QUALID']
		qtyid  = ui_flags['QTYID']
		QSDFG  = ui_flags['QSDFG']
		QSOFG  = ui_flags['QSOFG']
		VQOFG  = ui_flags['VQOFG']
		
		sqo    = ui_parms['SQO']
		wsqop  = ui_parms['WSQOP']
		wsfac = 2.30 / wsqop

		# preallocate output arrays (always needed)
		SOQUAL = ts[constituent + '/SOQUAL'] = zeros(simlen)
		SOQC   = ts[constituent + '/SOQC']   = zeros(simlen)
		SOQO   = ts[constituent + '/SOQO'] = zeros(simlen)
		
		# preallocate output arrays (QUALOF)
		SQO    = ts[constituent + '/SQO']    = zeros(simlen)
		SOQOC  = ts[constituent + '/SOQOC']  = zeros(simlen)

		# preallocate output arrays (QUALSD)
		SOQS   = ts[constituent + '/SOQS']   = zeros(simlen)
		SOQSP  = ts[constituent + '/SOQSP'] = zeros(simlen)

		# preallocate output arrays for atmospheric deposition
		IQADDR = ts[constituent + '/IQADDR']   = zeros(simlen)
		IQADWT = ts[constituent + '/IQADWT'] = zeros(simlen)
		IQADEP = ts[constituent + '/IQADEP'] = zeros(simlen)

		SLIQO  = ts[constituent + '/SLIQO'] = zeros(simlen)   # lateral inflow
		INFLOW = ts[constituent + '/INFLOW'] = zeros(simlen)  # total inflow

		
		# handle monthly tables
		u = uci['FLAGS']
		ts['POTFW'] = initm(siminfo, uci, ui_flags['VPFWFG'], 'MONTHLY_POTFW', ui_parms['POTFW'])
		ts['ACQOP'] = initm(siminfo, uci, ui_flags['VQOFG'], 'MONTHLY_ACQOP', ui_parms['ACQOP'])
		ts['SQOLIM'] = initm(siminfo, uci, ui_flags['VQOFG'], 'MONTHLY_SQOLIM', ui_parms['SQOLIM'])
		ts['IQADFX'] = initm(siminfo, uci, u['IQADFG'+ str((index*2) -1)], 'MONTHLY_IQADFX', 0.0)
		ts['IQADCN'] = initm(siminfo, uci, u['IQADFG'+ str(index*2)], 'MONTHLY_IQADCN', 0.0)
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
			if QSOFG != 0:  #  constituent n is simulated by association with overland flow; the value of qofp refers to the set of overland flow associated parameters to use
				if QSOFG >= 1:   # standard qualof simulation
					# washof ()
					''' Simulate accumulation of a quality constituent on the land surface and its removal using a constant unit rate and by direct washoff by overland flow'''
					if dayfg == 1:
						remqop = acqop / SQOLIM[loop]
						if QSOFG == 1 :   #update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
							sqo = acqop + sqo * (1.0 - remqop)

					# handle atmospheric deposition
					adfxfx = IQADFX[loop]  		            # dry deposition
					adcnfx = IQADCN[loop] * PREC[loop]  	# wet deposition

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
			IQADEP[loop] = adtot
			
	return errorsV, ERRMSG


	
'''
C
          IF (SLIQOX(QOFP) .GE. 1) THEN
C           lateral inflow of qualof
            SLIQO(QOFP)= SLIQO(QOFP)+ PAD(SLIQOX(QOFP)+IVL1)
C          ELSE
C           no lateral inflow
C            SLIQO(QOFP)= 0.0
          END IF
C

'''
'''
      DO 90 N= 1,NQUAL
          J= 2*(N- 1)+ 1
          IF ( (IQADFG(J) .NE. 0) .OR. (IQADFG(J+1) .NE. 0) ) THEN
C           error - non-qualof cannot have atmospheric deposition
            WRITE (CSTR,2010) (QUALID(I,N),I=1,3)
            I= 12
            CALL OMSTC (I,CSTR1)
            SGRP= 3
            CALL OMSG (MESSU,MSGFL,SCLU,SGRP,
     M                 ECOUNT)
            IQADFG(J)= 0
            IQADFG(J+1)= 0
          END IF
        END IF
C

        IF (QSOFG(N).GE.1) THEN
          SQO(NQOF)   = RVAL(1)
          ACQOP(NQOF) = RVAL(3)
          SQOLIM      = RVAL(4)
C         compute removal rate
          REMQOP(NQOF)= ACQOP(NQOF)/SQOLIM
          WSFAC(NQOF) = 2.30/RVAL(5)
        ELSE IF (QSOFG(N) .EQ. -1) THEN
C         special case for ches bay - allow constant conc qualof
C         this converts units from mg/l to lb/ac/in
          ACQOP(NQOF) = RVAL(3)*0.2266
          SQO(NQOF)   = 0.0
        END IF
C
************************* END PIQUAL
'''