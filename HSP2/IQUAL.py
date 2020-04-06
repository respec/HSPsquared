''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HIMPQUA.FOR module into Python''' 

from math import exp
from numpy import zeros, where, full
from numba import jit
from HSP2  import initmc


''' DESIGN NOTES
Each constituent will be in its own subdirectory in the HDF5 file.
IQUAL high level will contain list of constituents.

NEED to fix units conversions

NEED to add initmc() to HSP2

UNDEFINED: sliqsp
'''

ERRMSG = []

def iqual(store, general, ui, ts):
	''' Simulate washoff of quality constituents (other than solids, Heat, dox, and co2)
	using simple relationships with solids And/or water yield'''
	
	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = general['sim_delt'] / 60     # delt60 - simulation time interval in hours
	simlen = general['sim_len']
	tindex = general['tindex']

	SURO  = ts['SURO']
	SOSLD = ts['SOSLD']
	PREC  = ts['PREC']
	
	for name in ['SLIQSX', 'SLIQO', 'SLIQSP']:
		if name not in ts:
			ts[name] = full(simlen, -1.0E30)
	SLIQSX = ts['SLIQSX']
	SLIQO  = ts['SLIQO']
	SLIQSP = ts['SLIQSP']

	constituents = ui['CONSTITUENTS']   # (short) names of constituents
	slifac = ui['SLIFAC']
	
	DAYFG = where(tindex.hour==1, True, False)   # ??? need to check if minute == 0 or 1???
	DAYFG[0] = 1

	for constituent in constituents:     # simulate constituent
		# update UI values for this constituent here!
		qualid = ui[constituent + '/QUALID']
		qtyid  = ui[constituent + '/QTYID']
		QSDFG  = ui[constituent + '/QSDFG']
		QSOFG  = ui[constituent + '/QSOFG']
		VQOFG  = ui[constituent + '/VQOFG']
		
		sqo    = ui[constituent + '/SQO']
		wsqop  = ui[constituent + '/WSQOP']
		wsfac = 2.30 / wsqop

		# preallocate output arrays (always needed)
		SOQUAL = ts[constituent + '/SPQUAL'] = zeros(simlen)
		SOQC   = ts[constituent + '/SOQC']   = zeros(simlen)
		
		# preallocate output arrays (QUALOF)
		SQO    = ts[constituent + '/SQO']    = zeros(simlen)
		SOQOC  = ts[constituent + '/SOQOC']  = zeros(simlen)

		# preallocate output arrays (QUALSD)
		SOQS   = ts[constituent + '/SPQS']   = zeros(simlen)   
		
		# handle monthly tables 
		POTFW  = initmc(general, ui, ts, constituent, 'VPFWFG', 'POTFWM', 'POTFW')
		ACQOP  = initmc(general, ui, ts, constituent, 'VQOFG ', 'ACQOPM', 'ACQOP')
		SQOLIM = initmc(general, ui, ts, constituent, 'VQOFG ', 'SQOLIM', 'SQOLIM')
		ADFXFX = initmc(general, ui, ts, constituent, 'ADFXFG', 'ADFXMN', 'ADFXFX')
		ADCNFX = initmc(general, ui, ts, constituent, 'ADCNFG', 'ADCNMN', 'ADCNFX')
		REMQOP = initmc(general, ui, ts, constituent, '', 'REMQOPM', 'REMQOP')   #???

		for loop in range(simlen):
			suro   = SURO[loop]
			sosld  = SOSLD[loop]
			dayfg  = DAYFG[loop]
			sliqsx = SLIQSX[loop]
			sliqo  = SLIQO[loop]
			sliqsp = SLIQSP[loop]
				
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
			if QSOFG != 0:  #  constituent n is simulated by association with overland flow; the value of qofp refers to the set of overland flow associated parameters to use
				if QSOFG >= 1:   # standard qualof simulation
					# washof ()
					''' Simulate accumulation of a quality constituent on the land surface and its removal using a constant unit rate and by direct washoff by overland flow'''
					if dayfg == 1:
						if VQOFG == 1:
							acqop  = ACQOP[loop]
							remqop = REMQOP[loop] / SQOLIM[loop]
						if QSOFG == 1 :   #update storage due to accumulation and removal which occurs independent of runoff - units are qty/acre
							sqo = acqop + sqo * (1.0 - remqop)

					# handle atmospheric deposition
					adfxfx = ADFXFX[loop]  		            # dry deposition
					adcnfx = ADCNFX[loop] * PREC[loop]  	# wet deposition

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
					acqop = ACQOP[loop] * 0.2266
					soqo  = suro * acqop
					soqoc = acqop
					sqo   = 0.0
				suroqo = soqo

			# sum outflows of constituent n from the land surface
			SOQUAL[loop] = soqual = suroqs + suroqo
			SOQC[loop]   = soqual / suro  if SURO > 0.0 else -1.0e30
			SQO[loop]    = sqo
			SOQS[loop]   = soqs   
			SOQOC[loop]  = soqoc
			
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