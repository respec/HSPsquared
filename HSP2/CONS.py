''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''


"""
C       convert variables to external units
        FACTA= 1.0/CCONV(I)
C       rchres-wide variables
        PIFLX= CNIF(I,LEV)*FACTA
C       storages
        PRCON = CNST(I,1)*FACTA
        PRCONS= CNST(I,LEV)*FACTA
C       computed fluxes
        PCFLX1= CNCF1(I,LEV)*FACTA
"""

'''  TO DO

CONS is not specific to each RCHRES segment. So store it parallel to PERLND, IMPLND, etc in the HDF5 file


/CONS/CONDATA
/CONS/MONTHLY/COADFXM  # dry deposition
/CONS/MONTHLY/COADCNM  # wet deposition


ASSUME CONDATA as DataFrame:

		NAME	CONID  CON  CONCID  CONV  QTYID
		CONS01
		CONS02
		...
		CONSNN


ASSUME each RCHRES segment has CON active table:
/RCHRES/segment/CONACTIVE

		NAME	ACTIVE
		CONS01
		CONS02
		...
		CONSNN		

		
NOTE: NCONS derived from DataFrame length ????
NOTE: COADFG flags NOT saved.  Use logic: if ts present, use it; otherwise look for monthly table.

'''




from numpy import zeros
from adcalc import advect as advect
from numba import jit

def cons(store, general, ui, ts):
	''' Simulate behavior of conservative constituents; calculate concentration 
	of conservative constituents after advection'''
	
	simlen = ui['SIM_LEN']
	nexits = general['NEXITS']
	vol    = ui['VOL']
	conactive = ui['CONACTIVE']   # dict 

	SAREA  = ts['SAREA']
	PREC   = ts['PREC']
	VOL    = ts['VOL']
	O      = ts['O']
	SROVOL = ts['SROVOL']
	EROVOL = ts['ERVOL']
	SOVOL  = ts['SOVOL']   # dimension (simlen, nexits)
	EOVOL  = ts['EVOL']    # dimension (simlen, nexits)
	
	ocon = zeros((simlen, nexits))
	
	condata = store('/CONS/CONDATA')
	for r  in itertuples(condata):
		name   = r['NAME']   # arbitrary identification, default CONxx
		conid  = r['CONID']  # string name of the conservative constituent
		con    = r['CON']    # initial concentration of the conservative
		concid = r['CONCID'] # string which specifies the concentration units for the conservative constituent.
		conv   = r['CONV']   # conversion factor from QTYID/VOL to the desired concentration units 
		qtyid  = r['QTYID']	 # string which specifies the units for inflow or outflow of constituent; e.g. kg
		
		if not conactive[name]:
			continue

		# preallocate output arrays for performance - like in MATLAB etc.
		CON   = ts[(name, 'CON')]   = zeros(simlen)
		RCON  = ts[(name, 'RCON')]  = zeros(simlen)
		ROCON = ts[(name, 'ROCON')] = zeros(simlen)
		OCON  = ts[(name, 'OCON')]  = zeros((simlen, nexits))

		# get incoming flow of constituent or zeros; 
		ICON = ts[(n, 'ICON')] * conv

		# dry deposition; flag: COADFG; monthly COAFXM; value: COADFX
		COADFG1 = ui['COADFG1']    # table-type cons-ad-flags
		COADFX = getit()           # flag: COADFG; monthly COAFXM; value: COADFX
		COADFX *= delt60 / (24.0 * 43560.0) 
		
		# wet deposition; flag: COADFG; monthly COACNM; value COADCN
		COADFG2 = ui['COADFG2']    # table-type cons-ad-flags
		COADCN = getit()            # flag: COADFG; monthly COACNM; value COADCN

		loopsub(SAREA, PREC, VOL, COADFX, COADCN, ICON, simlen, conid, CON, ROCON, OCON, RCON, SROVOL, EROVOL, 
		SOVOL, EOVOL, conv)

	return  


@jit(nopython=True)
def loopsub(SAREA, PREC ,VOL, COADFX, COADCN, ICON, simlen, conid, CON, ROCON, OCON, RCON, SROVOL, EROVOL, SOVOL, EOVOL, conv):
	''' loop as function to allow Numba to cache compilation'''		
	
	for loop in range(simlen):
		sarea  = SAREA[loop]
		prec   = PREC[loop]
		vol    = VOL[loop]

		coadfx = COADFX[loop]
		coadcn = COADCN[loop]
		icon   = ICON[loop]
		con    = CON[loop]
		
		coaddr = sarea * conv   * coadfx    # dry deposition; 
		coadwt = prec  * sarea  * coadcn    # wet deposition; 
		incon  = icon  + coaddr + coadwt

		con, rocon, ocon = advect(loop, incon, con, VOL, SROVOL, EROVOL, SOVOL, EOVOL)
		
		CON[loop]    = con
		ROCON[loop]  = rocon / conv  # outflow
		OCON[loop,:] = ocon  / conv
		RCON[loop]   = con * vol / conv # total storage of constituent
	return	
	