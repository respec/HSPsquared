''' Copyright (c) 2020 by RESPEC, INC.
Authors: Robert Heaphy, Ph.D. and Paul Duda
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

'''  
		
NOTE: NCONS derived from DataFrame length ????
NOTE: COADFG flags NOT saved.  Use logic: if ts present, use it; otherwise look for monthly table.

'''

from numpy import zeros
from HSP2.ADCALC import advect
from numba import njit
from HSP2.utilities  import make_numba_dict, initm

ERRMSG = []

def cons(io_manager, siminfo, uci, ts):
	''' Simulate behavior of conservative constituents; calculate concentration 
	of conservative constituents after advection'''

	errorsV = zeros(len(ERRMSG), dtype=int)

	simlen = siminfo['steps']
	uunits = siminfo['units']

	AFACT = 43560.0
	if uunits == 2:
		# si units conversion constants, 1 hectare is 10000 sq m
		AFACT = 1000000.0

	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData
	svol = vol * AFACT

	ts['VOL'] = VOL
	ts['SROVOL'] = SROVOL
	ts['EROVOL'] = EROVOL
	for i in range(nexits):
		ts['SOVOL' + str(i + 1)] = SOVOL[:, i]
		ts['EOVOL' + str(i + 1)] = EOVOL[:, i]

	ui = make_numba_dict(uci)
	nexits = int(ui['NEXITS'])

	ui['simlen'] = siminfo['steps']
	ui['uunits'] = siminfo['units']
	ui['svol']   = svol
	ui['delt60'] = siminfo['delt'] / 60  # delt60 - simulation time interval in hours

	# vol    = ui['VOL']
	# conactive = ui['CONACTIVE']   # dict

	ncons = 1
	if 'PARAMETERS' in uci:
		if 'NCONS' in uci['PARAMETERS']:
			ncons = uci['PARAMETERS']['NCONS']

	for index in range(ncons):
		icon = str(index + 1)
		parms = uci['CONS' + icon]
		conid = parms['CONID']   # string name of the conservative constituent
		con   = parms['CON']     # initial concentration of the conservative
		concid= parms['CONCID']  # string which specifies the concentration units for the conservative constituent.
		ui['conv'] = parms['CONV']    # conversion factor from QTYID/VOL to the desired concentration units
		qtyid = parms['QTYID']   # string which specifies the units for inflow or outflow of constituent; e.g. kg
		name  = 'CONS' + icon    # arbitrary identification, default CONxx
		ui['icon'] = index + 1
		ui['con']  = con

		# # dry deposition; flag: COADFG; monthly COAFXM; value: COADFX
		# COADFG1 = ui['COADFG1']    # table-type cons-ad-flags
		# COADFX = getit()           # flag: COADFG; monthly COAFXM; value: COADFX
		# # wet deposition; flag: COADFG; monthly COACNM; value COADCN
		# COADFG2 = ui['COADFG2']    # table-type cons-ad-flags
		# COADCN = getit()            # flag: COADFG; monthly COACNM; value COADCN

		if 'FLAGS' in uci:
			u = uci['FLAGS']
			# get atmos dep timeseries
			coadfg1 = u['COADFG' + str((index * 2) - 1)]
			if coadfg1 > 0:
				ts['COADFX'] = initm(siminfo, uci, coadfg1, 'CONS' + str(index) + '_MONTHLY/COADFX', 0.0)
			elif coadfg1 == -1:
				ts['COADFX'] = ts['COADFX'+ str(index)]

			coadfg2 = u['COADFG' + str(index * 2)]
			if coadfg2 > 0:
				ts['COADCN'] = initm(siminfo, uci, coadfg2, 'CONS' + str(index) + '_MONTHLY/COADCN', 0.0)
			elif coadfg2 == -1:
				ts['COADCN'] = ts['COADCN' + str(index)]

		if 'COADFX' not in ts:
			ts['COADFX'] = zeros(simlen)
		if 'COADCN' not in ts:
			ts['COADCN'] = zeros(simlen)

		############################################################################
		errors = _cons_(ui, ts)  # run CONS simulation code
		############################################################################

		if nexits > 1:
			u = uci['SAVE']
			key1 = name + '_OCON'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['OCON']
			del u['OCON']

	return errorsV, ERRMSG


@njit(cache=True)
def _cons_(ui, ts):
	''' Simulate behavior of conservative constituents; calculate concentration
	of conservative constituents after advection'''

	simlen = int(ui['simlen'])
	nexits = int(ui['NEXITS'])
	uunits = int(ui['uunits'])
	conv   = ui['conv']
	svol   = ui['svol']
	delt60 = ui['delt60']
	name = 'CONS' + str(int(ui['icon']))
	con = ui['con']

	AFACT = 43560.0
	if uunits == 2:
		# si units conversion constants, 1 hectare is 10000 sq m
		AFACT = 1000000.0

	PREC = ts['PREC']
	SAREA = ts['SAREA']

	VOL = ts['VOL']
	SROVOL = ts['SROVOL']
	EROVOL = ts['EROVOL']
	SOVOL = zeros((simlen, nexits))
	EOVOL = zeros((simlen, nexits))
	for i in range(nexits):
		SOVOL[:, i] = ts['SOVOL' + str(i + 1)]
		EOVOL[:, i] = ts['EOVOL' + str(i + 1)]

	COADFX = ts['COADFX'] * delt60 / (24.0 * AFACT)
	COADCN = ts['COADCN']

	# preallocate output arrays (always needed)
	ROCON = ts[name + '_ROCON'] = zeros(simlen)
	CON = ts[name + '_CON'] = zeros(simlen)
	RCON = ts[name + '_RCON'] = zeros(simlen)

	# preallocate output arrays for atmospheric deposition
	COADDR = ts[name + '_COADDR'] = zeros(simlen)
	COADWT = ts[name + '_COADWT'] = zeros(simlen)
	COADEP = ts[name + '_COADEP'] = zeros(simlen)

	# get incoming flow of constituent or zeros;
	if (name + '_ICON') not in ts:
		ts[name + '_ICON'] = zeros(simlen)
	ICON = ts[name + '_ICON'] * conv * AFACT * VOL

	OCON = zeros((simlen, nexits))

	for loop in range(simlen):
		sarea  = SAREA[loop]
		prec   = PREC[loop]
		vol    = VOL[loop] * AFACT

		coadfx = COADFX[loop]
		coadcn = COADCN[loop]
		if vol > 0.0:
			icon = ICON[loop] / vol
		else:
			icon = ICON[loop]

		coaddr = sarea * conv   * coadfx    # dry deposition; 
		coadwt = prec  * sarea  * coadcn    # wet deposition;

		adtot = coaddr + coadwt  # total atmospheric deposition

		incon  = icon  + coaddr + coadwt

		srovol = SROVOL[loop]
		erovol = EROVOL[loop]
		sovol = SOVOL[loop, :]
		eovol = EOVOL[loop, :]
		con, rocon, ocon = advect(incon, con, nexits, svol, vol, srovol, erovol, sovol, eovol)

		svol = vol  # svol is volume at start of time step, update for next time thru

		CON[loop]    = con
		ROCON[loop]  = rocon / conv  # outflow
		OCON[loop,:] = ocon  / conv
		RCON[loop]   = con * vol / conv # total storage of constituent

		COADWT[loop] = coadwt
		COADDR[loop] = coaddr
		COADEP[loop] = adtot

	if nexits > 1:
		for i in range(nexits):
			ts[name + '_OCON' + str(i + 1)] = OCON[:, i]

	return	

def expand_CONS_masslinks(flags, uci, dat, recs):
	if flags['CONS']:
		ncons = 1
		if 'PARAMETERS' in uci:
			if 'NCONS' in uci['PARAMETERS']:
				ncons = uci['PARAMETERS']['NCONS']
		for i in range(1, ncons + 1):
			# ICONS                        loop for each cons
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'CONS'
			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'ROCON'
			else:
				rec['SMEMN'] = 'OCON'
			rec['SMEMSB1'] = dat.SMEMSB1  # first sub is exit number
			rec['SMEMSB2'] = dat.SMEMSB2
			rec['TMEMN'] = 'ICON'
			rec['TMEMSB1'] = dat.TMEMSB1
			rec['TMEMSB2'] = dat.TMEMSB2
			rec['SVOL'] = dat.SVOL
			recs.append(rec)
	return recs