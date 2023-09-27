''' Copyright (c) 2020 by RESPEC, INC.
Authors: Robert Heaphy, Ph.D. and Paul Duda
License: LGPL2
'''

'''
C       parameters for state variables with temperature units
        SFACTA= 1.8
        SFACTB= 32.0
C       parameter for heat balance components (kcal/m2 or btu/ft2)        
        XFACTA= 0.369
C
C       parameters for flux variables with entrained thermal
C       energy units
        IF (UUNITS .EQ. 1) THEN
C         english to english
          FFACTA= 112.37
          FFACTB= 0.0
        ELSE
C         metric to english
          FFACTA= 3.97E03
          FFACTB= 0.0
        END IF
'''

from HSP2.ADCALC import advect
from numpy import zeros, full, float64, int64
from numba import njit
from HSP2.utilities  import make_numba_dict, hourflag, hoursval, initm

		
# METRIC LAPSE DATA
mlapse = [0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0019, 0.0021, 0.0022, 0.0023, 0.0024,
  0.0026, 0.0026, 0.0027, 0.0027, 0.0028, 0.0028, 0.0027, 0.0026, 0.0024, 0.0023, 0.0022,
  0.0021, 0.0021, 0.0020]

ERRMSGS =('HTRCH: Water temperature is above 66 C (150 F) -- In most cases, this indicates an instability in advection','')     #ERRMSG0

def htrch(io_manager, siminfo, uci, ts):
	'''Simulate heat exchange and water temperature'''

	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData

	ts['VOL'] = VOL
	ts['SROVOL'] = SROVOL
	ts['EROVOL'] = EROVOL
	for i in range(nexits):
		ts['SOVOL' + str(i + 1)] = SOVOL[:, i]
		ts['EOVOL' + str(i + 1)] = EOVOL[:, i]

	simlen = siminfo['steps']

	ts['DAYFG'] = hourflag(siminfo, 0, dofirst=True).astype(float64)

	ui = make_numba_dict(uci)
	nexits = int(ui['NEXITS'])
	ui['simlen'] = siminfo['steps']
	ui['uunits'] = siminfo['units']
	ui['delt']   = siminfo['delt']
	ui['delt60'] = siminfo['delt'] / 60
	ui['errlen'] = len(ERRMSGS)
	ui['vol'] = vol

	bedflg = 0
	if 'BEDFLG' in ui:
		bedflg = ui['BEDFLG']
	tgrnd = 59.0
	if bedflg == 1 or bedflg == 2:
		tgrnd = ui['TGRND']
	tstop = 55
	if 'TSTOP' in ui:
		tstop = ui['TSTOP']
	# even though delh and deltt are not ts, numba requires them to be passed as such
	if 'DELH' in ui:
		delh = ui['DELH']
	else:
		delh = zeros(int(tstop))
	ts['DELH'] = delh
	if 'DELTT' in ui:
		deltt = ui['DELTT']
	else:
		deltt = zeros(int(tstop))
	ts['DELTT'] = deltt

	u = uci['PARAMETERS']
	# process optional monthly arrays to return interpolated data or constant array
	if 'TGFLG' in u:
		ts['TGRND'] = initm(siminfo, uci, u['TGFLG'], 'TGRND', tgrnd)
	else:
		ts['TGRND'] = full(simlen, tgrnd)

	ts['LAPSE'] = hoursval(siminfo, mlapse, lapselike=True)

	############################################################################
	errors = _htrch_(ui, ts)  # run HTRCH simulation code
	############################################################################

	if nexits > 1:
		u = uci['SAVE']
		key = 'OHEAT'
		for i in range(nexits):
			u[f'{key}{i + 1}'] = u[key]
		del u[key]

	return errors, ERRMSGS


@njit(cache=True)
def _htrch_(ui, ts):
	'''Simulate heat exchange and water temperature'''

	errorsV = zeros(int(ui['errlen'])).astype(int64)

	simlen = int(ui['simlen'])
	nexits = int(ui['NEXITS'])
	uunits = int(ui['uunits'])
	delt60 = ui['delt60']
	delt   = ui['delt']
	vol = ui['vol']
	adfg = int(ui['ADFG'])

	TW = ts['TW'] = zeros(simlen)
	AIRTMP = ts['AIRTMP'] = zeros(simlen)
	HTEXCH = ts['HTEXCH'] = zeros(simlen)
	ROHEAT = ts['ROHEAT'] = zeros(simlen)
	OHEAT = zeros((simlen, nexits))

	QTOTAL = ts['QTOTAL'] = zeros(simlen)
	QSOLAR = ts['QSOLAR'] = zeros(simlen)
	QLONGW = ts['QLONGW'] = zeros(simlen)
	QCON   = ts['QCON']   = zeros(simlen)
	QEVAP  = ts['QEVAP']  = zeros(simlen)
	QPREC  = ts['QPREC']  = zeros(simlen)
	QBED   = ts['QBED']   = zeros(simlen)

	DAYFG = ts['DAYFG'].astype(int64)

	HTWCNT = 0
	elev = ui['ELEV']
	eldat = ui['ELDAT']
	if uunits == 2:
		elev = elev * 3.281
		eldat = eldat * 3.281
	cfsaex = ui['CFSAEX']
	katrad = ui['KATRAD']
	kcond = ui['KCOND']
	kevap = ui['KEVAP']

	# for table HT-BED-FLAGS
	if 'BEDFLG' in ui:
		bedflg = ui['BEDFLG']
	else:
		bedflg = 0
	if 'TGFLG' in ui:
		tgflg = ui['TGFLG']
	else:
		tgflg = 2
	if 'TSTOP' in ui:
		tstop = ui['TSTOP']
	else:
		tstop = 55

	tgrnd = 59.0
	if bedflg == 1 or bedflg == 2:
		muddep = ui['MUDDEP']
		tgrnd = ui['TGRND']
		kmud = ui['KMUD'] * delt60  # convert rate coefficients from kcal/m2/C/hr to kcal/m2/C/ivl
		kgrnd = ui['KGRND'] * delt60
	else:
		muddep = 0.0

	if uunits == 1:			# input units are deg.F, but need deg.C
		muddep = muddep / 3.281
		tgrnd = (tgrnd - 32.0) * 0.555
		tstop = (tstop - 32.0) * 0.555

	# if bedflg == 3:
	delh = ts['DELH']
	deltt = ts['DELTT']

	if 'SHADFG' in ui:
		shadfg = ui['SHADFG']
	else:
		shadfg = 0

	# if SHADFG:
	# pshade()  # not implemented for now

	# calculate the pressure correction factor for conductive-convective heat transport
	cfpres = ((288.0 - 0.001981 * elev) / 288.0) ** 5.256

	tw = ui['TW']
	if uunits == 1:
		tw = (tw - 32.0) * 0.555

	airtmp = ui['AIRTMP']
	if uunits == 1:
		airtmp = (airtmp - 32.0) * 0.555

	AFACT = 43560.0
	if uunits == 2:
		# si units conversion constants, 1 hectare is 10000 sq m
		AFACT = 1000000.0
	svol = vol * AFACT
	rheat = tw * svol  # compute initial value of heat storage

	# if bedflg == 2:  # compute initial tmud and tmuddt for brock/caupp model
	tmud = tw  # assume tmud = tw and
	tmuddt = -0.1  # tmuddt is small + negative (at midnight)

	############### end of PTHRCH

	TGRND = ts['TGRND']

	if not 'IHEAT' in ts:
		ts['IHEAT'] = zeros(simlen)
	IHEAT = ts['IHEAT']  # kcal.vol/l.ivl; heat is relative to 0 degreees c
	if uunits == 1:
		IHEAT = IHEAT * 0.0089  # conv factor from rchrests.seq
	else:
		IHEAT = IHEAT * 0.4034 / 1000.0
		ts['IHEAT'] = IHEAT * 1000.0

	AVDEP = ts['AVDEP']
	AVDEPE = AVDEP if uunits == 1 else AVDEP * 3.28  # avdepe is the average depth in english units

	SOLRAD = ts['SOLRAD']
	if shadfg == 1:
		DSOLAR = ts['DSOLAR']
	PREC = ts['PREC']
	CLOUD = ts['CLOUD']
	GATMP = ts['GATMP']  # get gage air temperature
	DEWTMP = ts['DEWTMP']
	WIND = ts['WIND']  # get wind movement expressed in m/ivl
	LAPSE = ts['LAPSE']

	VOL = ts['VOL']
	SROVOL = ts['SROVOL']
	EROVOL = ts['EROVOL']
	SOVOL = zeros((simlen, nexits))
	EOVOL = zeros((simlen, nexits))
	for i in range(nexits):
		SOVOL[:, i] = ts['SOVOL' + str(i + 1)]
		EOVOL[:, i] = ts['EOVOL' + str(i + 1)]

	qsolar = 0.0

	for loop in range(simlen):

		tws = tw
		iheat  = IHEAT[loop]
		solrad = SOLRAD[loop]
		oheat  = 0.0

		vol = VOL[loop] * AFACT
		srovol = SROVOL[loop]
		erovol = EROVOL[loop]
		sovol  = SOVOL[loop,:]
		eovol  = EOVOL[loop,:]
		tw, roheat, oheat = advect(iheat, tw, nexits, svol, vol, srovol, erovol, sovol, eovol) # watertemp treated as a concentration
		svol = vol

		if tw > 66.0:
			if adfg < 2:
				errorsV[0] += 1  # Water temperature is above 66 C (150 F).  In most cases, this indicates an instability in advection.
				rtw = tw
			else:
				tw = tws

		# simulate heat exchange with the atmosphere

		# calculate solar radiation absorbed(qsolar); solrad, which is expressed in langleys/ivl, is the solar radiation at gage corrected for location of reach;

		if shadfg == 1:     # perform shading calculations
			if DAYFG:       # get total daily input radiation
				dsolar = DSOLAR[loop]
				# shadeh(srad)  		# use shade module to compute solar radiation absorbed by stream
				srad = 1.0
				qsolar = srad * 10.0   # 10.0 is the conversion from ly/ivl to kcal/m2.ivl.
		else:               # use constant shading factor
			# 0.97 accounts for surface reflection (assumed 3 percent);
			# cfsaex is the ratio of radiation incident to water surface to gage radiation values (accounts for regional differences, shading of water surface,etc)
			# 10.0 is the conversion from ly/ivl to kcal/m2.ivl.
			qsolar = 0.97 * cfsaex * solrad * 10.0

		# calculate heat transfer rates for water surface; units are kcal/m2.ivl

		# get quantity of precipitation and convert ft/ivl to m/ivl,
		prec = PREC[loop] * 0.0833  # / 12.0 to match HSPF precision
		if prec > 0.0:
			if uunits == 1:
				mprec = prec /3.2808
			else:
				mprec = prec / (3.2808*3.2808)
			# calculate heat added by precip, assuming temperature is equal to reach/res water temperature
			qprec = mprec * tw * 1000.0
		else:
			qprec = 0.0
			mprec = 0.0

		# calculate cloud cover factor for determination of atmospheric longwave radiation
		cloud = CLOUD[loop]
		cldfac = 1.0 + (0.0017 * (cloud**2))

		gatmp  = GATMP[loop]     # get gage air temperature
		gatmp  = (gatmp - 32.0) * 0.555
		dewtmp = DEWTMP[loop]
		dewtmp = (dewtmp - 32.0) * 0.555
		wind   = WIND[loop] * 1609.0  # 5280.0 / 3.28     # get wind movement expressed in m/ivl
		lapse  = LAPSE[loop]
		# ratemp -- correct air temperature for elevation differences
		#   find precipitation rate during the interval; prrat is expressed in m/min
		prrat = mprec / delt
		if prrat > 2.0e-5:  # use rain period lapse rate expressed as deg c/ft
			laps = 1.94e-03
		else:  # use dry period lapse rate expressed as deg c/ft
			laps = lapse
		# compute corrected air temperature for the end of the current interval; airtmp is expressed in degrees c
		airtmp = gatmp - laps * eldat

		avdepe = AVDEPE[loop]
		if avdepe > 0.17:
			# to degrees kelvin to calculate atmospheric longwave radiation
			twkelv = tw     + 273.16
			takelv = airtmp + 273.16

			# calculate net flux of longwave radiation in kcal/m2.ivl;
			# 4.73e-8 is the stephan-boltzmann constant multiplied by .97 to account for emissivity of water
			# katrad is the atmospheric longwave radiation coefficient
			# changed sign to make it consistent with other fluxes; ie, positive = gain of heat by reach; brb 6/95
			qlongw = 4.73e-8 * ((twkelv**4) - katrad * 1.0e-6 * cldfac * (takelv**6)) * delt60 * -1.0

			# calculate conductive-convective heat transport in kcal/m2.ivl
			# kcond is the heat transport coefficient for conduction-convection
			# changed sign to make it consistent with other fluxes; ie, positive = gain of heat by reach; brb 6/95)
			qcon = cfpres * kcond * 1.0e-4 * wind * (airtmp - tw)

			# water evaporated during interval in meters/ivl; kevap is the evaporation coefficient
			# vapor(dewtmp) is vapor pressure of air above water surface in millibars
			# vapor(tw) is saturation vapor pressure at the water surface in millibars
			evap = kevap * 1.0e-9 * wind * (vapor(tw) - vapor(dewtmp))

			# heat loss due to evaporation in kcal/m2.ivl
			# (597300. - 570.*tw) = latent heat of vaporization
			# (597.3 - .57*tw) multiplied by the density of water(1000 kg/m3);
			# changed sign of qevap to make it consistent with other fluxes; ie, positive = heat gain; brb 6/95
			qevap = (597300.0 - 570.0 * tw) * evap * -1.0
			# print('qevap')

			# bed conduction
			# if bedflg == 1 or bedflg == 2:
			# 	pass # tgrnd = ???  #user defined ts, monthly or single valued by TGFLG (1 for TS; 2 for constant; 3 for monthly)

			# compute conduction heat flux
			# print('bedflg')
			if bedflg == 1:   # one-layer bed conduction model
				tgrnd = TGRND[loop]
				if uunits == 1:
					tgrnd = (tgrnd - 32.0) * 0.555

				qbed = kmud * (tgrnd - tw)
			elif bedflg == 2:	# two-layer bed conduction model
				# Following is subrouting  #$BEDHT2
				'''Compute bed conduction heat flux using 2-interface model based on Caupp's and Brock's (1994) model of the Truckee.'''

				cpr = 1000.  # CPR = density * specific heat of water (and mud); CPR = 1 gm/cm3 * 1 kcal/kg/C * 1000 cm3.kg/m3/g = 1000 kcal/m3/C; this model uses CPR for both water and mud, per Caupp

				# mud temperature at center of current time step; tmuddt is slope of mud temperature curve
				tmud += tmuddt / 2.0

				# heat flux between mud and water based on water temperature in last time step and mud temperature at center of current time step
				# CMUD is the heat conductance coefficient (kcal/m2/C/ivl); it is an input parameter; Caupp uses 0.02 kcal/m2/C/s;
				# WQRRS = 0.001; "Oldman River CMUD" = 0.014
				bflux = (tmud - tw) * kmud   # kcal/m2/ivl

				# compute a new mud temperature slope using heat flux and heat capacity of mud/water and depth ("thermal capacity") of mud
				tmuddt = -bflux / cpr / muddep

				# mud temperature at center of time step
				bthalf = tmud

				# compute heat flux between ground and mud based on mud temperature at center of current time step and input ground temperature,
				# which can be estimated by the mean annual air temperature; this flux will be used to compute mud temperature at end of
				# current time step; the eqn. uses mud depth (m), CPR (kcal/m3/C),  24 hr/day, and streambed thermal gradient (KGRND kcal/C/ivl/m3)
				# to express heat flux (BEDINS) in units of C/ivl;  depth of water is used in error; should be depth of mud, per Caupp
				# (KGRND=0.1419 cal/C/hr/cm2 apparently assumes ground depth= 1 m ?)
				bedins = kgrnd * (tgrnd - tmud) / muddep / cpr

				# compute the new mud temperature at end of current time step; first, account for heat flux between water and mud
				# second, account for heat flux between ground and mud
				tmud = tmud + tmuddt / 2.0 + bedins

				# compute heat flux between mud and water using mud temperature at center of time step and current water temperature;
				# KMUD is the mud-water heat conductance coefficient (kcal/m2/C/ivl)
				qbed = (bthalf - tw) * kmud
				# end BEDHT2
			elif bedflg == 3:       # Jobson's bed conduction model;
				# set qbed to 0 initially in order to compute preliminary deltt which will be used later for computing qbed
				qbed = 0.0
			else:   # no bed conductance
				qbed = 0.0

			# calculate total heat exchange at water surface; qtotal in kcal/m2.ivl
			qtotal = qsolar + qlongw + qcon + qevap + qprec + qbed

			if abs(qtotal) > 1.0:    # if net heat flux > 1 kcal/m2.ivl, calculate new water temperature
				# solution technique requires sum of partial derivatives of qlongw, qcon, and qevap with respect to water temperature;
				# spd is derived by a series of three operations; the actual value of spd is not derived until the last operation
				spd = 18.92e-8 * (twkelv**3) * delt60 + cfpres * kcond * 1.0e-4 * wind
				spd = spd + kevap * 1.0e-9 * wind * 588750.0 * (0.4436 + tw * (28.63195e-3 + tw * (0.8e-3 + tw * (0.01124e-3 + tw * 0.00013e-3))))
				spd = 0.5 * spd

				# conversion factor to convert total heat exchange in kcal/m2.ivl to degrees c/ivl for the volume of water in the reach;
				cvqt = 3.281e-3 / avdepe  # 3.281e-3 = (1000 cal/kcal)*(1 m3/10e6 cm3)*(3.281 ft/m)

				# change in water temperaturE (if Jobsons bed conduction method is being used, this is a preliminary calculation of TW and DELTTW
				delttw = cvqt * qtotal / (1.0 + spd * cvqt)
				otw   = tw
				tw    = tw + delttw
				if tw < 0.04:
					delttw = delttw + 0.04 - tw
					tw    = 0.04

				if bedflg == 3:   # Jobson's bed conductance model
					deltt[0] = 0.0  if tws < -1.0e10 else tw - tws
					for i in range(tstop):
						qbed += delh[i] * deltt[i]
					qtotal = qtotal + qbed

					# recalculate change in water temperature
					delttw = cvqt * qtotal / (1.0 + spd * cvqt)
					tw     = otw + delttw
					if tw < 0.04:
						delttw = delttw + 0.04 - tw
						tw    = 0.04
				htexch = delttw * vol
			else:    # water temperature remains unchanged
				htexch = 0.0
		else:     # there is too little water in reach to simulate heat exchange with atmosphere
			# set water temp to air temp
			delttw = airtmp - tw
			tw     = airtmp
			if tw < 0.04:
				delttw = delttw + 0.04 - tw
				tw  = 0.04
			htexch = delttw * vol

			# set all atmospheric/bed fluxes to 0
			qtotal = qsolar = qlongw = qcon = qevap = qprec = qbed = 0.0

		# update deltt array for next time step of jobsons bed conductance model
		if bedflg == 3:
			deltt[0] = 0.0  if tws < -1.0e10 or tw < -1.0e10 else tw - tws
			for i in range(tstop, 0, -1):  # do 30 i= tstop, 2, -1
				deltt[i] = deltt[i-1]

		rheat = tw * vol     # calculate storage of thermal energy in rchres

		if uunits == 1:
			TW[loop]    = (tw * 9.0 / 5.0) + 32.0
			AIRTMP[loop]= (airtmp * 9.0 / 5.0) + 32.0
			HTEXCH[loop] = htexch * 407960. * 12. / 43560.
			ROHEAT[loop] = roheat / 0.0089
			OHEAT[loop] = oheat / 0.0089
			QTOTAL[loop] = qtotal * 0.369
			QSOLAR[loop] = qsolar * 0.369
			QLONGW[loop] = qlongw * 0.369
			QCON[loop] = qcon * 0.369
			QEVAP[loop] = qevap * 0.369
			QPREC[loop] = qprec * 0.369
			QBED[loop] = qbed * 0.369
		else:
			TW[loop] = tw
			AIRTMP[loop] = airtmp
			HTEXCH[loop] = htexch * 1000.
			ROHEAT[loop]= roheat * 1000.
			OHEAT[loop] = oheat * 1000.
			QTOTAL[loop] = qtotal
			QSOLAR[loop] = qsolar
			QLONGW[loop] = qlongw
			QCON[loop]   = qcon
			QEVAP[loop]  = qevap
			QPREC[loop]  = qprec
			QBED[loop]   = qbed

	if nexits > 1:
		for i in range(nexits):
			ts['OHEAT' + str(i+1)] = OHEAT[:, i]

	return errorsV


@njit(cache=True)
def vapor(tmp):
		'''	# define vapor function based on temperature (deg c); vapor pressure is expressed in millibars'''
		return 33.8639 * ((0.00738 * tmp + 0.8072)**8 - 0.000019 * abs(1.8 * tmp + 48.0) + 0.001316)


def expand_HTRCH_masslinks(flags, uci, dat, recs):
	if flags['HTRCH']:
		# IHEAT
		rec = {}
		rec['MFACTOR'] = dat.MFACTOR
		rec['SGRPN'] = 'HTRCH'
		if dat.SGRPN == "ROFLOW":
			rec['SMEMN'] = 'ROHEAT'
		else:
			rec['SMEMN'] = 'OHEAT'
		rec['SMEMSB1'] = dat.SMEMSB1
		rec['SMEMSB2'] = dat.SMEMSB2
		rec['TMEMN'] = 'IHEAT'
		rec['TMEMSB1'] = dat.TMEMSB1
		rec['TMEMSB2'] = dat.TMEMSB2
		rec['SVOL'] = dat.SVOL
		recs.append(rec)
	return recs