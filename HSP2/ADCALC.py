''' Copyright (c) 2020 by RESPEC, INC.
Authors: Robert Heaphy, Ph.D. and Paul Duda
License: LGPL2
'''

from numpy import zeros, array
from numba import njit
from HSP2.utilities import make_numba_dict

# The clean way to get calculated data from adcalc() into advert() is to use a closure. 
# This is not currently supported by Numba.

'''
split out the loop section of adcalc to allow numba caching
let omat be passed in advect to avoid recreating it each time
ADFLAG == 2  loop simplification as elif

'''
ERRMSG = []

def adcalc(io_manager, siminfo, uci, ts):
	'''Prepare to simulate advection of fully entrained constituents'''

	errorsV = zeros(len(ERRMSG), dtype=int)

	simlen = siminfo['steps']

	ui = make_numba_dict(uci)
	nexits = int(ui['NEXITS'])          # table type GEN-INFO
	ui['simlen'] = siminfo['steps']
	ui['delts'] = siminfo['delt'] * 60.0     # delts is the simulation interval in seconds
	ui['uunits']  = siminfo['units']

	# calculated timeseries for advect()
	if 'SROVOL' not in ts:
		ts['SROVOL'] = zeros(simlen)
	if 'EROVOL' not in ts:
		ts['EROVOL'] = zeros(simlen)

	############################################################################
	errors = _adcalc_(ui, ts)  # run ADCALC simulation code
	############################################################################

	if 'VOL' in ui:
		vol = ui['VOL']
	else:
		vol = 0.0

	VOL = ts['VOL']
	SROVOL = ts['SROVOL']
	EROVOL = ts['EROVOL']
	SOVOL = zeros((simlen, nexits))
	EOVOL = zeros((simlen, nexits))
	for i in range(nexits):
		SOVOL[:, i] = ts['SOVOL' + str(i + 1)]
		EOVOL[:, i] = ts['EOVOL' + str(i + 1)]

	uci['adcalcData'] = (nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL)

	return errorsV, ERRMSG


@njit(cache=True)
def _adcalc_(ui, ts):
	''' Internal adcalc() loop for Numba'''

	ADFG = ui['ADFG']  # table type ACTIVITY
	simlen = int(ui['simlen'])
	nexits = int(ui['NEXITS'])
	delts  = ui['delts']
	uunits = ui['uunits']

	# units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
	VFACT = 43560.0
	AFACT = 43560.0
	if uunits == 2:
		# si units conversion constants, 1 hectare is 10000 sq m, assumes area input in hectares, vol in Mm3
		VFACT = 1.0e6
		AFACT = 10000.0

	# table ADCALC-DATA
	if 'CRRAT' in ui:
		crrat = ui['CRRAT']
	else:
		crrat = 1.5
	if 'VOL' in ui:
		vol = ui['VOL'] * VFACT
	else:
		vol = 0.0

	ks = 0.0
	if ADFG == 2:
		ks = ui['KS']  # need to get from HYDR section

	VOL = ts['VOL']

	SROVOL = ts['SROVOL']
	EROVOL = ts['EROVOL']
	if 'SOVOL' in ts:
		SOVOL = ts['SOVOL']
	else:
		SOVOL = zeros((simlen, nexits))
		for index in range(nexits):
			ts['SOVOL' + str(index+1)] = SOVOL[:,index]
	if 'EOVOL' in ts:
		EOVOL = ts['EOVOL']
	else:
		EOVOL = zeros((simlen, nexits))
		for index in range(nexits):
			ts['EOVOL' + str(index + 1)] = EOVOL[:, index]

	ROS = ui['ROS']
	OS  = zeros(nexits)
	for index in range(nexits):
		OS[index] = ui['OS' + str(index + 1)]

	# external time series
	O = zeros((simlen, nexits))
	if nexits > 1:
		for i in range(nexits):
			O[:, i] = ts['O' + str(i + 1)]
	else:
		O[:, 0] = ts['RO']

	for loop in range(simlen):
		vols = VOL[loop-1] * VFACT  if loop > 0 else vol

		o  = O[loop]
		os = O[loop-1] if loop > 0 else OS
		ro = 0.0
		ros= 0.0
		for index in range(nexits):
			ro  += o[index]
			ros += os[index]

		# weighting factors to calculate mean outflow rate over ivl: constituents outflow rate at start (js) and end (cojs) of interval;
		if ADFG >= 1:         # first use standard method of computing advective weighting
			if ros > 0.0:     # calculate ratio of volume to outflow volume; 
				rat = vols / (ros * delts)
				if rat < crrat:
					# some outflow volume entered control volume as inflow during same interval; hence, 
					# concentration of inflowing material will affect outflow concentration => js will be < 1.0
					js = rat / crrat
				else:
					# all water in outflow volume was contained in control volume at beginning of ivl; 
					# mean rate of outflow over ivl will be wholly dependent upon rate of outflow of constituents at start of ivl
					js = 1.0
			else:     # reach/res has no outflow at start of ivl
				js = 0.0
			if ADFG == 2 and js > ks:
				# use same weighting as used in flow routing i.e., JS = KS; this was added 6/2010 by brb & pbd 
				# based on suggestion of Sen Bai of Tetra Tech -  ASCE J. Hydrol. Engr., Vol.15, No.3, March, 2010
				js = ks
		cojs = 1.0 - js        # cojs is the complement of js

		# calculate weighted volumes of outflow at start of ivl (srovol) and end of ivl (erovol)
		SROVOL[loop] = js   * ros * delts
		EROVOL[loop] = cojs * ro  * delts
		# if nexits > 1:  # determine weighted volume of outflow at start and end of ivl per exit
		for index in range(nexits):
			ts['SOVOL' + str(index + 1)][loop] = js * os[index] * delts
			ts['EOVOL' + str(index + 1)][loop] = cojs * o[index] * delts
	return


@njit(cache=True)
def advect(imat, conc, nexits, vols, vol, srovol, erovol, sovol, eovol):
	''' Simulate advection of constituent totally entrained in water.
	Originally designed to be called as: advect(loop, imat, conc, omat, *ui['adcalcData'])
	but unit conversions in the calling routine make this impractical'''
	
	# vols   = VOL[loop-1]  if loop > 0 else vol
	# vol    = VOL[loop]
	# srovol = SROVOL[loop]
	# erovol = EROVOL[loop]
	# sovol  = SOVOL[loop,:]
	# eovol  = EOVOL[loop,:]

	omat = zeros((nexits))
	if vol > 0.0:    # reach/res contains water
		concs = conc
		conc = (imat + concs * (vols - srovol)) / (vol + erovol)  # material entering during interval, weighted volume of outflow based on conditions at start of ivl (srovol), and weighted volume of outflow based on conditions at end of ivl (erovol)
		romat = srovol * concs + erovol * conc    # total material leaving reach/res in ivl
		if nexits > 1:                            # material leaving through each exit gate
			omat = sovol * concs + eovol * conc   # qty.vol/l.ivl, array calculation
	else:                                         # reach/res has gone dry during the interval
		romat = imat + (conc * vols)  	          # total material leaving during interval = inflow + initial material
		if nexits > 1 and srovol > 0:              # calculate material leaving through each exit gate
			omat = (sovol / srovol) * romat       # array calculation
		conc = -1.0e30			
	return conc, romat, omat
	

@njit(cache=True)
def oxrea(LKFG,wind,cforea,avvele,avdepe,tcginv,reamfg,reak,reakt,expred,exprev,len, delth,tw,delts,delt60,uunits):
	''' Calculate oxygen reaeration coefficient'''
	# DELTS  - ???
	# DELT60 - simulation time interval in hours
	# UUNITS - system of units   1-english, 2-metric

	if LKFG == 1:     # this reach/res is a lake or reservoir
		''' empirically reaeration coefficient based on windspeed, surface area, and volume;
		windsp is windspeed in m/sec; wind is wind movement in m/ivl'''
		windsp = wind / delts
		windf = windsp * (-0.46 + 0.136 * windsp)  if windsp > 6.0 else 2.0
		korea = (0.032808 * windf * cforea / avdepe) * delt60

	else:
		# calculate reaeration coefficient for free-flowing reach
		if reamfg == 1:
			# calculate reaeration coefficient based on energy dissipation principles (tsivoglou method)
			# convert length and drop in energy line along length of rchres to english units, if necessary
			if uunits == 2:
				lene = len * 3.28
				delthe = delth * 3.28
			else:
				lene   = len
				delthe = delth

			if abs(avvele) > 0.0:
				flotim = lene / avvele
				korea  = reakt * (delthe / flotim) * (tcginv**(tw - 20.)) * delts
			else:
				korea = 0.0
		else:
			if reamfg == 2:
				# calculate reaeration coefficient as a power function of average hydraulic
				# depth and velocity; determine exponents to depth and velocity terms and assign value to reak
				if avdepe <= 2.0:  # use owen's formulation for reaeration
					reak   = 0.906
					exprev = 0.67
					expred = -1.85
				else:
					# calculate transition depth; transition depth determines which method
					# of calculation is used given the current velocity
					trandp = 0.0  if avvele < 1.7 else 0.4263 * (avvele**2.9135)
					if avdepe - trandp <= 0.0:  # use churchill's formulation for reaeration
						reak   = 0.484
						exprev = 0.969
						expred = -1.673
					else:                       # use o'connor-dobbins formulation for reaeration
						reak   = 0.538
						exprev = 0.5
						expred = -1.5
			korea = reak * avvele**exprev * avdepe**expred * tcginv**(tw - 20.0) * delt60  if tw < 66 else  0.999

	if korea > 1.0:
		korea = 0.999
	return korea


@njit(cache=True)
def sink (vol, avdepe, kset, conc):
	''' calculate quantity of material settling out of the control volume; determine the change in concentration as a result of sinking'''
	if kset > 0.0 and avdepe > 0.17:
		# calculate concentration change due to outgoing material; snkout is expressed in mass/liter/ivl; kset is expressed as ft/ivl and avdepe as feet
		snkout = conc * (kset / avdepe)  if kset < avdepe else conc  # calculate portion of material which settles out of the control volume during time step; snkout is expressed as mass/liter.ivl; conc is the concentration of material in the control volume
		conc  -= snkout        # calculate remaining concentration of material in the control volume
		snkmat = snkout * vol    # find quantity of material that sinks out; units are  mass.ft3/l.ivl in english system, and mass.m3/l.ivl in metric system
	else:
		snkout = 0.0
		snkmat = 0.0
	return conc, snkmat