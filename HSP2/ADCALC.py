''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros

# The clean way to get calculated data from adcalc() into advert() is to use a closure. 
# This is not currently supported by Numba.

'''
split out the loop section of adcalc to allow numba caching
let omat be passed in advect to avoid recreating it each time
ADFLAG == 2  loop simplification as elif

'''

def adcalc(general, ui, ts):
	'''Prepare to simulate advection of fully entrained constituents'''

	simlen = general['SIMLEN']
	delts  = general['DELT'] * 60.0     # delts is the simulation interval in seconds
	nexits = ui['NEXITS']               # table type GEN-INFO
	ADFG   = ui['ADFG']                 # table type ACTIVITY

	# table ADCALC-DATA
	crrat  = ui['CRRAT']
	vol    = ui['VOL']
	
	# external time series
	O   = ts['O']  	 # total rate of outflow per exit: O[simlen, nexits]
	VOL = ts['VOL']
	
	# calculated timeseries for advect()
	SROVOL = ts['SROVOL'] = zeros(simlen)
	EROVOL = ts['ERVOL']  = zeros(simlen)
	SOVOL  = ts['SOVOL']  = zeros((simlen, nexits))
	EOVOL  = ts['EVOL']   = zeros((simlen, nexits))
	
	if ADFG == 2:
		ks = ui['KS']  if ADFG == 2 else 0.0     # need to get from HYDR section
	
	o  = zeros(nexits)
	os = zeros(nexits)
	
	adcalc_(simlen, delts, nexits, crrat, ks, vol, ADFG, O, VOL, SROVOL, EROVOL, SOVOL, EOVOL, o, os)
	ui['adcalcData'] = (nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL)
	return


#@jit(nopython=True)	
def adcalc_(simlen, delts, nexits, crrat, ks, vol, ADFG, O, VOL, SROVOL, EROVOL, SOVOL, EOVOL, o, os):	
	''' Internal adcalc() loop for Numba'''
	
	for loop in range(simlen):
		vols = VOL[loop-1]  if loop > 0 else vol
		o    = O[loop,:]
		os   = O[loop-1,:] if loop > 0 else O[loop,:]
		if nexits > 1:
			ro  = o.sum()
			ros = os.sum()
		
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
		if nexits > 1:  # determine weighted volume of outflow at start and end of ivl per exit
			SOVOL[loop,:] = js   * os * delts
			EOVOL[loop,:] = cojs * o  * delts
	return


#@jit(nopython=True)
def advect(loop, imat, conc, omat, nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL):
	''' Simulate advection of constituent totally entrained in water.
	called as: advect(loop, imat, conc, omat, *ui['adcalcData']) '''
	
	vols   = VOL[loop-1]  if loop > 0 else vol
	vol    = VOL[loop]
	srovol = SROVOL[loop]
	erovol = EROVOL[loop]
	sovol  = SOVOL[loop,:]
	eovol  = EOVOL[loop,:]
	
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
	

#@jit(nopython=True)
def oxrea(LKFG,wind,cforea,avvele,avdepe,tcginv, REAMFG,reak,reakt,expred,exprev,len, delth,tw,delts,delt60,uunits, korea):
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
	
	# calculate reaeration coefficient for free-flowing reach
	elif REAMFG == 1:
		# calculate reaeration coefficient based on energy dissipation principles (tsivoglou method)
		# convert length and drop in energy line along length of rchres to english units, if necessary
		lene   = len
		delthe = delth

		if abs(avvele) > 0.0:
			flotim = lene / avvele
			korea  = reakt * (delthe / flotim) * (tcginv**(tw - 20.)) * delts
		else:
			korea = 0.0
	elif REAMFG == 2:
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
