''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERTMP.FOR module into Python''' 


from numpy import zeros, where, ones, float64, full, int64
from numba import njit
from HSP2.utilities  import hoursval, initm, make_numba_dict


ERRMSG = ['SLTMP temperature less than -100C',   # MSG0
		'ULTMP temperature less than -100C',     # MSG1
		'LGTMP temperature less than -100C', 	# MSG2
		'SLTMP temperature greater than 100C', 	# MSG3
		'ULTMP temperature greater than 100C',  # MSG4
		'LGTMP temperature greater than 100C']  # MSG5

MINTMP = -100
MAXTMP = 100

def pstemp(io_manager, siminfo, uci, ts):
	'''Estimate soil temperatures in a pervious land segment'''
	simlen = siminfo['steps']
	
	ui = make_numba_dict(uci)
	ui['simlen'] = siminfo['steps']
	ui['uunits'] = siminfo['units']
	ui['delt'] = siminfo['delt']
	ui['errlen'] = len(ERRMSG)

	u = uci['PARAMETERS']
	if 'SLTVFG' in u:
		ts['ASLT'] = initm(siminfo, uci, u['SLTVFG'], 'MONTHLY_ASLT', u['ASLT'])
		ts['BSLT'] = initm(siminfo, uci, u['SLTVFG'], 'MONTHLY_BSLT', u['BSLT'])
	else:
		ts['ASLT'] = full(simlen, u['ASLT'])
		ts['BSLT'] = full(simlen, u['BSLT'])
	if 'ULTVFG' in u:
		ts['ULTP1'] = initm(siminfo, uci, u['ULTVFG'], 'MONTHLY_ULTP1', u['ULTP1'])
		ts['ULTP2'] = initm(siminfo, uci, u['ULTVFG'], 'MONTHLY_ULTP2', u['ULTP2'])
	else:
		ts['ULTP1'] = full(simlen, u['ULTP1'])
		ts['ULTP2'] = full(simlen, u['ULTP2'])
	if 'LGTVFG' in u:
		ts['LGTP1'] = initm(siminfo, uci, u['LGTVFG'], 'MONTHLY_LGTP1', u['LGTP1'])
		ts['LGTP2'] = initm(siminfo, uci, u['LGTVFG'], 'MONTHLY_LGTP2', u['LGTP2'])
	else:
		ts['LGTP1'] = full(simlen, u['LGTP1'])
		ts['LGTP2'] = full(simlen, u['LGTP2'])

	ts['HRFG'] = hoursval(siminfo, ones(24), dofirst=True).astype(float64)  # numba Dict limitation

	############################################################################
	errors = _pstemp_(ui, ts)  # run PSTEMP simulation code
	############################################################################

	return errors, ERRMSG


@njit(cache=True)
def _pstemp_(ui, ts):
	'''Estimate soil temperatures in a pervious land segment'''

	errorsV = zeros(int(ui['errlen'])).astype(int64)

	simlen = int(ui['simlen'])
	delt = ui['delt']
	uunits = ui['uunits']

	if 'TSOPFG' in ui:
		TSOPFG = ui['TSOPFG']
	else:
		TSOPFG = 0
	AIRTFG = int(ui['AIRTFG'])

	# initial conditions
	# may need to convert to C from F
	airtc  = 60.0
	sltmp  = 60.0
	ultmp  = 60.0
	lgtmp  = 60.0
	if uunits == 2:
		airtc = 16.0
		sltmp = 16.0
		ultmp = 16.0
		lgtmp = 16.0
	if 'AIRTC' in ui:
		airtc = ui['AIRTC']
	if 'SLTMP' in ui:
		sltmp = ui['SLTMP']
	if 'ULTMP' in ui:
		ultmp = ui['ULTMP']
	if 'LGTMP' in ui:
		lgtmp = ui['LGTMP']
	if uunits != 2:
		airtc = (airtc - 32.0) * 0.555
		sltmp = (sltmp - 32.0) * 0.555
		ultmp = (ultmp - 32.0) * 0.555
		lgtmp = (lgtmp - 32.0) * 0.555

	# preallocate storage
	AIRTC = ts['AIRTC'] = zeros(simlen)
	SLTMP = ts['SLTMP'] = zeros(simlen)
	ULTMP = ts['ULTMP'] = zeros(simlen)
	LGTMP = ts['LGTMP'] = zeros(simlen)
	
	AIRTMP = ts['AIRTMP']
	if uunits == 2:
		AIRTMP = (AIRTMP - 32.0) * 0.555

	ASLT  = ts['ASLT']
	BSLT  = ts['BSLT']
	ULTP1 = ts['ULTP1']
	ULTP2 = ts['ULTP2']
	LGTP1 = ts['LGTP1']
	LGTP2 = ts['LGTP2']

	if uunits != 2:
		if TSOPFG == 1:
			ULTP1= (ULTP1 - 32.0) * 0.5555 # trying to match HSPF precision here
			LGTP1= (LGTP1 - 32.0) * 0.5555
		else:
			ULTP2=  0.555 * ULTP2
			LGTP2 = 0.555 * LGTP2

	HRFG = ts['HRFG'].astype(int64)

	airts = AIRTMP[0]

	for loop in range(simlen):
		hrfg = HRFG[loop]
		airtmp = AIRTMP[loop]

		if uunits != 2:
			# convert to centigrade
			airtcs = (airts - 32.0) * 0.555
			airtc  = (airtmp - 32.0) * 0.555
		else:
			airtcs = airts
			airtc  = airtmp

		# determine soil temperatures - units are deg c temperature of surface layer is always estimated using a linear regression with air temperature
		if hrfg:    # it is time to update surface layer temperature
			if uunits != 2:
				aslt = (ASLT[loop]- 32.0) * 0.555
			else:
				aslt = ASLT[loop]
			bslt = BSLT[loop]
			sltmp = aslt + bslt * airtc

		if TSOPFG == 1: # compute subsurface temperature using regression and monthly values
			if hrfg:   # it is time to update subsurface temperatures temperature of upper layer is computed by regression with air temperature
				ault  = ULTP1[loop]
				bult  = ULTP2[loop]
				ultmp = ault + bult * airtc

				# temperature of lower layer and groundwater were interpolated from monthly values
				lgtmp = LGTP1[loop]
		else:   # tsopfg is 0 or 2
			''' compute subsurface temperatures using a mean departure from air temperature plus a smoothing factor -
			if tsopfg is 2, the lower/gw layer temperature is a function of upper layer temperature instead of air temperature'''

			ulsmo  = ULTP1[loop]
			ultdif = ULTP2[loop]
			ultmps = ultmp
			ultmp  = ultmps + ulsmo * (airtcs + ultdif - ultmps)

			lgsmo  = LGTP1[loop]
			lgtdif = LGTP2[loop]
			lgtmps = lgtmp

			if TSOPFG == 0:   # original method - lower/gw temp based on air temp
				lgtmp = lgtmps + lgsmo * (airtcs + lgtdif - lgtmps)
			else:             # new method for corps of engineers 10/93 -  TSOPFG=2   
				lgtmp = lgtmps + lgsmo * (ultmp + lgtdif - lgtmps)  #  lower/gw temp based on upper temp

		# check temperatures for invalid results
		if sltmp < MINTMP:
			sltmp = MINTMP
			errorsV[0] += 1
		if sltmp > MAXTMP:
			sltmp = MAXTMP
			errorsV[3] += 1
			
		if ultmp < MINTMP:
			ultmp = MINTMP
			errorsV[1] += 1
		if ultmp > MAXTMP:
			ultmp = MAXTMP
			errorsV[4] += 1
			
		if lgtmp < MINTMP:
			lgtmp = MINTMP
			errorsV[2] += 1
		if lgtmp > MAXTMP:
			lgtmp = MAXTMP
			errorsV[5] += 1

		# update airts for next interval if section atemp not active
		if AIRTFG == 0:
			airts = airtmp

		if uunits != 2:
			# Need to convert back to English units here
			AIRTC[loop] = (airtc * 9.0 / 5.0) + 32.0
			SLTMP[loop] = (sltmp * 9.0 / 5.0) + 32.0
			ULTMP[loop] = (ultmp * 9.0 / 5.0) + 32.0
			LGTMP[loop] = (lgtmp * 9.0 / 5.0) + 32.0
		else:
			AIRTC[loop] = airtc
			SLTMP[loop] = sltmp
			ULTMP[loop] = ultmp
			LGTMP[loop] = lgtmp

	return errorsV