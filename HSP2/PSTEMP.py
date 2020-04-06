''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERTMP.FOR module into Python''' 

''' NOTE: needs lots of Celcius temp conversions, in and out'''

from numpy import zeros, where
from numba import jit
from HSP2  import initm


ERRMSG = ['SLTMP temperature less than 100C',   # MSG0
		'ULTMP temperature less than 100C',     # MSG1
		'LGTMP temperature less than 100C', 	# MSG2
		'SLTMP temperature greater than 100C', 	# MSG3
		'ULTMP temperature greater than 100C',  # MSG4
		'LGTMP temperature greater than 100C']  # MSG5

MINTMP = -100
MAXTMP = 100

def pstemp(store, general, ui, ts):
	'''Estimate soil temperatures in a pervious land segment'''
	errorsV = zeros(len(ERRMSG), dtype=int)
	
	simlen = general['sim_len']
	tindex = general['tindex']
	
	TSOPFG = ui['TSOPFG']

	# may need to convert to C from F
	airtc  = ui['AIRTC']
	sltmp  = ui['SLTMP']
	ultmp  = ui['ULTMP']
	lgtmp  = ui['LGTMP']
	
	# preallocate storage
	AIRTC = ts['AIRTC'] = zeros(simlen)
	SLTMP = ts['SLTMP'] = zeros(simlen)
	ULTMP = ts['ULTMP'] = zeros(simlen)
	LGTMP = ts['LGTMP'] = zeros(simlen)
	
	AIRTMP = ts['AIRTMP']
	
	initm(general, ui, ts, 'SLTVFG', 'ASLTM', 'ASLT')
	initm(general, ui, ts, 'SLTVFG', 'BSLTM', 'BSLT')
	initm(general, ui, ts, 'ULTVFG', 'ULTP1M', 'ULTP1')
	initm(general, ui, ts, 'ULTVFG', 'ULTP2M', 'ULTP2')
	initm(general, ui, ts, 'LGTVFG', 'LGTP1M', 'LGTP1')
	initm(general, ui, ts, 'LGTVFG', 'LGTP2M', 'LGTP2')
	ASLT  = ts['ASLT']
	BSLT  = ts['BSLT']
	ULTP1 = ts['ULTP1']
	ULTP2 = ts['ULTP2']
	LGTP1 = ts['LGTP1']
	LGTP2 = ts['LGTP2']
	
	HRFG = where(tindex.minute==1, True, False)   # ??? need to check if minute == 0	
	
	arts = AIRTMP[0]  # arts not used ???

	for loop in range(simlen):
		hrfg = HRFG[loop]
		airtmp = AIRTMP[loop]

		# convert to centigrade  
		airtcs = (airts - 32.0)  * 0.555   # airts not defined ???
		airtc  = (airtmp - 32.0) * 0.555

		# determine soil temperatures - units are deg c temperature of surface layer is always estimated using a linear regression with air temperature
		if hrfg:    # it is time to update surface layer temperature
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
		if AIRTFG == 0:     # AIRTFG not defined  ???
			airts = airtmp   # airts not used   ???

	# Need to convert back to English units here
	AIRTC[loop] = airtc
	SLTMP[loop] = sltmp
	ULTMP[loop] = ultmp
	LGTMP[loop] = lgtmp

	return errorsV, ERRMSG