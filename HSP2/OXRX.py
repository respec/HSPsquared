''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros, array
from math import exp
from hrchrq import sink

def oxrx(general, ui, ts):
	''' simulate primary do, bod balances'''
	
	delt60 = general['DELT'] / 60.0
	nexits = general['nexits']

	vol  = ui['VOL']
	
	# table-type ox-genparm
	kbod20 = ui['KBOD20'] * delt60  # convert units from 1/hr to 1/ivl
	tcbod  = ui['TCBOD']
	kodset = ui['KODSET'] * delt60  # convert units from 1/hr to 1/ivl
	supsat = ui['SUPSAT']
	
	# table-type ox-init
	dox   = ui['dox']
	bod   = ui['bod']
	satdo = ui['satdo']
	
	# other required values
	REAMFG = ui['REAMFG']   # table-type ox-flags
	elev   = ui['ELEV']     # table-type elev
	
	cfpres = ((288.0 - 0.001981 * elev) / 288.0)**5.256  # pressure correction factor -
	ui['CFPRES'] = cfpres
	
	LKFG = ui['LKFG']
	if LKFG == 1:
		cforea = ui['CFOREA']   # reaeration parameter from table-type ox-cforea
	elif REAMFG == 1:            # tsivoglou method;  table-type ox-tsivoglou
		reakt  = ui['REAKT']
		tcginv = ui['TCGINV']
		len_   = ui['LEN']
		delth  = ui['DELTH']
	elif REAMFG == 2:	        # owen/churchill/o'connor-dobbins; table-type ox-tcginv
		tcginv = ui['TCGINV']
		reak   = ui['REAK']
		expred = ui['EXPRED']
	elif REAMFG == 3:           # user formula - table-type ox-reaparm
		tcginv = ui['TCGINV']
		reak   = ui['REAK']
		expred = ui['EXPRED']
		
	if BENRFG == 1:       # benthic release parms - table-type ox-benparm
		benod  = ui['BENOD'] * delt60  # convert units from 1/hr to 1/ivl
		tcben  = ui['TCBEN']
		expod  = ui['EXPOD']
		exprel = ui['EXPREL']		
		BRBOD  = array([ui['BRBOD1'] , ui['BRBOD2']])  * delt60  # convert units from 1/hr to 1/ivl

	rdox = dox * vol
	rbod = bod * vol

	odox = zeros(nexits)
	odob = zeros(nexits)

	#@jit(nopython = True)
	''' simulate primary do, bod balances'''
	def oxrx(idox, ibod, wind, avdepe, avvele, depcor, tw, BENRFG, advData):
		# advect dissolved oxygen
		dox, rodox, odox = advect(idox, dox, odox, *advData)

		# advect bod
		bod, robod, obod = advect(ibod, bod, robod, obod(1))

		if avdepe > 0.17:  # benthal influences are considered
			# sink bod
			bod, snkbod = sink(vol, avdepe, kodset, bod, snkbod)
			snkbod = -snkbod

			if BENRFG == 1:
				#$OXBEN   # simulate benthal oxygen demand and benthal release of bod, and compute associated fluxes
				# calculate amount of dissolved oxygen required to satisfy benthal oygen demand (mg/m2.ivl)
				benox = benod * (tcben**(tw -20.0)) * (1.0 -exp(-expod * dox))

				# adjust dissolved oxygen state variable to acount for oxygen lost to benthos, and compute concentration flux
				doben = dox
				dox   = dox - benox * depcor
				doben = benox * depcor  
				if dox >= 0.001:
					doben = benox * depcor
				else:
					dox = 0.0

				# calculate benthal release of bod; release is a function of dissolved oxygen
				# (dox) and a step function of stream velocity; brbod(1) is the aerobic benthal 
				# release rate; brbod(2) is the base increment to benthal release under 
				# decreasing do concentration; relbod is expressed as mg bod/m2.ivl
				relbod = (brbod1 + brbod2 * exp(-exprel * dox)) * scrfac

				# add release to bod state variable and compute concentration flux
				bod    = bod + relbod * depcor
				bodbnr = relbod * depcor
						
				# end #$OXBEN
				bendox = -doben * vol
				bnrbod = bodbnr * vol

		elif LKFG == 1:
			# calculate oxygen reaeration
			if not (GQFG == 1 and GQALFG(4) == 1):
				korea = oxrea(LKFG, wind,cforea,avvele,avdepe,tcginv, reamfg,reak,reakt,expred,exprev,len,
				delth,tw,delts,delt60,uunits, korea)

			# calculate oxygen saturation level for current water
			# temperature; satdo is expressed as mg oxygen per liter
			satdo = 14.652 + tw * (-0.41022 + tw * (0.007991 - 0.7777e-4 * tw))

			# adjust satdo to conform to prevalent atmospheric pressure
			# conditions; cfpres is the ratio of site pressure to sea level pressure
			satdo = cfpres * satdo
			if satdo < 0.0:
				# warning - this occurs only when water temperature is very high - above
				# about 66 c.  usually means error in input gatmp (or tw if htrch is not being simulated).      
				satdo = 0.0   # reset saturation level

			# compute dissolved oxygen value after reaeration,and the reaeration flux
			dorea  = korea * (satdo - dox)
			dox    = dox + dorea
			readox = dorea * vol

			#$BODDEC
			'''calculate concentration of oxygen required to satisfy computed bod decay'''
			bodox = (kbod20 * (tcbod**(tw -20.0))) * bod   # bodox is expressed as mg oxygen/liter.ivl
			if bodox > bod:
				bodox = bod

			# adjust dissolved oxygen state variable to acount for oxygen lost to bod decay, and compute concentration flux
			if bodox >= dox:
				bodox = dox
				dox   = 0.0
			else:
				dox = dox - bodox

			# adjust bod state variable to account for bod decayed
			bod -= bodox
			if bod < 0.0001:
				bod = 0.0
			# end #$BODDEC
			
			boddox = -bodox * vol
			decbod = -bodox * vol

		else:    # there is too little water to warrant simulation of quality processes
			bodox  = 0.0
			
			readox = 0.0
			boddox = 0.0
			bendox = 0.0
			decbod = 0.0
			bnrbod = 0.0
			snkbod = 0.0

		totdox = readox + boddox + bendox
		totbod = decbod + bnrbod + snkbod

		rdox = dox * vol
		rbod = bod * vol
		return dox, bod, satdo, rodo, robod, odo, obod
	return oxrx
