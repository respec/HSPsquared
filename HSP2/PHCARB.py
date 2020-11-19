''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from math import log10
from numpy import zeros, array

from HRCHRQ import benth
from HRCHUT import advect

def phcarb(general, ui, ts):
	''' simulate ph, carbon dioxide, total inorganic carbon,  and alkalinity'''

	simlen = general('SIMLEN')
	delt60   = general('DELT') / 60.0   # ???? check
	ncons  = ui['NCONS']
	nexits = ui['NEXITS']
	BENRFG = ui['BENRFG']  # table type BENTH-FLAG  from RQual
	
	
	# flags - table-type ph-parm1
	phcnt  = ui['PHCNT']    # is the maximum number of iterations to pH solution.
	alkcon = ui['ALKCON']   # ALKCON  is the number of the conservative substance which is alkalinity.
	
	# table-type ph-parm2
	cfcinv = ui['CFCINV']
	BRCO2  = array([ui['BRCO21'], ui['BRCO22']]) * delt60    # convert benthal releases from  /hr to  /ivl

	# table-type ph-init
	tic = ui['TIC']
	co2 = ui['CO2']
	ph   = ui['PH']
	

	anaer  = ui['ANAER']


	
	otic = zeros(nexits)
	oco2 = zeros(nexits)	
	
	if ncons <= 0:
		ncons = alkcon
	if alkcon > ncons:
		pass # errmsg: error - invalid no. for alkcon


	#@jit(nopython=True)
	def phcarb_(dox, bod, alk, itic, ico2, tw, avdepe, scrfac, advData):
		''' simulate ph, carbon dioxide, total inorganic carbon,  and alkalinity'''


		# advect total inorganic carbon
		tic, rotic, otic = advect(loop, itic, tic, otic, *advData)

		# advect carbon dioxide
		co2, roco2, oco2 = advect(loop, ico2, co2, oco2, *advData)

		if vol > 0.0:
			twkelv = tw + 273.16

			# convert tic, co2, and alk to molar concentrations for duration of phcarb section
			tic = tic / 12000.0
			co2 = co2 / 12000.0
			alk = alk / 50000.0

			if avdepe > 0.17:
				if BENRFG == 1:           # simulate benthal release of co2
					co2 *= 12000.0        # convert co2 to mg/l for use by benth
					co2, BRCO2 =  benth(dox, anaer, BRCO2, scrfac, depcor, co2, benco2)
					co2 /= 12000.0
				else:                     # benthal release of co2 is not considered
					benco2 = 0.0

				# calculate molar saturation concentration for co2 (satco2); first, calculate
				# henry's constant, s, for co2; s is defined as the molar concentration of 
				# atmospheric co2/partial pressure of co2; cfpres corrects the equation for
				# effects of elevation differences from sea level
				s = 10.0**((2385.73 / twkelv) - 14.0184 + 0.0152642 * twkelv)
				satco2 = 3.16e-04 * cfpres * s

				# calculate increase in co2 due to atmospheric invasion;  the co2 
				# invasion is based on oxygen reaeration rate for the control volume
				kcinv  = min(cfcinv * korea, 0.999)
				inv    = kcinv * (satco2 - co2)
				invco2 = inv * 12000.0

				# calculate net molar co2 change due to co2 invasion, zooplankton
				# excretion and respiration, phytoplankton and benthic algae 
				# respiration, bod decay, and benthal release of co2
				bodco2 = decco2
				phyco2 = pyco2
				zooco2 = zoco2
				balco2 = baco2
				totco2 = invco2 + zooco2 + phyco2 + balco2 + bodco2 + benco2
				deltcd = totco2 / 12000.0

				# calculate change in total inorganic carbon balance due to net co2 change
				tic = max(tic + deltcd, 0.0)

			else:
				# too little water to warrant simulation of quality processes; calculate
				# values of co2 and ph state variables based on only longitudinal advection
				invco2 = 0.0
				zooco2 = 0.0
				phyco2 = 0.0
				balco2 = 0.0
				bodco2 = 0.0
				benco2 = 0.0
				totco2 = 0.0   # invco2 + zooco2 + phyco2 + balco2 + bodco2 + benco2

			# calculate ionization product of water
			kwequ = 10.0**(-4470.99 / twkelv + 6.0875 - 0.01706 * twkelv)

			# calculate first dissociation constant of carbonic acid
			k1equ = 10.0**(-3404.71 / twkelv + 14.8435 - 0.032786 * twkelv)

			# calculate second dissociation constant of carbonic acid
			k2equ = 10.0**(-2902.39 / twkelv + 6.4980 - 0.02379 * twkelv)

			# assign values to variables and coefficients used in the solution algorithm
			if ph < 0.0:       # it is undefined (due to no water in reach)
				ph = 7.0
			hest   = 10.0**(-ph)
			hllim  = 0.0
			hulim  = 1.0
			coeff1 = alk + k1equ
			coeff2 = -kwequ + alk * k1equ + k1equ * k2equ - tic * k1equ
			coeff3 = -2.0 * k1equ * k2equ * tic - k1equ * kwequ + alk * k1equ * k2equ
			coeff4 = -k1equ * k2equ * kwequ

			#$PHCALC()     ''' calculate ph'''
			count  = 0
			while count <= phcnt:
				count = count + 1
				# evaluate quadratic and slope for solution equation
				quadh = (((hest + coeff1) * hest + coeff2) * hest + coeff3) * hest + coeff4
				dfdh  = ((4.0 * hest + 3.0 * coeff1) * hest + 2.0 * coeff2) * hest + coeff3
				if dfdh <= 0.0: # slope of solution equation is zero or negative
					# solution for hplus is not meaningful for such a slope
					# update values for hllim, hulim, and hest to force convergence
					if quadh < 0.0:
						if hest >= hllim:
							hllim = hest
							hest  = 10.0 * hest
						elif hest <= hulim:
							hulim = hest
							hest  = 0.10 * hest
				else:   # calculate new hydrogen ion concentration
					hplus = hest - quadh / dfdh
					if abs(hplus - hest) / hplus <= 0.10:
						break
					# adjust prior	estimate for next iteration	
					if   hplus <= hllim:    hest = (hest + hllim) / 2.0
					elif hplus >= hulim:    hest = (hest + hulim) / 2.0
					else:                   hest = hplus
			else:
				pass  # ERRMSG:  a satisfactory solution for ph has not been reached		
			ph = -log10(hplus)
			# end #$PHCALC()
			
			# calculate co2 concentration (molar)
			co2 = tic / (1.0 + k1equ / hplus + k1equ * k2equ / (hplus**2))

			# convert tic, co2, and alk from moles/liter to mg/liter
			tic *= 12000.0
			co2 *= 12000.0
			alk *= 50000.0
		else:    # reach/res has gone dry during the interval; set ph equal to an undefined value
			ph     = -1.0e30
			invco2 = 0.0
			zooco2 = 0.0
			phyco2 = 0.0
			balco2 = 0.0
			bodco2 = 0.0
			benco2 = 0.0
			totco2 = 0.0  # invco2 + zooco2 + phyco2 + balco2 + bodco2 + benco2
		
		return dox, bod, ph, tic, co2, rotic, roco2, otic, oco2, totco2 
	return pphcarb
