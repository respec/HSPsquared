''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import where
from HRCHOXR import oxrx
from HRCHNUT import nutrx
from HRCHPLK import plank
from HRCHPHC import phcarb

UUNITS = 1



def RQUAL(general, ui, ts):
	''' Simulate constituents involved in biochemical transformations'''

	simlen = general['SIMLEN']
	
	BENRFG = ui['BENFGX']   # table-type benth-flag

	# table type ACTIVITY
	NUTFG = ui['NUTFG']
	PLKFG = ui['PLKFG']
	PHFG  = ui['PLKFG']
	
	# get external time series
	PREC  = ts['PREC']
	SAREA = ts['SAREA']
	AVDEP = ts['AVDEP']
	AVVEL = ts['AVVEL']
	TW    = ts['TW'] 
	if LKFG = 1:
		WIND  = ts['WIND']
	
	nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL = ui['advectData']
	
	TW     = where(TW < -100.0, 20.0, TW)  # fix undefined temps if present
	AVDEPE = where(UUNITS == 2, AVDEP * 3.28, AVDEP)  # convert to english units) in feet
	AVVELE = where(UUNITS == 2, AVVEL * 3.28, AVVEL)  # convert to english units)
	DEPCOR = where(avdepe > 0.0, 3.28084e-3 / AVDEPE, -1.e30)  # # define conversion factor from mg/m2 to mg/l
	if BENRFG == 1: 
		scrvel = ui['SCRVEL']   # table-type scour-parms
		scrmul = ui['SCRMUL']   # table-type scour-parms
		SCRFAC = where(AVVELE > scrvel, scrmul, 1.0)   # calculate scouring factor
	
	# use Closures to capture 'ui' data to minimize calling arguments.
	#### OXRX  ####
	
	IDOX  = ts['IDOX']  # optional, input flow
	IBOD  = ts['IBOD']  # optional, input flow	
	
	# preallocate storage for OXRX calculated results
	DOX   = ts['DOX']   = zeros(simlen)   # concentration, state variable
	BOD   = ts['BOD']   = zeros(simlen)   # concentration, state variable
	SATDO = ts['SATDO'] = zeros(simlen)   # concentration, state variable
	RODOX = ts['RODOX'] = zeros(simlen)             # reach outflow of DOX
	ROBOD = ts['ROBOD'] = zeros(simlen)             # reach outflow of BOD
	ODOX  = ts['ODOX']  = zeros((simlen, nexits))   # reach outflow per gate of DOX
	OBOD  = ts['OBOD']  = zeros((simlen, nexits))   # reach outflow per gate of BOD
	
	oxrx = poxrx()  # returns Numba accelerated function in closure
	
	if NUTFG:
		# get NUTRX specific time series
		INO3 = ts['INO3']   # optional, input
		INH3 = ts['INH3']   # optional, input
		INO2 = ts['INO2']   # optional, input
		IPO4 = ts['IPO4']   # optional, input
		
		NUAFX = setit()  # NUAFXM monthly, constant or time series
		NUACN = setit()  # NUACNM monthly, constant or time series
		
		# preallocate storage for computed time series
		NO3   = ts['NO3']   = zeros(simlen)   # concentration, state variable
		NO2   = ts['NO2']   = zeros(simlen)   # concentration, state variable
		NH3   = ts['NH3']   = zeros(simlen)   # concentration, state variable
		PO4   = ts['PO4']   = zeros(simlen)   # concentration, state variable 
		TAM   = ts['TAM']   = zeros(simlen)   # concentration, state variable
		RONO3 = ts['RONO3'] = zeros(simlen)   # outflow
		RONO2 = ts['RONO2'] = zeros(simlen)   # outflow
		RONH3 = ts['RONH3'] = zeros(simlen)   # outflow
		ROPO4 = ts['ROPO4'] = zeros(simlen)   # outflow
		ONO3  = ts['ONO3']  = zeros((simlen, NEXITS))   # outflow
		ONO2  = ts['ONO2']  = zeros((simlen, NEXITS))   # outflow
		ONH3  = ts['ONH3']  = zeros((simlen, NEXITS))   # outflow
		OPO4  = ts['OPO4']  = zeros((simlen, NEXITS))   # outflow
		 
		nutrx = pnutrx()  # returns Numba accelerated function in closure		
		
	if PLKFG:
		# get PLANK specific time series
		IPHYTO = ts['IPHYTO']   # optional
		IZOO   = ts['IZOO']     # optional
		IORN   = ts['IORN']     # optional
		IORP   = ts['IORP']     # optional
		IORC   = ts['IORC']     # optional
		WASH   = ts['WASH']
		SOLRAD = ts['SOLRAD']
		 
		# preallocate arrays for better performance
		ORN    = ts['PKST3_ORN']    = zeros(simlen)  # state variable
		ORP    = ts['PKST3_ORP']    = zeros(simlen)  # state variable
		ORC    = ts['PKST3_ORC']    = zeros(simlen)  # state variable
		TORN   = ts['PKST3_TORN']   = zeros(simlen)  # state variable
		TORP   = ts['PKST3_TORP']   = zeros(simlen)  # state variable
		TORC   = ts['PKST3_TORC']   = zeros(simlen)  # state variable
		POTBOD = ts['PKST3_POTBOD'] = zeros(simlen)  # state variable
		
		PHYTO  = ts['PHYTO']        = zeros(simlen)  # concentration
		ZOO    = ts['ZOO']          = zeros(simlen)  # concentration
		BENAL  = ts['BENAL']        = zeros(simlen)  # concentration
		PHYCLA = ts['PHYCLA']       = zeros(simlen)  # concentration
		BALCLA = ts['BALCLA']       = zeros(simlen)  # concentration
		
		ROPHYTO  = ts['ROPHYTO']  = zeros(simlen)  # total outflow
		ROZOO    = ts['ROZOO']    = zeros(simlen)  # total outflow
		ROBENAL  = ts['ROBENAL']  = zeros(simlen)  # total outflow
		ROPHYCLA = ts['ROPHYCLA'] = zeros(simlen)  # total outflow
		ROBALCLA = ts['ROBALCLA'] = zeros(simlen)  # total outflow
		
		OPHYTO  = ts['OPHYTO']  = zeros((simlen, nexits)) # outflow by gate	
		OZOO    = ts['OZOO']    = zeros((simlen, nexits)) # outflow by gate	 	
		OBENAL  = ts['OBENAL']  = zeros((simlen, nexits)) # outflow by gate	 	
		OPHYCLA = ts['OPHYCLA'] = zeros((simlen, nexits)) # outflow by gate	 	
		OBALCLA = ts['OBALCLA'] = zeros((simlen, nexits)) # outflow by gate	 	
		
		BINV   = setit()   # ts (BINVFG==1), monthly (BINVFG)
		PLADFX = setit()   # time series, monthly(PLAFXM)
		PLADCN = setit()   # time series, monthly(PLAFXM)		
		
		plank = pplank()  # returns Numba accelerated function in closure
	
	if PHFG:
		# get PHCARB() specific external time series
		ALK  = ts['CON']    # ALCON only
		ITIC = ts['ITIC']   # or zero if not present
		ICO2 = ts['ICO2']   # or zero if not present

		
		# preallocate output arrays for speed
		PH     = ts['PH']    = zeros(simlen)            # state variable
		TIC    = ts['TIC']   = zeros(simlen)            # state variable
		CO2    = ts['CO2']   = zeros(simlen)            # state variable
		ROTIC  = ts['ROTIC'] = zeros(simlen)            # reach total outflow
		ROCO2  = ts['ROCO2'] = zeros(simlen)            # reach total outflow
		OTIC   = ts['OTIC']  = zeros((simlen, nexits))  # outflow by exit
		OCO2   = ts['OCO2']  = zeros((simlen, nexits))  # outflow by exit
		TOTCO2 = ts['TOTCO2'] = zeros(simlen)            #  ??? computed, but not returned???			

		phcarb = pphcarb()  # returns Numba accelerated function in closure
		
	############## master simulation loop  #####################	


	for loop in range(simlen):
		avdepe = AVDEPE[loop]
		avvele = AVVELE[loop]
		tw     = TW[loop]
		depcor = DEPCOR[loop]
		
		advData = nexits, vol, VOL[loop], SROVOL[loop], EROVOL[loop], SOVOL[loop], EOVOL[loop]		
		
		# simulate primary do and bod balances
		(dox, bod, satdo, rodo, robod, odo, 
		obod) = oxrx(
		IDOX[loop], IBOD[loop], WIND[loop], avdepe, avvele, tw, depcor, BENRFG, advData)   
		
		
	
		if NUTFG == 1:  # simulate primary inorganic nitrogen and phosphorus balances
			(dox, bod, NO3[loop], NO2[loop], NH3[loop], PO4[loop], TAM[loop], RONO3[loop],
			 RONO2[loop], RONH3[loop], ROPO4[loop], ONO3[loop], ONO2[loop], ONH3[loop],
			 OPO4[loop]) = nutrx(
			 dox, bod, tw, INO3[loop], INH3[loop], INO2[loop], IPO4[loop], NUAFX[loop],
			 NUACN[loop], PREC[loop], SAREA[loop], advData) 
			
			if PLKFG == 1:    # simulate plankton populations and associated reactions
				(dox, bod, ORN[loop], ORP[loop], ORC[loop], TORN[loop], TORP[loop],
				 TORC[loop], POTBOD[loop], PHYTO[loop], ZOO[loop], BENAL[loop], 
				 PHYCLA[loop], BALCLA[loop], ROPHYTO[loop], ROZOO[loop], ROBENAL[loop],
				 ROPHYCLA[loop], ROBALCLA[loop], OPHYTO[loop], OZOO[loop], 
				 OBENAL[loop], OPHYCLA[loop], OBALCLA[loop], BINV[loop], PLADFX[loop], 
				 PLADCN[loop]) = plank(
				 dox, bod, IPHYTO[loop], IZOO[loop], IORN[loop], IORP[loop], 
				 IORC[loop], tw, WASH[loop], SOLRAD[loop], PREC[loop], SAREA[loop], advData)
				
				if PHFG == 1:   # simulate ph and carbon species
					
					(dox, bod, PH[loop], TIC[loop], CO2[loop], ROTIC[loop],
					 ROCO2[loop], OTIC[loop], OCO2[loop], TOTCO2[loop], 	
					 ) = phcarb(
					 dox, bod, ALK[loop], ITIC[loop], ICO2[loop], tw, avdepe, SCRFAC[loop],  advData)
	
			
			# update totals of nutrients
			rno3  = no3 * vol
			rtam  = tam * vol
			rno2  = no2 * vol
			rpo4  = po4 * vol
			rnh4  = nh4 * vol
			rnh3  = nh3 * vol
			rrno3 = no3 * vol
			rrtam = tam * vol
			if ADNHFG == 1:  
				rrtam += rsnh4(4)  # add adsorbed suspended nh4 to dissolved
			
			rrno2 = no2 * vol
			rrpo4 = po4 * vol
			if ADPOFG == 1:  
				rrpo4 += rspo4(4) # add adsorbed suspended po4 to dissolved

		# check do level; if dox exceeds user specified level of supersaturation, then release excess do to the atmosphere
		doxs = dox
		if dox > supsat * satdo:
			dox = supsat * satdo
		readox = readox + (dox - doxs) * vol
		totdox = readox + boddox + bendox + nitdox + phydox + zoodox + baldox
		
		# update dissolved totals and totals of nutrients
		rdox = dox * vol
		rbod = bod * vol
	return


#@jit(nopython=True)
def benth (dox, anaer, BRCON, scrfac, depcor, conc):
	''' simulate benthal release of constituent'''
	# calculate benthal release of constituent; release is a step function of aerobic/anaerobic conditions, and stream velocity;
	# scrfac, the scouring factor dependent on stream velocity and depcor, the conversion factor from mg/m2 to mg/l,
	# both calculated in rqual; releas is expressed in mg/m2.ivl
	releas = BRCON[0] * scrfac * depcor  if dox > anaer else BRCON[1] * scrfac * depcor
	conc  += releas
	return conc, releas


#@jit(nopython=True)
def decbal(TAMFG, PO4FG, decnit, decpo4, tam, no3, po4):
	''' perform materials balance for transformation from organic to inorganic material by decay in reach water'''
	if TAMFG:
		tam += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
	else:
		no3 += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
	if PO4FG:   # add phosphorus transformed to inorganic phosphorus by biomass decomposition to po4 state variable
		po4 += decpo4
	return tam, no3, po4


#@jit(nopython=True)
def sink (vol, avdepe, kset, conc, snkmat):
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
