''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import where, zeros, array, float64
from numba import types
from numba.typed import Dict

from HSP2.utilities  import make_numba_dict
from HSP2.OXRX_Class import OXRX_Class
from HSP2.NUTRX_Class import NUTRX_Class
#from HSP2.PLANK_Class import PLANK_Class
#from HSP2.PHCARB import phcarb

ERRSMGS = ('Placeholder')

def rqual(store, siminfo, uci, uci_oxrx, uci_nutrx, uci_plank, ts):
	''' Simulate constituents involved in biochemical transformations'''

	# simulation information:
	delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	delts  = siminfo['delt'] * 60
	uunits = siminfo['units']

	siminfo_ = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
	for key in set(siminfo.keys()):
		value = siminfo[key]

		if type(value) in {int, float}:
			siminfo_[key] = float(value)
	
	# numba dictionary:
	ui = make_numba_dict(uci)

	BENRFG = int(ui['BENRFG'])   # table-type benth-flag

	# table type ACTIVITY
	NUTFG = int(ui['NUTFG'])
	PLKFG = int(ui['PLKFG'])
	PHFG  = int(ui['PHFG'])

	LKFG = int(ui['LKFG'])
	
	# get external time series
	PREC  = ts['PREC']
	SAREA = ts['SAREA']
	AVDEP = ts['AVDEP']
	AVVEL = ts['AVVEL']
	TW    = ts['TW'] 
	if LKFG == 1:
		WIND  = ts['WIND']
	else:
		WIND = zeros(simlen)
	
	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData
	
	TW     = where(TW < -100.0, 20.0, TW)  # fix undefined temps if present
	AVDEPE = where(uunits == 2, AVDEP * 3.28, AVDEP)  # convert to english units) in feet
	AVVELE = where(uunits == 2, AVVEL * 3.28, AVVEL)  # convert to english units)
	DEPCOR = where(AVDEPE > 0.0, 3.28084e-3 / AVDEPE, -1.e30)  # # define conversion factor from mg/m2 to mg/l
	if BENRFG == 1: 
		scrvel = ui['SCRVEL']   # table-type scour-parms
		scrmul = ui['SCRMUL']   # table-type scour-parms
		SCRFAC = where(AVVELE > scrvel, scrmul, 1.0)   # calculate scouring factor
	
	ts['SCRFAC'] = SCRFAC

	#### OXRX  ####
	
	if ('OXIF1' in ts):
		IDOX  = ts['OXIF1']  # optional, input flow
	else:
		IDOX = zeros(simlen)
	
	if ('OXIF2' in ts):
		IBOD  = ts['OXIF2']  # optional, input flow	
	else:
		IBOD = zeros(simlen)

	# preallocate storage for OXRX calculated results
	DOX   = ts['DOX']   = zeros(simlen)   # concentration, state variable
	BOD   = ts['BOD']   = zeros(simlen)   # concentration, state variable
	SATDO = ts['SATDO'] = zeros(simlen)   # concentration, state variable
	RODOX = ts['RODOX'] = zeros(simlen)             # reach outflow of DOX
	ROBOD = ts['ROBOD'] = zeros(simlen)             # reach outflow of BOD
	ODOX  = zeros((simlen, nexits))   # reach outflow per gate of DOX
	OBOD  = zeros((simlen, nexits))   # reach outflow per gate of BOD

	for i in range(nexits):
		ts['ODOX' + str(i + 1)] = zeros(simlen)
		ts['OBOD' + str(i + 1)] = zeros(simlen)

	# instantiate OXRX class:	
	ui_oxrx = make_numba_dict(uci_oxrx)
	OXRX = OXRX_Class(siminfo_, advectData, ui, ui_oxrx, ts)

	#oxrx = poxrx()  # returns Numba accelerated function in closure
	
	#NUTFG = 0		#TMR!!!
	if NUTFG:

		ui_nutrx = make_numba_dict(uci_nutrx)

		# get NUTRX specific input time series
		INO3 = zeros(simlen); INO2 = zeros(simlen)
		INH3 = zeros(simlen); IPO4 = zeros(simlen)

		if 'NUIF11' in ts:
			INO3 = ts['NUIF11']   # optional, input

		if 'NUIF12' in ts:				
			INH3 = ts['NUIF12']   # optional, input
		
		if 'NUIF13' in ts:
			INO2 = ts['NUIF13']   # optional, input

		if 'NUIF14' in ts:
			IPO4 = ts['NUIF14']   # optional, input

		# atmospheric deposition - create time series:
		NUADFX = zeros((simlen,4))
		NUADCN = zeros((simlen,4))
		

		#LTI NUAFX = setit()  # NUAFXM monthly, constant or time series
		#LTI NUACN = setit()  # NUACNM monthly, constant or time series

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
		ONO3  = zeros((simlen, nexits))   # outflow
		ONO2  = zeros((simlen, nexits))   # outflow
		ONH3  = zeros((simlen, nexits))   # outflow
		OPO4  = zeros((simlen, nexits))   # outflow

		for i in range(nexits):
			ts['ONO3' + str(i + 1)] = zeros(simlen)
			ts['ONO2' + str(i + 1)] = zeros(simlen)
			ts['ONH3' + str(i + 1)] = zeros(simlen)
			ts['OPO4' + str(i + 1)] = zeros(simlen)

		NUTRX = NUTRX_Class(siminfo_, advectData, ui, ui_nutrx, ts, 
							OXRX.dox, OXRX.bod, OXRX.korea)

	PLKFG = 0		#LTI!!!
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
		
		OPHYTO  = zeros((simlen, nexits)) # outflow by gate	
		OZOO    = zeros((simlen, nexits)) # outflow by gate	 	
		OBENAL  = zeros((simlen, nexits)) # outflow by gate	 	
		OPHYCLA = zeros((simlen, nexits)) # outflow by gate	 	
		OBALCLA = zeros((simlen, nexits)) # outflow by gate	 	
		
		for i in range(nexits):
			ts['OPHYTO' + str(i + 1)] = zeros(simlen)
			ts['OZOO' + str(i + 1)] = zeros(simlen)
			ts['OBENAL' + str(i + 1)] = zeros(simlen)
			ts['OPHYCLA' + str(i + 1)] = zeros(simlen)
			ts['OBALCLA' + str(i + 1)] = zeros(simlen)

		BINV   = setit()   # ts (BINVFG==1), monthly (BINVFG)
		PLADFX = setit()   # time series, monthly(PLAFXM)
		PLADCN = setit()   # time series, monthly(PLAFXM)		
		
		ui_plank = make_numba_dict(uci_plank)
		PLANK = PLANK_Class(store, siminfo, uci, ui_plank, ts, OXRX, NUTRX)

		#plank = pplank()  # returns Numba accelerated function in closure
	
	PHYFG = 0	#LTI!!!
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
		OTIC   = zeros((simlen, nexits))  # outflow by exit
		OCO2   = zeros((simlen, nexits))  # outflow by exit
		TOTCO2 = ts['TOTCO2'] = zeros(simlen)            #  ??? computed, but not returned???			

		for i in range(nexits):
			ts['OTIC' + str(i + 1)] = zeros(simlen)
			ts['OCO2' + str(i + 1)] = zeros(simlen)

		phcarb = pphcarb()  # returns Numba accelerated function in closure
		
	############## master simulation loop  #####################	


	for loop in range(simlen):
		avdepe = AVDEPE[loop]
		avvele = AVVELE[loop]
		tw     = TW[loop]
		if uunits == 1:
			tw = (tw - 32.0) * (5.0 / 9.0)

		depcor = DEPCOR[loop]
		
		advData = nexits, vol, VOL[loop], SROVOL[loop], EROVOL[loop], SOVOL[loop], EOVOL[loop]		
		
		# simulate primary do and bod balances
		OXRX.simulate(IDOX[loop], IBOD[loop], WIND[loop], SCRFAC[loop], avdepe, avvele, depcor, tw, advData)
	
		if NUTFG == 1:  # simulate primary inorganic nitrogen and phosphorus balances
			OXRX = NUTRX.simulate(tw, OXRX.dox, OXRX.bod, 
									INO3[loop], INH3[loop], INO2[loop], IPO4[loop], 
			 						NUAFX[loop], NUACN[loop], PREC[loop], SAREA[loop], advData)

			
			if PLKFG == 1:    # simulate plankton populations and associated reactions
				(OXRX, NUTRX) = PLANK.simulate(OXRX, NUTRX, tw, IPHYTO[loop], IZOO[loop], 
												IORN[loop], IORP[loop], IORC[loop], WASH[loop], SOLRAD[loop], PREC[loop], SAREA[loop], advData)

				'''
				(dox, bod, ORN[loop], ORP[loop], ORC[loop], TORN[loop], TORP[loop],
				 TORC[loop], POTBOD[loop], PHYTO[loop], ZOO[loop], BENAL[loop], 
				 PHYCLA[loop], BALCLA[loop], ROPHYTO[loop], ROZOO[loop], ROBENAL[loop],
				 ROPHYCLA[loop], ROBALCLA[loop], OPHYTO[loop], OZOO[loop], 
				 OBENAL[loop], OPHYCLA[loop], OBALCLA[loop], BINV[loop], PLADFX[loop], 
				 PLADCN[loop]) = plank_run(
				 dox, bod, IPHYTO[loop], IZOO[loop], IORN[loop], IORP[loop], 
				 IORC[loop], rsnh4, rspo4, tw, WASH[loop], SOLRAD[loop], PREC[loop], SAREA[loop], advData)
				 '''
				
				if PHFG == 1:   # simulate ph and carbon species
					
					(dox, bod, PH[loop], TIC[loop], CO2[loop], ROTIC[loop],
					 ROCO2[loop], OTIC[loop], OCO2[loop], TOTCO2[loop], 	
					 ) = phcarb(
					 dox, bod, ALK[loop], ITIC[loop], ICO2[loop], tw, avdepe, SCRFAC[loop],  advData)
	
				# check do level; if dox exceeds user specified level of supersaturation, then release excess do to the atmosphere
				OXRX.adjust_dox(vol, NUTRX.nitdox, PLANK.phydox, PLANK.zoodox, PLANK.baldox)
			
			# update totals of nutrients
			NUTRX.updateMass()

	return


# #@jit(nopython=True)
# def benth (dox, anaer, BRCON, scrfac, depcor, conc):
# 	''' simulate benthal release of constituent'''
# 	# calculate benthal release of constituent; release is a step function of aerobic/anaerobic conditions, and stream velocity;
# 	# scrfac, the scouring factor dependent on stream velocity and depcor, the conversion factor from mg/m2 to mg/l,
# 	# both calculated in rqual; releas is expressed in mg/m2.ivl
# 	releas = BRCON[0] * scrfac * depcor  if dox > anaer else BRCON[1] * scrfac * depcor
# 	conc  += releas
# 	return conc, releas


# #@jit(nopython=True)
# def decbal(TAMFG, PO4FG, decnit, decpo4, tam, no3, po4):
# 	''' perform materials balance for transformation from organic to inorganic material by decay in reach water'''
# 	if TAMFG:
# 		tam += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
# 	else:
# 		no3 += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
# 	if PO4FG:   # add phosphorus transformed to inorganic phosphorus by biomass decomposition to po4 state variable
# 		po4 += decpo4
# 	return tam, no3, po4


# #@jit(nopython=True)
# def sink (vol, avdepe, kset, conc, snkmat):
# 	''' calculate quantity of material settling out of the control volume; determine the change in concentration as a result of sinking'''
# 	if kset > 0.0 and avdepe > 0.17:
# 		# calculate concentration change due to outgoing material; snkout is expressed in mass/liter/ivl; kset is expressed as ft/ivl and avdepe as feet
# 		snkout = conc * (kset / avdepe)  if kset < avdepe else conc  # calculate portion of material which settles out of the control volume during time step; snkout is expressed as mass/liter.ivl; conc is the concentration of material in the control volume
# 		conc  -= snkout        # calculate remaining concentration of material in the control volume
# 		snkmat = snkout * vol    # find quantity of material that sinks out; units are  mass.ft3/l.ivl in english system, and mass.m3/l.ivl in metric system
# 	else:
# 		snkout = 0.0
# 		snkmat = 0.0		
# 	return conc, snkmat
