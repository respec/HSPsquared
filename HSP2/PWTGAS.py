''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERGAS.FOR module into Python''' 


from numpy import zeros, where
from numba import jit
from HSP2  import initm


ERRMSG = []

# english system
# parameters for variables with energy units
EFACTA = 407960.
EFACTB = 0.0

# parameters for variables with temperature units
TFACTA = 1.8
TFACTB = 32.0

# parameters for variables for dissolved gases with mass units
MFACTA = 0.2266
MFACTB = 0.
	
	
def pwtgas(store, general, ui, ts):
	''' Estimate water temperature, dissolved oxygen, and carbon dioxide in the outflows
	from a pervious landsegment. calculate associated fluxes through exit gates'''

	sim_len = general['SIM_LEN']
	delt    = general['DELT']
	tindex  = general['tindex']

	elevgc = ((288.0 - 0.00198 * ui['ELEV'])  /288.0)**5.256

	BCVFG  = ui['BCVFG']
	doxp   = ui['DOXP']
	adoxp  = ui['ADOXP']
	aco2P  = ui['ACO2P']
	sotmp  = ui['SOTMP']
	iotmp  = ui['IOTMP']
	aotmp  = ui['AOTMP']
	sodox  = ui['SODOX']
	soco2  = ui['SOCO2']
	iodox  = ui['IODOX']
	ioco2  = ui['IOCO2']
	aodox  = ui['AODOX']
	aoco2  = ui['AOCO2']
	sdlfac = ui['SDLFAC']
	slifac = ui['SLIFAC']
	ilifac = ui['ILIFAC']
	alifac = ui['ALIFAC']

	initm(general, ui, ts, 'IDVFG', 'IDOXPM', 'IDOXP')
	initm(general, ui, ts, 'ICVFG', 'ICO2PM', 'ICO2P')
	initm(general, ui, ts, 'GDVFG', 'ADOXPM', 'ADOXP')
	initm(general, ui, ts, 'GCVFG', 'ACO2PM', 'ACO2P')
	IDOXP = ts['IDOXP']
	ICO2P = ts['ICO2P']
	ADOXP = ts['ADOXP']
	ACO2P = ts['ACO2P']
	
	for name in ['WYIELD', 'SURO', 'IFWO', 'AGWO', 'SURLI', 'IDWLI', 'AGWLI']:
		if name not in ts:
			ts[name] = zeros(simlen)
	WYIELD = ts['WYIELD']
	SURO   = ts['SURO']
	IFWO   = ts['IFWO']
	AGWO   = ts['AGWO']
	SURLI  = ts['SURLI']
	IDWLI  = ts['IDWLI']
	AGWLI  = ts['AGWLI']	
	
	for name in ['SLTMP', 'ULTMP', 'LGTMP', 'SLITMP', 'SLIDOX', 'SLICO2']:
		if name not in ts:
			ts[name] = full(simlen, -1.0E30)
	SLTMP  = ts['SLTMP']
	ULTMP  = ts['ULTMP']
	LGTMP  = ts['LGTMP']
	SLITMP = ts['SLITMP']
	SLIDOX = ts['SLIDOX']
	SLICO2 = ts['SLICO2']

	# preallocate output arrays
	SPTMP  = ts['SPTMP']  = zeros(sim_len)
	IOTMP  = ts['IOTMP']  = zeros(sim_len)
	APTMP  = ts['APTMP']  = zeros(sim_len)
	SODOX  = ts['SODOX']  = zeros(sim_len)
	SOCO2  = ts['SOCO2']  = zeros(sim_len)
	IODOX  = ts['IODOX']  = zeros(sim_len)
	IOCO2  = ts['IOCO2']  = zeros(sim_len)
	AODOX  = ts['AODOX']  = zeros(sim_len)
	AOCO2  = ts['AOCO2']  = zeros(sim_len)
	SOHT   = ts['SOHT']   = zeros(sim_len)
	IOHT   = ts['IOHT']   = zeros(sim_len)
	AOHT   = ts['AOHT']   = zeros(sim_len)
	POHT   = ts['POHT']   = zeros(sim_len)
	SODOXM = ts['SODOXM'] = zeros(sim_len)
	IODOXM = ts['IODOXM'] = zeros(sim_len)
	IOCO2M = ts['IOCO2M'] = zeros(sim_len)
	AODOXM = ts['AODOXM'] = zeros(sim_len)
	AOCO2M = ts['AOCO2M'] = zeros(sim_len)
	PODOXM = ts['PODOXM'] = zeros(sim_len)
	POCO2M = ts['POCO2M'] = zeros(sim_len)

	DAYFG = where(tindex.hour==1, True, False)   # ??? need to check if minute == 0	
	
	for loop in range(sim_len):
		dayfg   = DAYFG[loop]
		suro   = SURO[loop]
		wyield = WYIELD[loop]
		ifwo   = IFWO[loop]
		agwo   = AGWO[loop]
		surli  = SURLI[loop]
		idwli  = IDWLI[loop]
		agwli  = AGWLI[loop]
		
		sotmp = -1.0e30
		sodox = -1.0e30
		soco2 = -1.0e30		
		if suro > 0.0:  # there is surface outflow
			# local surface outflow temp equals surface soil temp
			sotmp = sltmp
			if sotmp < 0.5:
				sotmp = 0.5  # min water temp
			if CSNOFG:      # effects of snow are considered
				# adjust surface outflow temperature if snowmelt is occurring
				if wyield > 0.0:
					sotmp = 0.5  # snowmelt is occuring - use min temp

			# oxygen calculation
			dummy = sotmp * (0.007991 - 0.77774E-4 * sotmp)
			sodox = (14.652 + sotmp * (-0.41022 + dummy)) * elevgc

			# carbon dioxide calculation
			abstmp = sotmp + 273.16
			dummy = 2385.73 / abstmp - 14.0184 + 0.0152642 * abstmp
			soco2 = 10.0**dummy * 3.16e-04 * elevgc * 12000.0

			if surli > 0.0 and slifac > 0.0:  # check for effects of lateral inflow
				if slitmp >= -1.0e10:   # there is temperature of surface lateral inflow
					sotmp = slitmp * slifac + sotmp * (1.0 - slifac)
				if slidox >= 0.0:    # there is do conc of surface lateral inflow
					sodox = slidox * slifac + sodox * (1.0 - slifac)
				if slico2 >= 0.0:  #there is co2 conc of surface lateral inflow
					soco2 = slico2 * slifac + soco2 * (1.0 - slifac)

		# get interflow lateral inflow temp and concentrations
		ilitmp = -1.0e30
		ilidox = -1.0e30
		ilico2 = -1.0e30
		if ifwli > 0.0:    # there is lateral inflow
			ilitmp = ILITMP[loop]
			ilidox = ILIDOX[loop]
			ilico2 = ILICO2[loop]

		if dayfg:    # it is the first interval of the day
			idoxp = IDOXP[loop]
			ico2p = ICO2P[loop]
			
		iotmp = -1.0e30
		iodox = -1.0e30
		ioco2 = -1.0e30
		if ifwo > 0.0:   # there is interflow outflow
			# local interflow outflow temp equals upper soil temp
			iotmp = ultmp
			if iotmp < 0.5:
				iotmp = 0.5    # min water temp

			iodox = idoxp
			ioco2 = ico2p
			
			if ifwli > 0.0 and ilifac > 0.0:
				# check for effects of lateral inflow
				if ilitmp >= -1.0e10:   # there is temperature of interflow lateral inflow
					iotmp = ilitmp * ilifac + iotmp * (1.0 - ilifac)
				if ilidox >= 0.0:       # there is do conc of interflow lateral inflow
					iodox = ilidox * ilifac + iodox * (1.0 - ilifac)
				if ilico2 >= 0.0:       # there is co2 conc of interflow lateral inflow
					ioco2 = ilico2 * ilifac + ioco2 * (1.0 - ilifac)

		# get baseflow lateral inflow temp and concentrations
		alitmp = -1.0e30
		alidox = -1.0e30
		alico2 = -1.0e30
		if agwli > 0.0:
			alitmp = ALITMP[loop]
			alidox = ALIDOX[loop]
			alico2 = ALICO2[loop]


		if dayfg:    #it is the first interval of the day
			if GCVFG:
				adoxp = ADOXP[loop]
				aco2p = ACO2P[loop]

		aotmp = -1.0e30
		aodox = -1.0e30
		aoco2 = -1.0e30
		if agwo > 0.0:   # there is baseflow
			aotmp = LGTMP[loop]  		# local baseflow temp equals lower/gw soil temp
			if aotmp < 0.5:     # min water temp
				aotmp = 0.5
			
			aodox = adoxp
			aoco2 = aco2p
			if agwli > 0.0 and alifac > 0.0:   #check for effects of lateral inflow
				if alitmp >= -1.0e10:   # there is temperature of baseflow lateral inflow
					aotmp = alitmp * alifac + aotmp * (1.0 - alifac)
				if alidox >= 0.0:	# there is do conc of baseflow lateral inflow
					aodox = alidox * alifac + aodox * (1.0 - alifac)
				if alico2 >= 0.0:    # there is co2 conc of baseflow lateral inflow
					aoco2 = alico2 * alifac + aoco2 * (1.0 - alifac)

		# compute the outflow of heat energy in water - units are deg. c-in./ivl
		soht = sotmp * suro
		ioht = iotmp * ifwo
		aoht = aotmp * agwo
		POHT[loop] = soht + ioht + aoht

		# calculate outflow mass of dox - units are mg-in./l-ivl
		sodoxm = sodox * suro
		iodoxm = iodox * ifwo
		aodoxm = aodox * agwo
		PODOXM[loop] = sodoxm + iodoxm + aodoxm

		# calculate outflow mass of co2 - units are mg-in./l-ivl
		soco2m = soco2 * suro
		ioco2m = ioco2 * ifwo
		aoco2m = aoco2 * agwo
		POCO2M[loop] = soco2m + ioco2m + aoco2m

    	SPTMP[loop]  = sptmp
    	IOTMP[loop]  = iotmp
    	APTMP[loop]  = aptmp
    	SODOX[loop]  = sodox
    	SOCO2[loop]  = soco2
    	IODOX[loop]  = iodox
    	IOCO2[loop]  = ioco2
    	AODOX[loop]  = aodox
    	AOCO2[loop]  = aoco2
    	SOHT[loop]   = soht
    	IOHT[loop]   = ioht
    	AOHT[loop]   = aoht

    	SODOXM[loop] = sodoxm
    	IODOXM[loop] = iodoxm
    	IOCO2M[loop] = ioco2m
    	AODOXM[loop] = aodoxm
    	AOCO2M[loop] = aoco2m

	return errorsV, ERRMSG