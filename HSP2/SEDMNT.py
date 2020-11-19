''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2

Conversion of HSPF HPERSED.FOR module into Python''' 

from numpy import zeros, where, int64
from numba import njit
from HSP2.utilities  import initm, make_numba_dict, hourflag

ERRMSG = []


# english system
MFACTA = 1.0

'''
PDETS= DETS*MFACTA # convert dimensional variables to external units
'''

def sedmnt(store, siminfo, uci, ts):
	''' Produce and remove sediment from the land surface'''
	
	errorsV = zeros(len(ERRMSG), dtype=int)

	delt   = siminfo['delt']
	simlen = siminfo['steps']
	tindex = siminfo['tindex']

	ui = make_numba_dict(uci)  # Note: all values converted to float automatically
	VSIVFG = ui['VSIVFG']
	SDOPFG = ui['SDOPFG']
	CSNOFG = int(ui['CSNOFG'])
	smpf   = ui['SMPF']
	krer   = ui['KRER']
	jrer   = ui['JRER']
	affix  = ui['AFFIX']
	nvsi   = ui['NVSI'] * delt / 1440.
	kser   = ui['KSER']
	jser   = ui['JSER']
	kger   = ui['KGER']
	jger   = ui['JGER']

	u = uci['PARAMETERS']
	ts['COVERI'] = initm(siminfo, uci, u['CRVFG'], 'MONTHLY_COVER', u['COVER'])
	COVERI = ts['COVERI']
	cover = COVERI[0]   # for numba
	ts['NVSI'] = initm(siminfo, uci, u['VSIVFG'], 'MONTHLY_NVSI', u['NVSI'])
	NVSI = ts['NVSI'] * delt / 1440.

	if 'RAINF' in ts:
		RAIN = ts['RAINF']
	else:
		RAIN = ts['PREC']
		
	for name in ['PREC', 'SLSED', 'SNOCOV', 'SURO', 'SURS']:
		if name not in ts:
			ts[name] = zeros(simlen)
	PREC   = ts['PREC']
	SLSED  = ts['SLSED']
	SNOCOV = ts['SNOCOV']
	SURO   = ts['SURO']
	SURS   = ts['SURS']
	
	# preallocate output arrays
	DETS  = ts['DETS']  = zeros(simlen)
	WSSD  = ts['WSSD']  = zeros(simlen)
	SCRSD = ts['SCRSD'] = zeros(simlen)
	SOSED = ts['SOSED'] = zeros(simlen)
	COVER = ts['COVER'] = zeros(simlen)

        # HSPF 12.5 has only one sediment block
	dets = ui['DETS']

	# BLOCK SPECIFIC VALUES
	# nblks = ui['NBLKS']
	
	# ''' the initial storages are repeated for each block'''
	# sursb = ui['SURSB']
	# uzsb  = ui['UZSB']
	# ifwsb = ui['IFWSB']
	# detsb  = ui['DETSB']
	
	# if nblks > 1:
	#	SUROB  = ts['SIROB']  # if NBLKS > 1
	#	SURB   = ts['SURSB']  # if NBLKS > 1

	#	DETSB  = ts['DETSB']  = zeros(nblks, simlen)
	#	WSSDB  = ts['WSSDB']  = zeros(nblks, simlen)
	#	SCRSDB = ts['SCRSDB'] = zeros(nblks, simlen)
	#	SOSDB  = ts['SOSDB']  = zeros(nblks, simlen)

	# initialize sediment transport capacity
	surs = SURS[0]
	delt60 = delt / 60.0  # simulation interval in hours
	stcap = delt60 * kser * (surs / delt60)**jser if SDOPFG else 0.0

	# DAYFG = where(tindex.hour==1, True, False)   # ??? need to check if minute == 0
	DAYFG = hourflag(siminfo, 0, dofirst=True).astype(bool)
	
	DRYDFG = 1
	
	for loop in range(simlen):
		dayfg  = DAYFG[loop]
		rain   = RAIN[loop]
		prec   = PREC[loop]
		suro   = SURO[loop]
		surs   = SURS[loop]
		snocov = SNOCOV[loop]
		slsed  = SLSED[loop]
		
		if dayfg:
			cover = COVERI[loop]
		
		# estimate the quantity of sediment particles detached from the soil surface by rainfall and augment the detached sediment storage
		# detach()
		if rain > 0.0:   # simulate detachment because it is raining
			# find the proportion of area shielded from raindrop impact by snowpack and other cover
			if CSNOFG == 1:     # snow is being considered
				cr = cover + (1.0 - cover) * snocov  if snocov > 0.0 else cover 
			else:
				cr = cover

			# calculate the rate of soil detachment, delt60= delt/60 - units are tons/acre-ivl
			det = delt60 * (1.0 - cr) * smpf * krer * (rain / delt60)**jrer
			dets = dets + det   # augment detached sediment storage - units are tons/acre
		else:    # no rain - either it is snowing or it is "dry"
			det = 0.0
		# end detach()
		
		if dayfg:     # it is the first interval of the day
			nvsi = NVSI[loop]
			if VSIVFG == 2 and DRYDFG:  # last day was dry, add a whole days load in first interval detailed output will show load added over the whole day.;
				dummy = nvsi * (1440. / delt)
				dets  = dets + dummy

		# augment the detached sediment storage by external(vertical) inputs of sediment - dets and detsb units are tons/acre
		dummy = slsed
		if VSIVFG <= 2: 
			dummy = dummy + nvsi
		dets = dets + dummy

		# washoff of detached sediment from the soil surface
		if SDOPFG:   # use method 1
			# sosed()
			''' Warning,  this method of computing sediment removal contains a dimensionally non-homogeneous term (surs+ suro).  this introduces additional dependence of the results on the simulation interval delt.  so far, it has only been used with delt of 15 and 5 minutes.'''
			''' Remove both detached surface sediment and soil matrix by surface Flow using method 1'''
			if suro > 0.0:  # surface runoff occurs, so sediment and soil matrix particles may be removed
				arg = surs + suro  		# get argument used in transport equations
				stcap = delt60 * kser * (arg / delt60)**jser  	# calculate capacity for removing detached sediment - units are tons/acre-ivl
				if stcap > dets: 
					wssd = dets * suro / arg   # there is insufficient detached storage, base sediment removal on that available, wssd is in tons/acre-ivl
				else:
					wssd = stcap * suro / arg    # there is sufficient detached storage, base sediment removal on the calculated capacity
				dets = dets - wssd
				
				scrsd = delt60 * kger * (arg/delt60)**jger  # calculate scour of matrix soil by surface runoff - units are tons/acre-ivl
				scrsd = scrsd * suro / arg
				
				sosed = wssd + scrsd            # total removal by runoff
			else:              #no runoff occurs, so no removal by runoff
				wssd  = 0.0
				scrsd = 0.0
				sosed = 0.0
				stcap = 0.0
			# end sosed()

		else:        # use method 2
			# sosed()
			''' Warning,  this method of computing sediment removal has not been tested.  but it is dimensionally homogeneous'''
			''' Flow using method 2; Remove both detached surface sediment and soil matrix by surface'''
			if suro > 0.0:  #surface runoff occurs, so sediment and soil matrix particles may be removed, delt60= delt/60
				# calculate capacity for removing detached sediment - units are tons/acre-ivl
				stcap = delt60 * kser * (suro / delt60)**jser
				if stcap > dets:   # there is insufficient detached storage, base sediment removal on that available, wssd is in tons/acre-ivl
					wssd = dets
					dets = 0.0
				else:  #there is sufficient detached storage, base sediment removal on the calculated capacity
					wssd = stcap
					dets = dets - wssd

				# calculate scour of matrix soil by surface runoff - units are tons/acre-ivl
				scrsd = delt60 * kger * (suro / delt60)**jger
				sosed = wssd+ scrsd                      #  total removal by runoff
			else:    # no runoff occurs, so no removal by runoff
				wssd  = 0.0
				scrsd = 0.0
				sosed = 0.0
				stcap = 0.0

		'''attach detached sediment on the surface to the soil matrix
		this code has been modified to allow dry day sed load code modification for chesapeake bay'''
		if  dayfg:    # first interval of new day
			if DRYDFG:  #yesterday was dry, attach detached sediment on the surface to the soil matrix
				# atach()
				''' simulate attachment or compaction of detached sediment on the surface.  the calculation is done at the start of each day, if the previous day was dry'''
				''' this subroutine was modified to allow optional sed loading on dry days (chesapeake bay)
				precipitation did not occur during the previous day
				the attachment of surface sediment to the soil matrix is taken into account by decreasing the storage of detached sediment'''

				dets = dets * (1.0 - affix)			
				# end atach()
			DRYDFG = 1  # assume today will be dry

		if prec > 0:   # today is wet
			DRYDFG = 0
			
		DETS[loop]  = dets
		WSSD[loop]  = wssd
		SCRSD[loop] = scrsd
		SOSED[loop] = sosed
		COVER[loop] = cover

	return errorsV, ERRMSG