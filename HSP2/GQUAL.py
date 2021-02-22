''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import array, zeros
from math import exp
from HSP2.utilities import initm, make_numba_dict, hoursval
from HSP2.ADCALC import advect, oxrea

ERRMSG = []

def gqual(store, siminfo, uci, ts):
	''' Simulate the behavior of a generalized quality constituent'''

	errorsV = zeros(len(ERRMSG), dtype=int)
	delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	delts = siminfo['delt'] * 60

	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData
	svol = vol * 43560

	ui = make_numba_dict(uci)

	UUNITS = 1  # assume english units for now

	# table-type gq-gendata
	ngqual = 1
	tempfg = 2
	phflag = 2
	roxfg  = 2
	cldfg  = 2
	sdfg   = 2
	phytfg = 2
	lat    = 0

	ui = uci['PARAMETERS']
	if 'NGQUAL' in ui:
		ngqual = ui['NGQUAL']
		tempfg = ui['TEMPFG']
		phflag = ui['PHFLAG']
		roxfg  = ui['ROXFG']
		cldfg  = ui['CLDFG']
		sdfg   = ui['SDFG']
		phytfg = ui['PHYTFG']
		lat    = ui['LAT']
	lkfg = 0
	ecnt = 0

	len_ = 0.0
	delth= 0.0
	if 'LEN' in ui:
		len_  = ui["LEN"] * 5280.0  # mi to feet
		delth = ui["DELTH"]
	ts['HRFG'] = hour24Flag(siminfo).astype(float)
	HRFG = ts['HRFG']

	# NGQ3 = NGQUAL * 3
	ddqal = zeros((8, ngqual+1))

	for index in range(1, ngqual+1):

		# update UI values for this constituent here!
		ui_parms = uci['GQUAL' + str(index)]

		if 'GQADFG' + str((index * 2) - 1) in ui_parms:
			# get atmos dep timeseries
			gqadfgf = ui_parms['GQADFG' + str((index * 2) - 1)]
			if gqadfgf > 0:
				ts['GQADFX'] = initm(siminfo, uci, gqadfgf, 'GQUAL' + str(index) + '_MONTHLY/GQADFX', 0.0)
			elif gqadfgf == -1:
				ts['GQADFX'] = ts['GQADFX' + str(index) + ' 1']
			gqadfgc = ui_parms['GQADFG' + str(index * 2)]
			if gqadfgc > 0:
				ts['GQADCN'] = initm(siminfo, uci, gqadfgc, 'IQUAL' + str(index) + '_MONTHLY/GQADCN', 0.0)
			elif gqadfgc == -1:
				ts['GQADCN'] = ts['GQADCN' + str(index) + ' 1']

			if UUNITS == 1:
				if 'GQADFX' in ts:
					ts['GQADFX'] *= delt60 / (24.0 * 43560.0)
			else:
				if 'GQADFX' in ts:
					ts['GQADFX'] *= delt60 / (24.0 * 10000.0)

		if 'GQADFX' not in ts:
			ts['GQADFX'] = zeros(simlen)
		if 'GQADCN' not in ts:
			ts['GQADCN'] = zeros(simlen)

		# table-type gq-qaldata
		qualid = ui_parms['GQID']
		dqal   = ui_parms['DQAL']
		concid = ui_parms['CONCID']
		conv   = ui_parms['CONV']
		qtyid  = ui_parms['QTYID']

		rdqal = dqal * vol
		cinv  = 1.0 / conv   # get reciprocal of unit conversion factor

		# get incoming flow of constituent or zeros;
		if ('GQUAL' + str(index) + '_IDQAL') not in ts:
			ts['GQUAL' + str(index) + '_IDQAL'] = zeros(simlen)
		IDQAL = ts['GQUAL' + str(index) + '_IDQAL'] 

		# process flags for this constituent

		# table-type gq-qalfg
		qalfg = zeros(8)
		qalfg[1] = ui_parms['QALFG1']
		qalfg[2] = ui_parms['QALFG2']
		qalfg[3] = ui_parms['QALFG3']
		qalfg[4] = ui_parms['QALFG4']
		qalfg[5] = ui_parms['QALFG5']
		qalfg[6] = ui_parms['QALFG6']
		qalfg[7] = ui_parms['QALFG7']

		# table-type gq-flg2
		gqpm2 = zeros(8)
		gqpm2[7] = 2
		if 'GQPM21' in ui_parms:
			gqpm2[1] = ui_parms['GQPM21']
			gqpm2[2] = ui_parms['GQPM22']
			gqpm2[3] = ui_parms['GQPM23']
			gqpm2[4] = ui_parms['GQPM24']
			gqpm2[5] = ui_parms['GQPM25']
			gqpm2[6] = ui_parms['GQPM26']
			gqpm2[7] = ui_parms['GQPM27']

		# process parameters for this constituent
		ka = 0.0
		kb = 0.0
		kn = 0.0
		thhyd = 0.0
		kox = 0.0
		thox = 0.0
		if qalfg[1] == 1:   # qual undergoes hydrolysis
			# HYDPM(1,I)  # table-type gq-hydpm
			ka = ui_parms['KA'] * delts   # convert rates from /sec to /ivl
			kb = ui_parms['KB'] * delts
			kn = ui_parms['KN'] * delts
			thhyd = ui_parms['THHYD']

		if qalfg[2] == 1:   # qual undergoes oxidation by free radical processes
			# ROXPM(1,I)  # table-type gq-roxpm
			kox  = ui_parms['KOX'] * delts  # convert rates from /sec to /ivl
			thox = ui_parms['THOX']

		photpm = zeros(21)
		if qalfg[3] == 1:   # qual undergoes photolysis
			# PHOTPM(1,I) # table-type gq-photpm
			if 'EXTENDEDS_PHOTPM' in uci:
				ttable = uci['EXTENDEDS_PHOTPM']
				photpm[1] = ttable['PHOTPM0']
				photpm[2] = ttable['PHOTPM1']
				photpm[3] = ttable['PHOTPM2']
				photpm[4] = ttable['PHOTPM3']
				photpm[5] = ttable['PHOTPM4']
				photpm[6] = ttable['PHOTPM5']
				photpm[7] = ttable['PHOTPM6']
				photpm[8] = ttable['PHOTPM7']
				photpm[9] = ttable['PHOTPM8']
				photpm[10] = ttable['PHOTPM9']
				photpm[11] = ttable['PHOTPM10']
				photpm[12] = ttable['PHOTPM11']
				photpm[13] = ttable['PHOTPM12']
				photpm[14] = ttable['PHOTPM13']
				photpm[15] = ttable['PHOTPM14']
				photpm[16] = ttable['PHOTPM15']
				photpm[17] = ttable['PHOTPM16']
				photpm[18] = ttable['PHOTPM17']
				photpm[19] = ttable['PHOTPM18']
				photpm[20] = ttable['PHOTPM19']

		cfgas = 0.0
		if qalfg[4] == 1:   # qual undergoes volatilization
			cfgas = ui_parms['CFGAS']     # table-type gq-cfgas

		biocon = 0.0
		thbio  = 0.0
		biop   = 0.0
		if qalfg[5] == 1:   # qual undergoes biodegradation
			# BIOPM(1,I)  # table-type gq-biopm
			biocon = ui_parms['BIOCON'] * delt60 / 24.0  # convert rate from /day to /ivl
			thbio  = ui_parms['THBIO']
			biop   = ui_parms['BIO']
			ts['BIO'] = zeros(simlen)
			ts['BIO'].fill(biop)
			# specifies source of biomass data using GQPM2(7,I)
			if gqpm2[7] == 1 or gqpm2[7] == 3:
				# BIOM = # from ts, monthly, constant
				ts['BIO'] = initm(siminfo, uci, ui_parms['GQPM27'], 'GQUAL' + str(index) + '_MONTHLY/BIO',
								ui_parms['BIO'])

		fstdec = 0.0
		thfst  = 0.0
		if qalfg[6] == 1:   #  qual undergoes "general" decay
			# GENPM(1,I)) # table-type gq-gendecay
			fstdec = ui_parms['FSTDEC'] * delt60 / 24.0 # convert rate from /day to /ivl
			thfst  = ui_parms['THFST']

		adpm1 = zeros(7)
		adpm2 = zeros(7)
		adpm3 = zeros(7)
		rsed  = zeros(7)
		sqal  = zeros(7)
		if qalfg[7] == 1:   # constituent is sediment-associated
			# get all required additional input
			# ADDCPM      # table-type gq-seddecay
			# convert rates from /day to /ivl
			addcpm1 = ui_parms['ADDCP1'] * delt60 / 24.0 # convert rate from /day to /ivl
			addcpm2 = ui_parms['ADDCP2']
			addcpm3 = ui_parms['ADDCP3'] * delt60 / 24.0 # convert rate from /day to /ivl
			addcpm4 = ui_parms['ADDCP4']

			# table-type gq-kd
			adpm1[1] = ui_parms['ADPM11']
			adpm1[2] = ui_parms['ADPM21']
			adpm1[3] = ui_parms['ADPM31']
			adpm1[4] = ui_parms['ADPM41']
			adpm1[5] = ui_parms['ADPM51']
			adpm1[6] = ui_parms['ADPM61']
	
			# gq-adrate
			adpm2[1] = ui_parms['ADPM12'] * delt60 / 24.0 # convert rate from /day to /ivl
			adpm2[2] = ui_parms['ADPM22'] * delt60 / 24.0 # convert rate from /day to /ivl
			adpm2[3] = ui_parms['ADPM32'] * delt60 / 24.0 # convert rate from /day to /ivl
			adpm2[4] = ui_parms['ADPM42'] * delt60 / 24.0 # convert rate from /day to /ivl
			adpm2[5] = ui_parms['ADPM52'] * delt60 / 24.0 # convert rate from /day to /ivl
			adpm2[6] = ui_parms['ADPM62'] * delt60 / 24.0 # convert rate from /day to /ivl

			# table-type gq-adtheta
			if 'ADPM13' in ui_parms:
				adpm3[1] = ui_parms['ADPM13']
				adpm3[2] = ui_parms['ADPM23']
				adpm3[3] = ui_parms['ADPM33']
				adpm3[4] = ui_parms['ADPM43']
				adpm3[5] = ui_parms['ADPM53']
				adpm3[6] = ui_parms['ADPM63']
			else:
				adpm3[1] = 1.07
				adpm3[2] = 1.07
				adpm3[3] = 1.07
				adpm3[4] = 1.07
				adpm3[5] = 1.07
				adpm3[6] = 1.07

			# table-type gq-sedconc
			sqal[1] = ui_parms['SQAL1']
			sqal[2] = ui_parms['SQAL2']
			sqal[3] = ui_parms['SQAL3']
			sqal[4] = ui_parms['SQAL4']
			sqal[5] = ui_parms['SQAL5']
			sqal[6] = ui_parms['SQAL6']

			# find the total quantity of material on various forms of sediment
			RSED1 = ts['RSED1']   # sediment storages - suspended sand
			RSED2 = ts['RSED2']   # sediment storages - suspended silt
			RSED3 = ts['RSED3']   # sediment storages - suspended clay
			RSED4 = ts['RSED4']   # sediment storages - bed sand
			RSED5 = ts['RSED5']   # sediment storages - bed silt
			RSED6 = ts['RSED6']   # sediment storages - bed clay

			rsed1 = RSED1[0]
			rsed2 = RSED2[0]
			rsed3 = RSED3[0]
			if 'SSED1' in ui:
				rsed1 = ui['SSED1']
				rsed2 = ui['SSED2']
				rsed3 = ui['SSED3']

			if UUNITS == 1:
				rsed[1] = RSED1[0] / 3.121E-08
				rsed[2] = RSED2[0] / 3.121E-08
				rsed[3] = RSED3[0] / 3.121E-08
				rsed[4] = RSED4[0] / 3.121E-08
				rsed[5] = RSED5[0] / 3.121E-08
				rsed[6] = RSED6[0] / 3.121E-08
			else:
				rsed[1] = RSED1[0] / 2.83E-08
				rsed[2] = RSED2[0] / 2.83E-08
				rsed[3] = RSED3[0] / 2.83E-08
				rsed[4] = RSED4[0] / 2.83E-08
				rsed[5] = RSED5[0] / 2.83E-08
				rsed[6] = RSED6[0] / 2.83E-08

			rsqal1 = sqal[1] * rsed1 * svol
			rsqal2 = sqal[2] * rsed2 * svol
			rsqal3 = sqal[3] * rsed3 * svol
			rsqal4 = rsqal1 + rsqal2 + rsqal3
			rsqal5 = sqal[4] * rsed[4]
			rsqal6 = sqal[5] * rsed[5]
			rsqal7 = sqal[6] * rsed[6]
			rsqal8 = rsqal5 + rsqal6 + rsqal7
			rsqal9 = rsqal1 + rsqal5
			rsqal10 = rsqal2 + rsqal6
			rsqal11 = rsqal3 + rsqal7
			rsqal12 = rsqal9 + rsqal10 + rsqal11
		else:
			# qual not sediment-associated
			rsqal12 = 0.0

		# find total quantity of qual in the rchres
		rrqal = rdqal + rsqal12
		gqst1 = rrqal

		# find values for global flags

		# gqalfg indicates whether any qual undergoes each of the decay processes or is sediment-associated

		# qalgfg indicates whether a qual undergoes any of the 6 decay processes
		qalgfg = 0
		if qalfg[1] > 0 or qalfg[2] > 0 or qalfg[3] > 0 or qalfg[4] > 0 or qalfg[5] > 0 or qalfg[6] > 0:
			qalgfg = 1

		# gdaufg indicates whether any constituent is a "daughter" compound through each of the 6 possible decay processes
		gdaufg = 0
		if gqpm2[1] > 0 or gqpm2[2] > 0 or gqpm2[3] > 0 or gqpm2[4] > 0 or gqpm2[5] > 0 or gqpm2[6] > 0:
			gdaufg = 1

		# daugfg indicates whether or not a given qual is a daughter compound
		daugfg = 0
		if gqpm2[1] > 0 or gqpm2[2] > 0 or gqpm2[3] > 0 or gqpm2[4] > 0 or gqpm2[5] > 0 or gqpm2[6] > 0:
			daugfg = 1

		# get initial value for all inputs which can be constant,
		# vary monthly, or be a time series-some might be over-ridden by
		# monthly values or time series

		# table-type gq-values
		if tempfg == 2 and "TWAT" in ui_parms:
			twat  = ui_parms["TWAT"]
		else:
			twat  = 60.0
		if phflag == 2 and "PHVAL" in ui_parms:
			phval = ui_parms["PHVAL"]
		else:
			phval = 7.0
		if roxfg == 2 and "ROC" in ui_parms:
			roc   = ui_parms["ROC"]
		else:
			roc   = 0.0
		if cldfg == 2 and "CLD" in ui_parms:
			cld   = ui_parms["CLD"]
		else:
			cld   = 0.0
		if sdfg == 2 and "SDCNC" in ui_parms:
			sdcnc = ui_parms["SDCNC"]
		else:
			sdcnc = 0.0
		if phytfg == 2 and "PHY" in ui_parms:
			phy   = ui_parms["PHY"]
		else:
			phy   = 0.0

		ts['TEMP']  = initm(siminfo, uci, tempfg, 'GQUAL' + str(index) + '_MONTHLY/WATEMP', twat)
		ts['PHVAL'] = initm(siminfo, uci, phflag, 'GQUAL' + str(index) + '_MONTHLY/PHVAL', phval)
		ts['ROC']   = initm(siminfo, uci, roxfg, 'GQUAL' + str(index) + '_MONTHLY/ROXYGEN', roc)

		alph = zeros(19)
		gamm = zeros(19)
		delta = zeros(19)
		kcld = zeros(19)
		fact1 = 0.0
		if qalfg[3] == 1:
			#  table-type gq-alpha
			if 'EXTENDEDS_ALPH' in uci:
				ttable = uci['EXTENDEDS_ALPH']
				alph[1] = ttable['ALPH0']
				alph[2] = ttable['ALPH1']
				alph[3] = ttable['ALPH2']
				alph[4] = ttable['ALPH3']
				alph[5] = ttable['ALPH4']
				alph[6] = ttable['ALPH5']
				alph[7] = ttable['ALPH6']
				alph[8] = ttable['ALPH7']
				alph[9] = ttable['ALPH8']
				alph[10] = ttable['ALPH9']
				alph[11] = ttable['ALPH10']
				alph[12] = ttable['ALPH11']
				alph[13] = ttable['ALPH12']
				alph[14] = ttable['ALPH13']
				alph[15] = ttable['ALPH14']
				alph[16] = ttable['ALPH15']
				alph[17] = ttable['ALPH16']
				alph[18] = ttable['ALPH17']
			#  table-type gq-gamma
			if 'EXTENDEDS_GAMM' in uci:
				ttable = uci['EXTENDEDS_GAMM']
				gamm[1] = ttable['GAMM0']
				gamm[2] = ttable['GAMM1']
				gamm[3] = ttable['GAMM2']
				gamm[4] = ttable['GAMM3']
				gamm[5] = ttable['GAMM4']
				gamm[6] = ttable['GAMM5']
				gamm[7] = ttable['GAMM6']
				gamm[8] = ttable['GAMM7']
				gamm[9] = ttable['GAMM8']
				gamm[10] = ttable['GAMM9']
				gamm[11] = ttable['GAMM10']
				gamm[12] = ttable['GAMM11']
				gamm[13] = ttable['GAMM12']
				gamm[14] = ttable['GAMM13']
				gamm[15] = ttable['GAMM14']
				gamm[16] = ttable['GAMM15']
				gamm[17] = ttable['GAMM16']
				gamm[18] = ttable['GAMM17']
			#  table-type gq-delta
			if 'EXTENDEDS_DEL' in uci:
				ttable = uci['EXTENDEDS_DEL']
				delta[1] = ttable['DEL0']
				delta[2] = ttable['DEL1']
				delta[3] = ttable['DEL2']
				delta[4] = ttable['DEL3']
				delta[5] = ttable['DEL4']
				delta[6] = ttable['DEL5']
				delta[7] = ttable['DEL6']
				delta[8] = ttable['DEL7']
				delta[9] = ttable['DEL8']
				delta[10] = ttable['DEL9']
				delta[11] = ttable['DEL10']
				delta[12] = ttable['DEL11']
				delta[13] = ttable['DEL12']
				delta[14] = ttable['DEL13']
				delta[15] = ttable['DEL14']
				delta[16] = ttable['DEL15']
				delta[17] = ttable['DEL16']
				delta[18] = ttable['DEL17']
			#  table-type gq-cldfact
			if 'EXTENDEDS_KCLD' in uci:
				ttable = uci['EXTENDEDS_KCLD']
				kcld[1] = ttable['KCLD0']
				kcld[2] = ttable['KCLD1']
				kcld[3] = ttable['KCLD2']
				kcld[4] = ttable['KCLD3']
				kcld[5] = ttable['KCLD4']
				kcld[6] = ttable['KCLD5']
				kcld[7] = ttable['KCLD6']
				kcld[8] = ttable['KCLD7']
				kcld[9] = ttable['KCLD8']
				kcld[10] = ttable['KCLD9']
				kcld[11] = ttable['KCLD10']
				kcld[12] = ttable['KCLD11']
				kcld[13] = ttable['KCLD12']
				kcld[14] = ttable['KCLD13']
				kcld[15] = ttable['KCLD14']
				kcld[16] = ttable['KCLD15']
				kcld[17] = ttable['KCLD16']
				kcld[18] = ttable['KCLD17']

			ts['CLD'] = initm(siminfo, uci, cldfg, 'GQUAL' + str(index) + '_MONTHLY/CLOUD', cld)
			ts['SDCNC'] = initm(siminfo, uci, sdfg, 'GQUAL' + str(index) + '_MONTHLY/SEDCONC', sdcnc)
			ts['PHY'] = initm(siminfo, uci, phytfg, 'GQUAL' + str(index) + '_MONTHLY/PHYTO', phy)

			htfg = int(ui['HTFG'])
			cfsaex = 1.0
			if htfg == 0:
				cfsaex = ui['CFSAEX']

			# fact1 is a pre-calculated value used in photolysis simulation
			fact1 = cfsaex * delt60 / 24.0

			# decide which set of light data to use
			light = (abs(int(lat)) + 5) // 10
			if light == 0:  # no table for equation, so use 10 deg table
				light = 1

			# # read the light data- 9 values to a line,
			# SGRP  = 50 + LIGHT
			# INITFG= 1
			# DO 210 L=1,4
			# 	LIT(K,L)  # FIRST 9; index K
			# 	LIT(K,L)  # SECOND 9; index K
			# # 210    CONTINUE
			#
			# # determine which season (set) of data to start with
			# litfg = 0
			#
			# # look one time-step ahead to see which "month" to use,
			# # because we might be on a month boundary, in which case
			# # datim will contain the earlier month
			# idelt = delt
			# DO 220 I=1,5
			# 	NEWDAT(I) =DATIM(I)
			# # 220    CONTINUE
			#
			# CALL ADDTIM()
			#
			# NEWMO = NEWDAT(2)
			# LSET = NEWMO/3
			# IF (LSET.EQ.0) THEN
			# 	LSET= 4
			# END IF
			#
			# # southern hemisphere is 2 seasons out of phase
			#
			# IF (LAT.LT.0) THEN
			# 	LSET= LSET + 2
			# 	IF (LSET.GT.4) THEN
			# 		LSET= LSET - 4
			# 	END IF
			# END IF

		reamfg = 0
		cforea = 0.0
		tcginv = 0.0
		reak   = 0.0
		reakt  = 0.0
		expred = 0.0
		exprev = 0.0
		if qalfg[4] == 1:
			# one or more constituents undergoes volatilization process- input required to compute reaeration coefficient

			# flags - table-type ox-flags
			reamfg = 2
			if 'REAMFG' in ui_parms:
				reamfg = ui_parms["REAMFG"]
			dopfg = 0
			if 'DOPFG' in ui_parms:
				dopfg  = ui_parms["DOPFG"]

			htfg = int(ui['HTFG'])
			if htfg == 0:
				elev = ui_parms["ELEV"]
				cfpres = ((288.0 - 0.001981 * elev) / 288.0) ** 5.256

			lkfg = int(ui['LKFG'])
			if lkfg == 1:
				# table-type ox-cforea
				cforea = 1.0
				if 'CFOREA' in ui_parms:
					cforea = ui_parms["CFOREA"]
				if 'CFOREA' in ui:
					cforea = ui["CFOREA"]
			else:
				if reamfg == 1:
					# tsivoglou method - table-type ox-tsivoglou
					reakt  = ui_parms["REAKT"]
					tcginv = ui_parms["TCGINV"]
				elif reamfg == 2:
					# owen/churchill/o'connor-dobbins  # table-type ox-tcginv
					tcginv = 1.047
					if "TCGINV" in ui_parms:
						tcginv = ui_parms["TCGINV"]
				elif reamfg == 3:
					# user formula - table-type ox-reaparm
					tcginv = ui_parms["TCGINV"]
					reak   = ui_parms["REAK"]
					expred = ui_parms["EXPRED"]
					exprev = ui_parms["EXPREV"]

		# process tables specifying relationship between "parent" and "daughter" compounds
		# table-type gq-daughter
		c = zeros((8,7))
		if 'C21' in ui_parms:
			c[2,1] = ui_parms("C21")
			c[3,1] = ui_parms("C31")
			c[4,1] = ui_parms("C41")
			c[5,1] = ui_parms("C51")
			c[6,1] = ui_parms("C61")
			c[7,1] = ui_parms("C71")
			c[3,2] = ui_parms("C32")
			c[4,2] = ui_parms("C42")
			c[5,2] = ui_parms("C52")
			c[6,2] = ui_parms("C62")
			c[7,2] = ui_parms("C72")
			c[4,3] = ui_parms("C43")
			c[5,3] = ui_parms("C53")
			c[6,3] = ui_parms("C63")
			c[7,3] = ui_parms("C73")
			c[5,4] = ui_parms("C54")
			c[6,4] = ui_parms("C64")
			c[7,4] = ui_parms("C74")
			c[6,5] = ui_parms("C65")
			c[7,5] = ui_parms("C75")
			c[7,6] = ui_parms("C76")

		if qalfg[7] == 1: #  one or more quals are sediment-associated
			sedfg = int(ui['SEDFG'])
			if sedfg == 0: # section sedtrn not active
				#ERRMSG
				pass

		hydrfg = int(ui['HYDRFG'])
		aux1fg = int(ui['AUX1FG'])
		aux2fg = int(ui['AUX2FG'])
		if hydrfg == 1:  # check that required options in section hydr have been selected
			if qalfg[3] == 1 and aux1fg == 0:
				# ERRMSG: error-simulation of photolysis requires aux1fg to be on to calculate average depth
				pass
			if qalfg[4] == 1:
				lkfg = int(ui['LKFG'])
				if lkfg == 0:
					if aux2fg == 0:
						# ERRMSG:  error-simulation of volatilization in a free flowing stream requires aux3fg on
						pass
				else:
					if aux1fg == 0:
						# ERRMG: error-simulation of volatilization in a lake requires aux1fg on to calculate average depth
						pass

		#####################  end PGQUAL

		# get input timeseries
		AVDEP = ts['AVDEP']
		PHVAL = ts['PHVAL']
		TW    = ts['TW']
		ROC   = ts['ROC']
		if 'SDCNC' in ts:
			SDCNC = ts['SDCNC']    # constant, monthly, ts; SDFG, note: interpolate to daily value only
		if 'PHY' in ts:
			PHYTO = ts['PHY']      # constant, monthly, ts; PHYTFG, note: interpolate to daily value only
		if 'CLD' in ts:
			CLD   = ts['CLD']      # constant, monthly, ts['CLOUD']
		WIND  = ts['WIND'] * 1609.0 # miles to meters
		AVVEL = ts['AVVEL']
		PREC  = ts['PREC']
		SAREA = ts['SAREA']
		GQADFX = ts['GQADFX']
		GQADCN = ts['GQADCN']
		if 'BIO' not in ts:
			ts['BIO'] = zeros(simlen)
		BIO    = ts['BIO']
		if 'ISQAL1' not in ts:
			ts['ISQAL1'] = zeros(simlen)
		ISQAL1 = ts['ISQAL1']
		if 'ISQAL2' not in ts:
			ts['ISQAL2'] = zeros(simlen)
		ISQAL2 = ts['ISQAL2']
		if 'ISQAL3' not in ts:
			ts['ISQAL3'] = zeros(simlen)
		ISQAL3 = ts['ISQAL3']
		DEPSCR1 = ts['DEPSCR1']
		DEPSCR2 = ts['DEPSCR2']
		DEPSCR3 = ts['DEPSCR3']
		ROSED1 = ts['ROSED1']
		ROSED2 = ts['ROSED2']
		ROSED3 = ts['ROSED3']

		OSED1 = zeros((simlen, nexits))
		OSED2 = zeros((simlen, nexits))
		OSED3 = zeros((simlen, nexits))

		# this number is used to adjust reaction rates for temperature
		# TW20 = TW - 20.0

		name = 'GQUAL' + str(index)  # arbitrary identification
		# preallocate output arrays (always needed)
		ADQAL1 = ts[name + '_ADQAL1'] = zeros(simlen)
		ADQAL2 = ts[name + '_ADQAL2'] = zeros(simlen)
		ADQAL3 = ts[name + '_ADQAL3'] = zeros(simlen)
		ADQAL4 = ts[name + '_ADQAL4'] = zeros(simlen)
		ADQAL5 = ts[name + '_ADQAL5'] = zeros(simlen)
		ADQAL6 = ts[name + '_ADQAL6'] = zeros(simlen)
		ADQAL7 = ts[name + '_ADQAL7'] = zeros(simlen)
		DDQAL1 = ts[name + '_DDQAL1'] = zeros(simlen)
		DDQAL2 = ts[name + '_DDQAL2'] = zeros(simlen)
		DDQAL3 = ts[name + '_DDQAL3'] = zeros(simlen)
		DDQAL4 = ts[name + '_DDQAL4'] = zeros(simlen)
		DDQAL5 = ts[name + '_DDQAL5'] = zeros(simlen)
		DDQAL6 = ts[name + '_DDQAL6'] = zeros(simlen)
		DDQAL7 = ts[name + '_DDQAL7'] = zeros(simlen)
		DQAL   = ts[name + '_DQAL'] = zeros(simlen)
		DSQAL1 = ts[name + '_DSQAL1'] = zeros(simlen)
		DSQAL2 = ts[name + '_DSQAL2'] = zeros(simlen)
		DSQAL3 = ts[name + '_DSQAL3'] = zeros(simlen)
		DSQAL4 = ts[name + '_DSQAL4'] = zeros(simlen)
		GQADDR = ts[name + '_GQADDR'] = zeros(simlen)
		GQADEP = ts[name + '_GQADEP'] = zeros(simlen)
		GQADWT = ts[name + '_GQADWT'] = zeros(simlen)
		ISQAL4 = ts[name + '_ISQAL4'] = zeros(simlen)
		PDQAL  = ts[name + '_PDQAL'] = zeros(simlen)
		RDQAL  = ts[name + '_RDQAL'] = zeros(simlen)
		RODQAL = ts[name + '_RODQAL'] = zeros(simlen)
		ROSQAL1= ts[name + '_ROSQAL1'] = zeros(simlen)
		ROSQAL2= ts[name + '_ROSQAL2'] = zeros(simlen)
		ROSQAL3= ts[name + '_ROSQAL3'] = zeros(simlen)
		ROSQAL4= ts[name + '_ROSQAL4'] = zeros(simlen)
		RRQAL  = ts[name + '_RRQAL'] = zeros(simlen)
		RSQAL1 = ts[name + '_RSQAL1'] = zeros(simlen)
		RSQAL2 = ts[name + '_RSQAL2'] = zeros(simlen)
		RSQAL3 = ts[name + '_RSQAL3'] = zeros(simlen)
		RSQAL4 = ts[name + '_RSQAL4'] = zeros(simlen)
		RSQAL5 = ts[name + '_RSQAL5'] = zeros(simlen)
		RSQAL6 = ts[name + '_RSQAL6'] = zeros(simlen)
		RSQAL7 = ts[name + '_RSQAL7'] = zeros(simlen)
		RSQAL8 = ts[name + '_RSQAL8'] = zeros(simlen)
		RSQAL9 = ts[name + '_RSQAL9'] = zeros(simlen)
		RSQAL10= ts[name + '_RSQAL10'] = zeros(simlen)
		RSQAL11= ts[name + '_RSQAL11'] = zeros(simlen)
		RSQAL12= ts[name + '_RSQAL12'] = zeros(simlen)
		SQAL1  = ts[name + '_SQAL1'] = zeros(simlen)
		SQAL2  = ts[name + '_SQAL2'] = zeros(simlen)
		SQAL3  = ts[name + '_SQAL3'] = zeros(simlen)
		SQAL4  = ts[name + '_SQAL4'] = zeros(simlen)
		SQAL5  = ts[name + '_SQAL5'] = zeros(simlen)
		SQAL6  = ts[name + '_SQAL6'] = zeros(simlen)
		SQDEC1 = ts[name + '_SQDEC1'] = zeros(simlen)
		SQDEC2 = ts[name + '_SQDEC2'] = zeros(simlen)
		SQDEC3 = ts[name + '_SQDEC3'] = zeros(simlen)
		SQDEC4 = ts[name + '_SQDEC4'] = zeros(simlen)
		SQDEC5 = ts[name + '_SQDEC5'] = zeros(simlen)
		SQDEC6 = ts[name + '_SQDEC6'] = zeros(simlen)
		SQDEC7 = ts[name + '_SQDEC7'] = zeros(simlen)
		TIQAL  = ts[name + '_TIQAL'] = zeros(simlen)
		TROQAL = ts[name + '_TROQAL'] = zeros(simlen)
		TOQAL  = zeros((simlen, nexits))
		ODQAL  = zeros((simlen, nexits))
		OSQAL1 = zeros((simlen, nexits))
		OSQAL2 = zeros((simlen, nexits))
		OSQAL3 = zeros((simlen, nexits))
		TOSQAL = zeros((simlen, nexits))

		if nexits > 1:
			u = uci['SAVE']
			key1 = name + '_ODQAL'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['ODQAL']
			del u['ODQAL']
			key1 = name + '_OSQAL1'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['OSQAL']
			key1 = name + '_OSQAL2'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['OSQAL']
			key1 = name + '_OSQAL3'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['OSQAL']
			del u['OSQAL']
			key1 = name + '_TOSQAL'
			for i in range(nexits):
				u[f'{key1}{i + 1}'] = u['TOSQAL']
			del u['TOSQAL']

		for loop in range(simlen):
			# within time loop

			# tw20 may be required for bed decay of qual even if tw is undefined (due to vol=0.0)
			tw   = TW[loop]
			tw = (tw - 32.0) * 5.0 / 9.0
			tw20 = tw - 20.0           # TW20[loop]
			if tw <= -10.0:
				tw20 = 0.0
			# correct unrealistically high values of tw calculated in htrch
			if tw >= 50.0:
				tw20 = 30.0
			prec = PREC[loop]
			sarea= SAREA[loop]
			vol  = VOL[loop] * 43560
			toqal = TOQAL[loop]
			tosqal = TOSQAL[loop]
			if UUNITS == 1:
				depscr1 = DEPSCR1[loop] / 3.121E-08
				depscr2 = DEPSCR2[loop] / 3.121E-08
				depscr3 = DEPSCR3[loop] / 3.121E-08
				rosed1 = ROSED1[loop] / 3.121E-08
				rosed2 = ROSED2[loop] / 3.121E-08
				rosed3 = ROSED3[loop] / 3.121E-08
				osed1 = OSED1[loop] / 3.121E-08
				osed2 = OSED2[loop] / 3.121E-08
				osed3 = OSED3[loop] / 3.121E-08
			else:
				depscr1 = DEPSCR1[loop] / 2.83E-08
				depscr2 = DEPSCR2[loop] / 2.83E-08
				depscr3 = DEPSCR3[loop] / 2.83E-08
				rosed1 = ROSED1[loop] / 2.83E-08
				rosed2 = ROSED2[loop] / 2.83E-08
				rosed3 = ROSED3[loop] / 2.83E-08
				osed1 = OSED1[loop] / 2.83E-08
				osed2 = OSED2[loop] / 2.83E-08
				osed3 = OSED3[loop] / 2.83E-08
			isqal1 = ISQAL1[loop]
			isqal2 = ISQAL2[loop]
			isqal3 = ISQAL3[loop]

			if UUNITS == 2:  # uci is in metric units
				avdepm = AVDEP[loop]
				avdepe = AVDEP[loop] * 3.28
				avvele = AVVEL[loop] * 3.28
			else:         # uci is in english units
				avdepm = AVDEP[loop] * 0.3048
				avdepe = AVDEP[loop]
				avvele = AVVEL[loop]

			fact2 = zeros(19)
			if qalfg[3] > 0:
				# one or more constituents undergoes photolysis decay
				if avdepe > 0.17:
					# depth of water in rchres is greater than two inches -
					# consider photolysis; this criteria will also be applied to other decay processes
					for l in range(1, 18):
						# evaluate the light extinction exponent- 2.76*klamda*d
						kl   = alph[l] + gamm[l] * sdcnc + delta[l] * phy
						expnt= 2.76 * kl * avdepm * 100.0
						# evaluate the cloud factor
						cldl= (10.0 - cld * kcld[l]) / 10.0
						if expnt <= -20.0:
							expnt = -20.
						if expnt >= 20.0:
							expnt = 20.
						# evaluate the precalculated factors fact2
						# lit is data from the seq file, just make zero for now
						#fact2[l] = cldl * lit[l,lset] * (1.0 - exp(-expnt)) / expnt
						fact2[l] = 0.0
				else:
					# depth of water in rchres is less than two inches -photolysis is not considered
					pass

			korea = 0.0
			if qalfg[4] > 0:
				# prepare to simulate volatilization by finding the oxygen reaeration coefficient
				wind = 0.0
				if lkfg == 1:
					wind =  WIND[loop]
				if avdepe > 0.17:   # rchres depth is sufficient to consider volatilization
					# compute oxygen reaeration rate-korea
					korea = oxrea(lkfg, wind, cforea, avvele, avdepe, tcginv, reamfg, reak, reakt, expred, exprev,
						  			len_, delth, tw, delts, delt60, UUNITS)
					# KOREA = OXREA(LKFG,WIND,CFOREA,AVVELE,AVDEPE,TCGINV,REAMFG,REAK,REAKT,EXPRED,EXPREV,LEN, DELTH,TWAT,DELTS,DELT60,UUNITS,KOREA)
				else:
					# rchres depth is not sufficient to consider volatilization
					pass

			# get data on inflow of dissolved material
			gqadfx = GQADFX[loop]
			gqadcn = GQADCN[loop]
			gqaddr = sarea * conv * gqadfx  # dry deposition;
			gqadwt = prec * sarea * gqadcn  # wet deposition;

			gqadep = gqaddr + gqadwt  # total atmospheric deposition
			idqal = IDQAL[loop] * conv
			indqal = idqal + gqaddr + gqadwt

			# simulate advection of dissolved material
			srovol = SROVOL[loop]
			erovol = EROVOL[loop]
			sovol = SOVOL[loop, :]
			eovol = EOVOL[loop, :]
			dqal, rodqal, odqal = advect(indqal, dqal, nexits, svol, vol, srovol, erovol, sovol, eovol)

			bio = biop
			if qalfg[5] > 0:
				# get biomass input, if required (for degradation)
				bio = BIO[loop]

			if avdepe > 0.17:   #  simulate decay of dissolved material
				hr = HRFG[loop]
				ddqal[:,index] = ddecay(qalfg, tw20, ka, kb, kn, thhyd, phval,kox,thox, roc, fact2, fact1, photpm, korea, cfgas,
					 			biocon, thbio, bio, fstdec, thfst, vol, dqal, hr, delt60)
				# ddqal[1,index] = DDECAY(QALFG(1,I),TW20,HYDPM(1,I),PHVAL,ROXPM(1,I),ROC,FACT2(1),FACT1,PHOTPM(1,I),KOREA,CFGAS(I),
				# 						BIOPM(1,I),BIO(I),GENPM(1,I),VOLSP,DQAL(I),HR,DELT60,DDQAL(1,I))

				pdqal = 0.0
				for k in range(1, 6):
					if gqpm2[k] == 1:    # this compound is a "daughter"-compute the contribution to it from its "parent(s)"
						itobe = index - 1
						for j in range(1,itobe):
							pdqal = pdqal + ddqal[k,j]*c[j,k]

				# update the concentration to account for decay and for input
				# from decay of "parents"- units are conc/l
				if vol > 0:
					dqal = dqal + (pdqal - ddqal[7,index])/vol
			else:
				# rchres depth is less than two inches - dissolved decay is not considered
				for l in range(1, 7):
					ddqal[l,index] = 0.0
				# 320      CONTINUE
				pdqal = 0.0

			adqal = zeros(8)
			dsqal1 = 0.0
			dsqal2 = 0.0
			dsqal3 = 0.0
			dsqal4 = 0.0
			osqal1 = 0.0
			osqal2 = 0.0
			osqal3 = 0.0
			osqal4 = 0.0
			rosqal1 = 0.0
			rosqal2 = 0.0
			rosqal3 = 0.0
			sqdec1 = 0.0
			sqdec2 = 0.0
			sqdec3 = 0.0
			sqdec4 = 0.0
			sqdec5 = 0.0
			sqdec6 = 0.0
			sqdec7 = 0.0
			# zero the accumulators
			isqal4 = 0.0
			dsqal4 = 0.0
			rosqal4 = 0.0

			if qalfg[7] == 1:   # this constituent is associated with sediment
				if nexits > 1:
					for n in range(1, nexits):
						tosqal[n] = 0.0

				# repeat for each sediment size fraction
				# get data on inflow of sediment-associated material

				# sand
				# advect this material, including calculation of deposition and scour
				ecnt, sqal[1], sqal[4], dsqal1, rosqal1, osqal1 = advqal(isqal1, rsed[1], rsed[4], depscr1, rosed1, osed1,
																	 nexits, rsqal1, rsqal5, ecnt)
				# GQECNT(1),SQAL(J,I),SQAL(J + 3,I),DSQAL(J,I), ROSQAL(J,I),OSQAL(1,J,I)) = ADVQAL (ISQAL(J,I),RSED(J),RSED(J + 3),\
				# DEPSCR(J),ROSED(J),OSED(1,J),NEXITS,RCHNO, MESSU,MSGFL,DATIM, GQID(1,I),J,RSQAL(J,I),RSQAL(J + 4,I),GQECNT(1),
				# SQAL(J,I),SQAL(J + 3,I),DSQAL(J,I),ROSQAL(J,I),OSQAL(1,J,I))

				iqal4   = isqal4 + isqal1
				dsqal4  = dsqal4 + dsqal1
				rosqal4 = rosqal4 + rosqal1
				if nexits > 1:
					for n in range(1, nexits):
						tosqal[n] = tosqal[n] +osqal1[n]

				# silt
				# advect this material, including calculation of deposition and scour
				ecnt, sqal[2], sqal[5], dsqal2, rosqal2, osqal2 = advqal(isqal2, rsed[2], rsed[5], depscr2, rosed2, osed2,
																	 nexits, rsqal2, rsqal6, ecnt)
				# GQECNT(1), SQAL(J, I), SQAL(J + 3, I), DSQAL(J, I), ROSQAL(J, I), OSQAL(1, J, I)) = ADVQAL(
				# 	ISQAL(J, I), RSED(J), RSED(J + 3), \
				# 	DEPSCR(J), ROSED(J), OSED(1, J), NEXITS, RCHNO, MESSU, MSGFL, DATIM, GQID(1, I), J, RSQAL(J, I),
				# 	RSQAL(J + 4, I), GQECNT(1),
				# 	SQAL(J, I), SQAL(J + 3, I), DSQAL(J, I), ROSQAL(J, I), OSQAL(1, J, I))

				iqal4 = isqal4 + isqal2
				dsqal4 = dsqal4 + dsqal2
				rosqal4 = rosqal4 + rosqal2
				if nexits > 1:
					for n in range(1, nexits):
						tosqal[n] = tosqal[n] + osqal2[n]

				# clay
				# advect this material, including calculation of deposition and scour
				ecnt, sqal[3], sqal[6], dsqal3, rosqal3, osqal3 = advqal(isqal3, rsed[3], rsed[6], depscr3, rosed3, osed3,
																	 nexits, rsqal3, rsqal7, ecnt)
				# GQECNT(1), SQAL(J, I), SQAL(J + 3, I), DSQAL(J, I), ROSQAL(J, I), OSQAL(1, J, I)) = ADVQAL(
				# 	ISQAL(J, I), RSED(J), RSED(J + 3), \
				# 	DEPSCR(J), ROSED(J), OSED(1, J), NEXITS, RCHNO, MESSU, MSGFL, DATIM, GQID(1, I), J, RSQAL(J, I),
				# 	RSQAL(J + 4, I), GQECNT(1),
				# 	SQAL(J, I), SQAL(J + 3, I), DSQAL(J, I), ROSQAL(J, I), OSQAL(1, J, I))

				iqal4 = isqal4 + isqal3
				dsqal4 = dsqal4 + dsqal3
				rosqal4 = rosqal4 + rosqal3
				if nexits > 1:
					for n in range(1, nexits):
						tosqal[n] = tosqal[n] + osqal3[n]

				tiqal  = idqal + isqal4
				troqal = rodqal + rosqal4
				if nexits > 1:
					for n in range(1, nexits):
						toqal[n] = odqal[n] + tosqal[n]

				if avdepe > 0.17:     # simulate decay on suspended sediment
					sqal[1], sqal[2], sqal[3], sqdec1, sqdec2, sqdec3 = adecay(addcpm1, addcpm2, tw20, rsed[1], rsed[2], rsed[3], sqal[1], sqal[2], sqal[3])
					# SQAL((1),I), SQDEC((1),I)) =  ADECAY(ADDCPM(1,I),TW20,RSED(1),SQAL((1),I),SQDEC((1),I))
				else:
					# rchres depth is less than two inches - decay of qual
					# associated with suspended sediment is not considered
					sqdec1 = 0.0
					sqdec2 = 0.0
					sqdec3 = 0.0

				# simulate decay on bed sediment
				sqal[4], sqal[5], sqal[6], sqdec4, sqdec5, sqdec6 = adecay(addcpm3, addcpm4, tw20, rsed[4], rsed[5], rsed[6], sqal[4], sqal[5], sqal[6])
				# SQAL((4),I), SQDEC((4),I)) = ADECAY(ADDCPM(3,I),TW20,RSED(4),SQAL((4),I),SQDEC((4),I))

				# get total decay
				sqdec7 = sqdec1 + sqdec2 + sqdec3 + sqdec4 + sqdec4 + sqdec4

				if avdepe > 0.17:  # simulate exchange due to adsorption and desorption
					dqal, sqal, adqal = adsdes(vol, rsed, adpm1, adpm2, adpm3, tw20, dqal, sqal)
					# DQAL(I), SQAL(1,I), ADQAL(1,I) = ADSDES(VOLSP,RSED(1),ADPM(1,1,I),TW20,DQAL(I),SQAL(1,I),ADQAL(1,I))
				else:
					# rchres depth is less than two inches - adsorption and
					# desorption of qual is not considered
					adqal[1] = 0.0
					adqal[2] = 0.0
					adqal[3] = 0.0
					adqal[4] = 0.0
					adqal[5] = 0.0
					adqal[6] = 0.0
					adqal[7] = 0.0

				# find total quantity of material on various forms of sediment
				rsqal4 = 0.0
				rsqal8 = 0.0
				rsqal12 = 0.0
				rsqal1 = sqal[1] * rsed[1]
				rsqal2 = sqal[2] * rsed[2]
				rsqal3 = sqal[3] * rsed[3]
				rsqal4 = rsqal1 + rsqal2 + rsqal3
				rsqal5 = sqal[4] * rsed[4]
				rsqal6 = sqal[5] * rsed[5]
				rsqal7 = sqal[6] * rsed[6]
				rsqal8 = rsqal5 + rsqal6 + rsqal7
				rsqal9 = rsqal1 + rsqal5
				rsqal10 = rsqal2 + rsqal6
				rsqal11 = rsqal3 + rsqal7
				rsqal12 = rsqal9 + rsqal10 + rsqal11
			else:
				# qual constituent not associated with sediment-total just
				# above should have been set to zero by run interpreter
				tiqal = idqal
				troqal = rodqal
				if nexits > 1:
					for n in range(1, nexits):
						toqal[n] = odqal[n]

			# find total quantity of qual in rchres
			rdqal = dqal * vol
			if qalfg[7] == 1:
				rrqal = rdqal + rsqal12
			else:
				rrqal = rdqal

			svol = vol  # svol is volume at start of time step, update for next time thru

			ADQAL1[loop] = adqal[1] 			# put values for this time step back into TS
			ADQAL2[loop] = adqal[2]
			ADQAL3[loop] = adqal[3]
			ADQAL4[loop] = adqal[4]
			ADQAL5[loop] = adqal[5]
			ADQAL6[loop] = adqal[6]
			ADQAL7[loop] = adqal[7]
			DDQAL1[loop] = ddqal[1,index] / conv
			DDQAL2[loop] = ddqal[2, index] / conv
			DDQAL3[loop] = ddqal[3, index] / conv
			DDQAL4[loop] = ddqal[4, index] / conv
			DDQAL5[loop] = ddqal[5, index] / conv
			DDQAL6[loop] = ddqal[6, index] / conv
			DDQAL7[loop] = ddqal[7, index] / conv
			DQAL[loop]   = dqal
			DSQAL1[loop] = dsqal1
			DSQAL2[loop] = dsqal2
			DSQAL3[loop] = dsqal3
			DSQAL4[loop] = dsqal4
			GQADDR[loop] = gqaddr
			GQADEP[loop] = gqadep
			GQADWT[loop] = gqadwt
			ISQAL4[loop] = isqal4
			ODQAL[loop]  = odqal / conv
			OSQAL1[loop] = osqal1
			OSQAL2[loop] = osqal2
			OSQAL3[loop] = osqal3
			PDQAL[loop]  = pdqal
			RDQAL[loop]  = rdqal / conv
			RODQAL[loop] = rodqal / conv
			ROSQAL1[loop]= rosqal1
			ROSQAL2[loop]= rosqal2
			ROSQAL3[loop]= rosqal3
			ROSQAL4[loop]= rosqal4
			RRQAL[loop]  = rrqal / conv
			RSQAL1[loop] = rsqal1
			RSQAL2[loop] = rsqal2
			RSQAL3[loop] = rsqal3
			RSQAL4[loop] = rsqal4
			RSQAL5[loop] = rsqal5
			RSQAL6[loop] = rsqal6
			RSQAL7[loop] = rsqal7
			RSQAL8[loop] = rsqal8
			RSQAL9[loop] = rsqal9
			RSQAL10[loop]= rsqal10
			RSQAL11[loop]= rsqal11
			RSQAL12[loop]= rsqal12
			SQAL1[loop]  = sqal[1]
			SQAL2[loop]  = sqal[2]
			SQAL3[loop]  = sqal[3]
			SQAL4[loop]  = sqal[4]
			SQAL5[loop]  = sqal[5]
			SQAL6[loop]  = sqal[6]
			SQDEC1[loop] = sqdec1
			SQDEC2[loop] = sqdec2
			SQDEC3[loop] = sqdec3
			SQDEC4[loop] = sqdec4
			SQDEC5[loop] = sqdec5
			SQDEC6[loop] = sqdec6
			SQDEC7[loop] = sqdec7
			TIQAL[loop]  = tiqal
			TOSQAL[loop] = tosqal
			TROQAL[loop] = troqal / conv

		if nexits > 1:
			for i in range(nexits):
				ts[name + '_ODQAL' + str(i + 1)] = ODQAL[:, i]
				ts[name + '_OSQAL1' + str(i + 1)] = OSQAL1[:, i]
				ts[name + '_OSQAL2' + str(i + 1)] = OSQAL2[:, i]
				ts[name + '_OSQAL3' + str(i + 1)] = OSQAL3[:, i]
				ts[name + '_TOSQAL' + str(i + 1)] = TOSQAL[:, i]

	return errorsV, ERRMSG


def adecay(addcpm1, addcpm2, tw20, rsed_sand, rsed_silt, rsed_clay, sqal_sand, sqal_silt, sqal_clay):
	# real  addcpm(2),rsed(3),sqal(3),sqdec(3),tw20
	''' simulate decay of material in adsorbed state'''

	sqdec_sand = 0.0
	sqdec_silt = 0.0
	sqdec_clay = 0.0
	if addcpm1 > 0.0:     # calculate temp-adjusted decay rate
		dk  = addcpm1 * addcpm2**tw20
		fact = 1.0 - exp(-dk)

		if sqal_sand > 1.0e-30:
			dconc    = sqal_sand * fact
			sqal_sand  = sqal_sand - dconc
			sqdec_sand = dconc * rsed_sand
		if sqal_silt > 1.0e-30:
			dconc    = sqal_silt * fact
			sqal_silt  = sqal_silt - dconc
			sqdec_silt = dconc * rsed_silt
		if sqal_clay > 1.0e-30:
			dconc    = sqal_clay * fact
			sqal_clay  = sqal_clay - dconc
			sqdec_clay = dconc * rsed_clay

	return  sqal_sand, sqal_silt, sqal_clay, sqdec_sand, sqdec_silt, sqdec_clay


def adsdes(vol,rsed,adpm1,adpm2,adpm3,tw20,dqal,sqal):
	#  adpm(6,3),adqal(7),dqal,rsed(6),sqal(6),tw20,vol

	''' simulate exchange of a constituent between the dissolved
	state and adsorbed state-note that 6 adsorption site classes are
	considered: 1- suspended sand  2- susp. silt  3- susp. clay
	4- bed sand  5- bed silt  6- bed clay'''

	ainv  = zeros(7)
	cainv = zeros(7)
	adqal = zeros(8)
	if vol > 0.0:     # adsorption/desorption can take place
		# first find the new dissolved conc.
		num   = vol	* dqal
		denom = vol
		for j in range(1, 6):
			if rsed[j] > 0.0:  # this sediment class is present-evaluate terms due to it
				# transfer rate, corrected for water temp
				akj  = adpm2[j] * adpm3[j]**tw20
				temp = 1.0 / (1.0 + akj)

				# calculate 1/a and c/a
				ainv[j]  = akj * adpm1[j] * temp
				cainv[j] = sqal[j] * temp

				# accumulate terms for numerator and denominator in dqal equation
				num   = num + (sqal[j] - cainv[j]) * rsed[j]
				denom = denom + rsed[j] * ainv[j]

		# calculate new dissolved concentration-units are conc/l
		dqal= num / denom

		# calculate new conc on each sed class and the corresponding adsorption/desorption flux
		adqal[7] = 0.0
		for j in range(1, 6):
			if rsed[j] > 0.0:	# this sediment class is present-calculate data pertaining to it
				# new concentration
				temp = cainv[j] + dqal * ainv[j]

				# quantity of material transferred
				adqal[j] = (temp - sqal[j]) * rsed[j]
				sqal[j]  = temp

				# accumulate total adsorption/desorption flux
				adqal[7] = adqal[7] + adqal[j]
			else:     # this sediment class is absent
				adqal[j] = 0.0
				# sqal(j) is unchanged-"undefined"
	else:    # no water, no adsorption/desorption
		for j in range(1, 7):
			adqal[j] = 0.0
			# sqal(1 thru 3) and dqal should already have been set to undefined values

	return dqal, sqal, adqal


def advqal(isqal,rsed,bsed,depscr,rosed,osed,nexits,rsqals,rbqals,ecnt):

	''' simulate the advective processes, including deposition and
	scour for the quality constituent attached to one sediment size fraction'''

	if depscr < 0.0:      # there was scour during the interval
		if bsed <= 0.0:   #  bed was scoured "clean"
			bqal  = -1.0e30
			dsqal = -1.0 * rbqals  # cbrb changed sign of dsqal; it should be negative for scour; fixed 4/2007 
		else:              # there is still bed material left
			bqal = rbqals / (bsed - depscr)
			dsqal= bqal * depscr

		# calculate concentration in suspension-under these conditions,
		# denominator should never be zero
		if rsed + rosed > 0.0:
			sqal   = (isqal + rsqals - dsqal) / (rsed + rosed)
		else:
			sqal   = 0.0
		rosqal = rosed * sqal
	else:           # there was deposition or no scour/deposition during the interval
		denom = rsed + depscr + rosed
		if denom <= 0.0:     # there was no sediment in suspension during the interval
			sqal   = -1.0e30
			rosqal = 0.0
			dsqal  = 0.0
			if abs(isqal) > 0.0 or abs(rsqals) > 0.0:
				pass # errmsg: error-under these conditions these values should be zero
		else:   # there was some suspended sediment during the interval
			# calculate conc on suspended sed
			sqal   = (isqal + rsqals) / denom
			rosqal = rosed * sqal
			dsqal  = depscr * sqal
			if rsed <= 0.0:
				# rchres ended up without any suspended sediment-revise
				# value for sqal, but values obtained for rsqal,
				# rosqal, and dsqal are still ok
				sqal = -1.0e30

		# calculate conditions on the bed
		if bsed <= 0.0:     # no bed sediments at end of interval
			bqal = -1.0e30
			if abs(dsqal) > 0.0 or abs(rbqals > 0.0):
				# errmsg:  zrerror-under this condition these values should be zero
				pass
		else:     # there is bed sediment at the end of the interval
			rbqal= dsqal + rbqals
			bqal = rbqal / bsed

	osqal = zeros(nexits)
	# osqal = array([0.0, 0.0, 0.0, 0.0, 0.0])
	if nexits > 1:   # we need to compute outflow through each individual exit
		if rosed <= 0.0:    # all zero
			for i in range(nexits):
				osqal[i]=0.0
		else:
			for i in range(nexits):
				osqal[i]= rosqal * osed[i] / rosed

	return ecnt, sqal, bqal, dsqal, rosqal, osqal


def ddecay (qalfg,tw20,ka,kb,kn,thhyd,phval,kox,thox,roc,fact2,fact1,photpm,korea,cfgas,biocon,thbio,
			bio,fstdec,thfst,volsp,dqal,hr,delt60):
	''' estimate decay of dissolved constituent'''

	# bio,biopm(2),cfgas,ddqal(7),delt60,dqal,fact1,fact2(18),genpm(2),hydpm(4),korea,photpm(20),phval, roc,roxpm(2),tw20,volsp

	ddqal = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	if dqal > 1.0e-25:     # simulate decay
		k = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

		k[1] = 0.0
		if qalfg[1] == 1:  # simulate hydrolysis
			khyd = ka * 10.0**(-phval) + kb * 10.0**(phval - 14.0) + kn
			k[1] = khyd * thhyd**tw20  # adjust for temperature

		k[2] = 0.0
		if qalfg[2] == 1:   # simulate oxidation by free radical processes
			krox = kox * roc
			k[2] = krox * thox**tw20  # adjust for temperature

		k[3] = 0.0
		if qalfg[3] == 1:     # simulate photolysis
			# go through summation over 18 wave-length intervals
			fact3 = 0.0
			for l in range(1, 18):
				fact3 = fact3 + fact2[l] * photpm[l]
			k[3] = fact1 * photpm[19] * fact3 * photpm[20]**tw20
		if delt60 < 24.0:
			if 18 > hr >= 6:  # it is a daylight hour; photolysis rate is doubled for this interval
				k[3] = 2.0 * k[3]
			else:     # it is not a daylight hour; photolysis does not occur
				k[3] = 0.0
		# else:
			# simulation interval is greater than 24 hours;
			# no correction is made to photolysis rate to
			# represent diurnal fluctuation

		# simulate volatilization
		k[4] = korea * cfgas  if qalfg[4] == 1 else 0.0

		# simulate biodegradation
		k[5] = biocon * bio * thbio**tw20  if qalfg[5] == 1 else 0.0

		# simulate simple first-order decay
		k[6] = fstdec * thfst**tw20  if qalfg[6] == 1 else 0.0

		# get total decay rate
		k7 = k[1] + k[2] + k[3] + k[4] + k[5] + k[6]

		# calculate the total change in material due to decay-units are conc*vol/l.ivl
		ddqal[7] = dqal * (1.0 - exp(-k7)) * volsp

		# prorate among the individual decay processes- the method used
		# for proration is linear, which is not strictly correct, but
		# should be a good approximation under most conditions
		for i in range(1, 7):
			if k7 > 0.0:
				ddqal[i] = k[i] / k7  * ddqal[7]
			else:
				ddqal[i] = 0.0

	return ddqal

def expand_GQUAL_masslinks(flags, uci, dat, recs):
	if flags['GQUAL']:
		ngqual = 1
		if 'PARAMETERS' in uci:
			ui = uci['PARAMETERS']
			if 'NGQUAL' in ui:
				ngqual = ui['NGQUAL']
		for i in range(1, ngqual+1):
			# IDQAL                            # loop for each gqual
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'GQUAL'
			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'RODQAL'
				rec['SMEMSB1'] = str(i)   # first sub is qual index
				rec['SMEMSB2'] = ''
			else:
				rec['SMEMN'] = 'ODQAL'
				rec['SMEMSB1'] = str(i)       # qual index
				rec['SMEMSB2'] = dat.SMEMSB1  # exit number
			rec['TMEMN'] = 'IDQAL'
			rec['TMEMSB1'] = dat.TMEMSB1
			rec['TMEMSB2'] = dat.TMEMSB2
			rec['SVOL'] = dat.SVOL
			recs.append(rec)
			# ISQAL1
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'GQUAL'
			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'ROSQAL'
				rec['SMEMSB1'] = '1'     # for sand
				rec['SMEMSB2'] = str(i)  # second sub is qual index
			else:
				rec['SMEMN'] = 'OSQAL'
				rec['SMEMSB1'] = str(i)  # qual i
				rec['SMEMSB2'] = '1' + dat.SMEMSB1 # for clay for exit number
			rec['TMEMN'] = 'ISQAL1'
			rec['TMEMSB1'] = dat.TMEMSB1
			rec['TMEMSB2'] = dat.TMEMSB2
			rec['SVOL'] = dat.SVOL
			recs.append(rec)
			# ISQAL2
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'GQUAL'
			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'ROSQAL'
				rec['SMEMSB1'] = '2'     # for silt
				rec['SMEMSB2'] = str(i)  # second sub is qual index
			else:
				rec['SMEMN'] = 'OSQAL'
				rec['SMEMSB1'] = str(i)  # qual i
				rec['SMEMSB2'] = '2' + dat.SMEMSB1 # for clay for exit number
			rec['TMEMN'] = 'ISQAL2'
			rec['TMEMSB1'] = dat.TMEMSB1
			rec['TMEMSB2'] = dat.TMEMSB2
			rec['SVOL'] = dat.SVOL
			recs.append(rec)
			# ISQAL3
			rec = {}
			rec['MFACTOR'] = dat.MFACTOR
			rec['SGRPN'] = 'GQUAL'
			if dat.SGRPN == "ROFLOW":
				rec['SMEMN'] = 'ROSQAL'
				rec['SMEMSB1'] = '3'     # for clay
				rec['SMEMSB2'] = str(i)  # second sub is qual index
			else:
				rec['SMEMN'] = 'OSQAL'
				rec['SMEMSB1'] = str(i)  # qual i
				rec['SMEMSB2'] = '3' + dat.SMEMSB1 # for clay for exit number
			rec['TMEMN'] = 'ISQAL3'
			rec['TMEMSB1'] = dat.TMEMSB1
			rec['TMEMSB2'] = dat.TMEMSB2
			rec['SVOL'] = dat.SVOL
			rec['INDEX'] = str(i)
			recs.append(rec)
	return recs

def hour24Flag(siminfo, dofirst=False):
    '''timeseries with hour values'''
    hours24 = zeros(24)
    for i in range(0,24):
        hours24[i] = i
    return hoursval(siminfo, hours24, dofirst)
