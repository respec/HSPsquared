''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import zeros, array
from numba import jit
from HSP2.RQUTIL import sink, decbal
from HSP2.utilities  import make_numba_dict

ERRMSGS=('Placeholder')

def nutrx(store, siminfo, uci, ts):
	''' Determine primary inorganic nitrogen and phosphorus balances'''

	errors = zeros(len(ERRMSGS), dtype=int)

	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData

	simlen = siminfo['steps']
	delt   = siminfo['delt']
	delt60 = siminfo['delt'] / 60
	uunits = siminfo['units']

	ui = make_numba_dict(uci)

	# table-type nut-flags
	TAMFG  = ui['NH3FG']
	NO2FG  = ui['NO2FG']
	PO4FG  = ui['PO4FG']
	AMVFG  = ui['AMVFG']
	DENFG  = ui['DENFG']
	ADNHFG = ui['ADNHFG']
	ADPOFG = ui['ADPOFG']
	PHFLAG = ui['PHFLAG']
	BENRFG = ui['BENRFG']
	
	# table-type nut-ad-flags
	NUADFG = zeros(7)
	for j in range(1, 7):
		NUADFG[j] = ui['NUADFG(' + str(j) + ')']

	#NUADFG = ui['NUADFG']    # dimension = 3 for  NO3, NH3, PO4

	if (TAMFG == 0 and (AMVFG == 1 or ADNHFG == 1)) or (PO4FG == 0 and ADPOFG == 1):
		pass
		# error - either: 1) tam is not being simulated, and nh3 volat. or
		# nh4 adsorption is being simulated; or 2) po4 is not being
		# simulated, and po4 adsorption is being simulated
		# ERRMSG:

	if (ADNHFG == 1 or ADPOFG == 1) and SEDFG == 0:
		pass
		# ERRMSG: error - sediment associated nh4 and/or po4 is being simulated,but sediment is not being simulated in section sedtrn

	uafxm = zeros((13,4))
	if NUADFG[1] > 0:
		uafxm[:,1] = ui['NUAFXM1']
	if NUADFG[2] > 0:
		uafxm[:,2] = ui['NUAFXM2']
	if NUADFG[3] > 0:
		uafxm[:,3] = ui['NUAFXM3']		
	
	# convert units to internal
	if uunits == 1:     # convert from lb/ac.day to mg.ft3/l.ft2.ivl
		uafxm[:,1]  *= 0.3677 * delt60 / 24.0		
		uafxm[:,2]  *= 0.3677 * delt60 / 24.0
		uafxm[:,3]  *= 0.3677 * delt60 / 24.0		
	else:	             # convert from kg/ha.day to mg.m3/l.m2.ivl		
		uafxm[:,1]  *= 0.1 * delt60 / 24.0		
		uafxm[:,2]  *= 0.1 * delt60 / 24.0
		uafxm[:,3]  *= 0.1 * delt60 / 24.0	
		
	# conversion factors - table-type conv-val1
	cvbo   = ui['CVBO']
	cvbpc  = ui['CVBPC']
	cvbpn  = ui['CVBPN']
	bpcntc = ui['BPCNTC']

	# calculate derived values
	cvbp = (31.0 * bpcntc) / (1200.0 * cvbpc)
	cvbn = 14.0 * cvbpn * cvbp / 31.0
	cvoc = bpcntc / (100.0 * cvbo)
	cvon = cvbn / cvbo
	cvop = cvbp / cvbo

	if BENRFG == 1 or PLKFG == 1:    # benthal release parms - table-type nut-benparm
		brnit1 = ui['BRNIT1']  * delt60    #  convert units from 1/hr to 1/ivl
		brnit2 = ui['BRNIT2']  * delt60    #  convert units from 1/hr to 1/ivl
		brpo41 = ui['BRPO41'] * delt60    #  convert units from 1/hr to 1/ivl

	# nitrification parameters - table-type nut-nitdenit
	ktam20 = ui['KTAM20'] * delt60     # convert units from 1/hr to 1/ivl
	kno220 = ui['KNO220'] * delt60     # convert units from 1/hr to 1/ivl
	tcnit  = ui['TCNIT']
	kno320 = ui['KNO320'] * delt60     # convert units from 1/hr to 1/ivl
	tcden  = ui['TCDEN']
	denoxt = ui['DENOXT']

	if TAMFG == 1 and AMVFG == 1:   # ammonia volatilization parameters table nut-nh3volat
		expnvg = ui['EXPNVG']
		expnvl = ui['EXPNVL']

	if TAMFG == 1 and PHFLAG == 3:     # monthly ph values table mon-phval, not in RCHRES.SEQ
		phvalm = ui['PHVALM']

	nupm3 = zeros(7)
	nuadpm = zeros(7)
	rsnh4 = zeros(13)
	rspo4 = zeros(13)

	if (TAMFG == 1 and ADNHFG == 1) or (PO4FG == 1 and ADPOFG == 1):
		# bed sediment concentrations of nh4 and po4 - table nut-bedconc, not in RCHRES.SEQ
		nupm3[:] = ui['NUPM3']  /1.0E6   # convert concentrations from mg/kg to internal units of mg/mg
		
		# initialize adsorbed nutrient mass storages in bed
		rsnh4[8] = 0.0
		rspo4[8] = 0.0
		"""
		do 70 i= 5,7
			rsnh4(i) = bnh4(i-4) * rsed(i)
			rspo4(i) = bpo4(i-4) * rsed(i)
			rsnh4(8) = rsnh4(8)  + rsnh4(i)
			rspo4(8) = rspo4(8)  + rspo4(i)
		"""
            
		# adsorption parameters - table-type nut-adsparm
		nuadpm[:] = ui['NUADPM']  # dimension 6; NH4 (sand, silt, cla) and PO4 (sand, silt, clay)

	# initial conditions - table-type nut-dinit
	dnust = zeros(7); dnust2 = zeros(7)
	dnust[1] = ui['NO3'];   dnust2[1] *= vol
	dnust[2] = ui['TAM'];	dnust2[2] *= vol
	dnust[3] = ui['NO2'];	dnust2[3] *= vol
	dnust[4] = ui['PO4'];	dnust2[4] *= vol
	
	if TAMFG == 1:  # do the tam-associated initial values (nh4 nh3 phval)
		phval = ui['PHVAL']
		# assume nh4 and nh3 are 0.99 x tam and 0.01 x tam respectively
		dnust[5] = 0.99 * dnust[2];   dnust2[5] = dnust[5] * vol
		dnust[6] = 0.01 * dnust[2];   dnust2[6] = dnust[6] * vol

	if (TAMFG == 1 and ADNHFG == 1) or (PO4FG == 1 and ADPOFG == 1):
		# suspended sediment concentrations of nh4 and po4 - table nut-adsinit
		# (input concentrations are mg/kg - these are converted to mg/mg for
		# internal computations)
		snh4[:] = ui['SNH4'] / 1.0e6  # suspended nh4 (sand, silt, clay) 
		spo4[:] = ui['SPO4'] / 1.0e6  # suspended po4 (sand, silt, clay) 
		# initialize adsorbed nutrient mass storages in suspension
		rsnh4[4] = 0.0
		rspo4[4] = 0.0
		for i in range(1, 4):
			rsnh4[i] = snh4[i] * rsed[i]
			rspo4[i] = spo4[i] * rsed[i]
			rsnh4[4] += rsnh4[i]
			rspo4[4] += rspo4[i]
		# initialize totals on sand, silt, clay, and grand total
		rsnh[9]  = rsnh[1] + rsnh[5]
		rsnh[10] = rsnh[2] + rsnh[6]
		rsnh[11] = rsnh[3] + rsnh[7]
		rsnh[12] = rsnh[4] + rsnh[8]
		rspo4[9]  = rspo4[1] + rspo4[5]
		rspo4[10] = rspo4[2] + rspo4[6]
		rspo4[11] = rspo4[3] + rspo4[7]
		rspo4[12] = rspo4[4] + rspo4[8]

	# initialize total storages of nutrients in reach
	nust = zeros((5,2))

	nust[1,1] = dnust2[1]
	nust[2,1] = dnust2[2]
	if ADNHFG == 1:
		nust[2,1] += rsnh[4]

	nust[3,1] = dnust2[3]
	nust[4,1] = dnust2[4]
	if ADPOFG == 1:
		nust[4,1] += rspo4[4]

	# initialize nutrient flux if nutrient is not simulated
	otam = ono2 = opo4 = zeros(nexits)
	rosnh4 = rospo4 = zeros(5)
	dspo4 = dsnh4 = zeros(5)
	adpo4 = adnh4 = zeros(5)
	ospo4 = osnh4 = zeros((nexits, 5))
	nucf1 = zeros((5,2))
	nucf2 = nucf3 = nucf8 = tnucf2 = zeros((5,3))
	nucf4 = zeros((8,2))
	nucf5 = zeros((9,2))
	nucf6 = zeros((2,2))
	nucf7 = zeros((7,2))

	if TAMFG == 0:
		nucf1[2,1] = 0.0
		otam[:] = 0.0   # dimension nexits

	if ADNHFG == 0:
		rosnh4[:]  = 0.0
		dsnh4[:]   = 0.0
		adnh4[:]   = 0.0
		osnh4[:,:] = 0.0

	if NO2FG == 0:
		nucf1[3,1] = 0.0
		ono2[:]    = 0.0

	if PO4FG == 0:
		nucf1[4,1] = 0.0
		opo4[:]    = 0.0

	if ADPOFG == 0:
		rospo4[:]  = 0.0
		dspo4[:]   = 0.0
		adpo4[:]   = 0.0
		ospo4[:,:] = 0.0

	# initialize nutrient process fluxes (including ads/des and dep/scour)
	nucf4[1:7,1] = 0.0
	nucf5[1:7,1] = 0.0
	nucf7[1:7,1] = 0.0
	
	nucf4[7,1] = 0.0
	nucf5[7,1] = 0.0
	nucf5[8,1] = 0.0
	nucf6[1,1] = 0.0

	nucf3[1:5,1:3] = 0.0
	nucf8[1:5,1:3] = 0.0

	return errors, ERRMSGS

	#@jit(nopython=True)	
	def nutrx(dox, bod, tw, ino3, inh3, ino2, ipo4, nuafx, nuacn, prec, sarea, advData):
		''' Determine primary inorganic nitrogen and phosphorus balances'''
		
		#compute atmospheric deposition influx
		for i in range(1,4):
			n= 2*(i-1)+ 1
			# dry deposition
			if nuadfg[n] <= -1:
				nuadfx = ts['nuadfx']
				nuaddr[i] = sarea*nuadfx
			elif nuadfg[n] >= 1:
				nuaddr[i] = sarea*dayval(nuafxm[mon,i],nuafxm[nxtmon,i],day, ndays)
			else:
				nuaddr[i] = 0.0
			# wet deposition
			if nuadfg[n+1] <= -1:
				nuadcn = ts['nuadcn']
				nuadwt[n]= prec*sarea*nuadcn
			elif (nuadfg[n+1] >= 1):
				nuadwt[i] = prec*sarea*dayval(nuacnm[mon,i],nuacnm[nxtmon,i],day,ndays)
			else:
				nuadwt[i] = 0.0
			nuadep[i]= nuaddr[i]+ nuadwt[i]

		# get inflowing material from pad
		if ino3fp > 0:
			ino3 = ui['INO3']  # else zero if missing
		tnuif[1] = ino3
		inno3 = ino3 + nuadep[1]

		# advect nitrate
		no3, rono3,ono3 = advect(inno3, no3,rono3, ono3)

		nucf1[1] = rono3
		if nexits > 1:
			tnucf2[:,1] = ono3[:]   # nexits

		if TAMFG:
			if itamfp > 0:
				itam= ui['ITAM']  # or zero if missing
			intam = itam + nuadep[2]
			# advect total ammonia
			tam, rotam, otam = advect(intam, tam, rotam,otam)

		if NO2FG:
			if ino2fp > 0:
				ino2 = ui['INO2']
			tnuif[3] = ino2
			# advect nitrite
			no2, rono2, ono2 =  advect(ino2, no2, rono2,ono2)
			tnucf1[3] = rono2
			if nexits > 1:
				tnucf2[:,3] = ono2[:]   # nexits
				
		if PO4FG:
			if ipo4fp > 0:
				ipo4 = ui['IPO4']  # or zero if missing
			inpo4 = ipo4 + nuadep[3]
			# advect ortho-phosphorus
			po4, ropo4, opo4 = advect(inpo4,po4,ropo4,opo4)

		if ADPOFG:       # advect adsorbed phosphate
			# zero the accumulators
			ispo4[4]  = 0.0
			dspo4[4]  = 0.0
			rospo4[4] = 0.0
			if nexits > 1:
				ospo4[:,4] = 0.0  # nexits

			# repeat for each sediment fraction
			for j in range(1, 4):       # get data on sediment-associated phosphate
				fpt = ispofp[j]
				if fpt:
					ispo4[j] = ui['ISPO4']   # else zero if missing

				nuecnt[3],spo4[j],dspo4[j], rospo4[j],ospo4[1,j] = \
					advnut(ispo4[j],rsed[j],rsed[j +3],depscr[j],rosed[j],osed[1,j],nexits, \
					rchno,messu,msgfl,datim,nutid[2],j,rspo4[j],rspo4[j + 4],bpo4[j], \
					nuecnt[3],spo4[j],dspo4[j], rospo4[j],ospo4[1,j]) 

				ispo4[4]  += ispo4[j]
				dspo4[4]  += dspo4[j]
				rospo4[4] += rospo4[j]
				if nexits > 1:
					ospo4[:,4] += ospo4[:,j]   # nexits
			tnuif[4]  = ipo4  + ispo4[4]
			tnucf1[4] = ropo4 + rospo4[4]
			if nexits > 1:
				tnucf2[:,4] = opo4[:]+ ospo4[:,4]  # nexits
		else:            # no adsorbed fraction
			tnuif[4]  = ipo4
			tnucf1[4] = ropo4
			if nexits > 1:
				tnucf2[:,4] = opo4[:]

		if TAMFG and ADNHFG:    # advect adsorbed ammonium
			# zero the accumulators
			isnh4[4]  = 0.0
			dsnh4[4]  = 0.0
			rosnh4[4] = 0.0
			if nexits > 1:
				osnh4[:,4] = 0.0   # nexits

			# repeat for each sediment fraction
			for j in range(1, 4):
				# get data on sediment-associated ammonium
				fpt = isnhfp[j]
				isnh4[j]= ui['OSNH4']  # or zero if not there

				nuecnt[3],snh4[j],dsnh4[j],rosnh4[j],osnh4[1,j] = \
					advnut (isnh4[j],rsed[j],rsed[j + 3],depscr[j],rosed[j],osed[1,j],nexits, \
						    rchno,messu,msgfl,datim, nutid[1],j,rsnh[j],rsnh4[j + 4],bnh4[j], \
							nuecnt[3],snh4[j],dsnh4[j],rosnh4[j],osnh4[1,j])

				isnh4[4]  = isnh4[4]  + isnh4[j]
				dsnh4[4]  = dsnh4[4]  + dsnh4[j]
				rosnh4[4] = rosnh4[4] + rosnh4[j]
				if nexits > 1:
					osnh4[:,4] = osnh4[:,4] + osnh4[:,j]   # nexits
			tnuif[2]  = itam + isnh4[4]
			tnucf1[2] = rotam + rosnh4[4]
			if nexits > 1:
				tnucf2[:,2] = otam[:] + osnh4[:,4]  # nexits
		else:                 # no adsorbed fraction
			tnuif[2]  = itam
			tnucf1[2] = rotam
			if nexits > 1:
					tnucf2[:,2] = otam[:]  # nexits

		if TAMFG:     # calculate ammonia ionization in water column
			# get ph values
			#TMR ph = ??? # last computed value, time series, monthly, constant; phflag
			# compute ammonia ionization
			nh3, nh4 = ammion(tw, hval, tam, nh3, nh4)

		if avdepe > 0.17:
			if BENRFG:
				# simulate benthal release of inorganic nitrogen and
				# ortho-phosphorus; and compute associated fluxes
				if TAMFG:
					tam, bentam = benth(dox,anaer,brtam,scrfac,depcor,tam,bentam)
					bnrtam = bentam * voL

				if PO4FG:
					po4, benpo4 = benth(dox,anaer,brpo4,scrfac,depcor, po4, benpo4)
					bnrpo4 = benpo4 * vol

			if TAMFG:
				if AMVFG:     # compute ammonia volatilization
					twkelv = tw + 273.16        # convert water temperature to degrees kelvin 
					avdepm = avdepe * 0.3048    # convert depth to meters
					tam, nh3vlt = nh3vol(expnvg,expnvl,korea,wind,delt60,delts,avdepm,twkelv,tw,phval,tam, nh3vlt)
					volnh3 = -nh3vlt * vol
				else:
					volnh3 = 0.0

				# calculate amount of nitrification; nitrification does not
				# take place if the do concentration is less than 2.0 mg/l
				tam,no2,no3,dox, dodemd,tamnit,no2ntc,no3nit = \
					nitrif(ktam20,tcnit,tw,no2fg,kno220,tam,no2,no3,dox,dodemd,tamnit,no2ntc,no3nit)

				# compute nitrification fluxes
				nitdox = -dodemd * vol
				nittam = -tamnit * vol
				nitno2 =  no2ntc * vol
				nitno3 =  no3nit * vol

			if DENFG:    # consider denitrification processes, and compute associated fluxes
				no3, no3de = denit(kno320, tcden, tw, dox, denoxt, no3, no3de)
				denno3 = -no3de * vol

			# calculate amount of inorganic constituents released by bod decay in reach water
			decnit = bodox * cvon
			decpo4 = bodox * cvop
			decco2 = bodox * cvoc

			# update state variables of inorganic constituents which
			# are end products of bod decay; and compute associated fluxes
			tam, no3, po4 = decbal(tamfg, po4fg, decnit, decpo4, tam, no3, po4)
			if TAMFG:
				bodtam = decnit * vol
			else:
				bodno3 = decnit * vol

			if PO4FG == 1:
				bodpo4 = decpo4 * vol

			if PO4FG and SEDFG and ADPOFG:   # compute adsorption/desorption of phosphate
				po4, spo4[1], dumxxx, adpo4[1] = addsnu(vol, rsed[1], adpopm[1], po4, spo4[1], dumxxx, adpo4[1])

			if TAMFG and SEDFG and ADNHFG:  # compute adsorption/desorption of ammonium
				# first compute ammonia ionization
				nh3, nh4 = ammion(tw, phval, tam, nh3, nh4)
				nh4, snh4[1], tam, adnh4[1] = addsnu(vol, rsed[1], adnhpm[1], nh4, snh4[1], tam, adnh4[1])
				# then re-compute ammonia ionization
				nh3, nh4 = ammion (tw, phval, tam, nh3, nh4)
		else:
			# too little water is in reach to warrant simulation of quality processes
			decnit = 0.0
			decpo4 = 0.0
			decco2 = 0.0
			nitdox = 0.0
			denbod = 0.0
			nittam = 0.0
			bnrtam = 0.0
			volnh3 = 0.0
			bodtam = 0.0
			nitno2 = 0.0
			nitno3 = 0.0
			denno3 = 0.0
			bodno3 = 0.0
			bnrpo4 = 0.0
			bodpo4 = 0.0
			for k in range(1, 5):
				adnh4[k] = 0.0
				adpo4[k] = 0.0
			# 80     continue
		
		totdox = readox + boddox + bendox + nitdox
		totbod = decbod + bnrbod + snkbod + denbod
		totno3 = nitno3 + denno3 + bodno3
		tottam = nittam + volnh3 + bnrtam + bodtam
		totpo4 = bnrpo4 + bodpo4

		if PO4FG and SEDFG and ADPOFG:  # find total quantity of phosphate on various forms of sediment
			totpm1 = 0.0
			totpm2 = 0.0
			totpm3 = 0.0
			for j in range(1, 4):
				rspo4[j]     = spo4[j] * rsed[j]         # compute mass of phosphate adsorbed to each suspended fraction
				rspo4[j + 4] = bpo4[j] * rsed[j + 3]     # compute mass of phosphate adsorbed to each bed fraction
				rspo4[j + 8] = rspo4[j] + rspo4[j + 4]   # compute total mass of phosphate on each sediment fraction
				
				totpm1 = totpm1 + rspo4[j]
				totpm2 = totpm2 + rspo4[j + 4]
				totpm3 = totpm3 + rspo4[j + 8]

			rspo4[4]  = totpm1	 # compute total suspended phosphate
			rspo4[8]  = totpm2   # compute total bed phosphate
			rspo4[12] = totpm3   # compute total sediment-associated phosphate

		if TAMFG and SEDFG and ADNHFG:    # find total amount of ammonium on various forms of sediment
			totnm1 = 0.0
			totnm2 = 0.0
			totnm3 = 0.0
			for j in range(1, 4):
				rsnh4[j]     = snh4[j]  * rsed[j]       # compute mass of ammonium adsorbed to each suspended fraction
				rsnh4[j + 4] = bnh4[j]  * rsed[j + 3]   # compute mass of ammonium adsorbed to each bed fraction
				rsnh4[j + 8] = rsnh4[j] + rsnh4[j + 4]  # compute total mass of ammonium on each sediment fraction
				
				totnm1 += rsnh4[j]
				totnm2 += rsnh4[j + 4]
				totnm3 += rsnh4[j + 8]
			rsnh4[4]  = totnm1      # compute total suspended ammonium
			rsnh4[8]  = totnm2		# compute total bed ammonium
			rsnh4[12] = totnm3      # compute total sediment-associated ammonium

		return (dox, bod, orn, orp, orc, torn, torp, torc, potbod, phyto, zoo, benal, \
		 phycla, balcla, rophyto, rozoo, robenal, rophycla, robalcla, ophyto, ozoo, \
		 obenal, ophycla, obalcla, binv, pladfx, pladcn)
	return nutrx



def addsnu(vol, rsed, adpm, dnut, snut, dnutxx, adnut):
	''' simulate exchange of nutrient (phosphate or ammonium) between the
	dissolved state and adsorption on suspended sediment- 3 adsorption
	sites are considered: 1- suspended sand  2- susp. silt
	3- susp. clay
	assumes instantaneous linear equilibrium'''

	if vol > 0.0:    # adsorption/desorption can take place
		# establish nutrient equilibrium between reach water and suspended sediment; first find the new dissolved nutrient conc. in reach water
		dnutin = dnut
		num    = vol * dnut
		denom  = vol

		for j in range(1, 4):
			if rsed[j] > 0.0:   # accumulate terms for numerator and denominator in dnut equation
				num   += snut[j] * rsed[j]
				denom += adpm[j] * rsed[j]

		dnut  = num / denom 		        # calculate new dissolved concentration-units are mg/l
		dnutxx= dnutxx - (dnutin - dnut)  	# also calculate new tam conc if doing nh4 adsorption

		# calculate new conc on each sed class and the corresponding adsorption/desorption flux
		adnut[4] = 0.0

		for j in range(1, 4):
			if rsed[j] > 0.0:    # this sediment class is present-calculate data pertaining to it
				temp = dnut * adpm[j]  # new concentration

				# quantity of material transferred
				# adnut[j]= (temp - snut[j])*rsed[j]
				snut[j] = temp

				# accumulate total adsorption/desorption flux above bed
				adnut[4] += adnut[j]

			else:    # this sediment class is absent
				adnut[j] = 0.0
				# snut[j] is unchanged-"undefined"

	else:   # no water, no adsorption/desorption
		adnut[1:8] = 0.0
		# snut(1 thru 3) and dnut should already have been set to undefined values

	return dnut, snut, dnutxx, adnut


def  advnut(isnut,rsed,bsed,depscr,rosed,osed,nexits, rchno,messu,msgfl,datim,
			nutid,j,rsnuts,rbnuts,bnut, ecnt,snut,dsnut,rosnut,osnut):

	''' simulate the advective processes, including deposition and scour for the
	inorganic nutrient adsorbed to one sediment size fraction'''

	if depscr < 0.0:   # there was sediment scour during the interval
		# compute flux of nutrient mass into water column with scoured sediment fraction
		dsnut = bnut * depscr

		# calculate concentration in suspension-under these conditions, denominator should never be zero
		snut   = (isnut + rsnuts - dsnut) / (rsed + rosed)
		rosnut = rosed * snut
	else:  # there was deposition or no scour/deposition during the interval
		denom = rsed + depscr + rosed
		if denom == 0.0:   # there was no sediment in suspension during the interval
			snut   = -1.0e30
			rosnut = 0.0
			dsnut  = 0.0

			# fix sed-nut problem caused by very small sediment loads that are stored in
			# wdm file as zero (due to wdm attribute tolr > 0.0) when adsorbed nut load
			# is not zero; changed comparison from 0.0 to 1.0e-3; this should not cause
			# any mass balance errors since the condition is not likely to exist over a
			# long period and will be insignificant compared to
			# the total mass over a printout period; note that 1.0e-3 mg*ft3/l is 0.028 mg
			# (a very, very small mass)
			if abs(isnut) > 1.0e-3 or abs(rsnuts) > 1.0e-3:
				pass
				# errmsg: error-under these conditions these values should be zero
		else:		# there was some suspended sediment during the interval
			# calculate conc on suspended sed
			snut  = (isnut + rsnuts) / denom
			rosnut= rosed * snut
			dsnut = depscr * snut

			if rsed == 0.0:
				# rchres ended up without any suspended sediment-revise
				# value for snut, but values obtained for rosnut, and dsnut are still ok
				snut = -1.0e30

		# calculate conditions on the bed
		if bsed == 0.0:
			# no bed sediments at end of interval
			if abs(dsnut) > 0.0 or abs(rbnuts) > 0.0:
				pass # errsg: error-under this condition these values should be zero

	if nexits > 1:
		# compute outflow through each individual exit
		if rosed == 0.0:        # all zero
			osnut[:] = 0.0
		else:
			osnut[:] = rosnut * osed[:] / rosed

	return ecnt, snut, dsnut, rosnut, osnut 




def ammion(tw, ph, tam, nh3, nh4):
	''' simulate ionization of ammonia to ammonium using empirical relationships developed by loehr, 1973'''

	if tam >= 0.0:   # tam is defined, compute fractions
		# adjust very low or high values of water temperature to fit limits of dat used to develop empirical relationship
		if   tw < 5.0:   twx = 5.0
		elif tw > 35.0:  twx = 35.0
		else:            twx = tw
		
		if   ph < 4.0:   phx = 4.0
		elif ph > 10.0:  phx = 10.0
		else:            phx = ph
				
		# compute ratio of ionization constant values for aqueous ammonia and water at current water temperatue
		ratio = (-3.39753 * log(0.02409 * twx)) * 1.0e9

		# compute fraction of total ammonia that is un-ionized
		frac = 10.0**(phx) / (10.0**phx + ratio)

		# update nh3 and nh4 state variables to account for ionization
		nh3 =  frac * tam
		nh4 =  tam - nh3
	else:     # tam conc undefined
		nh3 = -1.0e30
		nh4 = -1.0e30
	return nh3, nh4


def denit(kno320, tcden, tw, dox, denoxt, no3, denno3):
	''' calculate amount of denitrification; denitrification does not take place
	if the do concentration is above user-specified threshold do value (denoxt)'''

	if dox <= denoxt:      # calculate amount of no3 denitirified to nitrogen gas
		denno3 = 0.0
		if no3 > 0.001:
			denno3 = kno320 * (tcden**(tw - 20.0)) * no3
			no3    = no3 - denno3
			if no3 < 0.001:             # adjust amount of no3 denitrified so that no3 state variable is not a negative number; set no3 to a value of .001 mg/l
				denno3 = denno3 + no3 - 0.001
				no3    = 0.001
	else:
		denno3 = 0.0          # denitrification does not occur
	return no3, denno3


def hcintp (phval, tw, hcnh3):
	''' calculate henry's constant for ammonia based on ph and water temperature'''

	xtw    = array([4.44, 15.56, 26.67, 37.78])
	xhplus = array([1.0, 10.0, 100.0, 1000.0, 10000.0])
	yhenc  = array([0.000266, 0.000754, 0.00198, 0.00486, 0.00266, 0.00753, 0.0197,
	0.0480, 0.0263, 0.0734, 0.186, 0.428, 0.238, 0.586, 1.20, 2.05, 1.2, 1.94, 2.65, 3.31])  # dimensions: fortran 4,5

	# adjust very low or very high values of water temperature to fit limits of henry's contant data range
	if tw < 4.44:      # use low temperature range values for henry's constant (4.4 degrees c or 40 degrees f)
		twx = 4.44
	elif tw > 37.78:  # use high temperature range values for henry's constant (37.78 degrees c or 100 degrees f)
		twx = 37.78
	else:             # use unmodified water temperature value in interpolation
		twx = tw

	# convert ph value to a modified version of hydrogen ion concentration
	# because our interpolation routine cant seem to work with small numbers
	hplus = 10.0**(phval) * 1.0e-6

	# adjust very low or very high values of hydrogen ion concentration to fit limits of henry's constant data range
	if hplus > 10000.0:    # use low hydrogen ion concentration range values for henry's constant
		hplus = 10000.0
	elif hplus < 1.0:      # use high hydrogen ion concentration range values for henry's constant
		hplus = 1.0

	# perform two-dimensional interpolation of henry's constant values to estimate henry's
	# constant for water temperature and ph conditions in water column (based on p. 97 of numerical recipes)
	i4 = 4
	i5 = 5
	for i in range(4):        # do 10 i= 1, 4
		for j in range(5):    # do 20 j= 1, 5
			yhtmp[j] = yhenc[i,j]   # copy row into temporary storage
			# 20     continue
		# perform linear interpolation within row of values
		ytwtmp[i] = intrp1(xhplus, yhtmp, i5, hplus, ytwtmp[i])
	# 10   continue

	# do final interpolation in remaining dimension
	hcmf = intrp1(xtw, ytwtmp, i4, twx, hcmf)

	# convert henry's constant from molar fraction form to units of atm.m3/mole:  assume 
	# 1) dilute air and water solutions
	# 2) ideal gas law
	# 3) stp i.e., 1 atm total pressure
	# 4) 1 gram water = 1 cm3

	# xa(air)                        1
	# --------- * -----------------------------------------
	# xa(water)    (1.e+6 m3/g water)/(18.01 g/mole water)

	hcnh3 = hcmf * (18.01 * 1.e-6)

	return hcnh3




def intrp1(xarr, yarr, len, xval, yval):
	''' perform one-dimensional interpolation of henry's constant values for ammonia (based on p. 82 of numerical recipes)'''

	ns = 1
	dif = abs(xval-xarr[0])
	# find the index ns of the closest array entry
	for i in range(len):      # do 10 i= 1, len
		dift = abs(xval - xarr[i])
		if dift < dif:
			ns  = i
			dif = dift

		# initialize correction array values
		c[i] = yarr[i]
		d[i] = yarr[i]

	# select intial approximation of yval
	yval = yarr[ns]
	ns  = ns - 1
	# loop over the current values in correction value arrays (c & d) to update them	
	
	for j in range(len-1):                # do 30 j = 1, len -1
		for i in range (len - j):         # do 20 i = 1, len - j
			ho  = xarr[i] - xval
			hp  = xarr[i + j] - xval
			w   = c[i + 1] - d[i]
			den = ho - hp
			den = w / den
			# update correction array values
			d[i] = hp * den
		c[i] = ho * den
		# 20     continue
		
		# select correction to yval
		if 2 * ns < len-j:
			dyval = c[ns + 1]
		else:
			dyval= d[ns]
			ns   = ns - 1

		# compute yval
		yval = yval + dyval
	# 30   continue
	return yval



def nh3vol(expnvg, expnvl, korea, wind, delt60, delts, avdepm, twkelv, tw, phval, tam, nh3vlt):
	''' calculate ammonia volatilization using two-layer theory'''

	if tam > 0.0:
		# convert reaeration coefficient into units needed for computatuion
		# of bulk liquid film gas transfer coefficient (cm/hr) based on
		# average depth of water
		dokl = korea * (avdepm * 100.0) / delt60

		# compute bulk liquid film gas transfer coefficient for ammonia using
		# equation 183 of mccutcheon; 1.8789 equals the ratio of oxygen
		# molecule molecular weight to ammonia molecular weight
		nh3kl = dokl * 1.8789**(expnvl / 2.0)

		# convert wind speed from meters/ivl (wind) to meters/sec (windsp)
		windsp = wind / delts

		# compute bulk gas film gas transfer coefficient (cm/hr) for ammonia
		# using equation 184 of mccutcheon; the product of the expression
		# (700.*windsp) is expressed in cm/hr; 1.0578 equals the ratio of water
		# molecule molecular weight to ammonia molecular weight
		if windsp <= 0.0:
			windsp = 0.001
		nh3kg = 700.0 * windsp * 1.0578**(expnvg / 2.0)

		# compute henry's constant for ammonia as a function of temperature
		# hcinp() called only here
		hcnh3 = hcintp(phval, tw, hcnh3)

		# avoid divide by zero errors
		chk = nh3kl * hcnh3
		if chk > 0.0:
			# compute overall mass transfer coefficient for ammonia (kr) in cm/hr
			# using equation 177 of mccutcheon; first calculate the inverse of kr
			# (krinv); 8.21e-05 equals ideal gas constant value expressed as
			# atm/degrees k mole
			krinv = (1.0 / nh3kl) + ((8.21e-05) * twkelv) / (hcnh3 * nh3kg)
			kr    = (1.0 / krinv)

			# compute reach-specific gas transfer coefficient (units are /interval)
			knvol = (kr / (avdepm * 100.0)) * delt60
		else:              # korea or hcnh3 was zero (or less)
			knvol = 0.0     

		# compute ammonia flux out of reach due to volatilization;  assumes that
		# equilibrium concentration of ammonia is sufficiently small to be considered zero
		nh3vlt = knvol * tam
		if nh3vlt >= tam:
			nh3vlt = 0.99 * tam
			tam    = 0.01 * tam
		else:
			tam = tam - nh3vlt
	else:                # no ammonia present; hence, no volatilization occurs
		nh3vlt = 0.0
	return tam, nh3vlt


def nitrif(ktam20, tcnit, tw, no2fg, kno220, tam, no2, no3, dox, dodemd, tamnit, no2ntc, no3nit):
	''' calculate amount of nitrification; nitrification does not take place if the do concentration is less than 2.0 mg/l'''
	
	if dox >= 2.0:
		# calculate amount of tam oxidized to no2; tamnit is expressed as mg tam-n/l
		tamnit = 0.0
		if tam > 0.001:
			amnit = ktam20 * (tcnit**(tw - 20.0)) * tam
			tam   = tam - tamnit
			if tam < 0.001:       # adjust amount of tam oxidized so that tam state variable is not a negative number; set tam to a value of .001 mg/l
				tamnit = tamnit + tam - .001
				tam    = .001
		if NO2FG:            # calculate amount of no2 oxidized to no3; no2nit is expressed as mg no2-n/l
			no2nit = 0.0
			if no2 > 0.001:
				no2nit = kno220 * (tcnit**(tw - 20.0)) * no2

			# update no2 state variable to account for nitrification
			if no2nit > 0.0:
				if no2 + tamnit - no2nit <= 0.0:
					no2nit = 0.9 * (no2 + tamnit)
					no2    = 0.1 * (no2 + tamnit)
				else:
					no2 = no2 + tamnit - no2nit
			else:
				no2 = no2 + tamnit
			no2ntc = tamnit - no2nit
		else:                 # no2 is not simulated; tam oxidized is fully oxidized to no3
			no2nit = tamnit
			no2ntc = 0.0

		# update no3 state variable to account for nitrification and compute concentration flux of no3
		no3    = no3 + no2nit
		no3nit = no2nit

		# find oxygen demand due to nitrification
		dodemd = 3.22 * tamnit + 1.11 * no2nit

		if dox < dodemd:
			# adjust nitrification demands on oxygen so that dox will not be zero;  
			# routine proportionally reduces tam oxidation to no2 and no2 oxidation to no3
			rho = dox / dodemd
			if rho < 0.001:
				rho = 0.0
			rhoc3 = (1.0 - rho) * tamnit
			rhoc2 = (1.0 - rho) * no2nit
			tam   = tam + rhoc3
			if NO2FG:
				no2 = no2 - rhoc3 + rhoc2
			no3    = no3 - rhoc2
			dodemd = dox
			dox    = 0.0
			tamnit = tamnit - rhoc3
			no2nit = no2nit - rhoc2
			no3nit = no3nit - rhoc2
			if NO2FG:
				no2ntc = no2ntc - rhoc3 + rhoc2
		else:                            # projected do value is acceptable
			dox = dox - dodemd
	else:                                # nitrification does not occur
		tamnit = 0.0
		no2nit = 0.0
		dodemd = 0.0
		no2ntc = 0.0
		no3nit = 0.0
	return tam, no2, no3, dox, dodemd, tamnit, no2ntc, no3nit
