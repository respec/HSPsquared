''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''

from numpy import array, zeros, int64
from math import log10, exp
from HSP2.utilities import initm, make_numba_dict, hoursval, dayval

ERRMSGS=('Placeholder')

def  plank(store, siminfo, uci, ts):
	'''Simulate behavior of plankton populations and associated reactions'''

	errors = zeros(len(ERRMSGS), dtype=int)

	advectData = uci['advectData']
	(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData

	limit = ['', 'LIT', 'NON', 'TEM', 'NIT',' PO4', 'NONE', 'WAT']
	delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
	simlen = siminfo['steps']
	uunits = siminfo['units']
	
	ui = make_numba_dict(uci)
	
	# flags - table-type PLNK-FLAGS
	PHYFG  = ui['PHYFG']
	ZOOFG  = ui['ZOOFG']
	BALFG  = ui['BALFG']
	SDLTFG = ui['SDLTFG']
	AMRFG  = ui['AMRFG']
	DECFG  = ui['DECFG']
	NSFG   = ui['NSFG']
	ZFOOD  = ui['ZFOOD']
	BNPFG  = ui['BNPFG']

	HTFG   = ui['HTFG']
	TAMFG  = ui['NH3FG']
	PO4FG  = ui['PO4FG']

	bpcntc = ui['BPCNTC']
	cvbo  = ui['CVBO']
	cvbpc  = ui['CVBPC']
	
	if ZOOFG == 1 and PHYFG == 0:
		pass # ERRMSG: error - zooplankton cannot be simulated without phytoplankton
	if NSFG == 1 and TAMFG == 0:
		pass # ERRMSG: error - ammonia cannot be included in n supply if it is not
	if PO4FG == 0:
		pass # ERRMSG: error - phosphate must be simulated if plankton are being

	if BALFG == 2:   # user has selected multiple species with more complex kinetics
		# additional benthic algae flags - table-type BENAL-FLAG
		numbal  = ui['NUMBAL']
		BINVFG  = ui['BINVFG']
		BFIXFG1 = ui['BFIXFG1'] 
		BFIXFG2 = ui['BFIXFG2'] 
		BFIXFG3 = ui['BFIXFG3'] 
		BFIXFG4 = ui['BFIXFG4'] 
	else:
		numbal = BALFG          # single species or none
		
	if HTFG == 0:     # fraction of surface exposed - table-type surf-exposed
		cfsaex = ui['CFSAEX']

	# table-type plnk-parm1
	ratclp = ui['RATCLP']
	nonref = ui['NONREF']
	litsed = ui['LITSED']
	alnpr  = ui['ALNPR']
	extb   = ui['EXTB']
	malgr  = ui['MALGR'] * delt60
	paradf = ui['PARADF']
	refr = 1.0 - nonref       # define fraction of biomass which is refractory material	

	# compute derived conversion factors
	cvbc   = bpcntc / 100.0
	cvnrbo = nonref * cvbo

	cvbp = (31.0 * bpcntc) / (1200.0 * cvbpc)
	cvpb   = 31.0 / (1000.0 * cvbp)
	cvbcl  = 31.0 * ratclp / cvpb

	# table-type plnk-parm2
	cmmlt  = ui['CMMLT']
	cmmn   = ui['CMMN']
	cmmnp  = ui['CMMNP']
	cmmp   = ui['CMMP']
	talgrh = ui['TALGRH']
	talgrl = ui['TALGRL']
	talgrm = ui['TALGRM']

	# table-type plnk-parm3
	alr20 = ui['ALR20'] * delt60   	# convert rates from 1/hr to 1/ivl
	aldl  = ui['ALDL']  * delt60 
	oxald = ui['OXALD'] * delt60
	naldh = ui['NALDH'] * delt60
	paldh = ui['PALDH']

	# table-type plnk-parm4
	nmingr = ui['NMINGR']
	pmingr = ui['PMINGR']
	cmingr = ui['CMINGR']
	lmingr = ui['LMINGR']
	nminc  = ui['NMINC']

	# phytoplankton-specific parms - table-type phyto-parm
	# this table must always be input so that REFSET is read
	seed   = ui['SEED']
	mxstay = ui['MXSTAY']
	oref   = ui['OREF'] 
	claldh = ui['CLALDH']
	physet = ui['PHYSET'] * delt60	# change settling rates to units of 1/ivl
	refset = ui['REFSET'] * delt60	# change settling rates to units of 1/ivl

	if PHYFG == 1 and ZOOFG == 1:   # zooplankton-specific parameters  
		# table-type zoo-parm1
		mzoeat = ui['MZOEAT'] * delt60   # convert rates from 1/hr to 1/ivl
		zfil20 = ui['ZFIL20'] * delt60   # convert rates from 1/hr to 1/ivl
		zres20 = ui['ZRES20'] * delt60   # convert rates from 1/hr to 1/ivl
		zd     = ui['ZD']     * delt60   # convert rates from 1/hr to 1/ivl
		oxzd   = ui['OXZD']   * delt60   # convert rates from 1/hr to 1/ivl
		# table-type zoo-parm2
		tczfil = ui['TCZFIL']
		tczres = ui['TCZRES']
		zexdel = ui['ZEXDEL']
		zomass = ui['ZOMASS']	
	
	if BALFG >= 1:    #   benthic algae-specific parms; table-type benal-parm
		mbal   = ui['MBAL']   / cvpb   # convert maximum benthic algae to micromoles of phosphorus
		cfbalr = ui['CFBALR']
		cfbalg = ui['CFBALG']
		minbal = ui['MINBAL'] / cvpb   # convert maximum benthic algae to micromoles of phosphorus
		campr  = ui['CAMPR']
		fravl  = ui['FRAVL']
		nmaxfx = ui['NMAXFX']
	
		if BALFG == 2:  # user has selected multiple species with more complex kinetics
			mbalgr = zeros(numbal)  # ??? + 1
			tcbalg = zeros(numbal)
			cmmnb =  zeros(numbal)
			cmmpb =  zeros(numbal)
			cmmd1 =  zeros(numbal)
			cmmd2 =  zeros(numbal)
			cslit =  zeros(numbal)
			
			balr20 = zeros(numbal)
			tcbalr = zeros(numbal)
			cslof1 = zeros(numbal)
			cslof2 = zeros(numbal)
			grores = zeros(numbal)		
			for i in range(numbal):
				# species-specific growth parms - table type benal-grow
				mbalgr[i] = ui['MBALGR']  * delt60
				tcbalg[i] = ui['TCBALG']
				cmmnb[i] =  ui['CMMNB']
				cmmpb[i] =  ui['CMMPB']
				cmmd1[i] =  ui['CMMD1']
				cmmd2[i] =  ui['CMMD2']
				cslit[i] =  ui['CSLIT']
				# species-specific resp and scour parms - table type benal-resscr
				balr20[i] = ui['BALR20']  * delt60
				tcbalr[i] = ui['TCBALR']
				cslof1[i] = ui['CSLOF1']  * delt60
				cslof2[i] = ui['CSLOF2']
				grores[i] = ui['GRORES']

			#  grazing and disturbance parms - table-type benal-graze
			cremvl = ui['CREMVL']
			cmmbi  = ui['CMMBI']
			binv   = ui['BINV']
			tcgraz = ui['TCGRAZ']

			cremvl = (cremvl / cvpb) / hrpyr * delt60

			if SDLTFG == 2:	# turbidity regression parms - table-type benal-light
				ctrbq1 = ui['CTRBQ1']
				ctrbq2 = ui['CTRBQ2']
				cktrb1 = ui['CKTRB1']
				cktrb2 = ui['CKTRB2']	

			if BINVFG == 3:      # monthly benthic invertebrate density - table-type mon-binv
				binvm = ui['BINVM']

		# table-type benal-riff1
		frrif  = ui['FRRIF']	
		cmmv   = ui['CMMV']
		rifcq1 = ui['RIFCQ1']
		rifcq2 = ui['RIFCQ2']
		rifcq3 = ui['RIFCQ3']

		# table-type benal-riff2
		rifvf1 = ui['RIFVF1'] 
		rifvf2 = ui['RIFVF2']
		rifvf3 = ui['RIFVF3']
		rifvf4 = ui['RIFVF4']

		rifdf1 = ui['RIFDF1']
		rifdf2 = ui['RIFDF2']
		rifdf3 = ui['RIFDF3']
		rifdf4 = ui['RIFDF4']

	# table-type plnk-init
	phyto = ui['PHYTO']
	zoo   = ui['ZOO']
	#benal = ui['BENAL']
	orn   = ui['ORN']
	orp   = ui['ORP']
	orc   = ui['ORC']

	# atmospheric deposition flags
	PLADFG = array(ui['PLADFG'])   # six  PLADFG1 to PLADFG6;  table-type PLNK-AD-FLAGS
	PLAFXM = zeros(4)

	for j in range(1, 4):
		n = 2*(j - 1) + 1
		if PLADFG[n] > 0:  # monthly flux must be read
			PLAFXM[j] = array(ui[('PLAFXM',j)])
			if uunits == 1:     # convert from lb/ac.day to mg.ft3/l.ft2.ivl
				PLAFXM[j] *= 0.3677 * DELT60 / 24.0
			elif uunits == 2:	      # convert from kg/ha.day to mg.m3/l.m2.ivl
				PLAFXM[j] *= 0.1 * DELT60 / 24.0
	# Note: get PLAFX array from monthly (above), constant, or time series
	# PLAFX is dimension (simlen, 3)
	# same with PLADCN 

	# compute atmospheric deposition influx; [N, P, C]
	pladdr = zeros(4)
	pladwt = zeros(4)
	pladep = zeros(4)
	flxbal = zeros((4,numbal))

	for i in range(1,4):
		n = 2 * (i - 1) + 1
		pladdr[i] = SAREA * PLADFX[i]          # dry deposition
		pladwt[i] = PREC * SAREA * PLADCN[i]   # wet deposition
		pladep[i] = pladdr[i] + pladwt[i]	
	
	if PHYFG == 0:   # initialize fluxes of inactive constituent
		rophyt = 0.0
		ophyt[:] = 0.0	#nexits
		phydox = 0.0
		phybod = 0.0
		phytam = 0.0
		phyno3 = 0.0
		phypo4 = 0.0
		phyorn = 0.0
		phyorp = 0.0
		phyorc = 0.0
		pyco2  = 0.0
		dthphy = 0.0
		grophy = 0.0
		totphy = 0.0

	if ZOOFG == 1:   # convert zoo to mg/l
		zoo *= zomass
	else:   #  zooplankton not simulated, but use default values
		# initialize fluxes of inactive constituent
		rozoo = 0.0
		ozoo[:] = 0.0	#nexits
		zoodox = 0.0
		zoobod = 0.0
		zootam = 0.0
		zoono3 = 0.0
		zoopo4 = 0.0
		zooorn = 0.0
		zooorp = 0.0
		zooorc = 0.0
		zoophy = 0.0
		zoco2  = 0.0
		grozoo = 0.0
		dthzoo = 0.0
		totzoo = 0.0

	benal = zeros(numbal)
	if numbal == 1:      # single species
		benal[1] = ui['BENAL']       # points to  table-type plnk-init above for rvals
	elif numbal >= 2:      # multiple species - table-type benal-init
		for n in range(numbal):
			str_ba = 'BENAL' + str(n+1)
			benal[n] = ui[str_ba] 	# how to get multiples???
	else:                     # no benthic algae simulated
		baldox = 0.0
		balbod = 0.0
		baltam = 0.0
		balno3 = 0.0
		balpo4 = 0.0
		balorn = 0.0
		balorp = 0.0
		balorc = 0.0
		baco2 = 0.0
		flxbal[1:4,1:5] = 0.0

	# compute derived quantities
	phycla = phyto * cvbcl
	balcla = zeros(numbal)
	for i in range(numbal):
		balcla[i] = benal[i] * cvbcl

	if vol > 0.0:   # compute initial summary concentrations
		for i in range(1, 4):
			lsnh4[i] = rsnh4[i] / vol
			lspo4[i] = rspo4[i] / vol
		
		torn,torp,torc,potbod,tn,tp = \
			pksums (phyfg,zoofg,tamfg,no2fg,po4fg,adnhfg,adpofg, 
					cvbn,cvbp,cvbc,cvbo,cvnrbo,phyto,zoo,orn,orp,orc,no3,tam,no2,lsnh4[1],lsnh4[2],
					lsnh4[3],po4,lspo4[1],lspo4[2],lspo4[3],bod,torn,torp,torc,potbod,tn,tp)
	else:    # undefined summary concentrations
		torn   = -1.0e30
		torp   = -1.0e30
		torc   = -1.0e30
		potbod = -1.0e30
		tn     = -1.0e30
		tp     = -1.0e30

	return errors, ERRMSGS

	@jit(nopython = True)
	def plank(dox, bod, iphyto, izoo, iorn, iorp, iorc, tw, wash, solrad, prec, sarea, advData):
		'''Simulate behavior of plankton populations and associated reactions'''
		if PHYFG == 1:   # phytoplankton simulated
			# advecvt phytoplankton
			phyto, rophyt,ophyt = advplk(iphyto,vols,srovol,vol,erovol,sovol, eovol,nexits,oref,mxstay,seed,delts,phyto,rophyt,ophyt)
			phyto, snkphy = sink(vol,avdepe,physet, phyto,snkphy)
			snkphy = -snkphy

			if ZOOFG == 1:    # zooplankton on; advect zooplankton
				zoo, rozoo,ozoo = advplk (izoo,vols,srovol,vol,erovol,sovol,eovol,nexits,oref,mxstay,seed,delts,zoo,rozoo,ozoo)
		
		# advect organic nitrogen
		inorn = iorn + pladep[1]
		dor,roorn,oorn = advect (inorn, dor, roorn, oorn)
		dor, snkorn = sink(vol, avdepe, refset, dor, snkorn)
		snkorn = -snkorn

		# advect organic phosphorus
		inorp = iorp + pladep[2]
		orp, roorp, oorp = advect(inorp, orp, roorp, oorp)
		orp, snkorp = sink(vol, avdepe, refset, orp, snkorp)
		snkorp = -snkorp

		# advect total organic carbon
		inorc = iorc + pladep[3]
		orc, roorc, oorc = advect (inorc, orc, roorc, oorc)
		orc, snkorc = sink (vol, avdepe, refset, orc, snkorc)
		snkorc = -snkorc

		if avdepe > 0.17:   # enough water to warrant computation of water quality reactions
			if frrif < 1.0:                                          
				# make adjustments to average water velocity and depth for the  
				# portion of the reach that consists of riffle areas.
				if ro < rifcq1:    # below first cutoff flow
					i= 1
				elif ro < rifcq2:   # below second cutoff flow
					i= 2
				elif ro < rifcq3:   # below third cutoff flow
					i= 3
				else:                        # above third cutoff flow
					i= 4

				# calculate the adjusted velocity and depth for riffle sections
				balvel = rifvel[i] * avvele
				baldep = rifdep[i] * avdepe
			else:                      # use full depth and velocity
				balvel = avvele
				baldep = avdepe
														
			# calculate solar radiation absorbed; solrad is the solar radiation at gage,
			# corrected for location of reach; 0.97 accounts for surface reflection
			# (assumed 3 per cent); cfsaex is the ratio of radiation incident to water
			# surface to gage radiation values (accounts for regional differences, shading
			# of water surface, etc); inlit is a measure of light intensity immediately below
			# surface of reach/res and is expressed as ly/min, adjusted for fraction that is
			# photosynthetically active.

			inlit = 0.97 * cfsaex * solrad / delt * paradf
			if SDLTFG == 1:		 # estimate contribution of sediment to light extinction
				extsed = litsed * ssed[4]
			elif SDLTFG == 2:   # equations from dssamt for estimating the extinction coefficient based on discharge and turbidity
				# estimate turbidity based on linear regression on flow
				turb = ctrbq1 * ro**ctrbq2
		
				# estimate the portion of the extinction coefficient due to
				# sediment based upon a system-wide regression of total
				# extinction to turbidity, and then subtracting the
				# base extinction coefficient
				extsed = (cktrb1 * turb**cktrb2) - extb                         
				if extsed < 0.0:         # no effective sediment shading
					extsed = 0.0
			else:                        # sediment light extinction not considered
				extsed = 0.0

			# calculate contribution of phytoplankton to light extinction (self-shading)
			extcla = 0.00452 * phyto * cvbcl

			# calculate light available for algal growth,  litrch only called here
			phylit,ballit,cflit = litrch (inlit,extb,extcla,extsed,avdepe,baldep,phyfg,balfg,phylit,ballit,cflit)

			if PHYFG == 1:   # simulate phytoplankton, phyrx only called here
				(po4,no3,tam,dox,orn,orp,orc,bod,phyto,limphy,pyco2,phycla,dophy,bodphy,tamphy,no3phy,po4phy,phdth,phgro,phyorn,phyorp,phyorc) \
					= phyrx(phylit,tw,talgrl,talgrh,talgrm,malgr,cmmp, \
		                cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60, \
		                cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh, \
		 				naldh,claldh,aldl,aldh,anaer,oxald,alnpr, \
		 				cvbo,refr,cvnrbo,cvpb,cvbcl,limit,co2, \
						nmingr,pmingr,cmingr,lmingr,nminc, \
						po4,no3,tam,dox,orn,orp,orc,bod,phyto, \
						limphy,pyco2,phycla,dophy,bodphy,tamphy, \
						no3phy,po4phy,phdth,phgro,ornphy,orpphy,orcphy)

				# compute associated fluxes
				phydox = dophy *  vol
				phybod = bodphy * vol
				phytam = tamphy * vol
				phyno3 = no3phy * vol
				phypo4 = po4phy * vol
				dthphy = -phdth * vol
				grophy = phgro  * vol
				phyorn = ornphy * vol
				phyorp = orpphy * vol
				phyorc = orcphy * vol

				if ZOOFG == 1:    # simulate zooplankton, zorx only called here
					(dox,bod,zoo,orn,orp,orc,tam,no3,po4,zeat,zco2,dozoo,zbod,znit,zpo4,zogr,zdth,zorn,zorp,zorc) \
						= zorx(zfil20,tczfil,tw,phyto,mzoeat,zexdel,cvpb, \
							zres20,tczres,anaer,zomass,tamfg,refr, \
							zfood,zd,oxzd,cvbn,cvbp,cvbc,cvnrbo,cvbo, \
							dox,bod,zoo,orn,orp,orc,tam,no3,po4, \
							zeat,zoco2,dozoo,bodzoo,nitzoo,po4zoo, \
							zgro,zdth,zorn,zorp,zorc)

					# compute associated fluxes
					zoodox = -dozoo * vol
					zoobod = bodzoo * vol
					if TAMFG != 0:   # ammonia on, so nitrogen excretion goes to ammonia
						zootam = nitzoo * vol
					else:             # ammonia off, so nitrogen excretion goes to nitrate
						zoono3 = nitzoo * vol
					zoopo4 = po4zoo * vol
					zoophy = -zeat  * vol
					zooorn = zorn   * vol
					zooorp = zorp   * vol
					zooorc = zorc   * vol
					grozoo = zgro   * vol
					dthzoo = -zdth  * vol
					totzoo = grozoo + dthzoo

					# update phytoplankton state variable to account for zooplankton predation
					phyto = phyto - zeat
					# convert phytoplankton expressed as mg biomass/l to chlorophyll a expressed as ug/l
					phycla = phyto * cvbcl
				totphy = snkphy + zoophy + dthphy + grophy

			if BALFG == 1:     # simulate benthic algae
				(po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,ornbal,orpbla,orcbal) \
				 = balrx(ballit,tw,talgrl,talgrh,talgrm,malgr,cmmp, \
		                cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60, \
		                cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh, \
						naldh,aldl,aldh,anaer,oxald,cfbalg,cfbalr, \
						alnpr,cvbo,refr,cvnrbo,cvpb,mbal,depcor, \
						limit,cvbcl,co2,nmingr,pmingr,cmingr,lmingr,nminc, \
						po4,no3,tam,dox,orn,orp,orc,bod,benal[1], \
						limbal[1],baco2,balcla[1],dobalg,bodbal, \
						tambal,no3bal,po4bal,bgro[1],bdth[1],ornbal, \
						orpbal,orcbal)

				#compute associated fluxes
				baldox = dobalg * vol
				balbod = bodbal * vol
				baltam = tambal * vol
				balno3 = no3bal * vol
				balpo4 = po4bal * vol
				balorn = ornbal * vol
				balorp = orpbal * vol
				balorc = orcbal * vol
				grobal[1] = bgro[1]
				dthbal[1] = -bdth[1]
			elif BALFG == 2:   # simulate enhanced benthic algae equations from dssamt
				# then perform reactions, balrx2 only called here
				(po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,ornbal,orpbla,orcbal) \
				 = balrx2 (ballit,tw,tamfg,nsfg,delt60,cvbpn,phfg,decfg, \
							cvbpc,alnpr,cvbo,refr,cvnrbo,cvpb,depcor, \
							limit,cvbcl,co2,numbal,mbalgr,cmmpb,cmmnb, \
							balr20,tcbalg,balvel,cmmv,bfixfg,cslit,cmmd1, \
							cmmd2,tcbalr,frrif,cremvl,cmmbi,binv,tcgraz, \
							cslof1,cslof2,minbal,fravl,bnpfg,campr,nmingr, \
							pmingr,cmingr,lmingr,nminc,nmaxfx,grores, \
							po4,no3,tam,dox,orn,orp,orc,bod,benal[1], \
							limbal[1],baco2,balcla[1],dobalg,bodbal, \
							tambal,no3bal,po4bal,bgro[1],bdth[1],ornbal, \
							orpbal,orcbal)

				# compute associated fluxes
				baldox = dobalg * vol
				balbod = bodbal * vol
				baltam = tambal * vol
				balno3 = no3bal * vol
				balpo4 = po4bal * vol
				balorn = ornbal * vol
				balorp = orpbal * vol
				balorc = orcbal * vol
				for i in range(numbal):
					grobal[i] = bgro[i]
					dthbal[i] = -bdth[i]
		else:     # not enough water in reach/res to warrant simulation of quality processes
			phyorn = 0.0
			balorn = 0.0
			zooorn = 0.0
			phyorp = 0.0
			balorp = 0.0
			zooorp = 0.0
			phyorc = 0.0
			balorc = 0.0
			zooorc = 0.0
			pyco2  = 0.0
			baco2  = 0.0
			zoco2  = 0.0
			phydox = 0.0
			zoodox = 0.0
			baldox = 0.0
			phybod = 0.0
			zoobod = 0.0
			balbod = 0.0
			phytam = 0.0
			zootam = 0.0
			baltam = 0.0
			phyno3 = 0.0
			zoono3 = 0.0
			balno3 = 0.0
			phypo4 = 0.0
			zoopo4 = 0.0
			balpo4 = 0.0

			if PHYFG == 1:  # water scarcity limits phytoplankton growth
				limc  = 'WAT'
				#read (limc,1000) limphy
				phycla = phyto * cvbcl
				grophy = 0.0
				dthphy = 0.0
				zoophy = 0.0
				totphy= snkphy
			if BALFG == 1:   # water scarcity limits benthic algae growth
				limc = 'WAT'
				#read (limc,1000) limbal[1]
				balcla[1] = benal[1] * cvbcl
				grobal[1] = 0.0
				dthbal[1] = 0.0
			elif BALFG == 2: # water scarcity limits benthic algae growth
				limc = 'WAT'
				for i in range(numbal):
					#read (limc,1000) limbal[i]
					balcla[i] = benal[i] * cvbcl
					grobal[i] = 0.0
					dthbal[i] = 0.0
			if ZOOFG == 1:    # water scarcity limits zooplankton growth
				grozoo= 0.0
				dthzoo= 0.0
				totzoo= 0.0

		if BALFG:   # store final benthic sums and fluxes in common block
			tbenal[1] = 0.0
			grotba = 0.0
			dthtba = 0.0
			for i in range(numbal):
				flxbal[1,i] = grobal[i]
				flxbal[2,i] = dthbal[i]
				flxbal[3,i] = grobal[i] + dthbal[i]
				tbenal[1]   = tbenal[1] + benal[i]
				grotba      = grotba + grobal[i]
			tbenal[2] = tbenal[1] * cvbcl
			tottba    = grotba + dthtba

		# compute final process fluxes for oxygen, nutrients and organics
		totdox = readox + boddox + bendox + nitdox + phydox + zoodox + baldox
		totbod = decbod + bnrbod + snkbod + denbod + phybod + zoobod + balbod
		totno3 = nitno3 + denno3 + bodno3 + phyno3 + zoono3 + balno3
		tottam = nittam + volnh3 + bnrtam + bodtam + phytam + zootam + baltam
		totpo4 = bnrpo4 + bodpo4 + phypo4 + zoopo4 + balpo4

		totorn = snkorn + phyorn + zooorn + balorn
		totorp = snkorp + phyorp + zooorp + balorp
		totorc = snkorc + phyorc + zooorc + balorc

		# compute summaries of total organics, total n and p, and potbod concentrations
		if vol > 0.0:     # compute summary concentrations
			for i in range(1, 4):
				lsnh4[i] = rsnh4[i] / vol
				lspo4[i] = rspo4[i] / vol

			itorn,itorp,itorc,dumval,itotn,itotp = 	pksums (phyfg,zoofg,tamfg,no2fg,
			po4fg,adnhfg,adpofg,cvbn,cvbp,cvbc,cvbo,cvnrbo,phyto,zoo,orn,orp,orc,
			no3,tam,no2,lsnh4[1],lsnh4[2],lsnh4[3],po4, lspo4[1],lspo4[2],lspo4(3),
			bod,itorn,itorp,itorc,dumval,itotn,itotp)
		else:   #   undefined summary concentrations
			torn   = -1.0e30
			torp   = -1.0e30
			torc   = -1.0e30
			potbod = -1.0e30
			tn     = -1.0e30
			tp     = -1.0e30

		# total inflows
		inno3 = ino3 + nuadep[1]
		intam = itam + nuadep[2]
		inpo4 = ipo4 + nuadep[3]
		
		(itorn,itorp,itorc,dumval,itotn,itotp) = pksums (phyfg,zoofg,tamfg,no2fg,
		po4fg,adnhfg,adpofg,cvbn,cvbp,cvbc,cvbo,cvnrbo,iphyto,izoo,inorn,inorp,
		inorc,inno3,intam,ino2,isnh4[1],isnh4[2],isnh4[3], inpo4,ispo4[1],
		ispo4[2],ispo4[3],ibod, itorn,itorp,itorc,dumval,itotn,itotp)

		# total outflows
		(rotorn,rotorp,rotorc,dumval,rototn,rototp) = pksums (phyfg,zoofg,tamfg,no2fg,po4fg,
		adnhfg,adpofg,cvbn,cvbp,cvbc,cvbo,cvnrbo,rophyt,rozoo,roorn,roorp, roorc,rono3,
		rotam,rono2,rosnh4[1],rosnh4[2],rosnh4[3],ropo4,rospo4[1],rospo4[2],rospo4[3],robod,
		rotorn,rotorp,rotorc,dumval,rototn,rototp)

		if nexits > 1:   # outflows by exit
			for i in range(nexits):
				otorn[i],otorp[i],otorc[i],dumval,ototn[i], ototp[i] \
				= pksums(phyfg,zoofg,
				tamfg,no2fg,po4fg,adnhfg,adpofg,cvbn,cvbp,cvbc,cvbo,cvnrbo,ophyt[i],ozoo[i],
				oorn[i],oorp[i],oorc[i],ono3[i],otam[i],ono2[i], osnh4[i,1],osnh4[i,2],
				osnh4[i,3],opo4[i], ospo4[i,1],ospo4[i,2],ospo4[i,3],obod[i],otorn[i],
				otorp[i],otorc[i],dumval,ototn[i],ototp[i])

		# cbrb  added call to ammion to redistribute tam after algal influence
		nh3,nh4 = ammion(tw, phval, tam, nh3, nh4)
		
		return (dox, bod, orn, orp, orc, torn, torp, torc, potbod, phyto, zoo, benal, 
		 phycla, balcla, rophyto, rozoo, robenal, rophycla, robalcla, ophyto, ozoo, 
		 obenal, ophycla, obalcla, binv, pladfx, pladcn)

	return plank


def advplk(iplank,vols,srovol,vol,erovol,sovol,eovol,nexits,oref,mxstay,seed,delts,plank,roplk,oplk):
	''' advect plankton'''
	
	# calculate concentration of plankton not subject to advection during interval
	oflo = (srovol + erovol) / delts
	if oref > 0.0 and oflo / oref <= 100.0:
		stay = (mxstay - seed) * (2.0**(-oflo / oref)) + seed
	else:
		stay = seed

	if plank > stay:
		# convert stay to units of mass; this mass will be converted
		# back to units of concentration based on the volume of the
		# reach/res at the end of the interval
		mstay = stay * vols

		# determine concentration of plankton subject to advection;
		# this value is passed into subroutine advect
		plnkad = plank - stay

		# advect plankton
		plnkad, roplk, oplk = advect(iplank,plnkad,roplk,oplk)

		# determine final concentration of plankton in reach/res after advection
		plank = plnkad + mstay / vol  if vol > 0.0 else plnkad
	else:       # no plankton leaves the reach/res
		roplk   = 0.0
		oplk[:] = 0.0
		mstay = plank * vols
		plank = (mstay + iplank) / vol if vol > 0.0 else -1.0e30
	return plank, roplk, oplk


def algro(light,po4,no3,tw,talgrl,talgrh,talgrm,malgr, cmmp,cmmnp,TAMFG,AMRFG,tam,NSFG,cmmn,cmmlt,
	alr20,cflit,delt60,limit,nmingr,pmingr,lmingr,limr,gro,res):

	''' calculate unit growth and respiration rates for algae
	population; both are expressed in units of per interval'''

	if light > lmingr:    # sufficient light to support growth
		if po4 > pmingr and no3 > nmingr:  # sufficient nutrients to support growth
			if  talgrh > tw > talgrl:      # water temperature allows growth
				if tw < talgrm:  		  # calculate temperature correction fraction
					tcmalg = (tw - talgrl) / (talgrm - talgrl)
				else:
					# no temperature correction to maximum unit growth rate
					# is necessary; water temperature is in the optimum
					# range for phytoplankton growth
					tcmalg = 1.0

				# perform temperature correction to maximum unit growth
				# rate; units of malgrt are per interval
				malgrt = malgr * tcmalg

				# calculate maximum phosphorus limited unit growth rate
				grop = malgrt * po4 * no3 / ((po4 + cmmp) * (no3 + cmmnp))

				# calculate the maximum nitrogen limited unit growth rate
				if TAMFG:       # consider influence of tam on growth rate
					if AMRFG:   # calculate tam retardation to nitrogen limited growth rate
						malgn = malgrt - 0.757 * tam + 0.051 * no3

						# check that calculated unit growth rate does not
						# exceed maximum allowable growth rate
						if malgn > malgrt:
							malgn = malgrt
						else:
							# check that calculated unit growth rate is not
							# less than .001 of the maximum unit growth rate;
							# if it is, set the unit growth rate equal to .001
							# of the maximum unit growth rate
							lolim = 0.001 * malgrt
							if malgn < lolim:
								malgn = lolim
					else:    # ammonia retardation is not considered
						malgn = malgrt

					if NSFG:   # include tam in nitrogen pool for calculation of nitrogen limited growth rate
						mmn = no3 + tam
					else:     # tam is not included in nitrogen pool for calculation of nitrogen limited growth
						mmn = no3
				else:            # tam is not simulated
					malgn = malgrt
					mmn  = no3

				# calculate the maximum nitrogen limited unit growth rate
				gron = (malgn * mmn) / (cmmn + mmn)

				# calculate the maximum light limited unit growth rate
				grol = (malgrt * light) / (cmmlt + light)
				if grop < gron and grop < grol:
					gro = grop
					lim = 'po4'
				elif gron < grol:
					gro = gron
					lim = 'nit'
				else:
					gro = grol
					lim = 'lit'
				if gro < 0.000001 * delt60:
					gro = 0.0

				if gro > 0.95 * malgrt:
					# there is no limiting factor to cause less than maximum growth rate
					lim = 'none'

				# adjust growth rate if control volume is not entirely	
				# contained within the euphotic zone; e.g. if only one
				# half of the control volume is in the euphotic zone, gro
				# would be reduced to one half of its specified value
				gro = gro * cflit
			else:            # water temperature does not allow algal growth
				gro = 0.0
				lim = 'tem'
		else:               # no algal growth occurs; necessary nutrients are not available
			gro = 0.0
			lim = 'non'
	else:                    # no algal growth occurs; necessary light is not available
		gro = 0.0
		lim = 'lit'

	# calculate unit algal respiration rate; res is expressed in
	# units of per interval; alr20 is the respiration rate at 20 degrees c
	res = alr20 * tw / 20.0
	return lim, gro, res



def baldth(nsfg,no3,tam,po4,paldh,naldh,aldl,aldh,mbal,dox,anaer,oxald,bal,depcor,dthbal):
	''' calculate benthic algae death'''

	# determine whether to use high or low unit death rate; all
	# unit death rates are expressed in units of per interval

	# determine available inorganic nitrogen pool for test of nutrient scarcity
	nit = no3 + tam  if NSFG else no3
	if po4 > paldh and nit > naldh:   # unit death rate is not incremented by nutrient scarcity check for benthic algae overcrowding
		balmax = mbal * depcor
		if bal < balmax:   # unit death rate is not incremented by benthic algae overcrowding
			ald  = aldl
			slof = 0.0
		else:
			# augment unit death rate to account for benthic algae
			# overcrowding; set bal value equal to maximum; calculate
			# amount of benthic algae in excess of mbal; these benthic
			# algae are added to aldth
			ald  = aldh
			slof = bal - balmax
			bal  = balmax
	else:   # augment unit death rate to account for nutrient scarcity
		ald = aldh

	if dox < anaer:    # conditions are anaerobic, augment unit death rate
		ald += oxald

	# use unit death rate to compute death rate; dthbal is expressed
	# as umoles of phosphorus per liter per interval
	return (ald * bal) + slof     # dthbal


def balrx(ballit,tw,talgrl,talgrh,talgrm,malgr,cmmp, cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60,
	cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh, naldh,aldl,aldh,anaer,oxald,cfbalg,cfbalr,
	alnpr,cvbo,refr,cvnrbo,cvpb,mbal,depcor,limit,cvbcl,co2,nmingr,pmingr,cmingr,lmingr,
	nminc, po4,no3,tam,dox,orn,orp,orc,bod,benal, limbal,baco2,balcla,dobalg,bodbal,tambal,
	no3bal,po4bal,balgro,bdth,balorn,balorp,balorc):
	''' simulate behavior of benthic algae in units of umoles p per
	liter; these units are used internally within balrx so that
	algal subroutines may be shared by phyto and balrx; externally,
	the benthic algae population is expressed in terms of areal
	mass, since the population is resident entirely on the
	bottom surface'''

	# convert benal to units of umoles phosphorus/l (bal) for internal calculations
	bal = (benal / cvpb) * depcor

	# compute unit growth and respiration rates for benthic algae; determine growth limiting factor
	limbal,gro,res = algro(ballit,po4,no3,tw,talgrl,talgrh,talgrm,malgr,cmmp, cmmnp,tamfg,amrfg,
	tam,nsfg,cmmn,cmmlt,alr20, cflit,delt60,limit,nmingr,pmingr,lmingr,limbal,gro,res)

	# calculate net growth rate of algae; grobal is expressed as
	# umoles phosphorus per liter per interval; benthic algae growth
	# will be expressed in terms of volume rather than area for the
	# duration of the subroutines subordinate to balrx; the output
	# values for benthic algae are converted to either mg biomass per
	# sq meter or mg chla per sq meter, whichever the user
	# specifies; cfbalg and cfbalr are the specified ratio of benthic
	# algae growth rate to phytoplankton growth rate and ratio of
	# benthic algae respiration rate to phytoplankton respiration
	# rate, respectively

	grobal = (gro * cfbalg - res * cfbalr) * bal
	if grobal > 0.0:  # adjust growth rate to account for limitations imposed by availability of required nutrients
		grtotn = grobal
		grobal = grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,i0,grtotn,grobal)

	# calculate benthic algae death, baldth only called here
	dthbal = baldth(nsfg,no3,tam,po4,paldh,naldh,aldl,aldh,mbal,dox,anaer,oxald,bal,depcor,dthbal)

	bal += grobal  # determine the new benthic algae population

	# adjust net growth rate, if necessary, so that population does not fall below minimum level
	minbal = 0.0001 * depcor
	if bal < minbal:
		grobal += minbal - bal
		bal     = minbal
	bal -= dthbal

	# adjust death rate, if necessary, so that population does not drop below minimum level
	if bal < minbal:
		dthbal = dthbal - (minbal - bal)
		bal    = minbal

	# update do state variable to account for net effect of benthic algae photosynthesis and respiration
	dobalg = cvpb * cvbo * grobal
	# dox   = dox + (cvpb*cvbo*grobal)
	if dox > -dobalg:
		dox = dox + dobalg
	else:
		dobalg = -dox
		dox    = 0.0

	# calculate amount of refractory organic constituents which result from benthic algae death
	balorn = refr * dthbal * cvbpn * 0.014
	balorp = refr * dthbal * 0.031
	balorc = refr * dthbal * cvbpc * 0.012

	# calculate amount of nonrefractory organics (bod) which result from benthic algae death
	bodbal = cvnrbo * cvpb * dthbal

	# perform materials balance resulting from benthic algae death
	orn,orp,orc,bod = orgbal(balorn,balorp,balorc,bodbal,orn,orp,orc,bod)

	# perform materials balance resulting from uptake of nutrients by benthic algae
	po4,tam,no3,baco2,tambal,no3bal,po4bal = nutrup(grobal,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg,nminc,baco2,tambal,no3bal,po4bal)
	baco2 = -baco2

	# convert bal back to external units; benal is expressed as
	# mg biomass/m2 and balcla is expressed as ug chlorophyll a/m2
	benal  = (bal * cvpb) / depcor
	balgro = (grobal * cvpb) / depcor
	bdth   = (dthbal * cvpb) / depcor
	balcla = benal * cvbcl

	return po4,no3,tam,dox,orn,orp,orc,bod,benal, limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,balorn,balorp,balorc


def grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,nfixfg,grtotn, grow):
	''' check whether computed growth rate demands too much of any
	nutrient; adjust growth rate, if necessary, so that at least
	minimum allowed level of each nutrient remains after growth'''

	# calculate growth rate which results in minimum free po4
	# remaining after growth; uplimp is expressed as umoles
	# phosphorus per liter per interval
	uplimp = (po4 - pmingr) * 32.29

	# calculate growth rate which results in minimum free
	# inorganic nitrogen remaining after growth; uplimn is expressed
	# as umoles phosphorus per interval
	if NSFG == 0:    # tam is not considered as a possible nutrient
		uplimn = (no3 - nmingr) * 71.43 / cvbpn
	else:
		uplimn = (no3 + tam - nmingr) * 71.43 / cvbpn

	uplimc = 1.0e30
	if PHFG and PHFG != 2 and DECFG == 0:
		# phcarb is on, and co2 is being considered as a possible
		# limiting nutrient to algal growth
		if co2 >= 0.0:
			# calculate growth rate which results in minimum free
			# carbon dioxide remaining after growth; uplimc is expressed
			# as umoles phosphorus per liter per interval
			uplimc = (co2 - cmingr) * 83.33 / cvbpc

	# calculate difference between available nutrient concentrations and
	# nutrients needed for modeled growth; amount needed for growth may differ 
	# for n if nitrogen-fixation occurs in any of the algal types
	uplimp -= grow
	uplimn -= grtotn
	uplimc -= grow

	# check that calculated growth does not result in less than
	# minimum allowed concentrations of orthophosphate, inorganic
	# nitrogen, or carbon dioxide; if it does, adjust growth
	if NFIXFG:                       # n-fixation is not occurring for this algal type
		uplim = min(uplimp, uplimn, uplimc)
	else:                           # n-fixation is occurring, nitrogen does not limit growth
		uplim = min(uplimp, uplimc)
	if uplim < 0.0:   # reduce growth rate to limit
		grow += uplim
	return grow


def litrch(inlit, extb, extcla, extsed, avdepe, baldep, PHYFG, BALFG, phylit, ballit, cflit):
	''' calculate light correction factor to algal growth (cflit); 
	determine amount of light available to phytoplankton and benthic algae'''
	ln01 = 4.60517   # minus natural log 0.01
	if inlit > 0.0:
		# calculate extinction of light based on the base extinction
		# coefficient of the water incremented by self-shading effects
		# of phytoplankton and light extinction due to total sediment
		# suspension
		extco = extb + extcla + extsed

		# calculate euphotic depth; euphotic depth is the distance,
		# in feet, below the surface of the water body at which one
		# percent of incident light is present
		eudep = ln01 / extco
		if eudep < avdepe:
			# calculate fraction of layer which is contained in the
			# euphotic zone; this fraction, cflit, will be multiplied
			# by calculated growth in algro to assure that growth only
			# occurs in the euphotic zone
			cflit = eudep / avdepe
			if cflit < 0.0001:
				cflit = 0.0
		else:
			cflit = 1.0

		if PHYFG:
			# calculate amount of light available to phytoplankton; all
			# phytoplankton are assumed to be at mid-depth of the reach;
			# light is expressed as langleys per minute
			phylit = inlit * exp(-extco * (0.5 * min(eudep, avdepe)))
			if phylit < 0.0001:
				phylit = 0.0

		if BALFG:
			# calculate amount of light available to benthic algae; all
			# benthic algae are assumed to be at the bottom depth of the
			# reach;light is expressed as langleys per minute
			ballit = inlit * exp(-extco * baldep)
			if ballit < 0.0001:
				ballit = 0.0
	else:   # there is no incident solar radiation; algal growth cannot occur
		cflit  = 0.0
		phylit = 0.0
		ballit = 0.0
	return phylit, ballit, cflit


def nutrup(grow, NSFG, cvbpn, alnpr, cvbpc, PHFG, DECFG, nminc, po4, tam, no3, alco2, tamalg, no3alg, po4alg):
	''' perform materials balance for transformation from inorganic to
	organic material; uptake of po4, no3, tam, and co2 are considered'''

	# calculate po4 balance subsequent to algal uptake or release;
	# 0.031 is the conversion from umoles p per liter to mg of p per liter
	po4    = po4 - 0.031 * grow
	po4alg = -0.031 * grow
	tamalg = 0.0
	if NSFG:
		# calculate tam balance subsequent to algal uptake or release
		# express calculated growth rate in terms of equivalent
		# nitrogen; grown is expressed as umoles nitrogen per interval
		grown = grow * cvbpn
		if grow < 0.0:  # algal respiration exceeds growth
			# nitrogen released by respiration is released in the form of tam
			# no uptake or release of no3 occurs
			altam = grown
			alno3 = 0.0
		else:
			# calculate amount of n uptake which is no3 and amount which is tam
			alno3 = alnpr * grown
			altam = grown - alno3
			# check that computed uptake of no3 does not consume more
			# than 99 percent of available free no3; if it does, satisfy
			# excess demand with free tam; no3lim is expressed as umoles
			# n per liter per interval
			no3lim = 70.72 * no3
			if alno3 > no3lim:
				altam += alno3 - no3lim
				alno3  = no3lim
			else:
				# check that calculated uptake of tam does not consume
				# more than 99 percent of available free tam; if it does,
				# satisfy excess demand with free no3; tamlim is expressed
				# as umoles n per liter per interval
				tamlim = 70.72 * tam
				if altam > tamlim:
					alno3 += altam - tamlim
					altam = tamlim

		# calculate net uptake or release of tam by algae; .014 is
		# the conversion from umoles of n per liter per interval to
		# mg n per liter per interval
		tams   = tam
		tamalg = -0.014 * altam
		tam    = tam - 0.014 * altam
		if tam < nminc:
			tamalg = -tams
			tam    = 0.0

	else:   # all inorganic n is in the form of no3
		alno3 = grow * cvbpn

	# calculate no3 balance subsequent to algal uptake or release;
	# eliminate insignificant values of no3
	no3s   = no3
	no3alg = -0.014 * alno3
	no3    = no3 - 0.014 * alno3
	if no3 < nminc:
		no3alg = -no3s
		no3    = 0.0
		
	if PHFG and DECFG == 0:   # calculate amount of algal uptake of co2 in mg co2-c/liter
		alco2 = grow * cvbpc * 0.012
	else:
		alco2 = 0.0
	return po4, tam, no3, alco2, tamalg, no3alg, po4alg


def orgbal(dthorn, dthorp, dthorc, dthbod, orn, orp, orc, bod):
	''' perform materials balance for transformation from living to dead organic material'''

	# calculate dead refractory organic nitrogen balance subsequent to plankton death;
	# plankton death may be either algal death, zooplankton death, or phytoplankton
	# filtered by zooplankton but not assimilated
	orn += dthorn

	# calculate dead refractory organic phosphorus balance subsequent to plankton death
	orp += dthorp

	# calculate dead refractory organic carbon balance subsequent to plankton death
	orc += dthorc

	# calculate bod balance subsequent to plankton death
	bod += dthbod

	return orn, orp, orc, bod


def phydth(nsfg,no3,tam,po4,paldh,naldh,phycla,claldh,aldl,aldh,dox,anaer,oxald,stc,dthphy):
	''' calculate phytoplankton death'''

	# determine whether to use high or low unit death rate; all unit
	# death rates are expressed in units of per interval
	# determine available inorganic nitrogen pool for test of nutrient scarcity
	nit = no3 + tam  if nsfg else  no3
	if po4 > paldh and nit > naldh:
		# unit death rate is not incremented by nutrient scarcity
		# check for phytoplankton overcrowding
		if phycla < claldh:    # unit death rate is not incremented by phytoplankton overcrowding
			ald = aldl
		else:
			ald = aldh   # augment unit death rate to account for overcrowding
	else: 		# augment unit death rate to account for nutrient scarcity
		ald = aldh

	# augment unit death rate if conditions are anaerobic
	if dox < anaer:
		ald += oxald

	# use unit death rate to compute death rate; aldth is expressed
	# as umoles of phosphorus per liter per interval
	return ald * stc                   # dthphy


def phyrx(phylit,tw,talgrl,talgrh,talgrm,malgr,cmmp,cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60,cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh,
	naldh,claldh,aldl,aldh,anaer,oxald,alnpr,cvbo,refr,cvnrbo,cvpb,cvbcl,limit,co2,nmingr,pmingr,cmingr,lmingr,nminc,
	po4,no3,tam,dox,orn,orp,orc,bod,phyto,limphy,pyco2,phycla,dophy,bodphy,tamphy,no3phy,po4phy,phdth,phgro,phyorn,phyorp,phyorc):
	''' simulate behavior of phytoplankton, as standing crop, in units of umoles p per liter'''

	# convert phyto to units of umoles phosphorus (stc) and ug chlorophyll a/l (phycla) for internal calculations
	stc    = phyto / vpb
	phycla = phyto * cvbcl

	# compute unit growth and respiration rates for phytoplankton;
	# determine growth limiting factor
	limphy,gro,res = algro(phylit,po4,no3,tw,talgrl,talgrh,talgrm,malgr,cmmp,cmmnp,tamfg,amrfg,
	tam,nsfg,cmmn,cmmlt,alr20,cflit,delt60,limit,nmingr,pmingr,lmingr,limphy,gro,res)

	# calculate net growth rate of phytoplankton; grophy is
	# expressed as umol phosphorus per liter per interval
	grophy = (gro - res) * stc

	if grophy > 0.0:
		# adjust growth rate to account for limitations imposed by
		# availability of required nutrients
		grtotn = grophy
		nmingr,pmingr,cmingr,i0,grtotn,grophy = grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,i0,grtotn,grophy)

	# calculate phytoplankton death
	dthphy = phydth(nsfg,no3,tam,po4,paldh,naldh,phycla,claldh,aldl,aldh,dox,anaer,oxald,stc,dthphy)

	# determine the new phytoplankton population
	stc += grophy

	# adjust net growth rate, if necessary, so population does not fall below minimum level
	if stc < .0025:
		grophy -= (0.0025 - stc)
		stc     = 0.0025
	stc -= dthphy

	# adjust death rate, if necessary, so that population does not drop below minimum level
	if stc < .0025:
		dthphy -= (0.0025 - stc)
		stc     = 0.0025

	# update do state variable to account for net effect of phytoplankton photosynthesis and respiration
	dophy = cvpb * cvbo * grophy
	if dox > -dophy:
		dox += dophy
	else:
		dophy = -dox
		dox   = 0.0

	# calculate amount of refractory organic constituents which result from phytoplankton death
	phyorn = refr * dthphy * cvbpn * 0.014
	phyorp = refr * dthphy * 0.031
	phyorc = refr * dthphy * cvbpc * 0.012

	# calculate amount of nonrefractory organics (bod) which result from phytoplankton death
	phybd  = cvnrbo * cvpb * dthphy
	bodphy = phybd

	# perform materials balance resulting from phytoplankton death
	orn,orp,orc,bod = orgbal(phyorn,phyorp,phyorc,phybd, orn,orp,orc,bod)

	# perform materials balance resulting from uptake of nutrients by phytoplankton
	po4,tam,no3,pyco2,tamphy,no3phy,po4phy = nutrup (grophy,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg,nminc,po4,tam,no3,pyco2,tamphy,no3phy,po4phy)
	pyco2 = -pyco2

	# convert stc to units of mg biomass/l (phyto) and ug chlorophyll a/l (phycla)
	phyto  = stc    * cvpb
	phgro  = grophy * cvpb
	phdth  = dthphy * cvpb
	phycla = phyto  * cvbcl

	return po4,no3,tam,dox,orn,orp,orc,bod,phyto,limphy,pyco2,phycla,dophy,bodphy,tamphy,no3phy,po4phy,phdth,phgro,phyorn,phyorp,phyorc


def zorx(zfil20,tczfil,tw,phyto,mzoeat,zexdel,cvpb,zres20,tczres,anaer,zomass,tamfg,refr,
	zfood,zd,oxzd,cvbn,cvbp,cvbc,cvnrbo,cvbo,dox,bod,zoo,orn,orp,orc,tam,no3,po4,zeat,zco2,dozoo,zbod,znit,zpo4,
	zogr,zdth,zorn,zorp,zorc):
	''' calculate zooplankton population balance'''

	# calculate zooplankton unit grazing rate expressed as liters
	# of water filtered per mg zooplankton per interval
	zfil = zfil20 * (tczfil**(tw - 20.0))

	# calculate mass of phytoplankton biomass ingested per mg
	# zooplankton per interval
	zoeat = zfil * phyto

	# check that calculated unit ingestion rate does not exceed
	# maximum allowable unit ingestion rate (mzoeat); if it does,
	# set unit ingestion rate equal to mzoeat
	if zoeat >= mzoeat:
		zoeat = mzoeat
		# nonrefractory portion of excretion is only partially
		# decomposed; zexdec is the fraction of nonrefractory
		# material which is decomposed
		zexdec = zexdel
	else:
		# calculated unit ingestion rate is acceptable
		# all nonrefractory excretion is decomposed
		zexdec = 1.0

	# calculate phytoplankton consumed by zooplankton; zeat is
	# expressed as mg biomass per interval
	zeat = zoeat * zoo

	# check that calculated ingestion does not reduce phytoplankton
	# concentration to less than .0025 umoles p; if it does, adjust
	# ingestion so that .0025 umoles of phytoplankton (expressed as p) remain
	if phyto - zeat < 0.0025 * cvpb:
		zeat = phyto - (0.0025 * cvpb)

	# calculate zooplankton assimilation efficiency
	if zfood == 1:
		# calculate assimilation efficiency resulting from ingestion
		# of high quality food; zeff is dimensionless
		zeff = -0.06 * phyto + 1.03
		if zeff > 0.99:    		# set upper limit on efficiency at 99 percent
			zeff = 0.99
	elif zfood == 2:
		# calculate assimilation efficiency resulting from ingestion of medium quality food
		zeff = -0.03 * phyto + 0.47
		if zeff < 0.20:  		# set lower limit on efficiency at 20 percent
			zeff = 0.20
	else:
		# calculate assimilation efficiency resulting from ingestion of low quality food
		zeff = -0.013 * phyto + 0.17
		if zeff < 0.03:
			zeff= 0.03   # # set lower limit on efficiency at 3 percent

	# calculate zooplankton growth; zogr is expressed as mg biomass per liter per interval
	zogr = zeff * zeat

	# calculate total zooplankton excretion (zexmas),excretion
	# decomposed to inorganic constituents (zingex), excretion
	# released as dead refractory constituents (zrefex), and
	# excretion released as dead nonrefractory material (znrfex)
	zexmas = zeat - zogr
	zrefex = refr * zexmas
	zingex = zexdec * (zexmas - zrefex)
	znrfex = zexmas - zrefex - zingex

	# calculate zooplankton respiration; zres is expressed as mg
	# biomass per liter per interval
	zres = zres20 * (tczres**(tw - 20.0)) * zoo

	# calculate zooplankton death; zdth is expressed as mg biomass per liter per interval
	if dox > anaer:
		# calculate death using aerobic death rate
		zdth = zd * zoo
	else:
		# calculate death using sum of aerobic death rate and anaerobic increment
		zdth = (zd + oxzd) * zoo

	# calculate zooplankton population after growth, respiration,
	# and death; adjust respiration and death, if necessary, to
	# assure minimum population of zooplankton
	# first, account for net growth (growth - respiration)
	zoo += zogr - zres

	# maintain minimum population of .03 organisms per liter; zomass
	# is a user specified conversion factor from organisms/l to mg biomass/l
	lolim = 0.03 * zomass
	if zoo < lolim:
		zres = zres + zoo - lolim
		zoo  = lolim

	# subtract oxygen required to satisfy zooplankton respiration from do state variable
	dozoo = 1.1 * zres
	dox  -= dozoo
	zbod = 0.0
	if dox < 0.0:            # include oxygen deficit in bod value
		zbod = -dox
		dox  = 0.0

	# subtract computed zooplankton death from zooplankton state variable
	zoo = zoo - zdth
	if zoo < lolim:
		zdth = zdth + zoo - lolim
		zoo  = lolim

	# calculate amount of inorganic constituents which are released
	# by zooplankton respiration and inorganic excretion
	znit = (zingex + zres) * cvbn
	zpo4 = (zingex + zres) * cvbp
	zco2 = (zingex + zres) * cvbc

	# update state variables for inorganic constituents to account
	# for additions from zooplankton respiration and inorganic excretion
	tam, no3, po4 = decbal(tamfg, i1, znit, zpo4, tam, no3, po4)

	# calculate amount of refractory organic constituents which result from zooplankton death and excretion
	zorn = ((refr * zdth) + zrefex) * cvbn
	zorp = ((refr * zdth) + zrefex) * cvbp
	zorc = ((refr * zdth) + zrefex) * cvbc

	# calculate amount of nonrefractory organics (bod) which result from zooplankton death and excretion
	zbod = zbod + (zdth * cvnrbo) + (znrfex * cvbo)
	orn, orp, orc, bod = orgbal(zorn, zorp, zorc, zbod, orn, orp, orc, bod)
	return  dox,bod,zoo,orn,orp,orc,tam,no3,po4,zeat,zco2,dozoo,zbod,znit,zpo4,zogr,zdth,zorn,zorp,zorc


def pksums(PHYFG,ZOOFG,TAMFG,NO2FG,PO4FG,ADNHFG,ADPOFG,cvbn,cvbp,cvbc,cvbo,cvnrbo,phyto,zoo,orn,
	orp,orc,no3,tam,no2,snh41,snh42,snh43,po4,spo41,spo42,spo43,bod, torn,torp,torc,potbod,tn,tp):
	''' computes summaries of: total organic n, p, c; total n, p; potbod'''

	tval = bod / cvbo
	if PHYFG == 1:
		tval += phyto
		if zoofg == 1:
			tval += zoo

	torn   = orn + cvbn * tval
	torp   = orp + cvbp * tval
	torc   = orc + cvbc * tval
	potbod = bod
	tn     = torn + no3
	if TAMFG == 1:
		tn = tn + tam
	if NO2FG == 1:
		tn = tn + no2
	if ADNHFG == 1:
		tn = tn + snh41 + snh42 + snh43
	tp = torp
	if PO4FG == 1:
		tp = tp + po4
	if ADPOFG == 1:
		tp = tp + spo41 + spo42 + spo43
	if PHYFG == 1:
		potbod = potbod + (cvnrbo * phyto)
		if zoofg == 1:
			potbod = potbod + (cvnrbo * zoo)
	return torn, torp, torc, potbod, tn, tp


def algro2 (ballit,po4,no3,tw,mbalgr,cmmp,tamfg,tam, nsfg,cmmn,balr20,delt60,limit,tcbalg,balvel,
	cmmv,bfixfg,cslit,cmmd1,cmmd2,sumba,tcbalr, nmingr,pmingr,lmingr,nmaxfx,grores,
	nfixfg,limr,groba,resba):

	''' calculate unit growth and respiration rates for benthic algae using more
	complex kinetics; both are expressed in units of per interval'''

	if ballit > lmingr:      # sufficient light to support growth
		if po4 > pmingr and no3 > nmingr:   # sufficient nutrients to support growth

			# calculate temperature correction fraction
			tcmbag = tcbalg**(tw - 20.0)

			# calculate velocity limitation on nutrient availability
			grofv = balvel / (cmmv + balvel)

			# calculate phosphorus limited unit growth factor
			grofp = po4 * grofv / (cmmp + po4 * grofv)

			# calculate the nitrogen limited unit growth factor
			if TAMFG:   # consider influence of tam on growth rate
				if NSFG:
					# include tam in nitrogen pool for calculation of
					# nitrogen limited growth rate
					mmn = no3 + tam
				else:
					# tam is not included in nitrogen pool for calculation
					# of nitrogen limited growth
					mmn = no3
			else:
				mmn = no3  # tam is not simulated

			if BFIXFG != 1:   # calculate the maximum nitrogen limited unit growth rate
				nfixfg = 0
				grofn = (mmn * grofv) / (cmmn + mmn * grofv)
			else:
				# n-fixing blue-green algae; determine if available nitrogen 
				# concentrations are high enough to suppress nitrogen fixation
				if mmn >= nmaxfx:
					nfixfg = 0
					grofn = (mmn * grofv) / (cmmn + mmn * grofv)
				else:
					nfixfg = 1   # available nitrogen concentrations do not suppress n-fixation
					grofn = 1.0  # nitrogen does not limit growth rate

				# calculate the maximum light limited unit growth rate
				grofl = (ballit / cslit) * exp(1.0 - (ballit / cslit))

				# calculate density limitation on growth rate
				grofd = (cmmd1 * sumba + cmmd2) / (sumba + cmmd2)
				if grofp < grofn and grofp < grofl:  # phosphorus limited
					gromin = grofp
					lim = 'po4'
				elif grofn < grofl:                  # nitrogen limited
					gromin = grofn
					lim = 'nit'
				else:                                # light limited
					gromin = grofl
					lim = 'lit'
				if gromin > 0.95:  # there is no limiting factor to cause less than maximum growth rate
					lim = 'none'
				# calculate overall growth rate in units of per interval
				groba = mbalgr * tcmbag * gromin * grofd
				if groba < 1.0e-06 * delt60:
					groba = 0.0
		else:    # no algal growth occurs; necessary nutrients are not available
			groba = 0.0
			lim   ='non'
	else:        # no algal growth occurs; necessary light is not available
		groba = 0.0
		lim = 'lit'

	# calculate unit algal respiration rate in units of per interval
	# balr20 is the benthic algal respiration rate at 20 degrees c
	resba = balr20 * tcbalr**(tw - 20.0) + grores * groba
	return  nfixfg, groba, resba

def balrem(crem,sumba,sumbal,cmmbi,tcgraz,tw,binv,bal,cslof1,cslof2,balvel,remba):
	''' calculate benthic algae removal due to macroinvertebrates and scouring, based
	upon equations from dssamt. This subroutine provides similar functionality for the
	enhanced periphyton kinetics (balrx2) that baldth provides for the original hspf benthic algae (balrx).'''

	# calculate the total benthic algae removed per interval due to grazing and disturbance of macroinvertebrates
	reminv = crem * (sumba / (sumba + cmmbi)) * tcgraz**(tw - 20.0) * binv

	# calculate portion of total algae removed represented by the present benthic algal group
	remba = reminv * bal / sumbal

	# calculate scour loss rate of benthic algae in per interval
	maxvel = log(1.0 / cslof1) / cslof2
	slof   = cslof1 * exp(cslof2 * balvel) if balvel < maxvel else 1.0

	# calculate amount of present benthic algae group removed through grazing, disturbance, and scouring
	remba += slof * bal
	return remba


def nutup2 (grow,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg, bnpfg,campr,sumgrn,nminc,  po4,tam,no3, alco2,tamalg,no3alg,po4alg):
	''' perform materials balance for transformation from inorganic to
	organic material; uptake of po4, no3, tam, and co2 are considered.
	used instead of nutrup by the balrx2 subroutine; adds a
	calculated nitrogen preference function and adjustments for 
	n-fixation to the nutrup code.'''
	
	nmgpum = 0.0140067
	cmgpum = 0.012011
	pmgpum = 0.0309738
	prct99 = 0.99

	# calculate po4 balance subsequent to algal uptake or release;
	# pmgpum is the conversion from umoles p per liter to mg of p per liter
	po4    = po4 - pmgpum * grow
	po4alg = -pmgpum * grow
	tamalg = 0.0
	if NSFG:
		# calculate tam balance subsequent to algal uptake or release
		# express calculated growth rate in terms of equivalent
		# nitrogen; grown is expressed as umoles nitrogen per interval;
		# use accumulated growth that affects available n (sumgrn)
		grown = sumgrn * cvbpn
		if sumgrn < 0.0:      # algal respiration exceeds growth
			# nitrogen released by respiration is released in the form of tam; no uptake or release of no3 occurs
			altam = grown
			alno3 = 0.0
		else:   # calculate amount of n uptake which is no3 and amount which is tam
			# check nitrogen preference flag; if bnpfg.ne.1 use original
			# approach of %no3, if bnpfg.eq.1 use preference function
			# from dssamt/ssamiv
			if BNPFG == 1:
				# use dssamt ammonia preference function, equal to 1-alnpr
				alnpr2 = 1.0 - (campr * tam) / (campr * tam + no3)
			else:
				alnpr2 = alnpr
			alno3 = alnpr2 * grown
			altam = grown - alno3
			# check that computed uptake of no3 does not consume more
			# than 99 percent of available free no3; if it does, satisfy
			# excess demand with free tam; no3lim is expressed as umoles
			# n per liter per interval
			# no3lim= (prct99/nmgpum)*no3
			if alno3 > no3lim:
				altam = altam + alno3 - no3lim
				alno3 = no3lim
			else:
				# check that calculated uptake of tam does not consume
				# more than 99 percent of available free tam; if it does,
				# satisfy excess demand with free no3; tamlim is expressed
				# as umoles n per liter per interval
				tamlim = (prct99 / nmgpum) * tam
				if altam > tamlim:
					alno3 = alno3 + altam - tamlim
					altam = tamlim
		# calculate net uptake or release of tam by algae; .014 is
		# the conversion from umoles of n per liter per interval to
		# mg n per liter per interval
		tams   = tam
		tamalg = -nmgpum * altam
		tam    = tam - nmgpum * altam
		if tam < nminc:
			tamalg = -tams
			tam    = 0.0
	else:     # all inorganic n is in the form of no3
		alno3 = sumgrn * cvbpn

	# calculate no3 balance subsequent to algal uptake or release;
	# eliminate insignificant values of no3
	no3s   = no3
	no3alg = -nmgpum * alno3
	no3    = no3 - nmgpum * alno3
	if no3 < nminc:
		no3alg = -no3s
		no3    = 0.0

	# calculate amount of algal uptake of co2; alco2 is expressed as mg co2-c/liter
	alco2 = grow * cvbpc * cmgpum  if PHFG and DECFG == 0 else 0.0

	return po4, tam, no3, alco2, tamalg, no3alg, po4alg



def balrx2(ballit,tw,TAMFG,NSFG,delt60,cvbpn,PHFG,DECFG,cvbpc,alnpr,cvbo,refr,cvnrbo,cvpb,
	depcor,limit,cvbcl,co2,numbal,mbalgr,cmmpb,cmmnb,balr20,tcbalg,balvel,cmmv,BFIXFG,
	cslit,cmmd1,cmmd2,tcbalr,frrif,cremvl,cmmbi,binv,tcgraz,cslof1,cslof2,minbal,
	fravl,BNPFG,campr,nmingr,pmingr,cmingr,lmingr,nminc,nmaxfx,grores,
	po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,
	no3bal,po4bal,balgro,bdth,balorn,balorp,balorc):

	''' simulate behavior of up to four types of benthic algae using
	algorithms adapted from the dssamt model.  this subroutine was
	adapted from balrx'''
	nmgpum = 0.0140067
	cmgpum = 0.012011
	pmgpum = 0.0309738

	# compute total biomass of all benthic algal types
	sumba = 0.0
	for i in range(numbal):
		sumba = sumba + benal[i]
	
	# convert to umoles p/l
	sumbal = (sumba / cvpb) * depcor * frrif

	# initialize variables for cumulative net growth/removal
	# of benthic algae types, and for growth affecting available n conc.
	grotot = 0.0
	grtotn = 0.0
	sumgro = 0.0
	sumgrn = 0.0
	sumdth = 0.0

	for i in range(numbal):    # do 20 i = 1, numbal
		# convert benal to units of umoles phosphorus/l (bal) for
		# internal calculations, and adjust for % riffle to which
		# periphyton are limited
		bal[i] = (benal[i] / cvpb) * depcor * frrif

		# compute unit growth and respiration rates
		nfixfg,limbal[i],groba,resba = algro2 (ballit,po4,no3,tw,mbalgr[i],cmmpb[i],tamfg,tam,nsfg,cmmnb[i],balr20[i],
		delt60,limit,tcbalg[i],balvel,cmmv,bfixfg[i],cslit[i],cmmd1[i],cmmd2[i],sumba,tcbalr[i],
		nmingr,pmingr,lmingr,nmaxfx,grores[i], nfixfg,limbal[i],groba,resba)

		# calculate net growth rate of algae; grobal is expressed as
		# umoles phosphorus per liter per interval; benthic algae growth
		# will be expressed in terms of volume rather than area for the
		# duration of the subroutines subordinate to balrx; the output
		# values for benthic algae are converted to either mg biomass per
		# sq meter or mg chla per sq meter, whichever the user specifies
		grobal[i] = (groba- resba) * bal[i]

		# track growth that affects water column available n concentrations
		if NFIXFG != 1:     # algae are not fixing n, net growth affects concentrations
			groban = grobal[i]
		else:               # algae that are fixing n affect water column n through respiration only
			groban = -resba * bal[i]

		# calculate cumulative net growth of algal types simulated so far
		grotot = grotot + grobal[i]

		# calculate cumulative algal growth that affects available n
		grtotn = grtotn + groban

		if grobal[i] > 0.0 and grotot > 0.0:
			# check that cumulative growth rate of algal types does not exceed
			# limitations imposed by the availability of required nutrients;
			# if so, reduce growth rate of last algal type

			grotmp = grotot # set temporary variable for comparison purposes
			grotot = grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,nfixfg,grtotn,grotot)
			if grotot < 0.0:            # this should never happen
				grotot = 0.0

			# compare nutrient-checked growth to original cumulative total,
			if grotot < grotmp:
				# adjust growth rate of last algal type
				grobal[i] = grobal[i] - (grotmp - grotot)
				# track changes in growth that affect available n concentrations
				if nfixfg != 1:     # n-fixation not occurring, all growth affects n
					groban = grobal[i]
				else:  # n-fixation is occurring, proportionately adjust respiration (groban)
					groban = groban * grotot / grotmp

		# calculate benthic algae removal
		crem = cremvl * depcor * frrif
		cslof1[i], cslof2[i], balvel, dthbal[i] = balrem (crem,sumba,sumbal,cmmbi,tcgraz,tw,binv,bal[i],
		cslof1[i],cslof2[i],balvel, dthbal[i])

		# add the net growth
		bal[i] = bal[i] + grobal[i]

		balmin = minbal * depcor * frrif
		if bal[i] < balmin and grobal[i] < 0.0:   # adjust net growth rate so that population does not fall below minimum level
			grotm2 = grobal[i]      # set temporary variable for growth
			grobal[i] = grobal[i] + (balmin- bal[i])
			bal[i] = balmin
			# adjust growth that affects available n concentrations
			if NFIXFG != 1:     # n-fixation not occurring, all growth affects n
				groban = grobal[i]
			else:                # n-fixation is occurring, proportionately adjust respiration (groban)
				groban = groban * grobal[i] / grotm2

		# subtract death/removal
		bal[i] = bal[i] - dthbal[i]
		if bal[i] < balmin:  # adjust death rate so that population does not drop below minimum level
			dthbal[i] = dthbal[i] - (balmin- bal[i])
			if dthbal[i] < 0.0:
				dthbal[i] = 0.0
			bal[i] = balmin

		# calculate total net growth and removal of all benthic algae types
		sumgro = sumgro + grobal[i]
		sumdth = sumdth + dthbal[i]
		sumgrn = sumgrn + groban
		# update internal loop tracking variables for cumulative growth 
		# to account for grochk and minimum biomass adjustments
		grotot = sumgro
		grtotn = sumgrn
	# 20   continue

	# update do state variable to account for net effect of benthic algae photosynthesis and respiration
	dobalg = cvpb * cvbo * sumgro
	if dox > -dobalg:          # enough oxygen available to satisfy demand
		dox += dobalg
	else:                      # take only available oxygen
		dobalg = -dox
		dox    = 0.0

	# calculate amount of refractory organic constituents which result from benthic algae death
	balorn = refr * sumdth * cvbpn * nmgpum
	balorp = refr * sumdth * pmgpum
	balorc = refr * sumdth * cvbpc * cmgpum

	# add to orc the carbon associated with nutrients immediately
	# released to the available pool from removed benthic algal biomass
	balorc = balorc + fravl * (1.0 - refr) * sumdth * cvbpc * cmgpum

	# calculate amount of nonrefractory organics (bod) which result from benthic algae death
	bodbal = cvnrbo * (1.0 - fravl) * cvpb * sumdth

	# perform materials balance resulting from benthic algae death
	orn,orp,orc,bod = orgbal (balorn,balorp,balorc,bodbal,orn,orp,orc,bod)

	# perform materials balance resulting from uptake of nutrients by benthic algae
	po4,tam,no3,baco2,tambal,no3bal,po4bal = nutup2 (sumgro,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg,bnpfg,
	campr,sumgrn,nminc,po4,tam,no3,baco2,tambal,no3bal,po4bal)
	baco2 = -baco2

	# update available nutrient pools with nutrients immediately
	# released from benthic algal biomass removal processes; nutrients
	# immediately cycled are calculated as a fraction (fravl) of the
	# nonrefractory biomass
	avlpho = fravl * (1.0 - refr) * sumdth * pmgpum
	po4    = po4 + avlpho
	po4bal = po4bal + avlpho

	# nitrogen is released as tam if tam is simulated otherwise as no3
	avlnit = fravl * (1.0 - refr) * sumdth * cvbpn * nmgpum
	if TAMFG:
		tam = tam + avlnit
		tambal = tambal + avlnit
	else:
		no3 = no3 + avlnit
		no3bal = no3bal + avlnit

	# convert biomass to units of mg biomass/m2 and ug chlorophyll a/m2
	for i in range(numbal):
		benal[i]  = (bal[i] * cvpb) / (depcor * frrif)
		balgro[i] = (grobal[i] *cvpb) / (depcor * frrif)
		bdth[i]   =   (dthbal[i] *cvpb) / (depcor * frrif)
		balcla[i] =  benal[i] * cvbcl
		# 30   continue

	return  po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,balorn,balorp,balorc

