
from numpy import zeros, array
from numba import int32, float32, float64, boolean, char    # import the types
from numba.experimental import jitclass

from HSP2.RQUTIL import sink, decbal
from HSP2.utilities  import make_numba_dict, initm

class NUTRX_Class:

	# class variables:



	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, advectData, ui_rq, ui, ts, 
					dox, bod, korea):

		''' Initialize instance variables for nutrient simulation '''

		#self.ERRMSGS=('Placeholder')
		#self.errors = zeros(len(self.ERRMSGS), dtype=int)

		(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = uci_rq['advectData']
		
		delt60 = siminfo['delt'] / 60  # delt60 - simulation time interval in hours
		self.delt60 = delt60
		self.simlen = siminfo['steps']
		self.delts  = siminfo['delt'] * 60
		self.uunits = siminfo['units']

		self.nexits = nexits

		self.AFACT = 43560.0
		if self.uunits == 2:
			# si units conversion
			self.AFACT = 1000000.0

		self.vol = vol * self.AFACT
		self.svol = self.vol

		# table-type nut-flags
		self.TAMFG  = int(ui['NH3FG'])
		self.NO2FG  = int(ui['NO2FG'])
		self.PO4FG  = int(ui['PO4FG'])
		self.AMVFG  = int(ui['AMVFG'])
		self.DENFG  = int(ui['DENFG'])
		self.ADNHFG = int(ui['ADNHFG'])
		self.ADPOFG = int(ui['ADPOFG'])
		self.PHFLAG = int(ui['PHFLAG'])
		self.BENRFG = int(ui_rq['BENRFG'])

		# table-type nut-ad-flags
		self.NUADFG = zeros(7)
		for j in range(1, 7):
			self.NUADFG[j] = int(ui['NUADFG(' + str(j) + ')'])

		if (self.TAMFG == 0 and (self.AMVFG == 1 or self.ADNHFG == 1)) or (self.PO4FG == 0 and self.ADPOFG == 1):
			pass
			# error - either: 1) tam is not being simulated, and nh3 volat. or
			# nh4 adsorption is being simulated; or 2) po4 is not being
			# simulated, and po4 adsorption is being simulated
			# ERRMSG:

		if (self.ADNHFG == 1 or self.ADPOFG == 1) and self.SEDFG == 0:
			pass
			# ERRMSG: error - sediment associated nh4 and/or po4 is being simulated,but sediment is not being simulated in section sedtrn

		self.uafxm = zeros((13,4))
		sf = 1.0

		if self.uunits == 1:			# convert from lb/ac.day to mg.ft3/l.ft2.ivl
			sf = 0.3677 * delt60 / 24.0
		else:					# convert from kg/ha.day to mg.m3/l.m2.ivl	
			sf = 0.1 * delt60 / 24.0	

		# convert units to internal:
		if self.NUADFG[1] > 0:
			self.uafxm[:,1] = ui['NUAFXM1'] * sf
		if self.NUADFG[2] > 0:
			self.uafxm[:,2] = ui['NUAFXM2'] * sf
		if self.NUADFG[3] > 0:
			self.uafxm[:,3] = ui['NUAFXM3'] * sf

		# conversion factors - table-type conv-val1
		self.cvbo   = ui['CVBO']
		self.cvbpc  = ui['CVBPC']
		self.cvbpn  = ui['CVBPN']
		self.bpcntc = ui['BPCNTC']

		# calculate derived values
		self.cvbp = (31.0 * self.bpcntc) / (1200.0 * self.cvbpc)
		self.cvbn = 14.0 * self.cvbpn * self.cvbp / 31.0
		
		self.cvoc = self.bpcntc / (100.0 * self.cvbo)
		self.cvon = self.cvbn / self.cvbo
		self.cvop = self.cvbp / self.cvbo	

		if self.BENRFG == 1 or self.PLKFG == 1:    # benthal release parms - table-type nut-benparm
			self.brnit1 = ui['BRNIT1']  * delt60    #  convert units from 1/hr to 1/ivl
			self.brnit2 = ui['BRNIT2']  * delt60    #  convert units from 1/hr to 1/ivl
			self.brpo41 = ui['BRPO41'] * delt60    #  convert units from 1/hr to 1/ivl

		# nitrification parameters - table-type nut-nitdenit
		self.ktam20 = ui['KTAM20'] * delt60     # convert units from 1/hr to 1/ivl
		self.kno220 = ui['KNO220'] * delt60     # convert units from 1/hr to 1/ivl
		self.tcnit  = ui['TCNIT']
		self.kno320 = ui['KNO320'] * delt60     # convert units from 1/hr to 1/ivl
		self.tcden  = ui['TCDEN']
		self.denoxt = ui['DENOXT']

		if self.TAMFG == 1 and self.AMVFG == 1:   # ammonia volatilization parameters table nut-nh3volat
			self.expnvg = ui['EXPNVG']
			self.expnvl = ui['EXPNVL']

		if self.TAMFG == 1 and self.PHFLAG == 3:     # monthly ph values table mon-phval, not in RCHRES.SEQ
			self.phvalm = ui['PHVALM']

		self.nupm3 = zeros(7)
		self.nuadpm = zeros(7)
		self.rsnh4 = zeros(13)
		self.rspo4 = zeros(13)

		if (self.TAMFG == 1 and self.ADNHFG == 1) or (self.PO4FG == 1 and self.ADPOFG == 1):
			# bed sediment concentrations of nh4 and po4 - table nut-bedconc, not in RCHRES.SEQ
			self.nupm3[:] = ui['NUPM3']  /1.0E6   # convert concentrations from mg/kg to internal units of mg/mg
			
			# initialize adsorbed nutrient mass storages in bed
			self.rsnh4[8] = 0.0
			self.rspo4[8] = 0.0
			"""
			do 70 i= 5,7
				rsnh4(i) = bnh4(i-4) * rsed(i)
				rspo4(i) = bpo4(i-4) * rsed(i)
				rsnh4(8) = rsnh4(8)  + rsnh4(i)
				rspo4(8) = rspo4(8)  + rspo4(i)
			"""
				
			# adsorption parameters - table-type nut-adsparm
			self.nuadpm[:] = ui['NUADPM']  # dimension 6; NH4 (sand, silt, cla) and PO4 (sand, silt, clay)

		# initial conditions - table-type nut-dinit
		self.dnust = zeros(7); 		self.dnust2 = zeros(7)
		self.dnust[1] = ui['NO3'];	self.dnust2[1] *= self.vol
		self.dnust[2] = ui['TAM'];	self.dnust2[2] *= self.vol
		self.dnust[3] = ui['NO2'];	self.dnust2[3] *= self.vol
		self.dnust[4] = ui['PO4'];	self.dnust2[4] *= self.vol
		
		if self.TAMFG == 1:  # do the tam-associated initial values (nh4 nh3 phval)
			self.phval = ui['PHVAL']
			# assume nh4 and nh3 are 0.99 x tam and 0.01 x tam respectively
			self.dnust[5] = 0.99 * self.dnust[2];   self.dnust2[5] = self.dnust[5] * self.vol
			self.dnust[6] = 0.01 * self.dnust[2];   self.dnust2[6] = self.dnust[6] * self.vol

		if (self.TAMFG == 1 and self.ADNHFG == 1) or (self.PO4FG == 1 and self.ADPOFG == 1):
			# suspended sediment concentrations of nh4 and po4 - table nut-adsinit
			# (input concentrations are mg/kg - these are converted to mg/mg for
			# internal computations)
			self.snh4[:] = ui['SNH4'] / 1.0e6  # suspended nh4 (sand, silt, clay) 
			self.spo4[:] = ui['SPO4'] / 1.0e6  # suspended po4 (sand, silt, clay) 
			# initialize adsorbed nutrient mass storages in suspension
			self.rsnh4[4] = 0.0
			self.rspo4[4] = 0.0
			for i in range(1, 4):
				self.rsnh4[i] = self.snh4[i] * rsed[i]
				self.rspo4[i] = self.spo4[i] * rsed[i]
				self.rsnh4[4] += self.rsnh4[i]
				self.rspo4[4] += self.rspo4[i]
			# initialize totals on sand, silt, clay, and grand total
			self.rsnh[9]  = self.rsnh[1] + self.rsnh[5]
			self.rsnh[10] = self.rsnh[2] + self.rsnh[6]
			self.rsnh[11] = self.rsnh[3] + self.rsnh[7]
			self.rsnh[12] = self.rsnh[4] + self.rsnh[8]
			self.rspo4[9]  = self.rspo4[1] + self.rspo4[5]
			self.rspo4[10] = self.rspo4[2] + self.rspo4[6]
			self.rspo4[11] = self.rspo4[3] + self.rspo4[7]
			self.rspo4[12] = self.rspo4[4] + self.rspo4[8]

		# initialize total storages of nutrients in reach
		self.nust = zeros((5,2))

		self.nust[1,1] = self.dnust2[1]
		self.nust[2,1] = self.dnust2[2]
		if self.ADNHFG == 1:
			self.nust[2,1] += self.rsnh[4]

		self.nust[3,1] = self.dnust2[3]
		self.nust[4,1] = self.dnust2[4]
		if self.ADPOFG == 1:
			self.nust[4,1] += self.rspo4[4]

		# initialize nutrient flux if nutrient is not simulated
		self.otam   = zeros(nexits); self.ono2 = zeros(nexits); self.opo4 = zeros(nexits)
		self.rosnh4 = zeros(5); self.rospo4 = zeros(5)
		self.dspo4  = zeros(5); self.dsnh4 = zeros(5)
		self.adpo4  = zeros(5); self.adnh4 = zeros(5)
		self.ospo4  = zeros((nexits, 5)); self.osnh4 = zeros((nexits, 5))
		self.nucf1 = zeros((5,2))
		self.nucf2 = zeros((5,3))
		self.nucf3 = zeros((5,3))
		self.nucf8 = zeros((5,3))
		self.tnucf2 = zeros((5,3))
		self.nucf4 = zeros((8,2))
		self.nucf5 = zeros((9,2))
		self.nucf6 = zeros((2,2))
		self.nucf7 = zeros((7,2))

		if self.TAMFG == 0:
			self.nucf1[2,1] = 0.0
			self.otam[:] = 0.0   # dimension nexits

		if self.ADNHFG == 0:
			self.rosnh4[:]  = 0.0
			self.dsnh4[:]   = 0.0
			self.adnh4[:]   = 0.0
			self.osnh4[:,:] = 0.0

		if self.NO2FG == 0:
			self.nucf1[3,1] = 0.0
			self.ono2[:]    = 0.0

		if self.PO4FG == 0:
			self.nucf1[4,1] = 0.0
			self.opo4[:]    = 0.0

		if self.ADPOFG == 0:
			self.rospo4[:]  = 0.0
			self.dspo4[:]   = 0.0
			self.adpo4[:]   = 0.0
			self.ospo4[:,:] = 0.0

		# initialize nutrient process fluxes (including ads/des and dep/scour)
		self.nucf4[1:7,1] = 0.0
		self.nucf5[1:7,1] = 0.0
		self.nucf7[1:7,1] = 0.0
		
		self.nucf4[7,1] = 0.0
		self.nucf5[7,1] = 0.0
		self.nucf5[8,1] = 0.0
		self.nucf6[1,1] = 0.0

		self.nucf3[1:5,1:3] = 0.0
		self.nucf8[1:5,1:3] = 0.0

		self.tnuif = zeros(5)

		# via OXRX module:
		self.dox = dox
		self.bod = bod
		self.korea = korea

		# initialize total nutrient masses:
		x = updateMass()


	def simulate(self, tw, dox, bod, ino3, itam, ino2, ipo4, nuafx, nuacn, prec, sarea, advectData):
		''' Determine primary inorganic nitrogen and phosphorus balances'''

		# hydraulics:
		(nexits, vol_, vol, srovol, erovol, sovol, eovol) = advectData
		
		self.vol = vol * self.AFACT

		# Update OXRX current states:
		self.dox = dox
		self.bod = bod

		# set instance inflow variables:
		self.ino3 = ino3
		self.itam = itam
		self.ino2 = ino2
		self.ipo4 = ipo4

		#compute atmospheric deposition influx
		nuadep = zeros(7)
		nuaddr = nuadwt = zeros(7)

		for i in range(1,4):
			n= 2*(i-1)+ 1
			# dry deposition
			if self.NUADFG[n] <= -1:
				nuadfx = ts['nuadfx']
				self.nuaddr[i] = sarea*nuadfx
			elif self.NUADFG[n] >= 1:
				nuaddr[i] = sarea*dayval(nuafxm[mon,i],nuafxm[nxtmon,i],day, ndays)
			else:
				self.nuaddr[i] = 0.0
			# wet deposition
			if self.NUADFG[n+1] <= -1:
				nuadcn = ts['nuadcn']
				nuadwt[n]= prec*sarea*nuadcn
			elif self.NUADFG[n+1] >= 1:
				nuadwt[i] = prec*sarea*dayval(nuacnm[mon,i],nuacnm[nxtmon,i],day,ndays)
			else:
				nuadwt[i] = 0.0

			nuadep[i] = nuaddr[i] + nuadwt[i]

		# advect nitrate
		self.tnuif[1] = ino3
		inno3 = ino3 + nuadep[1]

		self.no3, self.rono3, self.ono3 = advect(inno3, self.no3, self.rono3, self.ono3)

		self.nucf1[1] = self.rono3
		if self.nexits > 1:
			self.tnucf2[:,1] = self.ono3[:]   # nexits

		# advect total ammonia:		
		if self.TAMFG:
			intam = self.itam + nuadep[2]

			self.tam, self.rotam, self.otam = advect(intam, self.tam, self.rotam, self.otam)
			
			self.tnucf1[2] = self.rotam
			if self.nexits > 1:
				self.tnucf2[:,2] = self.otam[:]

		# advect nitrite:
		if self.NO2FG:
			self.tnuif[3] = ino2

			self.no2, self.rono2, self.ono2 =  advect(ino2, self.no2, self.rono2, self.ono2)
			
			self.tnucf1[3] = self.rono2
			if self.nexits > 1:
				self.tnucf2[:,3] = self.ono2[:]   # nexits

		# advect dissolved PO4:
		if self.PO4FG:
			inpo4 = ipo4 + nuadep[3]
			po4, ropo4, opo4 = advect(inpo4,self.po4,self.ropo4,self.opo4)		

		# advect adsorbed PO4:
		if self.ADPOFG:
			# zero the accumulators
			self.ispo4[4]  = 0.0
			self.dspo4[4]  = 0.0
			self.rospo4[4] = 0.0
			if self.nexits > 1:
				self.ospo4[:,4] = 0.0  # nexits

			# repeat for each sediment fraction (LTI)
			for j in range(1, 4):       # get data on sediment-associated phosphate
				fpt = ispofp[j]
				if fpt:
					ispo4[j] = ui['ISPO4']   # else zero if missing

				(self.nuecnt[3], self.spo4[j], self.dspo4[j], self.rospo4[j], self.ospo4[1,j]) \
					= advnut(self.ispo4[j],self.rsed[j],self.rsed[j+3],self.depscr[j],self.rosed[j],self.osed[1,j],self.nexits, 
								self.rspo4[j],self.rspo4[j + 4],self.bpo4[j], self.nuecnt[3],self.spo4[j],self.dspo4[j], self.rospo4[j],self.ospo4[1,j])

				self.ispo4[4]  += self.ispo4[j]
				self.dspo4[4]  += self.dspo4[j]
				self.rospo4[4] += self.rospo4[j]
				if self.nexits > 1:
					self.ospo4[:,4] += self.ospo4[:,j]   # nexits
			self.tnuif[4]  = self.ipo4  + self.ispo4[4]
			self.tnucf1[4] = self.ropo4 + self.rospo4[4]
			if self.nexits > 1:
				self.tnucf2[:,4] = self.opo4[:]+ self.ospo4[:,4]  # nexits
		else:            # no adsorbed fraction
			self.tnuif[4]  = self.ipo4
			self.tnucf1[4] = self.ropo4
			if self.nexits > 1:
				self.tnucf2[:,4] = self.opo4[:]

		# advect adsorbed ammonium
		if self.TAMFG and self.ADNHFG:    # advect adsorbed ammonium
			self.isnh4 = self.dsnh4 = self.rosnh4 = self.osnh4 = zeros(5)

			# zero the accumulators
			self.isnh4[4]  = 0.0; 
			self.dsnh4[4]  = 0.0
			self.rosnh4[4] = 0.0
			if self.nexits > 1:
				self.osnh4[:,4] = 0.0   # nexits

			# repeat for each sediment fraction
			for j in range(1, 4):
				# get data on sediment-associated ammonium
				fpt = isnhfp[j]
				self.isnh4[j]= ui['OSNH4']  # or zero if not there

				(self.nuecnt[3],self.snh4[j],self.dsnh4[j],self.rosnh4[j],self.osnh4[1,j]) \
					= advnut(self.isnh4[j],self.rsed[j],self.rsed[j + 3],self.depscr[j],self.rosed[j],self.osed[1,j],self.nexits,
						    	self.rsnh[j],self.rsnh4[j + 4],self.bnh4[j],self.nuecnt[3],self.snh4[j],self.dsnh4[j],self.rosnh4[j],self.osnh4[1,j])

				self.isnh4[4]  += self.isnh4[j]
				self.dsnh4[4]  += self.dsnh4[j]
				self.rosnh4[4] += self.rosnh4[j]

				if self.nexits > 1:
					self.osnh4[:,4] = self.osnh4[:,j]   # nexits
			
			self.tnuif[2]  = self.itam + self.isnh4[4]
			self.tnucf1[2] = self.rotam + self.rosnh4[4]
			if self.nexits > 1:
				self.tnucf2[:,2] = self.otam[:] + self.osnh4[:,4]  # nexits
		
		else:                 # no adsorbed fraction
			self.tnuif[2]  = self.itam
			self.tnucf1[2] = self.rotam
			if self.nexits > 1:
				self.tnucf2[:,2] = self.otam[:]  # nexits


 		# calculate ammonia ionization in water column
		if self.TAMFG:
			# get ph values
			#TMR ph = ??? # last computed value, time series, monthly, constant; phflag
			# compute ammonia ionization
			(self.nh3, self.nh4) = ammion(tw, phval, self.tam, self.nh3, self.nh4)

		if avdepe > 0.17:
			if self.BENRFG:
				# simulate benthal release of inorganic nitrogen and
				# ortho-phosphorus; and compute associated fluxes
				if self.TAMFG:
					(self.tam, self.bentam) = benth(self.dox,self.anaer,self.brtam,scrfac,depcor,tam,bentam)
					self.bnrtam = self.bentam * self.voL

				if self.PO4FG:
					self.po4, self.benpo4 = benth(dox,anaer,brpo4,scrfac,depcor, po4, benpo4)
					self.bnrpo4 = self.benpo4 * self.vol

			if self.TAMFG:
				if self.AMVFG:     # compute ammonia volatilization
					twkelv = tw + 273.16        # convert water temperature to degrees kelvin 
					avdepm = avdepe * 0.3048    # convert depth to meters
					self.tam, self.nh3vlt = nh3vol(self.expnvg,self.expnvl,self.korea,wind,self.delt60,self.delts,avdepm,twkelv,tw,phval,self.tam)
					self.volnh3 = -self.nh3vlt * self.vol
				else:
					self.volnh3 = 0.0

				# calculate amount of nitrification; nitrification does not
				# take place if the do concentration is less than 2.0 mg/l
				(self.tam,self.no2,self.no3,self.dox, self.dodemd,self.tamnit,self.no2ntc,self.no3nit) = \
					nitrif(self.ktam20,self.tcnit,tw,self.NO2FG,self.kno220,
							self.tam,self.no2,self.no3,self.dox,self.dodemd,self.tamnit,self.no2ntc,self.no3nit)

				# compute nitrification fluxes
				self.nitdox = -self.dodemd * vol
				self.nittam = -self.tamnit * vol
				self.nitno2 =  self.no2ntc * vol
				self.nitno3 =  self.no3nit * vol

			if self.DENFG:    # consider denitrification processes, and compute associated fluxes
				self.no3de = 0.0
				(self.no3, self.no3de) = denit(self.kno320, self.tcden, tw, self.dox, self.denoxt, self.no3)
				self.denno3 = -self.no3de * vol

			# calculate amount of inorganic constituents released by bod decay in reach water
			self.decnit = self.bodox * self.cvon
			self.decpo4 = self.bodox * self.cvop
			self.decco2 = self.bodox * self.cvoc

			# update state variables of inorganic constituents which
			# are end products of bod decay; and compute associated fluxes
			(self.tam, self.no3, self.po4) = decbal(self.TAMFG, self.PO4FG, self.decnit, self.decpo4, 
													self.tam, self.no3, self.po4)
			if self.TAMFG:
				self.bodtam = self.decnit * vol
			else:
				self.bodno3 = self.decnit * vol

			if PO4FG == 1:
				self.bodpo4 = self.decpo4 * vol

			if self.PO4FG and self.SEDFG and self.ADPOFG:   # compute adsorption/desorption of phosphate
				dumxxx = 0.0

				(self.po4, self.spo4[1], dumxxx, self.adpo4[1]) \
					= addsnu(vol, self.rsed[1], self.adpopm[1], self.po4, self.spo4[1], dumxxx, self.adpo4[1])

			if self.TAMFG and self.SEDFG and self.ADNHFG:  # compute adsorption/desorption of ammonium
				# first compute ammonia ionization
				(self.nh3, self.nh4) = ammion(tw, phval, self.tam, self.nh3, self.nh4)
				(self.nh4, self.snh4[1], self.tam, self.adnh4[1]) \
					= addsnu(vol, self.rsed[1], self.adnhpm[1], self.nh4, self.snh4[1], self.tam, self.adnh4[1])
				# then re-compute ammonia ionization
				(self.nh3, self.nh4) = ammion(tw, phval, self.tam, self.nh3, self.nh4)
		else:
			# too little water is in reach to warrant simulation of quality processes
			self.decnit = self.decpo4 = self.decco2 = 0.0
			self.nitdox = self.denbod = self.nittam = 0.0
			self.bnrtam = self.volnh3 = self.bodtam = 0.0
			self.nitno2 = self.nitno3 = self.denno3 = 0.0
			self.bodno3 = 0.0
			self.bnrpo4 = self.bodpo4 = 0.0

			self.adnh4[1:5] = 0.0
			self.adpo4[1:5] = 0.0
		
		self.totdox = self.readox + self.boddox + self.bendox + self.nitdox
		self.totbod = self.decbod + self.bnrbod + self.snkbod + self.denbod
		self.totno3 = self.nitno3 + self.denno3 + self.bodno3
		self.tottam = self.nittam + self.volnh3 + self.bnrtam + self.bodtam
		self.totpo4 = self.bnrpo4 + self.bodpo4

		if self.PO4FG and self.SEDFG and self.ADPOFG:  # find total quantity of phosphate on various forms of sediment
			totpm1 = totpm2 = totpm3 = 0.0
			
			for j in range(1, 4):
				self.rspo4[j]     = self.spo4[j] * self.rsed[j]         # compute mass of phosphate adsorbed to each suspended fraction
				self.rspo4[j + 4] = self.bpo4[j] * self.rsed[j + 3]     # compute mass of phosphate adsorbed to each bed fraction
				self.rspo4[j + 8] = self.rspo4[j] + self.rspo4[j + 4]   # compute total mass of phosphate on each sediment fraction
				
				totpm1 += self.rspo4[j]
				totpm2 += self.rspo4[j + 4]
				totpm3 += self.rspo4[j + 8]

			rspo4[4]  = totpm1	 # compute total suspended phosphate
			rspo4[8]  = totpm2   # compute total bed phosphate
			rspo4[12] = totpm3   # compute total sediment-associated phosphate

		# calculate total amount of ammonium on various forms of sediment
		if TAMFG and SEDFG and ADNHFG:
			totnm1 = totnm2 = totnm3 = 0.0

			for j in range(1, 4):
				self.rsnh4[j]     = self.snh4[j]  * self.rsed[j]       # compute mass of ammonium adsorbed to each suspended fraction
				self.rsnh4[j + 4] = self.bnh4[j]  * self.rsed[j + 3]   # compute mass of ammonium adsorbed to each bed fraction
				self.rsnh4[j + 8] = self.rsnh4[j] + self.rsnh4[j + 4]  # compute total mass of ammonium on each sediment fraction
				
				totnm1 += self.rsnh4[j]
				totnm2 += self.rsnh4[j + 4]
				totnm3 += self.rsnh4[j + 8]
			
			self.rsnh4[4]  = totnm1      # compute total suspended ammonium
			self.rsnh4[8]  = totnm2		# compute total bed ammonium
			self.rsnh4[12] = totnm3      # compute total sediment-associated ammonium


	def updateMass():
		# calculate total resident mass of nutrient species
		vol = self.vol

		self.rno3 = self.no3 * vol
		self.rtam  = self.tam * vol
		self.rno2  = self.no2 * vol
		self.rpo4  = self.po4 * vol
		self.rnh4  = self.nh4 * vol
		self.rnh3  = self.nh3 * vol
		self.rrno3 = self.no3 * vol
		self.rrtam = self.tam * vol	

		if self.ADNHFG == 1:  
			self.rrtam += self.rsnh4(4)  # add adsorbed suspended nh4 to dissolved
			
		self.rrno2 = self.no2 * vol
		self.rrpo4 = self.po4 * vol
		if self.ADPOFG == 1:  
			self.rrpo4 += self.rspo4(4) # add adsorbed suspended po4 to dissolved	

		return

	#--------------------------------------------------------------
	#	static methods
	#--------------------------------------------------------------
	@staticmethod
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


	@staticmethod
	def  advnut(isnut,rsed,bsed,depscr,rosed,osed,nexits,
					rsnuts,rbnuts,bnut,ecnt,snut,dsnut,rosnut,osnut):

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

	@staticmethod
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

	@staticmethod
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


	@staticmethod
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


	@staticmethod
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

	@staticmethod
	def nh3vol(expnvg, expnvl, korea, wind, delt60, delts, avdepm, twkelv, tw, phval, tam):
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

	@staticmethod
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