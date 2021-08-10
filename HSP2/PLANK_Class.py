
from numpy import zeros, array
from numba import int32, float32, float64, boolean    # import the types
from numba.experimental import jitclass

from HSP2.RQUTIL import sink, decbal
from HSP2.utilities  import make_numba_dict, initm

numba_vars = [
	('errors', int32[:]), ('ERRMSGS', char[:]),
	('delt60', float32), ('simlen', int32), ('delts', float32), ('uunits'), int32),
	('nexits', nexits), ('AFACT', float32), ('vol', float32), ('svol', float32),

]


@jitclass(numba_vars)
class PLANK_Class:

	# class variables:

	ERRMSGS=('Placeholder')

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, store, siminfo, uci_rq, ui, ts, OXRX, NUTRX):

		''' Initialize instance variables for lower food web simulation '''

		self.errors = zeros(len(self.ERRMSGS), dtype=int)

		(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = uci_rq['advectData']
		
		ui_rq = make_numba_dict(uci_rq)

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

		vol = vol * self.AFACT
		self.vol = vol
		self.svol = vol

		# flags - table-type PLNK-FLAGS
		self.PHYFG  = int(ui['PHYFG'])
		self.ZOOFG  = int(ui['ZOOFG'])
		self.BALFG  = int(ui['BALFG'])
		self.SDLTFG = int(ui['SDLTFG'])
		self.AMRFG  = int(ui['AMRFG'])
		self.DECFG  = int(ui['DECFG'])
		self.NSFG   = int(ui['NSFG'])
		self.ZFOOD  = int(ui['ZFOOD'])
		self.BNPFG  = int(ui['BNPFG'])

		self.HTFG   = NUTRX.HTFG
		self.TAMFG  = NUTRX.NH3FG
		self.NO2FG  = NUTRX.NO2FG
		self.PO4FG  = NUTRX.PO4FG

		self.ADNHFG = NUTRX.ADNHFG
		self.ADPOFG = NUTRX.ADPOFG

		self.bpcntc = ui['BPCNTC']
		self.cvbo  = ui['CVBO']
		self.cvbpc  = ui['CVBPC']
		self.cvbpn  = ui['CVBPN']

		if self.ZOOFG == 1 and self.PHYFG == 0:
			pass # ERRMSG: error - zooplankton cannot be simulated without phytoplankton
		if self.NSFG == 1 and self.TAMFG == 0:
			pass # ERRMSG: error - ammonia cannot be included in n supply if it is not
		if self.PO4FG == 0:
			pass # ERRMSG: error - phosphate must be simulated if plankton are being

		self.numbal = 0
		if self.BALFG == 2:   # user has selected multiple species with more complex kinetics
			# additional benthic algae flags - table-type BENAL-FLAG
			self.numbal  = int(ui['NUMBAL'])
			self.BINVFG  = int(ui['BINVFG'])
			self.BFIXFG1 = int(ui['BFIXFG1'])
			self.BFIXFG2 = int(ui['BFIXFG2'])
			self.BFIXFG3 = int(ui['BFIXFG3'])
			self.BFIXFG4 = int(ui['BFIXFG4'])
		else:
			self.numbal = BALFG          # single species or none
			
		if self.HTFG == 0:     # fraction of surface exposed - table-type surf-exposed
			self.cfsaex = ui['CFSAEX']

		# table-type plnk-parm1
		self.ratclp = ui['RATCLP']
		self.nonref = ui['NONREF']
		self.litsed = ui['LITSED']
		self.alnpr  = ui['ALNPR']
		self.extb   = ui['EXTB']
		self.malgr  = ui['MALGR'] * delt60
		self.paradf = ui['PARADF']
		self.refr = 1.0 - self.nonref       # define fraction of biomass which is refractory material	

		# compute derived conversion factors
		self.cvbc   = bpcntc / 100.0
		self.cvnrbo = self.nonref * self.cvbo

		self.cvbp = (31.0 * bpcntc) / (1200.0 * cvbpc)
		self.cvbn = 14.0 * cvbpn * self.cvbp / 31.0
		self.cvpb   = 31.0 / (1000.0 * self.cvbp)
		cvbcl  = 31.0 * self.ratclp / self.cvpb

		# table-type plnk-parm2
		self.cmmlt  = ui['CMMLT']
		self.cmmn   = ui['CMMN']
		self.cmmnp  = ui['CMMNP']
		self.cmmp   = ui['CMMP']
		self.talgrh = ui['TALGRH']
		self.talgrl = ui['TALGRL']
		self.talgrm = ui['TALGRM']

		# table-type plnk-parm3
		self.alr20 = ui['ALR20'] * delt60   	# convert rates from 1/hr to 1/ivl
		self.aldl  = ui['ALDL']  * delt60 
		self.oxald = ui['OXALD'] * delt60
		self.naldh = ui['NALDH'] * delt60
		self.paldh = ui['PALDH']

		# table-type plnk-parm4
		self.nmingr = ui['NMINGR']
		self.pmingr = ui['PMINGR']
		self.cmingr = ui['CMINGR']
		self.lmingr = ui['LMINGR']
		self.nminc  = ui['NMINC']

		# phytoplankton-specific parms - table-type phyto-parm
		# this table must always be input so that REFSET is read
		self.seed   = ui['SEED']
		self.mxstay = ui['MXSTAY']
		self.oref   = ui['OREF'] 
		self.claldh = ui['CLALDH']
		self.physet = ui['PHYSET'] * delt60	# change settling rates to units of 1/ivl
		self.refset = ui['REFSET'] * delt60	# change settling rates to units of 1/ivl

		if self.PHYFG == 1 and self.ZOOFG == 1:   # zooplankton-specific parameters  
			# table-type zoo-parm1
			self.mzoeat = ui['MZOEAT'] * delt60   # convert rates from 1/hr to 1/ivl
			self.zfil20 = ui['ZFIL20'] * delt60   # convert rates from 1/hr to 1/ivl
			self.zres20 = ui['ZRES20'] * delt60   # convert rates from 1/hr to 1/ivl
			self.zd     = ui['ZD']     * delt60   # convert rates from 1/hr to 1/ivl
			self.oxzd   = ui['OXZD']   * delt60   # convert rates from 1/hr to 1/ivl
			# table-type zoo-parm2
			self.tczfil = ui['TCZFIL']
			self.tczres = ui['TCZRES']
			self.zexdel = ui['ZEXDEL']
			self.zomass = ui['ZOMASS']

		if self.BALFG >= 1:    #   benthic algae-specific parms; table-type benal-parm
			self.mbal   = ui['MBAL']   / cvpb   # convert maximum benthic algae to micromoles of phosphorus
			self.cfbalr = ui['CFBALR']
			self.cfbalg = ui['CFBALG']
			self.minbal = ui['MINBAL'] / cvpb   # convert maximum benthic algae to micromoles of phosphorus
			self.campr  = ui['CAMPR']
			self.fravl  = ui['FRAVL']
			self.nmaxfx = ui['NMAXFX']
		
			if self.BALFG == 2:  # user has selected multiple species with more complex kinetics
				self.mbalgr = self.tcbalg = zeros(numbal)
				self.cmmnb =  self.cmmpb  =  zeros(numbal)
				self.cmmd1 =  self.cmmd2  =  zeros(numbal)
				self.cslit =  zeros(numbal)
				
				self.balr20 = zeros(numbal)
				self.tcbalr = zeros(numbal)
				self.cslof1 = zeros(numbal)
				self.cslof2 = zeros(numbal)
				self.grores = zeros(numbal)		
				
				for i in range(self.numbal):
					# species-specific growth parms - table type benal-grow
					self.mbalgr[i] = ui['MBALGR']  * delt60
					self.tcbalg[i] = ui['TCBALG']
					self.cmmnb[i] =  ui['CMMNB']
					self.cmmpb[i] =  ui['CMMPB']
					self.cmmd1[i] =  ui['CMMD1']
					self.cmmd2[i] =  ui['CMMD2']
					self.cslit[i] =  ui['CSLIT']
					# species-specific resp and scour parms - table type benal-resscr
					self.balr20[i] = ui['BALR20']  * delt60
					self.tcbalr[i] = ui['TCBALR']
					self.cslof1[i] = ui['CSLOF1']  * delt60
					self.cslof2[i] = ui['CSLOF2']
					self.grores[i] = ui['GRORES']

				#  grazing and disturbance parms - table-type benal-graze
				self.cremvl = ui['CREMVL']
				self.cmmbi  = ui['CMMBI']
				self.binv   = ui['BINV']
				self.tcgraz = ui['TCGRAZ']

				self.cremvl = (self.cremvl / cvpb) / hrpyr * delt60		#TMR

				if self.SDLTFG == 2:	# turbidity regression parms - table-type benal-light
					self.ctrbq1 = ui['CTRBQ1']
					self.ctrbq2 = ui['CTRBQ2']
					self.cktrb1 = ui['CKTRB1']
					self.cktrb2 = ui['CKTRB2']	

				if self.BINVFG == 3:      # monthly benthic invertebrate density - table-type mon-binv
					self.binvm = ui['BINVM']

			# table-type benal-riff1
			self.frrif  = ui['FRRIF']	
			self.cmmv   = ui['CMMV']
			self.rifcq1 = ui['RIFCQ1']
			self.rifcq2 = ui['RIFCQ2']
			self.rifcq3 = ui['RIFCQ3']

			# table-type benal-riff2
			self.rifvf1 = ui['RIFVF1'] 
			self.rifvf2 = ui['RIFVF2']
			self.rifvf3 = ui['RIFVF3']
			self.rifvf4 = ui['RIFVF4']

			self.rifdf1 = ui['RIFDF1']
			self.rifdf2 = ui['RIFDF2']
			self.rifdf3 = ui['RIFDF3']
			self.rifdf4 = ui['RIFDF4']

		# table-type plnk-init
		self.phyto = ui['PHYTO']
		self.zoo   = ui['ZOO']
		#benal = ui['BENAL']
		self.orn   = ui['ORN']
		self.orp   = ui['ORP']
		self.orc   = ui['ORC']

		# atmospheric deposition flags
		self.PLADFG = zeros(7)
		for j in range(1,7):
			self.PLADFG[j] = ui['PLADFG(' + str(j) + ')']

		# variable initialization:
		if self.PHYFG == 0:   # initialize fluxes of inactive constituent
			self.rophyt = 0.0
			self.ophyt[:] = 0.0	#nexits
			self.phydox = self.phybod = 0.0
			self.phytam = self.phyno3 = self.phypo4 = 0.0
			self.phyorn = self.phyorp = self.phyorc = 0.0
			self.pyco2  = 0.0
			self.dthphy = self.grophy = self.totphy = 0.0

		self.ozoo = zeros(nexits)

		if self.ZOOFG == 1:   # convert zoo to mg/l
			self.zoo *= self.zomass

		else:   #  zooplankton not simulated, but use default values
			# initialize fluxes of inactive constituent
			self.rozoo = 0.0
			self.ozoo[:] = 0.0	#nexits
			self.zoodox = self.zoobod = 0.0
			self.zootam = self.zoono3 = self.zoopo4 = 0.0
			self.zooorn = self.zooorp = self.zooorc = 0.0
			self.zoophy = 0.0
			self.zoco2  = 0.0
			self.grozoo = self.dthzoo = self.totzoo = 0.0

		self.benal = zeros(self.numbal)
		self.flxbal = zeros((4,5))

		if self.numbal == 1:      # single species
			self.benal[1] = ui['BENAL']       # points to  table-type plnk-init above for rvals
		elif self.numbal >= 2:      # multiple species - table-type benal-init
			for n in range(self.numbal):
				self.benal[n] = ui['BENAL' + str(n+1)]
		else:                     # no benthic algae simulated
			self.baldox = self.balbod = 0.0
			self.baltam = self.balno3 = self.balpo4 = 0.0
			self.balorn = self.balorp = self.balorc = 0.0
			self.baco2 = 0.0
			self.flxbal[1:4,1:5] = 0.0

		# compute derived quantities
		self.phycla = self.phyto * cvbcl

		self.balcla = zeros(4)
		for i in range(numbal):
			self.balcla[i] = self.benal[i] * cvbcl

		self.lsnh4 = self.lspo4 = zeros(4)

		if vol > 0.0:   # compute initial summary concentrations
			for i in range(1, 4):
				self.lsnh4[i] = NUTRX.rsnh4[i] / vol
				self.lspo4[i] = NUTRX.rspo4[i] / vol

		# calculate summary concentrations:
		pksums(vol)

	def simulate(self, OXRX, NUTRX, tw, iphyto, izoo, iorn, iorp, iorc, wash, solrad, prec, sarea, advData):
		''' Wrapper for "_simulate", which is numba-accelerated '''

		# hydraulics:
		(nexits, vol_, vol, srovol, erovol, sovol, eovol) = advData
		self.vol = vol

		# store OXRX and NUTRX current states:
		self.dox = OXRX.dox
		self.bod = OXRX.bod

		# call numba-accelerated method:
		_simulate(self)

		# update OXRX and NUTRX states:
		OXRX.dox = self.dox
		OXRX.bod = self.bod

		NUTRX.orn = self.orn
		NUTRX.orp = self.orp
		NUTRX.orc = self.orc
		NUTRX.torn = self.torn
		NUTRX.torp = self.torp
		NUTRX.torc = self.torc

		# return updated classes:
		return OXRX, NUTRX

	def _simulate():
		pass


	def pksums(vol):
	''' computes summaries of: total organic n, p, c; total n, p; potbod'''

		# undefined summary concentrations:
		if (vol <= 0):
			self.torn   = -1.0e30
			self.torp   = -1.0e30
			self.torc   = -1.0e30
			self.potbod = -1.0e30
			self.tn     = -1.0e30
			self.tp     = -1.0e30

			return

		# Calculate sums:
		tval = self.bod / self.cvbo
		if self.PHYFG == 1:
			tval += self.phyto
			if self.ZOOFG == 1:
				tval += self.zoo

		self.torn   = self.orn + self.cvbn * tval
		self.torp   = self.orp + self.cvbp * tval
		self.torc   = self.orc + self.cvbc * tval
		self.potbod = self.bod

		# total N
		self.tn     = self.torn + no3
		if self.TAMFG == 1:
			self.tn += self.tam
		if self.NO2FG == 1:
			self.tn += self.no2
		if self.ADNHFG == 1:
			for i in range(1, 4):
				self.tn += self.lsnh4[i]

		# total P
		self.tp = self.torp
		if self.PO4FG == 1:
			self.tp += self.po4
		if self.ADPOFG == 1:
			for i in range(1, 4):
				self.tp += self.lspo4[i]			
		if self.PHYFG == 1:
			self.potbod += (cvnrbo * self.phyto)
			
			if self.ZOOFG == 1:
				self.potbod += (cvnrbo * self.zoo)

		return			