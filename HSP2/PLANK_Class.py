import numpy as np
from numpy import zeros, array
from math import log, exp
import numba as nb
from numba.experimental import jitclass

from HSP2.ADCALC import advect
from HSP2.OXRX_Class import OXRX_Class
from HSP2.NUTRX_Class import NUTRX_Class
from HSP2.RQUTIL import sink, decbal
from HSP2.utilities  import make_numba_dict, initm

spec = [
	('OXRX', OXRX_Class.class_type.instance_type),
	('NUTRX', NUTRX_Class.class_type.instance_type),
	('aldh', nb.float64),
	('aldl', nb.float64),
	('alnpr', nb.float64),
	('alr20', nb.float64),
	('AMRFG', nb.int32),
	('baco2', nb.float64),
	('balbod', nb.float64),
	('balcla', nb.float64[:]),
	('baldep', nb.float64),
	('baldox', nb.float64),
	('BALFG', nb.int32),
	('ballit', nb.float64),
	('balno3', nb.float64),
	('balorc', nb.float64),
	('balorn', nb.float64),
	('balorp', nb.float64),
	('balpo4', nb.float64),
	('balr20', nb.float64[:]),
	('baltam', nb.float64),
	('balvel', nb.float64),
	('benal', nb.float64[:]),
	('BFIXFG', nb.int32[:]),
	('BINVFG', nb.int32),
	('BNPFG', nb.int32),
	('bpcntc', nb.float64),
	('campr', nb.float64),
	('cbnrbo', nb.float64),
	('cfbalg', nb.float64),
	('cfbalr', nb.float64),
	('cflit', nb.float64),
	('cfsaex', nb.float64),
	('cktrb1', nb.float64),
	('cktrb2', nb.float64),
	('claldh', nb.float64),
	('cmingr', nb.float64),
	('cmmbi', nb.float64),
	('cmmd1', nb.float64[:]),
	('cmmd2', nb.float64[:]),
	('cmmlt', nb.float64),
	('cmmn', nb.float64),
	('cmmnb', nb.float64[:]),
	('cmmnp', nb.float64),
	('cmmp', nb.float64),
	('cmmpb', nb.float64[:]),
	('cmmv', nb.float64),
	('co2', nb.float64),
	('conv', nb.float64),
	('cremvl', nb.float64),
	('cslit', nb.float64[:]),
	('cslof1', nb.float64[:]),
	('cslof2', nb.float64[:]),
	('ctrbq1', nb.float64),
	('ctrbq2', nb.float64),
	('cvbc', nb.float64),
	('cvbcl', nb.float64),
	('cvbn', nb.float64),
	('cvbo', nb.float64),
	('cvbp', nb.float64),
	('cvbpc', nb.float64),
	('cvbpn', nb.float64),
	('cvnrbo', nb.float64),
	('cvpb', nb.float64),
	('DECFG', nb.int32),
	('delt', nb.float64),
	('delt60', nb.float64),
	('delts', nb.float64),
	('dthbal', nb.float64[:]),
	('dthphy', nb.float64),
	('dthtba', nb.float64),
	('dthzoo', nb.float64),
	('errors', nb.int64[:]),
	('extb', nb.float64),
	('flxbal', nb.float64[:,:]),
	('fravl', nb.float64),
	('frrif', nb.float64),
	('grobal', nb.float64[:]),
	('grophy', nb.float64),
	('grores', nb.float64[:]),
	('grotba', nb.float64),
	('grozoo', nb.float64),
	('HTFG', nb.int32),
	('iorc', nb.float64),
	('iorn', nb.float64),
	('iorp', nb.float64),
	('iphyto', nb.float64),
	('itorc', nb.float64),
	('itorn', nb.float64),
	('itorp', nb.float64),
	('itotn', nb.float64),
	('itotp', nb.float64),
	('izoo', nb.float64),
	('limbal', nb.int32[:]),
	('limphy', nb.int32),
	('litsed', nb.float64),
	('lmingr', nb.float64),
	('lsnh4', nb.float64[:]),
	('lspo4', nb.float64[:]),
	('malgr', nb.float64),
	('mbal', nb.float64),
	('mbalgr', nb.float64[:]),
	('minbal', nb.float64),
	('mxstay', nb.float64),
	('mzoeat', nb.float64),
	('naldh', nb.float64),
	('nexits', nb.int32),
	('nmaxfx', nb.float64),
	('nminc', nb.float64),
	('nmingr', nb.float64),
	('nonref', nb.float64),
	('NSFG', nb.int32),
	('numbal', nb.int32),
	('oorc', nb.float64[:]),
	('oorn', nb.float64[:]),
	('oorp', nb.float64[:]),
	('ophyt', nb.float64[:]),
	('orc', nb.float64),
	('oref', nb.float64),
	('orn', nb.float64),
	('orp', nb.float64),
	('otorc', nb.float64[:]),
	('otorn', nb.float64[:]),
	('otorp', nb.float64[:]),
	('ototn', nb.float64[:]),
	('ototp', nb.float64[:]),
	('oxald', nb.float64),
	('oxzd', nb.float64),
	('ozoo', nb.float64[:]),
	('paldh', nb.float64),
	('paradf', nb.float64),
	('PHFG', nb.int32),
	('phybod', nb.float64),
	('phycla', nb.float64),
	('phydox', nb.float64),
	('PHYFG', nb.int32),
	('phylit', nb.float64),
	('phyno3', nb.float64),
	('phyorc', nb.float64),
	('phyorn', nb.float64),
	('phyorp', nb.float64),
	('phypo4', nb.float64),
	('physet', nb.float64),
	('phytam', nb.float64),
	('phyto', nb.float64),
	('pmingr', nb.float64),
	('potbod', nb.float64),
	('pyco2', nb.float64),
	('ratclp', nb.float64),
	('refr', nb.float64),
	('refset', nb.float64),
	('rifcq1', nb.float64),
	('rifcq2', nb.float64),
	('rifcq3', nb.float64),
	('rifdf', nb.float64[:]),
	('rifvf', nb.float64[:]),
	('roorc', nb.float64),
	('roorn', nb.float64),
	('roorp', nb.float64),
	('rophyt', nb.float64),
	('rotorc', nb.float64),
	('rotorn', nb.float64),
	('rotorp', nb.float64),
	('rototn', nb.float64),
	('rototp', nb.float64),
	('rozoo', nb.float64),
	('SDLTFG', nb.int32),
	('seed', nb.float64),
	('simlen', nb.int32),
	('snkphy', nb.float64),
	('snkorc', nb.float64),
	('snkorn', nb.float64),
	('snkorp', nb.float64),
	('svol', nb.float64),
	('talgrh', nb.float64),
	('talgrl', nb.float64),
	('talgrm', nb.float64),
	('tbenal', nb.float64[:]),
	('tcbalg', nb.float64[:]),
	('tcbalr', nb.float64[:]),
	('tcgraz', nb.float64),
	('tczfil', nb.float64),
	('tczres', nb.float64),
	('tn', nb.float64),
	('torc', nb.float64),
	('torn', nb.float64),
	('torp', nb.float64),
	('totbod', nb.float64),
	('totdox', nb.float64),
	('totno3', nb.float64),
	('totorc', nb.float64),
	('totorn', nb.float64),
	('totorp', nb.float64),
	('totphy', nb.float64),
	('totpo4', nb.float64),
	('tottam', nb.float64),
	('tottba', nb.float64),
	('totzoo', nb.float64),
	('tp', nb.float64),
	('uunits', nb.int32),
	('vol', nb.float64),
	('zd', nb.float64),
	('zexdel', nb.float64),
	('zfil20', nb.float64),
	('ZFOOD', nb.int32),
	('zoco2', nb.float64),
	('zomass', nb.float64),
	('zoo', nb.float64),
	('zoobod', nb.float64),
	('zoodox', nb.float64),
	('ZOOFG', nb.int32),
	('zoono3', nb.float64),
	('zooorc', nb.float64),
	('zooorn', nb.float64),
	('zooorp', nb.float64),
	('zoophy', nb.float64),
	('zoopo4', nb.float64),
	('zootam', nb.float64),
	('zres20', nb.float64),
]

@jitclass(spec)
class PLANK_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, nexits, vol, ui_rq, ui, ts, OXRX, NUTRX):

		''' Initialize instance variables for lower food web simulation '''

		self.errors = zeros(int(ui['errlen']), dtype=np.int64)
		
		'''
		self.limit = array(8, type=nb.char) #np.chararray(8, itemsize=4)
		self.limit[1] = 'LIT'
		self.limit[2] = 'NON'
		self.limit[3] = 'TEM'
		self.limit[4] = 'NIT'
		self.limit[5] = 'PO4'
		self.limit[6] = 'NONE'
		self.limit[7] = 'WAT'
		'''

		self.delt = siminfo['delt']
		delt60 = siminfo['delt'] / 60.0  # delt60 - simulation time interval in hours
		self.delt60 = delt60
		self.simlen = int(siminfo['steps'])
		self.delts  = siminfo['delt'] * 60
		self.uunits = int(siminfo['units'])

		self.nexits = int(nexits)

		self.vol = vol
		self.svol = self.vol

		# inflow/outflow conversion factor:
		if self.uunits == 2:		# SI conversion: (g/m3)*(m3/ivld) --> [kg/ivld]
			self.conv = 1.0e-3
		else:						# Eng. conversion: (g/m3)*(ft3/ivld) --> [lb/ivld]
			self.conv = 6.2428e-5

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

		self.HTFG   = int(ui_rq['HTFG'])
		self.PHFG   = int(ui_rq['PHFG'])

		self.bpcntc = NUTRX.bpcntc
		self.cvbo  = NUTRX.cvbo
		self.cvbpc  = NUTRX.cvbpc
		self.cvbpn  = NUTRX.cvbpn

		if self.ZOOFG == 1 and self.PHYFG == 0:
			self.errors[0] += 1 
			# ERRMSG: error - zooplankton cannot be simulated without phytoplankton
		if self.NSFG == 1 and NUTRX.TAMFG == 0:
			self.errors[1] += 1 
			# ERRMSG: error - ammonia cannot be included in n supply if it is not
		if NUTRX.PO4FG == 0:
			self.errors[2] += 1 
			# ERRMSG: error - phosphate must be simulated if plankton are being

		self.numbal = 0
		self.BFIXFG = zeros(5, dtype=np.int32)

		if self.BALFG == 2:   # user has selected multiple species with more complex kinetics
			# additional benthic algae flags - table-type BENAL-FLAG
			self.numbal  = int(ui['NUMBAL'])
			self.BINVFG  = int(ui['BINVFG'])

			for i in range(1,5):
				self.BFIXFG[i] = int(ui['BFIXFG' + str(i)])
		else:
			self.numbal = self.BALFG          # single species or none
			
		self.cfsaex = 1.0
		if self.HTFG > 0 and 'CFSAEX' in ui_rq:		# via heat-parm input table
			self.cfsaex = ui_rq['CFSAEX']		
		elif 'CFSAEX' in ui:     				# fraction of surface exposed - table-type surf-exposed
			self.cfsaex = ui['CFSAEX']

		# table-type plnk-parm1
		self.ratclp = ui['RATCLP']
		self.nonref = ui['NONREF']
		self.litsed = ui['LITSED']
		self.alnpr  = ui['ALNPR']
		self.extb   = ui['EXTB']
		self.malgr  = ui['MALGR'] * self.delt60
		self.paradf = ui['PARADF']
		self.refr = 1.0 - self.nonref       # define fraction of biomass which is refractory material	

		# compute derived conversion factors
		self.cvbc   = self.bpcntc / 100.0
		self.cvnrbo = self.nonref * self.cvbo

		self.cvbp = (31.0 * self.bpcntc) / (1200.0 * self.cvbpc)
		self.cvbn = 14.0 * self.cvbpn * self.cvbp / 31.0
		self.cvpb   = 31.0 / (1000.0 * self.cvbp)
		self.cvbcl  = 31.0 * self.ratclp / self.cvpb

		# table-type plnk-parm2
		self.cmmlt  = ui['CMMLT']
		self.cmmn   = ui['CMMN']
		self.cmmnp  = ui['CMMNP']
		self.cmmp   = ui['CMMP']

		self.talgrh = ui['TALGRH']
		self.talgrl = ui['TALGRL']
		self.talgrm = ui['TALGRM']

		if self.uunits == 1:
			self.talgrh = (self.talgrh - 32.0) * 0.555		
			self.talgrl = (self.talgrl - 32.0) * 0.555		
			self.talgrm = (self.talgrm - 32.0) * 0.555		

		# table-type plnk-parm3
		self.alr20 = ui['ALR20'] * delt60   	# convert rates from 1/hr to 1/ivl
		self.aldh  = ui['ALDH']  * delt60 
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
			self.mbal   = ui['MBAL']   / self.cvpb   # convert maximum benthic algae to micromoles of phosphorus
			self.cfbalr = ui['CFBALR']
			self.cfbalg = ui['CFBALG']
			self.minbal = ui['MINBAL'] / self.cvpb   # convert maximum benthic algae to micromoles of phosphorus
			self.campr  = ui['CAMPR']
			self.fravl  = ui['FRAVL']
			self.nmaxfx = ui['NMAXFX']
		
			self.mbalgr = zeros(self.numbal)
			self.tcbalg = zeros(self.numbal)
			self.cmmnb  = zeros(self.numbal)
			self.cmmpb  = zeros(self.numbal)
			self.cmmd1  = zeros(self.numbal)
			self.cmmd2  = zeros(self.numbal)
			self.cslit  = zeros(self.numbal)
			
			self.balr20 = zeros(self.numbal)
			self.tcbalr = zeros(self.numbal)
			self.cslof1 = zeros(self.numbal)
			self.cslof2 = zeros(self.numbal)
			self.grores = zeros(self.numbal)		

			if self.BALFG == 2:  # user has selected multiple species with more complex kinetics				
				for i in range(self.numbal):
					# species-specific growth parms - table type benal-grow
					self.mbalgr[i] = ui['MBALGR'] * self.delt60
					self.tcbalg[i] = ui['TCBALG']
					self.cmmnb[i] =  ui['CMMNB']
					self.cmmpb[i] =  ui['CMMPB']
					self.cmmd1[i] =  ui['CMMD1']
					self.cmmd2[i] =  ui['CMMD2']
					self.cslit[i] =  ui['CSLIT']
					# species-specific resp and scour parms - table type benal-resscr
					self.balr20[i] = ui['BALR20'] * self.delt60
					self.tcbalr[i] = ui['TCBALR']
					self.cslof1[i] = ui['CSLOF1'] * self.delt60
					self.cslof2[i] = ui['CSLOF2']
					self.grores[i] = ui['GRORES']

				#  grazing and disturbance parms - table-type benal-graze
				self.cremvl = ui['CREMVL']
				self.cmmbi  = ui['CMMBI']
				self.tcgraz = ui['TCGRAZ']

				hrpyr = 8760.0		#constant
				self.cremvl = (self.cremvl / self.cvpb) / hrpyr * self.delt60

				if self.SDLTFG == 2:	# turbidity regression parms - table-type benal-light
					self.ctrbq1 = ui['CTRBQ1']
					self.ctrbq2 = ui['CTRBQ2']
					self.cktrb1 = ui['CKTRB1']
					self.cktrb2 = ui['CKTRB2']	

			# table-type benal-riff1
			self.frrif  = ui['FRRIF']	
			self.cmmv   = ui['CMMV']
			self.rifcq1 = ui['RIFCQ1']
			self.rifcq2 = ui['RIFCQ2']
			self.rifcq3 = ui['RIFCQ3']

			# table-type benal-riff2
			self.rifvf = zeros(5);	self.rifdf = zeros(5)

			for i in range(1,5):
				self.rifvf[i] = ui['RIFVF' + str(i)]
				self.rifdf[i] = ui['RIFDF' + str(i)]

		# table-type plnk-init
		self.phyto = ui['PHYTO']
		self.zoo   = ui['ZOO']
		#benal = ui['BENAL']
		self.orn   = ui['ORN']
		self.orp   = ui['ORP']
		self.orc   = ui['ORC']

		# variable initialization:
		if self.PHYFG == 0:   # initialize fluxes of inactive constituent
			self.rophyt = 0.0
			self.ophyt[:] = 0.0	#nexits
			self.phydox = self.phybod = 0.0
			self.phytam = 0.0; self.phyno3 = 0.0; self.phypo4 = 0.0
			self.phyorn = 0.0; self.phyorp = 0.0; self.phyorc = 0.0
			self.pyco2  = 0.0
			self.dthphy = 0.0; self.grophy = 0.0; self.totphy = 0.0

		self.rozoo = 0.0
		self.ozoo = zeros(nexits)

		if self.ZOOFG == 1:   # convert zoo to mg/l
			self.zoo *= self.zomass

		else:   #  zooplankton not simulated, but use default values
			# initialize fluxes of inactive constituent
			self.rozoo = 0.0
			self.ozoo[:] = 0.0	#nexits
			self.zoodox = 0.0; self.zoobod = 0.0
			self.zootam = 0.0; self.zoono3 = 0.0; self.zoopo4 = 0.0
			self.zooorn = 0.0; self.zooorp = 0.0; self.zooorc = 0.0
			self.zoophy = 0.0
			self.zoco2  = 0.0
			self.grozoo = 0.0; self.dthzoo = 0.0; self.totzoo = 0.0

		# benthic algae initialization:
		self.benal = zeros(self.numbal)
		self.flxbal = zeros((4,5))

		if self.numbal == 1:      # single species
			self.benal[0] = ui['BENAL']       # points to  table-type plnk-init above for rvals
		elif self.numbal >= 2:      # multiple species - table-type benal-init
			for n in range(self.numbal):
				self.benal[n] = ui['BENAL' + str(n+1)]
		else:                     # no benthic algae simulated
			self.baldox = 0.0; self.balbod = 0.0
			self.baltam = 0.0; self.balno3 = 0.0; self.balpo4 = 0.0
			self.balorn = 0.0; self.balorp = 0.0; self.balorc = 0.0
			self.baco2 = 0.0

		# compute derived quantities
		self.phycla = self.phyto * self.cvbcl

		self.balcla = zeros(self.numbal)
		for i in range(self.numbal):
			self.balcla[i] = self.benal[i] * self.cvbcl

		self.lsnh4 = zeros(4)
		self.lspo4 = zeros(4)

		if self.vol > 0.0:   # compute initial summary concentrations
			for i in range(1, 4):
				self.lsnh4[i] = NUTRX.rsnh4[i] / self.vol
				self.lspo4[i] = NUTRX.rspo4[i] / self.vol

		# calculate summary concentrations:
		(self.torn, self.torp, self.torc, self.potbod, self.tn, self.tp) \
			= self.pksums(NUTRX,self.phyto,self.zoo,self.orn,self.orp,self.orc,
						NUTRX.no3,NUTRX.tam,NUTRX.no2,self.lsnh4,NUTRX.po4,self.lspo4,OXRX.bod)

		return

	def simulate(self, tw, phval, co2, tss, OXRX, NUTRX, iphyto, izoo, iorn, iorp, iorc, 
					wash, solrad, avdepe, avvele, depcor, ro, binv,
				 	pladep_orn, pladep_orp, pladep_orc, advData):
		
		''' '''

		# hydraulics:
		(nexits, vols, vol, srovol, erovol, sovol, eovol) = advData
		self.vol = vol

		# initialize temp. vars for DO/BOD and NUTRX states:
		po4 = NUTRX.po4
		no3 = NUTRX.no3
		no2 = NUTRX.no2
		tam = NUTRX.tam
		
		dox = OXRX.dox
		bod = OXRX.bod		

		self.co2 = co2

		# inflows: convert from [mass/ivld] to [conc.*vol/ivld]
		self.iphyto = iphyto / self.conv
		self.izoo = izoo / self.conv
		self.iorn = iorn / self.conv
		self.iorp = iorp / self.conv
		self.iorc = iorc / self.conv

		#-----------------------------------------------------------
		#	advection:
		#-----------------------------------------------------------
		if self.PHYFG == 1:

			# advect phytoplankton
			(self.phyto, self.rophyt, self.ophyt) \
				= self.advplk(self.iphyto,self.svol,srovol,self.vol,erovol,sovol,eovol,nexits,
								self.oref,self.mxstay,self.seed,self.delts,self.phyto)

			(self.phyto, snkphy) = sink(self.vol,avdepe,self.physet,self.phyto)
			self.snkphy = -snkphy

			if self.ZOOFG == 1:    # zooplankton on; advect zooplankton
				(self.zoo, self.rozoo, self.ozoo) \
					= self.advplk(self.izoo,self.svol,srovol,vol,erovol,sovol,eovol,nexits,
									self.oref,self.mxstay,self.seed,self.delts,self.zoo)

		# advect organic nitrogen
		inorn = self.iorn + pladep_orn
		(self.orn,self.roorn,self.oorn) = advect (inorn, self.orn, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)
		(self.orn, snkorn) = sink(self.vol, avdepe, self.refset, self.orn)
		self.snkorn = -snkorn

		# advect organic phosphorus
		inorp = self.iorp + pladep_orp
		(self.orp, self.roorp, self.oorp) = advect(inorp, self.orp, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)
		(self.orp, snkorp) = sink(self.vol, avdepe, self.refset, self.orp)
		self.snkorp = -snkorp

		# advect total organic carbon
		inorc = self.iorc + pladep_orc
		(self.orc, self.roorc, self.oorc) = advect (inorc, self.orc, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)
		(self.orc, snkorc) = sink(self.vol, avdepe, self.refset, self.orc)
		self.snkorc = -snkorc

		if avdepe > 0.17:   # enough water to warrant computation of water quality reactions
			
			if self.BALFG > 0:
				if self.frrif < 1.0:                                          
					# make adjustments to average water velocity and depth for the  
					# portion of the reach that consists of riffle areas.
					if ro < self.rifcq1:    # below first cutoff flow
						i= 1
					elif ro < self.rifcq2:   # below second cutoff flow
						i= 2
					elif ro < self.rifcq3:   # below third cutoff flow
						i= 3
					else:                        # above third cutoff flow
						i= 4

					# calculate the adjusted velocity and depth for riffle sections
					self.balvel = self.rifvf[i] * avvele
					self.baldep = self.rifdf[i] * avdepe
				else:                      # use full depth and velocity
					self.balvel = avvele
					self.baldep = avdepe
			else:
				self.balvel = 0.0
				self.baldep = 0.0

			# calculate solar radiation absorbed; solrad is the solar radiation at gage,
			# corrected for location of reach; 0.97 accounts for surface reflection
			# (assumed 3 per cent); cfsaex is the ratio of radiation incident to water
			# surface to gage radiation values (accounts for regional differences, shading
			# of water surface, etc); inlit is a measure of light intensity immediately below
			# surface of reach/res and is expressed as ly/min, adjusted for fraction that is
			# photosynthetically active.

			inlit = 0.97 * self.cfsaex * solrad / self.delt * self.paradf
			extsed = 0.0

			if self.SDLTFG == 1:		 # estimate contribution of sediment to light extinction
				extsed = self.litsed * tss
			elif self.SDLTFG == 2:   # equations from dssamt for estimating the extinction coefficient based on discharge and turbidity
				# estimate turbidity based on linear regression on flow
				turb = self.ctrbq1 * ro**self.ctrbq2
		
				# estimate the portion of the extinction coefficient due to
				# sediment based upon a system-wide regression of total
				# extinction to turbidity, and then subtracting the
				# base extinction coefficient
				extsed = (self.cktrb1 * turb**self.cktrb2) - self.extb                         
				if extsed < 0.0:         # no effective sediment shading
					extsed = 0.0
			else:                        # sediment light extinction not considered
				extsed = 0.0

			# calculate contribution of phytoplankton to light extinction (self-shading)
			extcla = 0.00452 * self.phyto * self.cvbcl

			# calculate light available for algal growth,  litrch only called here
			(self.phylit, self.ballit, self.cflit) = self.litrch (inlit,self.extb,extcla,extsed,avdepe,self.baldep,self.PHYFG,self.BALFG)

			#-----------------------------------------------------------
			#	sestonic algae growth & respiration
			#-----------------------------------------------------------
			if self.PHYFG == 1:   # simulate phytoplankton, phyrx only called here
				(po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.phyto,self.limphy,self.pyco2,self.phycla,
				dophy,bodphy,tamphy,no3phy,po4phy,phdth,phgro,ornphy,orpphy,orcphy) \
					= self.phyrx(self.phylit,tw,self.talgrl,self.talgrh,self.talgrm,self.malgr,self.cmmp, \
							self.cmmnp,NUTRX.TAMFG,self.AMRFG,self.NSFG,self.cmmn,self.cmmlt,self.delt60, \
							self.cflit,self.alr20,self.cvbpn,self.PHFG,self.DECFG,self.cvbpc,self.paldh, \
							self.naldh,self.claldh,self.aldl,self.aldh,NUTRX.anaer,self.oxald,self.alnpr, \
							self.cvbo,self.refr,self.cvnrbo,self.cvpb,self.cvbcl,self.co2, \
							self.nmingr,self.pmingr,self.cmingr,self.lmingr,self.nminc, \
							po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.phyto)

				# compute associated fluxes
				self.phydox = dophy *  self.vol
				self.phybod = bodphy * self.vol
				self.phytam = tamphy * self.vol
				self.phyno3 = no3phy * self.vol
				self.phypo4 = po4phy * self.vol
				self.dthphy = -phdth * self.vol
				self.grophy = phgro  * self.vol
				self.phyorn = ornphy * self.vol
				self.phyorp = orpphy * self.vol
				self.phyorc = orcphy * self.vol

				#-----------------------------------------------------------
				#	zooplankton growth & death:
				#-----------------------------------------------------------
				if self.ZOOFG == 1:    # simulate zooplankton, zorx only called here
					(dox,bod,self.zoo,self.orn,self.orp,self.orc,tam,no3,po4,zeat,self.zoco2,dozoo,bodzoo,nitzoo,po4zoo,zgro,zdth,zorn,zorp,zorc) \
						= self.zorx(self.zfil20,self.tczfil,tw,self.phyto,self.mzoeat,self.zexdel,self.cvpb, \
							self.zres20,self.tczres,NUTRX.anaer,self.zomass,NUTRX.TAMFG,self.refr, \
							self.ZFOOD,self.zd,self.oxzd,self.cvbn,self.cvbp,self.cvbc,self.cvnrbo,self.cvbo, \
							dox,bod,self.zoo,self.orn,self.orp,self.orc,tam,no3,po4)

					# compute associated fluxes
					self.zoodox = -dozoo * self.vol
					self.zoobod = bodzoo * self.vol

					if NUTRX.TAMFG != 0:   # ammonia on, so nitrogen excretion goes to ammonia
						self.zootam = nitzoo * self.vol
						self.zoono3 = 0.0
					else:             # ammonia off, so nitrogen excretion goes to nitrate
						self.zoono3 = nitzoo * self.vol
						self.zootam = 0.0
					
					self.zoopo4 = po4zoo * self.vol
					self.zoophy = -zeat  * self.vol
					self.zooorn = zorn   * self.vol
					self.zooorp = zorp   * self.vol
					self.zooorc = zorc   * self.vol
					self.grozoo = zgro   * self.vol
					self.dthzoo = -zdth  * self.vol
					self.totzoo = self.grozoo + self.dthzoo

					# update phytoplankton state variable to account for zooplankton predation
					self.phyto = self.phyto - zeat

					# convert phytoplankton expressed as mg biomass/l to chlorophyll a expressed as ug/l
					self.phycla = self.phyto * self.cvbcl
				
				self.totphy = self.snkphy + self.zoophy + self.dthphy + self.grophy

			#-----------------------------------------------------------
			#	benthic algae growth & respiration
			#-----------------------------------------------------------
			self.limbal = zeros(self.numbal, dtype=np.int32)

			if self.BALFG > 0:
				bgro = zeros(self.numbal)
				bdth = zeros(self.numbal)

				if self.BALFG == 1:     # simulate benthic algae
					(po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.benal[0],self.limbal[0],self.baco2,self.balcla[0],
						dobalg,bodbal,tambal,no3bal,po4bal,bgro[0],bdth[0],ornbal,orpbal,orcbal) \
						= self.balrx(self.ballit,tw,self.talgrl,self.talgrh,self.talgrm,self.malgr,self.cmmp, \
							self.cmmnp,NUTRX.TAMFG,self.AMRFG,self.NSFG,self.cmmn,self.cmmlt,self.delt60, \
							self.cflit,self.alr20,self.cvbpn,self.PHFG,self.DECFG,self.cvbpc,self.paldh, \
							self.naldh,self.aldl,self.aldh,NUTRX.anaer,self.oxald,self.cfbalg,self.cfbalr, \
							self.alnpr,self.cvbo,self.refr,self.cvnrbo,self.cvpb,self.mbal,depcor, \
							self.cvbcl,self.co2,self.nmingr,self.pmingr,self.cmingr,self.lmingr,self.nminc, \
							po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.benal[0])

				elif self.BALFG == 2:   # simulate enhanced benthic algae equations from dssamt (!)
					# then perform reactions, balrx2 only called here
					(po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.benal,self.limbal,self.baco2,self.balcla,
						dobalg,bodbal,tambal,no3bal,po4bal,bgro,bdth,ornbal,orpbal,orcbal) \
						= self.balrx2 (self.ballit,tw,NUTRX.TAMFG,self.NSFG,self.delt60,self.cvbpn,self.PHFG,self.DECFG, \
								self.cvbpc,self.alnpr,self.cvbo,self.refr,self.cvnrbo,self.cvpb,depcor, \
								self.cvbcl,co2,self.numbal,self.mbalgr,self.cmmpb,self.cmmnb, \
								self.balr20,self.tcbalg,self.balvel,self.cmmv,self.BFIXFG,self.cslit,self.cmmd1, \
								self.cmmd2,self.tcbalr,self.frrif,self.cremvl,self.cmmbi,binv,self.tcgraz, \
								self.cslof1,self.cslof2,self.minbal,self.fravl,self.BNPFG,self.campr,self.nmingr, \
								self.pmingr,self.cmingr,self.lmingr,self.nminc,self.nmaxfx,self.grores, \
								po4,no3,tam,dox,self.orn,self.orp,self.orc,bod,self.benal)

				#compute associated fluxes
				self.baldox = dobalg * self.vol
				self.balbod = bodbal * self.vol
				self.baltam = tambal * self.vol
				self.balno3 = no3bal * self.vol
				self.balpo4 = po4bal * self.vol
				self.balorn = ornbal * self.vol
				self.balorp = orpbal * self.vol
				self.balorc = orcbal * self.vol

				self.grobal = zeros(self.numbal)
				self.dthbal = zeros(self.numbal)

				for i in range(self.numbal):
					self.grobal[i] = bgro[i]
					self.dthbal[i] = -bdth[i]

		else:     # not enough water in reach/res to warrant simulation of quality processes
			self.phyorn = 0.0
			self.balorn = 0.0
			self.zooorn = 0.0
			self.phyorp = 0.0
			self.balorp = 0.0
			self.zooorp = 0.0
			self.phyorc = 0.0
			self.balorc = 0.0
			self.zooorc = 0.0
			self.pyco2  = 0.0
			self.baco2  = 0.0
			self.zoco2  = 0.0
			self.phydox = 0.0
			self.zoodox = 0.0
			self.baldox = 0.0
			self.phybod = 0.0
			self.zoobod = 0.0
			self.balbod = 0.0
			self.phytam = 0.0
			self.zootam = 0.0
			self.baltam = 0.0
			self.phyno3 = 0.0
			self.zoono3 = 0.0
			self.balno3 = 0.0
			self.phypo4 = 0.0
			self.zoopo4 = 0.0
			self.balpo4 = 0.0

			if self.PHYFG == 1:  # water scarcity limits phytoplankton growth
				self.limphy  = 1  #'WAT'
				self.phycla = self.phyto * self.cvbcl
				self.grophy = 0.0
				self.dthphy = 0.0
				self.zoophy = 0.0
				self.totphy = self.snkphy

			if self.BALFG > 0:   # water scarcity limits benthic algae growth
				limc = 1  #'WAT'
				self.limbal = zeros(self.numbal, dtype=np.int32)
				self.balcla = zeros(self.numbal)
				self.grobal = zeros(self.numbal)
				self.dthbal = zeros(self.numbal)

				for i in range(self.numbal):
					self.balcla[i] = self.benal[i] * self.cvbcl
					self.limbal[i] = limc

			if self.ZOOFG == 1:    # water scarcity limits zooplankton growth
				self.grozoo= 0.0
				self.dthzoo= 0.0
				self.totzoo= 0.0

		#-----------------------------------------------------------
		#	store final benthic sums and fluxes
		#-----------------------------------------------------------
		if self.BALFG > 0:
			self.tbenal = zeros(3)
			self.grotba = 0.0
			self.dthtba = 0.0
			
			for i in range(self.numbal):
				self.flxbal[1,i] = self.grobal[i]
				self.flxbal[2,i] = self.dthbal[i]
				self.flxbal[3,i] = self.grobal[i] + self.dthbal[i]
				self.tbenal[1]  += self.benal[i]
				self.grotba     += self.grobal[i]
				self.dthtba     += self.dthbal[i]
			
			self.tbenal[2] = self.tbenal[1] * self.cvbcl
			self.tottba    = self.grotba + self.dthtba

		#-----------------------------------------------------------
		# compute final process fluxes for oxygen, nutrients and organics
		#-----------------------------------------------------------
		self.totdox = OXRX.readox + OXRX.boddox + OXRX.bendox + NUTRX.nitdox + self.phydox + self.zoodox + self.baldox
		self.totbod = OXRX.decbod + OXRX.bnrbod + OXRX.snkbod + NUTRX.denbod + self.phybod + self.zoobod + self.balbod
		self.totno3 = NUTRX.nitno3 + NUTRX.denno3 + NUTRX.bodno3 + self.phyno3 + self.zoono3 + self.balno3
		self.tottam = NUTRX.nittam + NUTRX.volnh3 + NUTRX.bnrtam + NUTRX.bodtam + self.phytam + self.zootam + self.baltam
		self.totpo4 = NUTRX.bnrpo4 + NUTRX.bodpo4 + self.phypo4 + self.zoopo4 + self.balpo4

		self.totorn = self.snkorn + self.phyorn + self.zooorn + self.balorn
		self.totorp = self.snkorp + self.phyorp + self.zooorp + self.balorp
		self.totorc = self.snkorc + self.phyorc + self.zooorc + self.balorc

		#-----------------------------------------------------------
		# compute summaries of total organics, total n and p, and potbod
		#-----------------------------------------------------------

		# concentrations:
		if self.vol > 0.0:
			for i in range(1, 4):
				self.lsnh4[i] = NUTRX.rsnh4[i] / self.vol
				self.lspo4[i] = NUTRX.rspo4[i] / self.vol

			(self.torn, self.torp, self.torc, self.potbod, self.tn, self.tp) \
				= self.pksums(NUTRX,self.phyto,self.zoo,self.orn,self.orp,self.orc,
								no3,tam,no2,self.lsnh4,po4,self.lspo4,bod)
		else:
			self.torn   = -1.0e30
			self.torp   = -1.0e30
			self.torc   = -1.0e30
			self.potbod = -1.0e30
			self.tn     = -1.0e30
			self.tp     = -1.0e30

		(self.itorn, self.itorp, self.itorc, dumval, self.itotn, self.itotp) \
			= self.pksums(NUTRX,self.iphyto,self.izoo,self.iorn,self.iorp,self.iorc,
							NUTRX.ino3,NUTRX.itam,NUTRX.ino2,NUTRX.isnh4,NUTRX.ipo4,NUTRX.ispo4,OXRX.ibod)

		# total outflows:
		(self.rotorn, self.rotorp, self.rotorc, dumval, self.rototn, self.rototp) \
			= self.pksums(NUTRX,self.rophyt,self.rozoo,self.roorn,self.roorp,self.roorc,
							NUTRX.rono3,NUTRX.rotam,NUTRX.rono2,NUTRX.rosnh4,NUTRX.ropo4,NUTRX.rospo4,OXRX.robod)

		# outflows by exit:
		self.otorn = zeros(nexits); self.otorp = zeros(nexits); self.otorc = zeros(nexits)
		self.ototn = zeros(nexits); self.ototp = zeros(nexits)

		if nexits > 1:
			for i in range(nexits):
				(self.otorn[i], self.otorp[i], self.otorc[i], dumval, self.ototn[i], self.ototp[i]) \
					= self.pksums(NUTRX,self.ophyt[i],self.ozoo[i],self.oorn[i],self.oorp[i],self.oorc[i],
									NUTRX.ono3[i],NUTRX.otam[i],NUTRX.ono2[i],NUTRX.osnh4[i],NUTRX.opo4[i],NUTRX.ospo4[i],OXRX.obod[i])

		#-----------------------------------------------------------
		# update DO/BOD and nutrient states (for OXRX/NUTRX):
		#-----------------------------------------------------------
		OXRX.dox = dox
		OXRX.bod = bod

		NUTRX.po4 = po4
		NUTRX.no3 = no3
		NUTRX.no2 = no2
		NUTRX.tam = tam

		# redistribute TAM after algal influence:
		(NUTRX.nh3,NUTRX.nh4) = NUTRX.ammion(tw, phval, tam)

		self.svol = self.vol  # svol is volume at start of time step, update for next time thru

		return OXRX, NUTRX


	@staticmethod
	def advplk(iplank,vols,srovol,vol,erovol,sovol,eovol,nexits,oref,mxstay,seed,delts,plank):
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
			(plnkad, roplk, oplk) = advect(iplank,plnkad,nexits,vols,vol,srovol,erovol,sovol,eovol)

			# determine final concentration of plankton in reach/res after advection
			plank = plnkad + mstay / vol  if vol > 0.0 else plnkad
		else:       # no plankton leaves the reach/res
			roplk   = 0.0
			oplk = zeros(nexits)
			mstay = plank * vols
			plank = (mstay + iplank) / vol if vol > 0.0 else -1.0e30

		return plank, roplk, oplk

	@staticmethod
	def algro(light,po4,no3,tw,talgrl,talgrh,talgrm,malgr, cmmp,cmmnp,
				TAMFG,AMRFG,tam,NSFG,cmmn,cmmlt,alr20,cflit,delt60,
				nmingr,pmingr,lmingr):

		''' calculate unit growth and respiration rates for algae
		population; both are expressed in units of per interval'''
		lim = -999

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
						lim = 6  #'po4'
					elif gron < grol:
						gro = gron
						lim = 5  #'nit'
					else:
						gro = grol
						lim = 4  #'lit'
					if gro < 0.000001 * delt60:
						gro = 0.0

					if gro > 0.95 * malgrt:
						# there is no limiting factor to cause less than maximum growth rate
						lim = 7  #'none'

					# adjust growth rate if control volume is not entirely	
					# contained within the euphotic zone; e.g. if only one
					# half of the control volume is in the euphotic zone, gro
					# would be reduced to one half of its specified value
					gro = gro * cflit
				else:            # water temperature does not allow algal growth
					gro = 0.0
					lim = 3  #'tem'
			else:               # no algal growth occurs; necessary nutrients are not available
				gro = 0.0
				lim = 2  #'non'
		else:                    # no algal growth occurs; necessary light is not available
			gro = 0.0
			lim = 4  #'lit'

		# calculate unit algal respiration rate; res is expressed in
		# units of per interval; alr20 is the respiration rate at 20 degrees c
		res = alr20 * tw / 20.0
		return lim, gro, res


	@staticmethod
	def baldth(nsfg,no3,tam,po4,paldh,naldh,aldl,aldh,mbal,dox,anaer,oxald,bal,depcor):
		''' calculate benthic algae death'''

		slof = 0.0
		# determine whether to use high or low unit death rate; all
		# unit death rates are expressed in units of per interval

		# determine available inorganic nitrogen pool for test of nutrient scarcity
		nit = no3 + tam  if nsfg > 0 else no3
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
			# check for overcrowding
			balmax = mbal * depcor
			if  bal < balmax:
				slof = 0.0
			else:
				slof = bal - balmax
				bal = balmax

		if dox < anaer:    # conditions are anaerobic, augment unit death rate
			ald += oxald

		# use unit death rate to compute death rate; dthbal is expressed
		# as umoles of phosphorus per liter per interval
		return (ald * bal) + slof, bal     # dthbal


	def balrx(self, ballit,tw,talgrl,talgrh,talgrm,malgr,cmmp, cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60,
				cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh, naldh,aldl,aldh,anaer,oxald,cfbalg,cfbalr,
				alnpr,cvbo,refr,cvnrbo,cvpb,mbal,depcor,cvbcl,baco2,nmingr,pmingr,cmingr,lmingr,
				nminc, po4,no3,tam,dox,orn,orp,orc,bod,benal):
		''' simulate behavior of benthic algae in units of umoles p per
		liter; these units are used internally within balrx so that
		algal subroutines may be shared by phyto and balrx; externally,
		the benthic algae population is expressed in terms of areal
		mass, since the population is resident entirely on the
		bottom surface'''

		# convert benal to units of umoles phosphorus/l (bal) for internal calculations
		i0 = 0
		bal = (benal / cvpb) * depcor

		# compute unit growth and respiration rates for benthic algae; determine growth limiting factor
		(limbal,gro,res) \
			= self.algro(ballit,po4,no3,tw,talgrl,talgrh,talgrm,malgr,cmmp, cmmnp,tamfg,amrfg,
					tam,nsfg,cmmn,cmmlt,alr20, cflit,delt60,nmingr,pmingr,lmingr)

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
			grobal = self.grochk (po4,no3,tam,phfg,decfg,baco2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,i0,grtotn,grobal)

		# calculate benthic algae death, baldth only called here
		(dthbal, bal) = self.baldth(nsfg,no3,tam,po4,paldh,naldh,aldl,aldh,mbal,dox,anaer,oxald,bal,depcor)

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
		orn,orp,orc,bod = self.orgbal(balorn,balorp,balorc,bodbal,orn,orp,orc,bod)

		# perform materials balance resulting from uptake of nutrients by benthic algae
		po4,tam,no3,baco2,tambal,no3bal,po4bal = self.nutrup(grobal,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg,nminc,po4,tam,no3,baco2)
		baco2 = -baco2

		# convert bal back to external units; benal is expressed as
		# mg biomass/m2 and balcla is expressed as ug chlorophyll a/m2
		benal  = (bal * cvpb) / depcor
		balgro = (grobal * cvpb) / depcor
		bdth   = (dthbal * cvpb) / depcor
		balcla = benal * cvbcl

		return po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,balorn,balorp,balorc

	@staticmethod
	def grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,nfixfg,grtotn,grow):
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
		if nsfg == 0:    # tam is not considered as a possible nutrient
			uplimn = (no3 - nmingr) * 71.43 / cvbpn
		else:
			uplimn = (no3 + tam - nmingr) * 71.43 / cvbpn

		uplimc = 1.0e30
		if phfg > 0 and phfg != 2 and decfg == 0:
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
		if nfixfg:                       # n-fixation is not occurring for this algal type
			uplim = min(uplimp, uplimn, uplimc)
		else:                           # n-fixation is occurring, nitrogen does not limit growth
			uplim = min(uplimp, uplimc)
		if uplim < 0.0:   # reduce growth rate to limit
			grow += uplim
		return grow


	@staticmethod
	def litrch(inlit, extb, extcla, extsed, avdepe, baldep, PHYFG, BALFG):
		''' calculate light correction factor to algal growth (cflit); 
		determine amount of light available to phytoplankton and benthic algae'''
		ln01 = 4.60517   # minus natural log 0.01

		cflit  = 0.0
		phylit = 0.0
		ballit = 0.0

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
			pass

		return phylit, ballit, cflit

	@staticmethod
	def nutrup(grow, NSFG, cvbpn, alnpr, cvbpc, PHFG, DECFG, nminc, po4, tam, no3, alco2):
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

	@staticmethod
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

	@staticmethod
	def phydth(nsfg,no3,tam,po4,paldh,naldh,phycla,claldh,aldl,aldh,dox,anaer,oxald,stc):
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


	def phyrx(self,phylit,tw,talgrl,talgrh,talgrm,malgr,cmmp,cmmnp,tamfg,amrfg,nsfg,cmmn,cmmlt,delt60,cflit,alr20,cvbpn,phfg,decfg,cvbpc,paldh,
				naldh,claldh,aldl,aldh,anaer,oxald,alnpr,cvbo,refr,cvnrbo,cvpb,cvbcl,co2,nmingr,pmingr,cmingr,lmingr,nminc,
				po4,no3,tam,dox,orn,orp,orc,bod,phyto):
		''' simulate behavior of phytoplankton, as standing crop, in units of umoles p per liter'''

		# convert phyto to units of umoles phosphorus (stc) and ug chlorophyll a/l (phycla) for internal calculations
		stc    = phyto / cvpb
		phycla = phyto * cvbcl

		# compute unit growth and respiration rates for phytoplankton;
		# determine growth limiting factor
		(limphy,gro,res) \
			= self.algro(phylit,po4,no3,tw,talgrl,talgrh,talgrm,malgr,cmmp,cmmnp,tamfg,amrfg,
					tam,nsfg,cmmn,cmmlt,alr20,cflit,delt60,nmingr,pmingr,lmingr)

		# calculate net growth rate of phytoplankton; grophy is
		# expressed as umol phosphorus per liter per interval
		grophy = (gro - res) * stc

		if grophy > 0.0:
			# adjust growth rate to account for limitations imposed by
			# availability of required nutrients
			grtotn = grophy
			i0 = 0

			grophy = self.grochk (po4,no3,tam,phfg,decfg,co2,cvbpc,cvbpn,nsfg,nmingr,pmingr,cmingr,i0,grtotn,grophy)

		# calculate phytoplankton death
		dthphy = self.phydth(nsfg,no3,tam,po4,paldh,naldh,phycla,claldh,aldl,aldh,dox,anaer,oxald,stc)

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
		orn,orp,orc,bod = self.orgbal(phyorn,phyorp,phyorc,phybd, orn,orp,orc,bod)

		# perform materials balance resulting from uptake of nutrients by phytoplankton
		(po4,tam,no3,pyco2,tamphy,no3phy,po4phy) = self.nutrup(grophy,nsfg,cvbpn,alnpr,cvbpc,phfg,decfg,nminc,po4,tam,no3,co2)
		pyco2 = -pyco2

		# convert stc to units of mg biomass/l (phyto) and ug chlorophyll a/l (phycla)
		phyto  = stc    * cvpb
		phgro  = grophy * cvpb
		phdth  = dthphy * cvpb
		phycla = phyto  * cvbcl

		return po4,no3,tam,dox,orn,orp,orc,bod,phyto,limphy,pyco2,phycla,dophy,bodphy,tamphy,no3phy,po4phy,phdth,phgro,phyorn,phyorp,phyorc


	def zorx(self,zfil20,tczfil,tw,phyto,mzoeat,zexdel,cvpb,zres20,tczres,anaer,zomass,tamfg,refr,
				zfood,zd,oxzd,cvbn,cvbp,cvbc,cvnrbo,cvbo,dox,bod,zoo,orn,orp,orc,tam,no3,po4):
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
		i1 = 1
		tam, no3, po4 = decbal(tamfg, i1, znit, zpo4, tam, no3, po4)

		# calculate amount of refractory organic constituents which result from zooplankton death and excretion
		zorn = ((refr * zdth) + zrefex) * cvbn
		zorp = ((refr * zdth) + zrefex) * cvbp
		zorc = ((refr * zdth) + zrefex) * cvbc

		# calculate amount of nonrefractory organics (bod) which result from zooplankton death and excretion
		zbod = zbod + (zdth * cvnrbo) + (znrfex * cvbo)
		orn, orp, orc, bod = self.orgbal(zorn, zorp, zorc, zbod, orn, orp, orc, bod)

		return  dox,bod,zoo,orn,orp,orc,tam,no3,po4,zeat,zco2,dozoo,zbod,znit,zpo4,zogr,zdth,zorn,zorp,zorc

	def pksums(self, NUTRX, phyto, zoo, orn, orp, orc, no3, tam, no2, snh4, po4, spo4, bod):
		''' computes summaries of: total organic n, p, c; total n, p; potbod'''

		# undefined summary concentrations:
		torn   = 0.0
		torp   = 0.0
		torc   = 0.0
		potbod = 0.0
		tn     = 0.0
		tp     = 0.0

		if (self.vol <= 0):
			return torn, torp, torc, potbod, tn, tp

		# Calculate sums:
		tval = bod / self.cvbo

		if self.PHYFG == 1:
			tval += phyto
			if self.ZOOFG == 1:
				tval += zoo

		torn   = orn + self.cvbn * tval
		torp   = orp + self.cvbp * tval
		torc   = orc + self.cvbc * tval
		potbod = bod

		# total N
		tn     = torn + no3
		if NUTRX.TAMFG == 1:
			tn += tam
		if NUTRX.NO2FG == 1:
			tn += no2
		if NUTRX.ADNHFG == 1:
			for i in range(1, 4):
				tn += snh4[i]

		# total P
		tp = torp
		if NUTRX.PO4FG == 1:
			tp += po4
		if NUTRX.ADPOFG == 1:
			for i in range(1, 4):
				tp += spo4[i]

		# potential BOD		
		if self.PHYFG == 1:
			potbod += (self.cvnrbo * phyto)
			
			if self.ZOOFG == 1:
				potbod += (self.cvnrbo * zoo)

		return torn, torp, torc, potbod, tn, tp

	@staticmethod
	def algro2 (ballit,po4,no3,tw,mbalgr,cmmp,TAMFG,tam,NSFG,cmmn,balr20,delt60,tcbalg,balvel,
				cmmv,BFIXFG,cslit,cmmd1,cmmd2,sumba,tcbalr,nmingr,pmingr,lmingr,nmaxfx,grores):

		''' calculate unit growth and respiration rates for benthic algae using more
		complex kinetics; both are expressed in units of per interval'''

		groba = 0.0
		NFIXFG = 0
		lim = -999		

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
					NFIXFG = 0
					grofn = (mmn * grofv) / (cmmn + mmn * grofv)
				else:
					# n-fixing blue-green algae; determine if available nitrogen 
					# concentrations are high enough to suppress nitrogen fixation
					if mmn >= nmaxfx:
						NFIXFG = 0
						grofn = (mmn * grofv) / (cmmn + mmn * grofv)
					else:
						NFIXFG = 1   # available nitrogen concentrations do not suppress n-fixation
						grofn = 1.0  # nitrogen does not limit growth rate

					# calculate the maximum light limited unit growth rate
					grofl = (ballit / cslit) * exp(1.0 - (ballit / cslit))

					# calculate density limitation on growth rate
					grofd = (cmmd1 * sumba + cmmd2) / (sumba + cmmd2)
					if grofp < grofn and grofp < grofl:  # phosphorus limited
						gromin = grofp
						lim = 5  #'po4'
					elif grofn < grofl:                  # nitrogen limited
						gromin = grofn
						lim = 4  #'nit'
					else:                                # light limited
						gromin = grofl
						lim = 1  #'lit'
					if gromin > 0.95:  # there is no limiting factor to cause less than maximum growth rate
						lim = 6  #'none'
					# calculate overall growth rate in units of per interval
					groba = mbalgr * tcmbag * gromin * grofd
					if groba < 1.0e-06 * delt60:
						groba = 0.0
			else:    # no algal growth occurs; necessary nutrients are not available
				groba = 0.0
				lim   = 2  #'non'
		else:        # no algal growth occurs; necessary light is not available
			groba = 0.0
			lim = 1  #'lit'

		# calculate unit algal respiration rate in units of per interval
		# balr20 is the benthic algal respiration rate at 20 degrees c
		resba = balr20 * tcbalr**(tw - 20.0) + grores * groba
		return  NFIXFG, lim, groba, resba

	@staticmethod
	def balrem(crem,sumba,sumbal,cmmbi,tcgraz,tw,binv,bal,cslof1,cslof2,balvel):
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

	@staticmethod
	def nutup2 (grow,NSFG,cvbpn,alnpr,cvbpc,PHFG,DECFG,BNPFG,campr,sumgrn,nminc,po4,tam,no3):
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
				no3lim= (prct99/nmgpum)*no3
				
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


	def balrx2(self,ballit,tw,TAMFG,NSFG,delt60,cvbpn,PHFG,DECFG,cvbpc,alnpr,cvbo,refr,cvnrbo,cvpb,
		depcor,cvbcl,co2,numbal,mbalgr,cmmpb,cmmnb,balr20,tcbalg,balvel,cmmv,BFIXFG,
		cslit,cmmd1,cmmd2,tcbalr,frrif,cremvl,cmmbi,binv,tcgraz,cslof1,cslof2,minbal,
		fravl,BNPFG,campr,nmingr,pmingr,cmingr,lmingr,nminc,nmaxfx,grores,
		po4,no3,tam,dox,orn,orp,orc,bod,benal):

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

		bal = zeros(numbal)
		grobal = zeros(numbal)
		dthbal = zeros(numbal)
		limbal = zeros(numbal, dtype=np.int32)

		for i in range(numbal):    # do 20 i = 1, numbal
			# convert benal to units of umoles phosphorus/l (bal) for
			# internal calculations, and adjust for % riffle to which
			# periphyton are limited
			bal[i] = (benal[i] / cvpb) * depcor * frrif

			# compute unit growth and respiration rates
			(NFIXFG,limbal[i],groba,resba) \
				= self.algro2 (ballit,po4,no3,tw,mbalgr[i],cmmpb[i],TAMFG,tam,NSFG,cmmnb[i],balr20[i],
								delt60,tcbalg[i],balvel,cmmv,BFIXFG[i],cslit[i],cmmd1[i],cmmd2[i],sumba,tcbalr[i],
								nmingr,pmingr,lmingr,nmaxfx,grores[i])

			# calculate net growth rate of algae; grobal is expressed as
			# umoles phosphorus per liter per interval; benthic algae growth
			# will be expressed in terms of volume rather than area for the
			# duration of the subroutines subordinate to balrx; the output
			# values for benthic algae are converted to either mg biomass per
			# sq meter or mg chla per sq meter, whichever the user specifies
			grobal[i] = (groba - resba) * bal[i]

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
				grotot = self.grochk (po4,no3,tam,PHFG,DECFG,co2,cvbpc,cvbpn,NSFG,nmingr,pmingr,cmingr,NFIXFG,grtotn,grotot)
				if grotot < 0.0:            # this should never happen
					grotot = 0.0

				# compare nutrient-checked growth to original cumulative total,
				if grotot < grotmp:
					# adjust growth rate of last algal type
					grobal[i] = grobal[i] - (grotmp - grotot)
					# track changes in growth that affect available n concentrations
					if NFIXFG != 1:     # n-fixation not occurring, all growth affects n
						groban = grobal[i]
					else:  # n-fixation is occurring, proportionately adjust respiration (groban)
						groban = groban * grotot / grotmp

			# calculate benthic algae removal
			crem = cremvl * depcor * frrif
			dthbal[i] = self.balrem (crem,sumba,sumbal,cmmbi,tcgraz,tw,binv,bal[i],cslof1[i],cslof2[i],balvel)

			# add the net growth
			bal[i] += grobal[i]
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
			bal[i] -= dthbal[i]
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
		(orn,orp,orc,bod) = self.orgbal (balorn,balorp,balorc,bodbal,orn,orp,orc,bod)

		# perform materials balance resulting from uptake of nutrients by benthic algae
		(po4,tam,no3,baco2,tambal,no3bal,po4bal) \
			= self.nutup2 (sumgro,NSFG,cvbpn,alnpr,cvbpc,PHFG,DECFG,BNPFG,
						campr,sumgrn,nminc,po4,tam,no3)
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
		balgro = zeros(numbal)
		bdth = zeros(numbal)
		balcla = zeros(numbal)

		for i in range(numbal):
			benal[i]  = (bal[i] * cvpb) / (depcor * frrif)
			balgro[i] = (grobal[i] *cvpb) / (depcor * frrif)
			bdth[i]   =   (dthbal[i] *cvpb) / (depcor * frrif)
			balcla[i] =  benal[i] * cvbcl

		return  po4,no3,tam,dox,orn,orp,orc,bod,benal,limbal,baco2,balcla,dobalg,bodbal,tambal,no3bal,po4bal,balgro,bdth,balorn,balorp,balorc