from numpy import zeros, array
from numba import njit, int32, float32, float64, char
from numba.experimental import jitclass
from math import exp

from HSP2.ADCALC import advect, oxrea
from HSP2.RQUTIL import sink
from HSP2.utilities	 import make_numba_dict

spec = [
	('AFACT', float32),
	('benod', float32),
	('benox', float32),
	('BENRFG', int32),
	('bod', float32),
	('bodbnr', float32),
	('bodox', float32),
	('BRBOD', float64[:]),
	('cforea', float32),
	('cfpres', float32),
	('decbod', float32),
	('delt60', float32),
	('delth', float32),
	('delts', float32),
	('doben', float32),
	('dorea', float32),
	('dox', float32),
	('errors', int32[:]),
	('expod', float32),
	('expred', float32),
	('exprel', float32),
	('kbod20', float32),
	('kodset', float32),
	('korea', float32),
	('len_', float32),
	('LKFG', int32),
	('nexits', int32),
	('obod', float64[:]),
	('odox', float64[:]),
	('rbod', float32),
	('rdox', float32),
	('readox', float32),
	('reak', float32),
	('reakt', float32),
	('REAMFG', int32),
	('rdox', float32),
	('rbod', float32),
	('relbod', float32),
	('robod', float32),
	('rodox', float32),
	('satdo', float32),
	('simlen', int32),
	('snkbod', float32),
	('supsat', float32),
	('svol', float32),
	('tcben', float32),
	('tcbod', float32),
	('tcginv', float32),
	('totdox', float32),
	('totbod', float32),
	('uunits', int32),
	('vol', float32),
]

#@jitclass(spec)
class OXRX_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, advectData, ui_rq, ui, ts):

		''' Initialize variables for primary DO, BOD balances '''

		#self.ERRMSGS = array('Placeholder')
		#self.errors = zeros(len(self.ERRMSGS), dtype=int32)

		(nexits, vol, VOL, SROVOL, EROVOL, SOVOL, EOVOL) = advectData
		
		delt60 = siminfo['delt'] / 60.0  # delt60 - simulation time interval in hours
		self.delt60 = delt60
		self.simlen = int(siminfo['steps'])
		self.delts  = siminfo['delt'] * 60
		self.uunits = int(siminfo['units'])

		self.nexits = int(nexits)

		self.AFACT = 43560.0
		if self.uunits == 2:
			# si units conversion
			self.AFACT = 1000000.0

		self.vol = vol * self.AFACT
		self.svol = self.vol

		# gqual flags
		self.GQFG = int(ui_rq['GQFG'])
		self.GQALFG4 = int(ui_rq['GQALFG4'])

		# table-type ox-genparm
		self.kbod20 = ui['KBOD20'] * delt60	 # convert units from 1/hr to 1/ivl
		self.tcbod	= ui['TCBOD']
		self.kodset = ui['KODSET'] * delt60	 # convert units from 1/hr to 1/ivl
		self.supsat = ui['SUPSAT']
		
		# table-type ox-init
		self.dox   = ui['DOX']
		self.bod   = ui['BOD']
		self.satdo = ui['SATDO']
		
		# other required values
		self.BENRFG = int(ui_rq['BENRFG'])	# via table-type benth-flag
		self.REAMFG = int(ui['REAMFG'])	 # table-type ox-flags
		elev = ui['ELEV']	 # table-type elev
		
		self.cfpres = ((288.0 - 0.001981 * elev) / 288.0)**5.256  # pressure correction factor -
		ui['CFPRES'] = self.cfpres

		self.LKFG = int(ui_rq['LKFG'])
		if self.LKFG == 1:
			self.cforea = ui['CFOREA']	 # reaeration parameter from table-type ox-cforea
		
		elif self.REAMFG == 1:			 # tsivoglou method;  table-type ox-tsivoglou
			self.reakt	= ui['REAKT']
			self.tcginv = ui['TCGINV']
			
			self.len_	= ui['LEN'] * 5280.0  # mi to feet
			self.delth	= ui['DELTH']
			if self.uunits == 2:
				self.len_ = ui['LEN'] * 1000.0  # length of reach, in meters
				self.delth = ui['DELTH'] * 1000.0  # convert to meters

		elif self.REAMFG == 2:			# owen/churchill/o'connor-dobbins; table-type ox-tcginv
			self.tcginv = ui['TCGINV']
			self.reak	= ui['REAK']
			self.expred = ui['EXPRED']
		
		elif self.REAMFG == 3:			# user formula - table-type ox-reaparm
			self.tcginv = ui['TCGINV']
			self.reak	= ui['REAK']
			self.expred = ui['EXPRED']
			
		if self.BENRFG == 1:		  # benthic release parms - table-type ox-benparm
			self.benod	= ui['BENOD'] * self.delt60	# convert units from 1/hr to 1/ivl
			self.tcben	= ui['TCBEN']
			self.expod	= ui['EXPOD']
			self.exprel = ui['EXPREL']

			self.BRBOD = zeros(2)
			self.BRBOD[0] = ui['BRBOD1'] * self.delt60		# convert units from 1/hr to 1/ivl
			self.BRBOD[1] = ui['BRBOD2'] * self.delt60		# convert units from 1/hr to 1/ivl

			#self.BRBOD	= array([ui['BRBOD1'] , ui['BRBOD2']])	* self.delt60  # convert units from 1/hr to 1/ivl

		self.snkbod = 0.0

		self.rdox = self.dox * self.vol
		self.rbod = self.bod * self.vol

		self.odox = zeros(nexits)
		self.obod = zeros(nexits)

		self.korea = 0.0

	#-------------------------------------------------------------------
	# simulation (single timestep):
	#-------------------------------------------------------------------

	def simulate(self, idox, ibod, wind, scrfac, avdepe, avvele, depcor, tw, advectData):

		# hydraulics:
		(nexits, vol_, vol, srovol, erovol, sovol, eovol) = advectData

		self.vol = vol * self.AFACT

		# advect dissolved oxygen
		(self.dox, self.rodox, self.odox) = \
			advect(idox, self.dox, nexits, self.svol, vol, srovol, erovol, sovol, eovol)

		# advect bod
		(self.bod, self.robod, self.obod) = \
			advect(ibod, self.bod, nexits, self.svol, vol, srovol, erovol, sovol, eovol)

		self.svol = vol

		# initialize variables:
		self.bodox	= 0.0			
		self.readox = 0.0
		self.boddox = 0.0
		self.bendox = 0.0
		self.decbod = 0.0

		if avdepe > 0.17:	# benthal influences are considered
			# sink bod
			self.bod, self.snkbod = sink(self.vol, avdepe, self.kodset, self.bod)
			self.snkbod = -self.snkbod

			if self.BENRFG == 1:
				#$OXBEN	  # simulate benthal oxygen demand and benthal release of bod, and compute associated fluxes
				# calculate amount of dissolved oxygen required to satisfy benthal oygen demand (mg/m2.ivl)
				self.benox = self.benod * (self.tcben**(tw -20.0)) * (1.0 -exp(-self.expod * self.dox))

				# adjust dissolved oxygen state variable to acount for oxygen lost to benthos, and compute concentration flux
				self.doben = self.dox
				self.dox   = self.dox - (self.benox * depcor)
				self.doben = self.benox * depcor  
				if self.dox >= 0.001:
					self.doben = self.benox * depcor
				else:
					self.dox = 0.0

				# calculate benthal release of bod; release is a function of dissolved oxygen
				# (dox) and a step function of stream velocity; brbod(1) is the aerobic benthal 
				# release rate; brbod(2) is the base increment to benthal release under 
				# decreasing do concentration; relbod is expressed as mg bod/m2.ivl
				self.relbod = (self.BRBOD[0] + self.BRBOD[1] * exp(-self.exprel * self.dox)) * scrfac

				# add release to bod state variable and compute concentration flux
				self.bod	= self.bod + self.relbod * depcor
				self.bodbnr = self.relbod * depcor
						
				# end #$OXBEN
				self.bendox = -self.doben * vol
				self.bnrbod = self.bodbnr * vol

		elif self.LKFG == 1:
			# calculate oxygen reaeration
			if not (self.GQFG == 1 and self.GQALFG4 == 1):
				self.korea = oxrea(
					self.LKFG, wind,self.cforea,avvele,avdepe,self.tcginv, 
					self.REAMFG,self.reak,self.reakt,self.expred,self.exprev,self.len_,
					self.delth,tw,self.delts,self.self.delt60,self.uunits, self.korea)

			# calculate oxygen saturation level for current water
			# temperature; satdo is expressed as mg oxygen per liter
			self.satdo = 14.652 + tw * (-0.41022 + tw * (0.007991 - 0.7777e-4 * tw))

			# adjust satdo to conform to prevalent atmospheric pressure
			# conditions; cfpres is the ratio of site pressure to sea level pressure
			self.satdo = self.cfpres * self.satdo
			if self.satdo < 0.0:
				# warning - this occurs only when water temperature is very high - above
				# about 66 c.  usually means error in input gatmp (or tw if htrch is not being simulated).		
				self.satdo = 0.0   # reset saturation level

			# compute dissolved oxygen value after reaeration,and the reaeration flux
			self.dorea  = self.korea * (self.satdo - self.dox)
			self.dox	= self.dox + dorea
			self.readox = dorea * vol

			#$BODDEC
			'''calculate concentration of oxygen required to satisfy computed bod decay'''
			self.bodox = (self.kbod20 * (self.tcbod**(tw -20.0))) * self.bod   # bodox is expressed as mg oxygen/liter.ivl
			if self.bodox > self.bod:
				self.bodox = self.bod

			# adjust dissolved oxygen state variable to acount for oxygen lost to bod decay, and compute concentration flux
			if self.bodox >= self.dox:
				self.bodox = self.dox
				self.dox   = 0.0
			else:
				self.dox = self.dox - self.bodox

			# adjust bod state variable to account for bod decayed
			self.bod -= self.bodox
			if self.bod < 0.0001:
				self.bod = 0.0
			# end #$BODDEC
			
			self.boddox = -self.bodox * vol
			self.decbod = -self.bodox * vol

		else:	 # there is too little water to warrant simulation of quality processes
			self.bodox	= 0.0
			
			self.readox = 0.0
			self.boddox = 0.0
			self.bendox = 0.0
			self.decbod = 0.0
			self.bnrbod = 0.0
			self.snkbod = 0.0

		self.totdox = self.readox + self.boddox + self.bendox
		self.totbod = self.decbod + self.bnrbod + self.snkbod

		self.rdox = self.dox * vol
		self.rbod = self.bod * vol

		return

	#-------------------------------------------------------------------
	# simulation (single timestep):
	#-------------------------------------------------------------------
	def adjust_dox(vol, nitdox, phydox, zoodox, baldox):
		# if dox exceeds user specified level of supersaturation, then release excess do to the atmosphere

		doxs = self.dox

		if self.dox > self.supsat * self.satdo:
			self.dox = self.supsat * self.satdo
		self.readox = self.readox + (self.dox - doxs) * vol
		self.totdox = self.readox + self.boddox + self.bendox \
					+ nitdox + phydox + zoodox + baldox
		
		# update dissolved totals and totals of nutrients
		self.rdox = self.dox * vol
		self.rbod = self.bod * vol


	#-------------------------------------------------------------------
	# mass links:
	#-------------------------------------------------------------------
	@staticmethod
	def expand_OXRX_masslinks(flags, uci, dat, recs):
		if flags['OXRX']:
				for i in range(1, 2):
					rec = {}
					rec['MFACTOR'] = dat.MFACTOR
					rec['SGRPN'] = 'OXRX'
					if dat.SGRPN == "ROFLOW":
						rec['SMEMN'] = 'OXCF1'
						rec['SMEMSB1'] = str(i)
						rec['SMEMSB2'] = '1'
					else:
						rec['SMEMN'] = 'OXCF2'
						rec['SMEMSB1'] = dat.SMEMSB1  # first sub is exit number
						rec['SMEMSB2'] = str(i)
					
					rec['TMEMN'] = 'OXIF'
					rec['TMEMSB1'] = str(i)
					rec['TMEMSB2'] = '1'
					rec['SVOL'] = dat.SVOL
					recs.append(rec)