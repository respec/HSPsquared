import numpy as np
from numpy import zeros, array
import numba as nb
from numba.typed import Dict
from numba.experimental import jitclass
from math import exp

from HSP2.ADCALC import advect, oxrea
from HSP2.RQUTIL import sink
from HSP2.utilities	 import make_numba_dict

spec = [
	('AFACT', nb.float64),
	('benod', nb.float64),
	('bendox', nb.float64),
	('benox', nb.float64),
	('BENRFG', nb.int32),
	('bnrbod', nb.float64),
	('bod', nb.float64),
	('bodbnr', nb.float64),
	('boddox', nb.float64),
	('bodox', nb.float64),
	('BRBOD', nb.float64[:]),
	('cforea', nb.float64),
	('cfpres', nb.float64),
	('conv', nb.float64),
	('decbod', nb.float64),
	('delt60', nb.float64),
	('delth', nb.float64),
	('delts', nb.float64),
	('doben', nb.float64),
	('dorea', nb.float64),
	('dox', nb.float64),
	('errors', nb.int64[:]),
	('expod', nb.float64),
	('expred', nb.float64),
	('exprel', nb.float64),
	('exprev', nb.float64),
	('GQFG', nb.int32),
	('GQALFG4', nb.int32),
	('idox', nb.float64),
	('ibod', nb.float64),
	('kbod20', nb.float64),
	('kodset', nb.float64),
	('korea', nb.float64),
	('len_', nb.float64),
	('LKFG', nb.int32),
	('nexits', nb.int32),
	('obod', nb.float64[:]),
	('odox', nb.float64[:]),
	('rbod', nb.float64),
	('rdox', nb.float64),
	('readox', nb.float64),
	('reak', nb.float64),
	('reakt', nb.float64),
	('REAMFG', nb.int32),
	('rdox', nb.float64),
	('rbod', nb.float64),
	('relbod', nb.float64),
	('robod', nb.float64),
	('rodox', nb.float64),
	('satdo', nb.float64),
	('simlen', nb.int32),
	('snkbod', nb.float64),
	('supsat', nb.float64),
	('svol', nb.float64),
	('tcben', nb.float64),
	('tcbod', nb.float64),
	('tcginv', nb.float64),
	('totdox', nb.float64),
	('totbod', nb.float64),
	('uunits', nb.int32),
	('vol', nb.float64)
]

@jitclass(spec)
class OXRX_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, nexits, vol, ui_rq, ui, ts):

		''' Initialize variables for primary DO, BOD balances '''

		self.errors = zeros(int(ui['errlen']), dtype=np.int64)

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

		self.cforea = 0.0
		self.delth = 0.0
		self.reak = 0.0; self.reakt = 1.0
		self.expred = 0.0; self.exprev = 0.0
		self.expod = 0.0; self.exprel = 0.0

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
			self.exprev = ui['EXPREV']
			
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

		return

	#-------------------------------------------------------------------
	# simulation (single timestep):
	#-------------------------------------------------------------------

	def simulate(self, oxif1, oxif2, wind, scrfac, avdepe, avvele, depcor, tw, advectData):

		# hydraulics:
		(nexits, vols, vol, srovol, erovol, sovol, eovol) = advectData

		self.vol = vol

		# inflows: convert from [mass/ivld] to [conc.*vol/ivld]
		self.idox = oxif1 / self.conv
		self.ibod = oxif2 / self.conv

		# advect dissolved oxygen
		(self.dox, self.rodox, self.odox) = \
			advect(self.idox, self.dox, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

		# advect bod
		(self.bod, self.robod, self.obod) = \
			advect(self.ibod, self.bod, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

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
				self.bendox = -self.doben * self.vol
				self.bnrbod = self.bodbnr * self.vol

			if self.LKFG != 1:
				wind = 0.0

			# calculate oxygen reaeration
			self.korea = oxrea(
					self.LKFG,wind,self.cforea,avvele,avdepe,self.tcginv,
					self.REAMFG,self.reak,self.reakt,self.expred,self.exprev,self.len_,
					self.delth,tw,self.delts,self.delt60,self.uunits)

			# calculate oxygen saturation level for current water
			# temperature; satdo is expressed as mg oxygen per liter
			self.satdo = 14.652 + tw * (-0.41022 + tw * (0.007991 - 0.7777e-4 * tw))

			# adjust satdo to conform to prevalent atmospheric pressure
			# conditions; cfpres is the ratio of site pressure to sea level pressure
			self.satdo = self.cfpres * self.satdo

			if self.satdo < 0.0:
				self.errors[0] += 1
				# warning - this occurs only when water temperature is very high - above
				# about 66 c.  usually means error in input gatmp (or tw if htrch is not being simulated).		
				self.satdo = 0.0   # reset saturation level

			# compute dissolved oxygen value after reaeration,and the reaeration flux
			dorea  = self.korea * (self.satdo - self.dox)
			self.dox	= self.dox + dorea
			self.readox = dorea * self.vol

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
			
			self.boddox = -self.bodox * self.vol
			self.decbod = -self.bodox * self.vol

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

		self.rdox = self.dox * self.vol
		self.rbod = self.bod * self.vol

		self.svol = self.vol  # svol is volume at start of time step, update for next time thru

		return