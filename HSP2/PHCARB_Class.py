import numpy as np
from numpy import zeros, array
import numba as nb
from numba.typed import Dict
from numba.experimental import jitclass
from math import log10

from HSP2.ADCALC import advect
from HSP2.RQUTIL import benth

spec = [
    ('alk', nb.float64),
	('alkcon', nb.int32),
	('anaer', nb.float64),
	('benco2', nb.float64),
	('benrfg', nb.int32),
	('brco2', nb.float64[:]),
	('cfcinv', nb.float64),
	('co2', nb.float64),
	('conv', nb.float64),
	('delt60', nb.float64),
	('delth', nb.float64),
	('delts', nb.float64),
	('errors', nb.int64[:]),
	('ico2', nb.float64),
	('invco2', nb.float64),
	('itic', nb.float64),
	('ncons', nb.int32),
	('nexits', nb.int32),
	('oco2', nb.float64[:]),
	('otic', nb.float64[:]),
	('ph', nb.float64),
	('phcnt', nb.int32),
	('roco2', nb.float64),
	('rotic', nb.float64),
	('satco2', nb.float64),
	('simlen', nb.int32),
	('svol', nb.float64),
	('tic', nb.float64),
	('totco2', nb.float64),
	('uunits', nb.int32),
	('vol', nb.float64),
]

@jitclass(spec)
class PHCARB_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, nexits, vol, ui_rq, ui_nutrx, ui, ts):

		''' Initialize variables for pH, carbon dioxide, total inorganic carbon,  and alkalinity '''

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

		# required values from other modules:
		self.benrfg = int(ui_rq['BENRFG'])	# via table-type benth-flag
		self.anaer = int(ui_nutrx['ANAER'])
		self.satco2 = -999.0

		# flags - table-type ph-parm1
		self.phcnt  = int(ui['PHCNT'])    # is the maximum number of iterations to pH solution.
		self.alkcon = int(ui['ALKCON'])   # ALKCON  is the number of the conservative substance which is alkalinity.

		ncons = int(ui_rq['NCONS'])
		if self.alkcon > ncons:
			self.errors[0] += 1
			# ERRMSG: Invalid CONS index specified for ALKCON (i.e., ALKCON > NCONS).

		# flags - table-type ph-parm2
		self.cfcinv = ui['CFCINV']

		self.brco2 = zeros(2)
		for i in range(2):
			self.brco2[i] = ui['BRCO2' + str(i+1)] * self.delt60

		# table-type ph-init
		self.tic = ui['TIC']
		self.co2 = ui['CO2']
		self.ph   = ui['PH']

		# initialize outflows:
		self.roco2 = 0.0
		self.rotic = 0.0

		self.oco2 = zeros(nexits)
		self.otic = zeros(nexits)

		return

	#-------------------------------------------------------------------
	# simulation (single timestep):
	#-------------------------------------------------------------------

	def simulate(self, tw, OXRX, NUTRX, PLANK, phif1, phif2, alk, avdepe, scrfac, depcor, advectData):

		''' simulate ph, carbon dioxide, total inorganic carbon, and alkalinity'''

		# hydraulics:
		(nexits, vols, vol, srovol, erovol, sovol, eovol) = advectData

		self.vol = vol

		# inflows: convert from [mass/ivld] to [conc.*vol/ivld]
		self.itic = phif1 / self.conv
		self.ico2 = phif2 / self.conv

		# advect TIC:
		(self.tic, self.rotic, self.otic) = \
			advect(self.itic, self.tic, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

		# advect CO2:
		(self.co2, self.roco2, self.oco2) = \
			advect(self.ico2, self.co2, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

		if vol > 0.0:
			twkelv = tw + 273.16

			# convert tic, co2, and alk to molar concentrations for duration of phcarb section
			self.tic = self.tic / 12000.0
			self.co2 = self.co2 / 12000.0
			self.alk = alk / 50000.0

			if avdepe > 0.17:
				if self.benrfg == 1:  # simulate benthal release of co2
					self.co2 *= 12000.0  # convert co2 to mg/l for use by benth
					self.co2, self.benco2 = benth(OXRX.dox, self.anaer, self.brco2, scrfac, depcor, self.co2)
					self.co2 /= 12000.0
				else:  # benthal release of co2 is not considered
					self.benco2 = 0.0

				# calculate molar saturation concentration for co2 (satco2); first, calculate
				# henry's constant, s, for co2; s is defined as the molar concentration of
				# atmospheric co2/partial pressure of co2; cfpres corrects the equation for
				# effects of elevation differences from sea level
				s = 10.0 ** ((2385.73 / twkelv) - 14.0184 + 0.0152642 * twkelv)
				self.satco2 = 3.16e-04 * OXRX.cfpres * s

				# calculate increase in co2 due to atmospheric invasion;  the co2
				# invasion is based on oxygen reaeration rate for the control volume
				kcinv = min(self.cfcinv * OXRX.korea, 0.999)
				inv = kcinv * (self.satco2 - self.co2)
				self.invco2 = inv * 12000.0

				# calculate net molar co2 change due to co2 invasion, zooplankton
				# excretion and respiration, phytoplankton and benthic algae
				# respiration, bod decay, and benthal release of co2
				bodco2 = NUTRX.decco2
				phyco2 = PLANK.pyco2
				zooco2 = PLANK.zoco2
				balco2 = PLANK.baco2
				self.totco2 = self.invco2 + zooco2 + phyco2 + balco2 + bodco2 + self.benco2
				deltcd = self.totco2 / 12000.0

				# calculate change in total inorganic carbon balance due to net co2 change
				self.tic = max(self.tic + deltcd, 0.0)

			else:
				# too little water to warrant simulation of quality processes; calculate
				# values of co2 and ph state variables based on only longitudinal advection
				self.invco2 = 0.0
				zooco2 = 0.0
				phyco2 = 0.0
				balco2 = 0.0
				bodco2 = 0.0
				self.benco2 = 0.0
				self.totco2 = 0.0  # invco2 + zooco2 + phyco2 + balco2 + bodco2 + benco2

			# calculate ionization product of water
			kwequ = 10.0 ** (-4470.99 / twkelv + 6.0875 - 0.01706 * twkelv)

			# calculate first dissociation constant of carbonic acid
			k1equ = 10.0 ** (-3404.71 / twkelv + 14.8435 - 0.032786 * twkelv)

			# calculate second dissociation constant of carbonic acid
			k2equ = 10.0 ** (-2902.39 / twkelv + 6.4980 - 0.02379 * twkelv)

			# assign values to variables and coefficients used in the solution algorithm
			if self.ph < 0.0:  # it is undefined (due to no water in reach)
				self.ph = 7.0
			hest = 10.0 ** (-self.ph)
			hllim = 0.0
			hulim = 1.0
			coeff1 = self.alk + k1equ
			coeff2 = -kwequ + self.alk * k1equ + k1equ * k2equ - self.tic * k1equ
			coeff3 = -2.0 * k1equ * k2equ * self.tic - k1equ * kwequ + self.alk * k1equ * k2equ
			coeff4 = -k1equ * k2equ * kwequ

			# $PHCALC()     ''' calculate ph'''
			count = 0
			while count <= self.phcnt:
				count = count + 1
				# evaluate quadratic and slope for solution equation
				quadh = (((hest + coeff1) * hest + coeff2) * hest + coeff3) * hest + coeff4
				dfdh = ((4.0 * hest + 3.0 * coeff1) * hest + 2.0 * coeff2) * hest + coeff3
				if dfdh <= 0.0:  # slope of solution equation is zero or negative
					# solution for hplus is not meaningful for such a slope
					# update values for hllim, hulim, and hest to force convergence
					if quadh < 0.0:
						if hest >= hllim:
							hllim = hest
							hest = 10.0 * hest
						elif hest <= hulim:
							hulim = hest
							hest = 0.10 * hest
				else:  # calculate new hydrogen ion concentration
					hplus = hest - quadh / dfdh
					if abs(hplus - hest) / hplus <= 0.10:
						break
					# adjust prior	estimate for next iteration
					if hplus <= hllim:
						hest = (hest + hllim) / 2.0
					elif hplus >= hulim:
						hest = (hest + hulim) / 2.0
					else:
						hest = hplus
			else:
				self.errors[1] += 1  # ERRMSG:  a satisfactory solution for ph has not been reached

			self.ph = -log10(hplus)
			# end #$PHCALC()

			# calculate co2 concentration (molar)
			self.co2 = self.tic / (1.0 + k1equ / hplus + k1equ * k2equ / (hplus ** 2))

			# convert tic, co2, and alk from moles/liter to mg/liter
			self.tic *= 12000.0
			self.co2 *= 12000.0
			self.alk *= 50000.0
		else:  # reach/res has gone dry during the interval; set ph equal to an undefined value
			self.ph = -1.0e30
			self.invco2 = 0.0
			zooco2 = 0.0
			phyco2 = 0.0
			balco2 = 0.0
			bodco2 = 0.0
			self.benco2 = 0.0
			self.totco2 = 0.0  # invco2 + zooco2 + phyco2 + balco2 + bodco2 + benco2

		self.svol = self.vol  # svol is volume at start of time step, update for next time thru

		return

