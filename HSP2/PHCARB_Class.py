import numpy as np
from numpy import zeros, array
import numba as nb
from numba.typed import Dict
from numba.experimental import jitclass
from math import exp, log

from HSP2.ADCALC import advect
from HSP2.RQUTIL import benth

spec = [
	('AFACT', nb.float64),
	('alkcon', nb.int32),
	('anaer', nb.int32),
	('BENRFG', nb.int32),
	('brco2', nb.float64[:]),
	('cfcinv', nb.float64),
	('co2', nb.float64),
	('conv', nb.float64),
	('delt60', nb.float64),
	('delth', nb.float64),
	('delts', nb.float64),
	('errors', nb.int64[:]),
	('ico2', nb.float64),
	('itic', nb.float64),
	('ncons', nb.int32),
	('nexits', nb.int32),
	('oco2', nb.float64[:]),
	('otic', nb.float64[:]),
	('ph', nb.float64),
	('phcnt', nb.int32),
	('roco2', nb.float64),
	('rotic', nb.float64),
	('uunits', nb.int32),
	('svol', nb.float64),
	('tic', nb.float64),
	('vol', nb.float64),
]

@jitclass(spec)
class PHCARB_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, nexits, vol, ui_rq, ui, ts):

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
		self.BENRFG = int(ui_rq['BENRFG'])	# via table-type benth-flag
		self.anaer = int(ui_rq['ANAER'])

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

	def simulate(self, phif1, phif2, avdepe, tw, dox, advectData):

		''' simulate ph, carbon dioxide, total inorganic carbon, and alkalinity'''

		# hydraulics:
		(nexits, vols, vol, srovol, erovol, sovol, eovol) = advectData

		self.vol = vol

		# inflows: convert from [mass/ivld] to [conc.*vol/ivld]
		self.itic = phif1 / self.conv
		self.ico2 = phif2 / self.conv

		# advect TIC:
		(self.tic, self.rotic, self.otic) = \
			advect(self.itic, self.bod, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

		# advect CO2:
		(self.co2, self.roco2, self.oco2) = \
			advect(self.ico2, self.co2, nexits, self.svol, self.vol, srovol, erovol, sovol, eovol)

		if self.CONSFG == 0:
			pass
		else:
			# con(alkcon) is available from section cons
			pass


		if self.vol > 0.0:
			pass

		self.svol = self.vol  # svol is volume at start of time step, update for next time thru

		return

	#-------------------------------------------------------------------
	# utility functions:
	#-------------------------------------------------------------------