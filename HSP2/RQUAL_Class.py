import numpy as np
from numpy import where, zeros, array
from math import log
import numba as nb
from numba.experimental import jitclass

from HSP2.OXRX_Class import OXRX_Class
from HSP2.NUTRX_Class import NUTRX_Class
from HSP2.PLANK_Class import PLANK_Class
#from HSP2.PHCARB_Class import PHCARB_Class
from HSP2.utilities  import make_numba_dict, initm

spec = [
	('OXRX', OXRX_Class.class_type.instance_type),
	('NUTRX', NUTRX_Class.class_type.instance_type),
	('PLANK', PLANK_Class.class_type.instance_type),
	('AFACT', nb.float64),
	('ALK', nb.float64[:]),
	('AVDEP', nb.float64[:]),
	('AVDEPE', nb.float64[:]),
	('AVVEL', nb.float64[:]),
	('AVVELE', nb.float64[:]),
	('BALCLA', nb.float64[:]),
	('BENAL1', nb.float64[:]),
	('BENRFG', nb.int32),
	('BOD', nb.float64[:]),
	('CO2', nb.float64[:]),
	('delt60', nb.float64),
	('delts', nb.float64),
	('DEPCOR', nb.float64[:]),
	('DOX', nb.float64[:]),
	('EOVOL', nb.float64[:,:]),
	('EROVOL', nb.float64[:]),
	('IBOD', nb.float64[:]),
	('ICO2', nb.float64[:]),
	('IDOX', nb.float64[:]),
	('INH4', nb.float64[:]),
	('INO2', nb.float64[:]),
	('INO3', nb.float64[:]),
	('IORC', nb.float64[:]),
	('IORN', nb.float64[:]),
	('IORP', nb.float64[:]),
	('IPHYT', nb.float64[:]),
	('IPO4', nb.float64[:]),
	('ISNH41', nb.float64[:]),
	('ISNH42', nb.float64[:]),
	('ISNH43', nb.float64[:]),
	('ISPO41', nb.float64[:]),
	('ISPO42', nb.float64[:]),
	('ISPO43', nb.float64[:]),
	('ITIC', nb.float64[:]),
	('IZOO', nb.float64[:]),
	('LKFG', nb.int32),
	('nexits', nb.int32),
	('NH3', nb.float64[:]),
	('NH4', nb.float64[:]),
	('NO2', nb.float64[:]),
	('NO3', nb.float64[:]),
	('NUADCN', nb.float64[:]),
	('NUADFG', nb.int32[:]),
	('NUADFX', nb.float64[:]),
	('NUTFG', nb.int32),
	('OBALCLA', nb.float64[:,:]),
	('OBENAL', nb.float64[:,:]),
	('OBOD', nb.float64[:,:]),
	('OCO2', nb.float64[:,:]),
	('ODOX', nb.float64[:,:]),
	('ONO2', nb.float64[:,:]),
	('ONO3', nb.float64[:,:]),
	('OPHYCLA', nb.float64[:,:]),
	('OPHYT', nb.float64[:,:]),
	('OPO4', nb.float64[:,:]),
	('OORC', nb.float64[:,:]),
	('OORN', nb.float64[:,:]),
	('OORP', nb.float64[:,:]),
	('ORC', nb.float64[:]),
	('ORN', nb.float64[:]),
	('ORP', nb.float64[:]),
	('OTAM', nb.float64[:,:]),
	('OTIC', nb.float64[:,:]),
	('OZOO', nb.float64[:,:]),
	('PH', nb.float64[:]),
	('PHFG', nb.int32),
	('PHYCLA', nb.float64[:]),
	('PHYTO', nb.float64[:]),
	('PKIF1', nb.float64[:]),
	('PKIF2', nb.float64[:]),
	('PKIF3', nb.float64[:]),
	('PKIF4', nb.float64[:]),
	('PKIF5', nb.float64[:]),
	('PLKFG', nb.int32),
	('PO4', nb.float64[:]),
	('POTBOD', nb.float64[:]),
	('PREC', nb.float64[:]),
	('RNH3', nb.float64[:]),
	('RNH4', nb.float64[:]),
	('RNO2', nb.float64[:]),
	('RNO3', nb.float64[:]),
	('RO', nb.float64[:]),
	('ROBALCLA', nb.float64[:]),
	('ROBENAL', nb.float64[:]),
	('ROBOD', nb.float64[:]),
	('ROCO2', nb.float64[:]),
	('RODOX', nb.float64[:]),
	('RONO2', nb.float64[:]),
	('RONO3', nb.float64[:]),
	('ROORC', nb.float64[:]),
	('ROORN', nb.float64[:]),
	('ROORP', nb.float64[:]),
	('ROPHYCLA', nb.float64[:]),
	('ROPHYT', nb.float64[:]),
	('ROPO4', nb.float64[:]),
	('ROSNH41', nb.float64[:]),
	('ROSNH42', nb.float64[:]),
	('ROSNH43', nb.float64[:]),
	('ROSPO41', nb.float64[:]),
	('ROSPO42', nb.float64[:]),
	('ROSPO43', nb.float64[:]),
	('ROTAM', nb.float64[:]),
	('ROTIC', nb.float64[:]),
	('ROTORC', nb.float64[:]),
	('ROTORN', nb.float64[:]),
	('ROTORP', nb.float64[:]),
	('ROZOO', nb.float64[:]),
	('RPO4', nb.float64[:]),
	('RTAM', nb.float64[:]),
	('SAREA', nb.float64[:]),
	('SATDO', nb.float64[:]),
	('SCRFAC', nb.float64[:]),
	('SEDFG', nb.int32),
	('simlen', nb.int32),
	('SOLRAD', nb.float64[:]),
	('SOVOL', nb.float64[:,:]),
	('SSED4', nb.float64[:]),
	('SROVOL', nb.float64[:]),
	('svol', nb.float64),
	('TAM', nb.float64[:]),
	('TBENAL1', nb.float64[:]),
	('TBENAL2', nb.float64[:]),
	('TIC', nb.float64[:]),
	('TN', nb.float64[:]),
	('TORC', nb.float64[:]),
	('TORN', nb.float64[:]),
	('TORP', nb.float64[:]),
	('TOTCO2', nb.float64[:]),
	('TP', nb.float64[:]),
	('TW', nb.float64[:]),
	('uunits', nb.int32),
	('vol', nb.float64),
	('VOL', nb.float64[:]),
	('WASH', nb.float64[:]),
	('WIND', nb.float64[:]),
	('ZOO', nb.float64[:])
]

@jitclass(spec)
class RQUAL_Class:

	#-------------------------------------------------------------------
	# class initialization:
	#-------------------------------------------------------------------
	def __init__(self, siminfo, ui, ui_oxrx, ui_nutrx, ui_plank, ui_phcarb, ts):
	
		''' Initialize instance variables for rqual wrapper '''

		# simulation data:
		delt60 = siminfo['delt'] / 60.0  # delt60 - simulation time interval in hours
		simlen = int(siminfo['steps'])

		self.delt60 = delt60
		self.simlen = simlen
		self.delts  = siminfo['delt'] * 60
		self.uunits = int(siminfo['units'])

		# hydaulic results:
		#(nexits, vol, VOL, self.SROVOL, self.EROVOL, self.SOVOL, self.EOVOL) = advectData

		self.AFACT = 43560.0
		if self.uunits == 2:
			# si units conversion
			self.AFACT = 1000000.0

		nexits = int(ui['nexits'])
		self.nexits = nexits

		self.VOL = ts['VOL'] * self.AFACT
		self.SROVOL = ts['SROVOL']
		self.EROVOL = ts['EROVOL']
		self.SOVOL = zeros((self.simlen, nexits))
		self.EOVOL = zeros((simlen, nexits))
		for i in range(nexits):
			self.SOVOL[:, i] = ts['SOVOL' + str(i + 1)]
			self.EOVOL[:, i] = ts['EOVOL' + str(i + 1)]

		self.vol = ui['vol'] * self.AFACT
		self.svol = self.vol

		# initialize flags: 
		self.BENRFG = int(ui['BENRFG'])   # table-type benth-flag

		# table type ACTIVITY
		self.NUTFG = int(ui['NUTFG'])
		self.PLKFG = int(ui['PLKFG'])
		self.PHFG  = int(ui['PHFG'])
		
		self.SEDFG = int(ui['SEDFG'])
		self.LKFG = int(ui['LKFG'])

		#TO-DO! - remove hardcode once PHCARB class is ready for testing
		self.PHFG = 0

		# get external time series
		self.PREC  = ts['PREC']
		self.SAREA = ts['SAREA'] #dbg * self.AFACT

		self.RO = ts['RO']		
		self.AVDEP = ts['AVDEP']
		self.AVVEL = ts['AVVEL']
		self.TW    = ts['TW'] 
		
		if 'WIND' in ts:
			self.WIND  = ts['WIND'] * 1609.0 # miles/ivld to meters/ivld
		else:
			self.WIND = zeros(simlen)

		# initialize time series for physical variables:
		self.AVDEPE = zeros(simlen)
		self.AVVELE = zeros(simlen)
		self.DEPCOR = zeros(simlen)
		self.SCRFAC = zeros(simlen)

		for t in range(simlen):
			# fix undefined temps if present
			self.TW[t] = 20.0 if self.TW[t] < -100.0 else self.TW[t]
			
			# convert depth/velocity to english units, if necessary:
			self.AVDEPE[t] = self.AVDEP[t]
			self.AVVELE[t] = self.AVVEL[t]

			if self.uunits == 2:
				self.AVDEPE[t] *= 3.28
				self.AVVELE[t] *= 3.28

			# define conversion factor from mg/m2 to mg/l
			if self.AVDEPE[t] > 0.0:
				self.DEPCOR[t] = 3.28084e-3 / self.AVDEPE[t]
			else:
				self.DEPCOR[t] = -1.0e30

			# calculate scouring factor
			if self.BENRFG == 1:
				scrvel = ui['SCRVEL']	# table-type scour-parms
				scrmul = ui['SCRMUL']	# table-type scour-parms
				
				if self.AVVELE[t] > scrvel:
					self.SCRFAC[t] = scrmul
				else:
					self.SCRFAC[t] = 1.0

		#-------------------------------------------------------
		# OXRX - initialize:
		#-------------------------------------------------------
		
		if ('OXIF1' in ts):
			self.IDOX  = ts['OXIF1']  # optional, input flow
		else:
			self.IDOX = zeros(simlen)
		
		if ('OXIF2' in ts):
			self.IBOD  = ts['OXIF2']  # optional, input flow	
		else:
			self.IBOD = zeros(simlen)

		# OXRX - instantiate:
		self.OXRX = OXRX_Class(siminfo, self.nexits, self.vol, ui, ui_oxrx, ts)

		# OXRX - preallocate arrays for computed time series:
		self.DOX   = ts['DOX']   = zeros(simlen)   # concentration, state variable
		self.BOD   = ts['BOD']   = zeros(simlen)   # concentration, state variable
		self.SATDO = ts['SATDO'] = zeros(simlen)   # concentration, state variable
		
		self.RODOX = ts['OXCF11'] = zeros(simlen)   # reach outflow of DOX
		self.ROBOD = ts['OXCF12'] = zeros(simlen)   # reach outflow of BOD

		if nexits > 1:
			for i in range(nexits):
				ts['OXCF2' + str(i + 1) + ' 1'] = zeros(simlen)	# DOX outflow by exit
				ts['OXCF2' + str(i + 1) + ' 2'] = zeros(simlen)	# BOD outflow by exit

		#-------------------------------------------------------
		# NUTRX - initialize:
		#-------------------------------------------------------		
		if self.NUTFG == 1:

			# nutrient inflows:
			self.INO3 = zeros(simlen); self.INO2 = zeros(simlen)
			self.INH4 = zeros(simlen); self.IPO4 = zeros(simlen)

			if 'NUIF11' in ts:
				self.INO3 = ts['NUIF11']   # optional, input

			if 'NUIF12' in ts:				
				self.INH4 = ts['NUIF12']   # optional, input
			
			if 'NUIF13' in ts:
				self.INO2 = ts['NUIF13']   # optional, input

			if 'NUIF14' in ts:
				self.IPO4 = ts['NUIF14']   # optional, input

			# sediment-adsorbed (NH4, PO4):
			self.ISNH41 = zeros(simlen)
			self.ISNH42 = zeros(simlen)
			self.ISNH43 = zeros(simlen)
			self.ISPO41 = zeros(simlen)
			self.ISPO42 = zeros(simlen)
			self.ISPO43 = zeros(simlen)

			if 'NUIF21 1' in ts:  self.ISNH41 = ts['NUIF21 1']
			if 'NUIF22 1' in ts:  self.ISNH42 = ts['NUIF22 1']
			if 'NUIF23 1' in ts:  self.ISNH43 = ts['NUIF23 1']

			if 'NUIF21 2' in ts:  self.ISPO41 = ts['NUIF21 2']
			if 'NUIF22 2' in ts:  self.ISPO42 = ts['NUIF22 2']
			if 'NUIF23 2' in ts:  self.ISPO43 = ts['NUIF23 2']

			# NUTRX - instantiate class:
			self.NUTRX = NUTRX_Class(siminfo, self.nexits, self.vol, ui, ui_nutrx, ts, self.OXRX)

			# NUTRX - preallocate storage for computed time series
			self.NO3   = ts['NO3']   = zeros(simlen)   # concentration, state variable
			self.TAM   = ts['TAM']   = zeros(simlen)   # concentration, state variable
			self.NO2   = ts['NO2']   = zeros(simlen)   # concentration, state variable
			self.PO4   = ts['PO4']   = zeros(simlen)   # concentration, state variable 
			self.NH4   = ts['NH4']   = zeros(simlen)   # concentration, state variable
			self.NH3   = ts['NH3']   = zeros(simlen)   # concentration, state variable

			#	inflows:			
			#self.NUIF11  = ts['NUIF1_NO3'] = zeros(simlen)  # total outflow
			#self.NUIF12  = ts['NUIF1_TAM'] = zeros(simlen)  # total outflow
			#self.NUIF13  = ts['NUIF1_NO2'] = zeros(simlen)  # total outflow
			#self.NUIF14  = ts['NUIF1_PO4'] = zeros(simlen)  # total outflow

			#	total outflows:
			self.RONO3 = ts['NUCF11'] = zeros(simlen)   # outflow
			self.ROTAM = ts['NUCF12'] = zeros(simlen)   # outflow
			self.RONO2 = ts['NUCF13'] = zeros(simlen)   # outflow
			self.ROPO4 = ts['NUCF14'] = zeros(simlen)   # outflow
			
			if self.NUTRX.ADNHFG > 0:
				self.ROSNH41 = ts['NUCF21 1'] = zeros(simlen)	# sand
				self.ROSNH42 = ts['NUCF22 1'] = zeros(simlen)	# silt
				self.ROSNH43 = ts['NUCF23 1'] = zeros(simlen)	# clay

			if self.NUTRX.ADPOFG > 0:
				self.ROSPO41 = ts['NUCF21 2'] = zeros(simlen)	# sand
				self.ROSPO42 = ts['NUCF22 2'] = zeros(simlen)	# silt
				self.ROSPO43 = ts['NUCF23 2'] = zeros(simlen)	# clay

			# exit outflows:
			if nexits > 1:
				for i in range(nexits):
					ts['NUCF9' + str(i + 1) + ' 1'] = zeros(simlen)
					ts['NUCF9' + str(i + 1) + ' 2'] = zeros(simlen)
					ts['NUCF9' + str(i + 1) + ' 3'] = zeros(simlen)
					ts['NUCF9' + str(i + 1) + ' 4'] = zeros(simlen)

					if self.NUTRX.ADNHFG > 0:
						ts['OSNH4' + str(i + 1) + ' 1'] = zeros(simlen)	# sand
						ts['OSNH4' + str(i + 1) + ' 2'] = zeros(simlen)	# silt
						ts['OSNH4' + str(i + 1) + ' 3'] = zeros(simlen)	# clay
					
					if self.NUTRX.ADPOFG > 0:
						ts['OSPO4' + str(i + 1) + ' 1'] = zeros(simlen)	# sand
						ts['OSPO4' + str(i + 1) + ' 2'] = zeros(simlen)	# silt
						ts['OSPO4' + str(i + 1) + ' 3'] = zeros(simlen)	# clay

			self.RNO3 = ts['RNO3'] = zeros(simlen)
			self.RTAM = ts['RTAM'] = zeros(simlen)
			self.RNO2 = ts['RNO2'] = zeros(simlen)
			self.RPO4 = ts['RPO4'] = zeros(simlen)
			self.RNH4 = ts['RNH4'] = zeros(simlen)
			self.RNH3 = ts['RNH3'] = zeros(simlen)
			
			#-------------------------------------------------------
			# PLANK - simulate biological components:
			#-------------------------------------------------------
			if self.PLKFG == 1:
				
				# get PLANK specific time series
				self.IPHYT = zeros(simlen);	self.IZOO = zeros(simlen)
				self.IORN = zeros(simlen);	self.IORP = zeros(simlen);	self.IORC = zeros(simlen)

				if 'PKIF1' in ts:	self.IPHYT = ts['PKIF1']   # optional input
				if 'PKIF2' in ts:	self.IZOO  = ts['PKIF2']   # optional input
				if 'PKIF3' in ts:	self.IORN  = ts['PKIF3']   # optional input
				if 'PKIF4' in ts:	self.IORP  = ts['PKIF4']   # optional input
				if 'PKIF5' in ts:	self.IORC  = ts['PKIF5']   # optional input

				self.WASH = zeros(simlen);	self.SOLRAD = zeros(simlen)
				if 'WASH' in ts:	self.WASH = ts['WASH']
				if 'SOLRAD' in ts:	self.SOLRAD = ts['SOLRAD']

				# total suspended sediment conc:
				self.SSED4 = zeros(simlen)
				if 'SSED4' in ts:	self.SSED4 = ts['SSED4']

				# PLANK - instantiate class:
				self.PLANK = PLANK_Class(siminfo, self.nexits, self.vol, ui, ui_plank, ts, self.OXRX, self.NUTRX)

				# PLANK - preallocate storage for computed time series
				
				self.PHYTO  = ts['PHYTO']        = zeros(simlen)  # concentration
				self.ZOO    = ts['ZOO']          = zeros(simlen)  # concentration
				self.BENAL1 = ts['BENAL1']      = zeros(simlen)  # concentration
				self.TBENAL1= ts['TBENAL1']      = zeros(simlen)  # concentration
				self.TBENAL2= ts['TBENAL2']      = zeros(simlen)  # concentration
				self.PHYCLA = ts['PHYCLA']       = zeros(simlen)  # concentration
			
				self.ORN    = ts['ORN']    = zeros(simlen)  # state variable
				self.ORP    = ts['ORP']    = zeros(simlen)  # state variable
				self.ORC    = ts['ORC']    = zeros(simlen)  # state variable
				self.TORN   = ts['TORN']   = zeros(simlen)  # state variable
				self.TORP   = ts['TORP']   = zeros(simlen)  # state variable
				self.TORC   = ts['TORC']   = zeros(simlen)  # state variable
				self.POTBOD = ts['POTBOD'] = zeros(simlen)  # state variable
				self.TN     = ts['TN']     = zeros(simlen)  # state variable
				self.TP     = ts['TP']     = zeros(simlen)  # state variable

				#	inflows:
				self.PKIF1  = ts['PKIF_PHYT'] = zeros(simlen)  # total outflow
				self.PKIF2  = ts['PKIF_ZOO'] = zeros(simlen)  # total outflow
				self.PKIF3  = ts['PKIF_ORN'] = zeros(simlen)  # total outflow
				self.PKIF4  = ts['PKIF_ORP'] = zeros(simlen)  # total outflow
				self.PKIF5  = ts['PKIF_ORC'] = zeros(simlen)  # total outflow

				#	outflows:
				self.ROPHYT   = ts['PKCF11'] = zeros(simlen)  # total outflow
				self.ROZOO    = ts['PKCF12']  = zeros(simlen)  # total outflow
				self.ROORN    = ts['PKCF13']  = zeros(simlen)  # total outflow
				self.ROORP    = ts['PKCF14']  = zeros(simlen)  # total outflow
				self.ROORC    = ts['PKCF15']  = zeros(simlen)  # total outflow

				self.ROTORN   = ts['ROTORN'] = zeros(simlen)  # total outflow
				self.ROTORP   = ts['ROTORP'] = zeros(simlen)  # total outflow
				self.ROTORC   = ts['ROTORC'] = zeros(simlen)  # total outflow
				#self.ROTN     = ts['ROTN']   = zeros(simlen)  # total outflow
				#self.ROTP     = ts['ROTP']   = zeros(simlen)  # total outflow

				if nexits > 1:
					for i in range(nexits):
						ts['PKCF2' + str(i + 1) + ' 1'] = zeros(simlen)	# OPHYT
						ts['PKCF2' + str(i + 1) + ' 2'] = zeros(simlen)	# OZOO
						ts['PKCF2' + str(i + 1) + ' 3'] = zeros(simlen)	# OORN
						ts['PKCF2' + str(i + 1) + ' 4'] = zeros(simlen)	# OORP
						ts['PKCF2' + str(i + 1) + ' 5'] = zeros(simlen)	# OORC
				
				#-------------------------------------------------------
				# PHCARB - initialize:
				#-------------------------------------------------------
				if self.PHFG == 1:

					# get PHCARB() specific external time series
					self.ALK = zeros(simlen)
					self.ITIC = zeros(simlen)
					self.ICO2 = zeros(simlen)

					if 'CON' in ts:		self.ALK = ts['CON']
					if 'ITIC' in ts:	self.ITIC = ts['ITIC']
					if 'ICO2' in ts:	self.CO2 = ts['ICO2']

					# preallocate output arrays for speed
					self.PH     = ts['PH']    = zeros(simlen)            # state variable
					self.TIC    = ts['TIC']   = zeros(simlen)            # state variable
					self.CO2    = ts['CO2']   = zeros(simlen)            # state variable
					self.ROTIC  = ts['ROTIC'] = zeros(simlen)            # reach total outflow
					self.ROCO2  = ts['ROCO2'] = zeros(simlen)            # reach total outflow
					self.OTIC   = zeros((simlen, nexits))  # outflow by exit
					self.OCO2   = zeros((simlen, nexits))  # outflow by exit
					self.TOTCO2 = ts['TOTCO2'] = zeros(simlen)            #  ??? computed, but not returned???			

					for i in range(nexits):
						ts['OTIC' + str(i + 1)] = zeros(simlen)
						ts['OCO2' + str(i + 1)] = zeros(simlen)

					#PHCARB = PHCARB_Class(siminfo, self.nexits, self.vol, ui, ui_phcarb, ts)

		return

	def simulate(self, ts):

		for loop in range(self.simlen):

			#-------------------------------------------------------
			# define inputs for current step:			
			#-------------------------------------------------------
			ro = self.RO[loop]
			avdepe = self.AVDEPE[loop]
			avvele = self.AVVELE[loop]
			scrfac = self.SCRFAC[loop]

			tw     = self.TW[loop]
			if self.uunits == 1:
				tw = (tw - 32.0) * (5.0 / 9.0)

			wind = self.WIND[loop]
			wind_r = wind
			if self.LKFG == 0:	wind_r = 0

			depcor = self.DEPCOR[loop]			

			svol = self.vol
			self.svol = svol

			self.vol = self.VOL[loop]
			advData = self.nexits, self.svol, self.vol, self.SROVOL[loop], self.EROVOL[loop], self.SOVOL[loop], self.EOVOL[loop]

			idox = self.IDOX[loop]
			ibod = self.IBOD[loop]

			# define initial CO2 conentration value for NUTRX:	#TO-DO! - needs implementation
			co2 = -999.0
			#co2 = PHCARB.co2

			# define initial pH concentration (for use in NUTRX):	#TO-DO! - needs implementation
			phval = 7.0

			#if PHFG == 1:
			#	if PHFLAG == 1:
			#		phval = PHCARB.ph

			#-------------------------------------------------------
			# OXRX - simulate do and bod balances:
			#-------------------------------------------------------
			self.OXRX.simulate(idox, ibod, wind_r, scrfac, avdepe, avvele, depcor, tw, advData)

			# initialize DO/BOD process quantities:
			nitdox = 0.0
			denbod = 0.0
			phydox = 0.0
			zoodox = 0.0
			baldox = 0.0

			#-------------------------------------------------------
			# NUTRX - simulate primary nutrient (N/P) balances:
			#-------------------------------------------------------
			if self.NUTFG == 1:
				
				# sediment results:
				depscr = zeros(5)
				rosed = zeros(5)
				osed = zeros((self.nexits,5))

				if self.SEDFG == 1:
					for j in range(1, 5):
						rosed[j] = ts['ROSED' + str(j)][loop]
						depscr[j] = ts['DEPSCR' + str(j)][loop]
						
						if self.nexits > 1:
							for i in range(self.nexits):
								osed[i,j] = ts['OSED' + str(j) + str(i+1)][loop]
						else:
							osed[0,j] = rosed[j]

				# sediment-associated nutrient inflows:
				isnh4 = zeros(5)
				isnh4[1] = self.ISNH41[loop]
				isnh4[2] = self.ISNH42[loop]
				isnh4[3] = self.ISNH43[loop]
				
				for j in range(1,4):
					isnh4[4] += isnh4[j]

				ispo4 = zeros(5)
				ispo4[1] = self.ISPO41[loop]
				ispo4[2] = self.ISPO42[loop]
				ispo4[3] = self.ISPO43[loop]
				
				for j in range(1,4):
					ispo4[4] += ispo4[j]

				# simulate nutrients:
				self.OXRX = self.NUTRX.simulate(loop, tw, wind, phval, self.OXRX, 
								self.INO3[loop], self.INH4[loop], self.INO2[loop], self.IPO4[loop], isnh4, ispo4,
								self.PREC[loop], self.SAREA[loop], scrfac, avdepe, depcor, depscr, rosed, osed, advData)
				

				# update DO / BOD totals:
				nitdox = self.NUTRX.nitdox
				denbod = self.NUTRX.denbod

				#-------------------------------------------------------
				# PLANK - simulate plankton components & associated reactions
				#-------------------------------------------------------				
				if self.PLKFG == 1:
					
					co2 = 0.0
					#if self.PHFG == 1: co2 = PHCARB.co2		#TO-DO!
					
					(self.OXRX, self.NUTRX) \
						=	self.PLANK.simulate(tw, phval, co2, self.SSED4[loop], self.OXRX, self.NUTRX,
										self.IPHYT[loop], self.IZOO[loop], 
										self.IORN[loop], self.IORP[loop], self.IORC[loop], 
										self.WASH[loop], self.SOLRAD[loop], self.PREC[loop], 
										self.SAREA[loop], avdepe, avvele, depcor, ro, advData)

					phydox = self.PLANK.phydox
					zoodox = self.PLANK.zoodox
					baldox = self.PLANK.baldox
					
					#-------------------------------------------------------
					# PHCARB - simulate: (TO-DO! - needs class implementation)
					#-------------------------------------------------------
					if self.PHFG == 1:
						#self.PHCARB.simulate()
						
						# update pH and CO2 concentration for use in NUTRX/PLANK:
						#if ui_nutrx['PHFLAG'] == 1:
						#	phval = PHCARB.ph
						
						#co2 = PHCARB.co2
						i = 0
						pass

				#-------------------------------------------------------
				# NUTRX - update masses (TO-DO! - removed for now; fails on numba compile)
				#-------------------------------------------------------
				'''
				self.NUTRX.rno3 = self.NUTRX.no3 * self.vol
				self.NUTRX.rtam = self.NUTRX.tam * self.vol
				self.NUTRX.rno2 = self.NUTRX.no2 * self.vol
				self.NUTRX.rpo4 = self.NUTRX.po4 * self.vol
				self.NUTRX.rnh4 = self.NUTRX.nh4 * self.vol
				self.NUTRX.rnh3 = self.NUTRX.nh3 * self.vol
				
				self.NUTRX.rrno3 = self.NUTRX.no3 * self.vol
				self.NUTRX.rrtam = self.NUTRX.tam * self.vol

				if self.NUTRX.ADNHFG == 1:  
					self.NUTRX.rrtam += self.NUTRX.rsnh4[4]  # add adsorbed suspended nh4 to dissolved
					
				self.NUTRX.rrno2 = self.NUTRX.no2 * self.vol
				self.NUTRX.rrpo4 = self.NUTRX.po4 * self.vol

				if self.NUTRX.ADPOFG == 1:  
					self.NUTRX.rrpo4 += self.NUTRX.rspo4[4] # add adsorbed suspended po4 to dissolved	
				'''

				#self.NUTRX.update_mass()

			#-------------------------------------------------------
			# OXRX - finalize DO and calculate totals
			#-------------------------------------------------------

			# check do level; if dox exceeds user specified level of supersaturation, then release excess do to the atmosphere								
			dox = self.OXRX.dox
			doxs = dox

			if dox > (self.OXRX.supsat * self.OXRX.satdo):
				dox = self.OXRX.supsat * self.OXRX.satdo

			self.OXRX.readox += (dox - doxs) * self.vol
			self.OXRX.totdox = self.OXRX.readox + self.OXRX.boddox + self.OXRX.bendox \
								+ nitdox + phydox + zoodox + baldox

			self.OXRX.dox = dox
			self.OXRX.rdox = self.OXRX.dox * self.vol
			self.OXRX.rbod = self.OXRX.bod * self.vol			

			#self.OXRX.adjust_dox(nitdox, denbod, phydox, zoodox, baldox)

			# udate initial volume for next step:
			#self.svol = self.vol

			#-------------------------------------------------------
			# Store time series results (all WQ modules):
			#-------------------------------------------------------
			
			# OXRX results:
			self.DOX[loop] = self.OXRX.dox
			self.BOD[loop] = self.OXRX.bod
			self.SATDO[loop] = self.OXRX.satdo
			
			#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
			self.RODOX[loop] = self.OXRX.rodox * self.OXRX.conv
			self.ROBOD[loop] = self.OXRX.robod * self.OXRX.conv

			if self.nexits > 1:
				for i in range(self.nexits):
					ts['OXCF2' + str(i + 1) + ' 1'][loop] = self.OXRX.odox[i] * self.OXRX.conv
					ts['OXCF2' + str(i + 1) + ' 2'][loop] = self.OXRX.obod[i] * self.OXRX.conv

			# NUTRX results:
			if self.NUTFG == 1:
				self.NO3[loop] = self.NUTRX.no3
				self.TAM[loop] = self.NUTRX.tam
				self.NO2[loop] = self.NUTRX.no2
				self.PO4[loop] = self.NUTRX.po4
				self.NH4[loop] = self.NUTRX.nh4
				self.NH3[loop] = self.NUTRX.nh3

				#	inflows (lb/ivld or kg/ivld):

				#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
				conv = self.NUTRX.conv
				self.RONO3[loop] = self.NUTRX.rono3 * conv
				self.ROTAM[loop] = self.NUTRX.rotam * conv
				self.RONO2[loop] = self.NUTRX.rono2 * conv
				self.ROPO4[loop] = self.NUTRX.ropo4 * conv

				if self.NUTRX.ADNHFG > 0:
					self.ROSNH41[loop] = self.NUTRX.rosnh4[1] * conv
					self.ROSNH42[loop] = self.NUTRX.rosnh4[2] * conv
					self.ROSNH43[loop] = self.NUTRX.rosnh4[3] * conv

				if self.NUTRX.ADPOFG > 0:
					self.ROSPO41[loop] = self.NUTRX.rospo4[1] * conv
					self.ROSPO42[loop] = self.NUTRX.rospo4[2] * conv
					self.ROSPO43[loop] = self.NUTRX.rospo4[3] * conv

				# exit outflows:
				if self.nexits > 1:
					for i in range(self.nexits):
						ts['NUCF9' + str(i + 1) + ' 1'][loop] = self.NUTRX.ono3[i] * conv
						ts['NUCF9' + str(i + 1) + ' 2'][loop] = self.NUTRX.otam[i] * conv
						ts['NUCF9' + str(i + 1) + ' 3'][loop] = self.NUTRX.ono2[i] * conv
						ts['NUCF9' + str(i + 1) + ' 4'][loop] = self.NUTRX.opo4[i] * conv

						if self.NUTRX.ADNHFG > 0:
							ts['OSNH4' + str(i + 1) + ' 1'][loop] = self.NUTRX.osnh4[i,1] * conv	# sand
							ts['OSNH4' + str(i + 1) + ' 2'][loop] = self.NUTRX.osnh4[i,2] * conv	# silt
							ts['OSNH4' + str(i + 1) + ' 3'][loop] = self.NUTRX.osnh4[i,3] * conv	# clay
						
						if self.NUTRX.ADPOFG > 0:
							ts['OSPO4' + str(i + 1) + ' 1'][loop] = self.NUTRX.ospo4[i,1] * conv	# sand
							ts['OSPO4' + str(i + 1) + ' 2'][loop] = self.NUTRX.ospo4[i,2] * conv	# silt
							ts['OSPO4' + str(i + 1) + ' 3'][loop] = self.NUTRX.ospo4[i,3] * conv	# clay

				#	mass storages:
				self.RNO3[loop] = self.NUTRX.no3 * self.vol
				self.RTAM[loop] = self.NUTRX.tam * self.vol
				self.RNO2[loop] = self.NUTRX.no2 * self.vol
				self.RPO4[loop] = self.NUTRX.po4 * self.vol
				self.RNH4[loop] = self.NUTRX.nh4 * self.vol
				self.RNH3[loop] = self.NUTRX.nh3 * self.vol


				# PLANK results:
				if self.PLKFG == 1:

					self.PHYTO[loop] = self.PLANK.phyto
					self.ZOO[loop] = self.PLANK.zoo
					if self.PLANK.BALFG:
						self.BENAL1[loop] = self.PLANK.benal[0]
						self.TBENAL1[loop] = self.PLANK.tbenal[1]
						self.TBENAL2[loop] = self.PLANK.tbenal[2]
					self.PHYCLA[loop] = self.PLANK.phycla

					self.ORN[loop] = self.PLANK.orn
					self.ORP[loop] = self.PLANK.orp
					self.ORC[loop] = self.PLANK.orc
					self.TORN[loop] = self.PLANK.torn
					self.TORP[loop] = self.PLANK.torp
					self.TORC[loop] = self.PLANK.torc
					self.POTBOD[loop] = self.PLANK.potbod
					self.TN[loop] = self.PLANK.tn
					self.TP[loop] = self.PLANK.tp

					#	inflows (lb/ivld or kg/ivld):
					self.PKIF1[loop] = self.PLANK.iphyto
					self.PKIF2[loop] = self.PLANK.izoo
					self.PKIF3[loop] = self.PLANK.iorn
					self.PKIF4[loop] = self.PLANK.iorp
					self.PKIF5[loop] = self.PLANK.iorc

					#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
					conv = self.PLANK.conv

					self.ROPHYT[loop] = self.PLANK.rophyt * conv
					self.ROZOO[loop]  = self.PLANK.rozoo * conv
					self.ROORN[loop]  = self.PLANK.roorn * conv
					self.ROORP[loop]  = self.PLANK.roorp * conv
					self.ROORC[loop]  = self.PLANK.roorc * conv

					self.ROTORN[loop]  = self.PLANK.rotorn * conv
					self.ROTORP[loop]  = self.PLANK.rotorp * conv
					self.ROTORC[loop]  = self.PLANK.rotorc * conv

					#	exit outflows:
					if self.nexits > 1:
						for i in range(self.nexits):
							ts['PKCF2' + str(i + 1) + ' 1'][loop] = self.PLANK.ophyt[i] * conv
							ts['PKCF2' + str(i + 1) + ' 2'][loop] = self.PLANK.ozoo[i] * conv
							ts['PKCF2' + str(i + 1) + ' 3'][loop] = self.PLANK.oorn[i] * conv
							ts['PKCF2' + str(i + 1) + ' 4'][loop] = self.PLANK.oorp[i] * conv
							ts['PKCF2' + str(i + 1) + ' 5'][loop] = self.PLANK.oorc[i] * conv

					# PHCARB results:
					if self.PHFG == 1:
						pass
			
		return