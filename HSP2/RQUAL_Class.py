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
	('BENAL', nb.float64[:]),
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
	('PLKFG', nb.int32),
	('PO4', nb.float64[:]),
	('POTBOD', nb.float64[:]),
	('PREC', nb.float64[:]),
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
	('ROTAM', nb.float64[:]),
	('ROTIC', nb.float64[:]),
	('ROZOO', nb.float64[:]),
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
	('TORC', nb.float64[:]),
	('TORN', nb.float64[:]),
	('TORP', nb.float64[:]),
	('TOTCO2', nb.float64[:]),
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

		self.DOX   = ts['DOX']   = zeros(simlen)   # concentration, state variable
		self.BOD   = ts['BOD']   = zeros(simlen)   # concentration, state variable
		self.SATDO = ts['SATDO'] = zeros(simlen)   # concentration, state variable
		self.RODOX = ts['RODOX'] = zeros(simlen)   # reach outflow of DOX
		self.ROBOD = ts['ROBOD'] = zeros(simlen)   # reach outflow of BOD
		self.ODOX  = zeros((simlen, nexits))   # reach outflow per gate of DOX
		self.OBOD  = zeros((simlen, nexits))   # reach outflow per gate of BOD

		for i in range(nexits):
			ts['ODOX' + str(i + 1)] = zeros(simlen)
			ts['OBOD' + str(i + 1)] = zeros(simlen)

		# OXRX - instantiate:
		self.OXRX = OXRX_Class(siminfo, self.nexits, self.vol, ui, ui_oxrx, ts)

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

			# atmospheric deposition - create time series (TO-DO! - needs implementation):
			'''
			self.NUADFX = zeros((simlen,4))
			self.NUADCN = zeros((simlen,4))
			self.NUADFG = zeros(7, dtype=np.int32)

			for i in range(1, 4):
				n = 2*(i - 1) + 1
				nuadfg_dd = int(ui_nutrx['NUADFG(' + str(n) + ')'])
				nuadfg_wd = int(ui_nutrx['NUADFG(' + str(n+1) + ')'])

				if (nuadfg_dd < 0):
					pass

				if (nuadfg_wd < 0):
					pass
			'''
			# NUAFX = setit()  # NUAFXM monthly, constant or time series
			# NUACN = setit()  # NUACNM monthly, constant or time series

			# preallocate storage for computed time series
			self.NO3   = ts['NO3']   = zeros(simlen)   # concentration, state variable
			self.NO2   = ts['NO2']   = zeros(simlen)   # concentration, state variable
			self.NH3   = ts['NH3']   = zeros(simlen)   # concentration, state variable
			self.NH4   = ts['NH4']   = zeros(simlen)   # concentration, state variable
			self.PO4   = ts['PO4']   = zeros(simlen)   # concentration, state variable 
			self.TAM   = ts['TAM']   = zeros(simlen)   # concentration, state variable
			self.RONO3 = ts['RONO3'] = zeros(simlen)   # outflow
			self.RONO2 = ts['RONO2'] = zeros(simlen)   # outflow
			self.ROTAM = ts['ROTAM'] = zeros(simlen)   # outflow
			self.ROPO4 = ts['ROPO4'] = zeros(simlen)   # outflow
			self.ONO3  = zeros((simlen, nexits))   # outflow
			self.ONO2  = zeros((simlen, nexits))   # outflow
			self.OTAM  = zeros((simlen, nexits))   # outflow
			self.OPO4  = zeros((simlen, nexits))   # outflow

			for i in range(nexits):
				ts['ONO3' + str(i + 1)] = zeros(simlen)
				ts['ONO2' + str(i + 1)] = zeros(simlen)
				ts['OTAM' + str(i + 1)] = zeros(simlen)
				ts['OPO4' + str(i + 1)] = zeros(simlen)

			# NUTRX - initialize:
			self.NUTRX = NUTRX_Class(siminfo, self.nexits, self.vol, ui, ui_nutrx, ts, self.OXRX)

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

				# preallocate arrays for better performance
				self.ORN    = ts['PKST3_ORN']    = zeros(simlen)  # state variable
				self.ORP    = ts['PKST3_ORP']    = zeros(simlen)  # state variable
				self.ORC    = ts['PKST3_ORC']    = zeros(simlen)  # state variable
				self.TORN   = ts['PKST3_TORN']   = zeros(simlen)  # state variable
				self.TORP   = ts['PKST3_TORP']   = zeros(simlen)  # state variable
				self.TORC   = ts['PKST3_TORC']   = zeros(simlen)  # state variable
				self.POTBOD = ts['PKST3_POTBOD'] = zeros(simlen)  # state variable
			
				self.PHYTO  = ts['PHYTO']        = zeros(simlen)  # concentration
				self.PHYCLA = ts['PHYCLA']       = zeros(simlen)  # concentration
				self.ZOO    = ts['ZOO']          = zeros(simlen)  # concentration
				self.TBENAL1= ts['TBENAL1']      = zeros(simlen)  # concentration
				self.TBENAL2= ts['TBENAL2']      = zeros(simlen)  # concentration
			
				self.ROPHYT   = ts['ROPHYT'] = zeros(simlen)  # total outflow
				self.ROZOO    = ts['ROZOO']  = zeros(simlen)  # total outflow
				self.ROORN    = ts['ROORN']  = zeros(simlen)  # total outflow
				self.ROORP    = ts['ROORP']  = zeros(simlen)  # total outflow
				self.ROORC    = ts['ROORC']  = zeros(simlen)  # total outflow
				
				self.OPHYT   = zeros((simlen, nexits)) # outflow by gate	
				self.OZOO    = zeros((simlen, nexits)) # outflow by gate	 	
				self.OORN    = zeros((simlen, nexits)) # outflow by gate	 	
				self.OORP    = zeros((simlen, nexits)) # outflow by gate	 	
				self.OORC    = zeros((simlen, nexits)) # outflow by gate	 	

				for i in range(nexits):
					ts['OPHYT' + str(i + 1)] = zeros(simlen)
					ts['OZOO' + str(i + 1)] = zeros(simlen)
					ts['OORN' + str(i + 1)] = zeros(simlen)
					ts['OORP' + str(i + 1)] = zeros(simlen)
					ts['OORC' + str(i + 1)] = zeros(simlen)

				# PLANK - initialize:
				self.PLANK = PLANK_Class(siminfo, self.nexits, self.vol, ui, ui_plank, ts, self.OXRX, self.NUTRX)
				
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
								osed[i,j] = ts['OSED' + str(i+1)][loop]
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
			self.RODOX[loop] = self.OXRX.rodox
			self.ROBOD[loop] = self.OXRX.robod

			if self.nexits > 1:
				for i in range(self.nexits):
					ts['ODOX' + str(i + 1)][loop] = self.OXRX.odox[i]
					ts['OBOD' + str(i + 1)][loop] = self.OXRX.obod[i]

			# NUTRX results:
			if self.NUTFG == 1:
				self.NO3[loop] = self.NUTRX.no3
				self.NO2[loop] = self.NUTRX.no2
				self.TAM[loop] = self.NUTRX.tam
				self.PO4[loop] = self.NUTRX.po4
				self.NH3[loop] = self.NUTRX.nh3

				self.RONO3[loop] = self.NUTRX.rono3
				self.RONO2[loop] = self.NUTRX.rono2
				self.ROTAM[loop] = self.NUTRX.rotam
				self.ROPO4[loop] = self.NUTRX.ropo4

				if self.nexits > 1:
					for i in range(self.nexits):
						ts['ONO3' + str(i + 1)][loop] = self.NUTRX.ono3[i]
						ts['ONO2' + str(i + 1)][loop] = self.NUTRX.ono2[i]
						ts['OTAM' + str(i + 1)][loop] = self.NUTRX.otam[i]
						ts['OPO4' + str(i + 1)][loop] = self.NUTRX.opo4[i]

				# PLANK results:
				if self.PLKFG == 1:

					self.ORN[loop] = self.PLANK.orn
					self.ORP[loop] = self.PLANK.orp
					self.ORC[loop] = self.PLANK.orc
					self.TORN[loop] = self.PLANK.torn
					self.TORP[loop] = self.PLANK.torp
					self.TORC[loop] = self.PLANK.torc

					self.PHYTO[loop] = self.PLANK.phyto
					self.PHYCLA[loop] = self.PLANK.phycla
					self.TBENAL1[loop] = self.PLANK.tbenal[1]
					self.TBENAL2[loop] = self.PLANK.tbenal[2]
					self.ZOO[loop] = self.PLANK.zoo

					if self.nexits > 1:
						for i in range(self.nexits):
							ts['OPHYT' + str(i + 1)][loop] = self.PLANK.ophyt[i]
							ts['OZOO' + str(i + 1)][loop] = self.PLANK.ozoo[i]
							ts['OORN' + str(i + 1)][loop] = self.PLANK.oorn[i]
							ts['OORP' + str(i + 1)][loop] = self.PLANK.oorp[i]
							ts['OORC' + str(i + 1)][loop] = self.PLANK.oorc[i]	

					# PHCARB results:
					if self.PHFG == 1:
						pass

		return