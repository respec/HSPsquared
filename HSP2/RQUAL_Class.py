import numpy as np
from numpy import where, zeros, array
from math import log
import numba as nb
from numba.experimental import jitclass

from HSP2.OXRX_Class import OXRX_Class
from HSP2.NUTRX_Class import NUTRX_Class
from HSP2.PLANK_Class import PLANK_Class
from HSP2.PHCARB_Class import PHCARB_Class
from HSP2.utilities  import make_numba_dict, initm

spec = [
	('OXRX', OXRX_Class.class_type.instance_type),
	('NUTRX', NUTRX_Class.class_type.instance_type),
	('PLANK', PLANK_Class.class_type.instance_type),
	('PHCARB', PHCARB_Class.class_type.instance_type),
	('AFACT', nb.float64),
	('ALK', nb.float64[:]),
	('AVDEP', nb.float64[:]),
	('AVDEPE', nb.float64[:]),
	('AVVEL', nb.float64[:]),
	('AVVELE', nb.float64[:]),
	('BALCLA1', nb.float64[:]),
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
    ('NUADDR1', nb.float64[:]),
	('NUADDR2', nb.float64[:]),
	('NUADDR3', nb.float64[:]),
    ('NUADWT1', nb.float64[:]),
	('NUADWT2', nb.float64[:]),
	('NUADWT3', nb.float64[:]),
    ('NUADEP1', nb.float64[:]),
	('NUADEP2', nb.float64[:]),
	('NUADEP3', nb.float64[:]),
	('NUCF4_NITNO3', nb.float64[:]),
	('NUCF4_DENNO3', nb.float64[:]),
	('NUCF4_BODNO3', nb.float64[:]),
	('NUCF4_TOTNO3', nb.float64[:]),
	('NUCF4_PHYNO3', nb.float64[:]),
	('NUCF4_ZOONO3', nb.float64[:]),
	('NUCF4_BALNO3', nb.float64[:]),
	('NUCF5_NITTAM', nb.float64[:]),
	('NUCF5_VOLNH3', nb.float64[:]),
	('NUCF5_BNRTAM', nb.float64[:]),
	('NUCF5_BODTAM', nb.float64[:]),
	('NUCF5_TOTTAM', nb.float64[:]),
	('NUCF5_PHYTAM', nb.float64[:]),
	('NUCF5_ZOOTAM', nb.float64[:]),
	('NUCF5_BALTAM', nb.float64[:]),
	('NUCF6_NITNO2', nb.float64[:]),
	('NUCF6_TOTNO2', nb.float64[:]),
	('NUCF7_BNRPO4', nb.float64[:]),
	('NUCF7_BODPO4', nb.float64[:]),
	('NUCF7_TOTPO4', nb.float64[:]),
	('NUCF7_PHYPO4', nb.float64[:]),
	('NUCF7_ZOOPO4', nb.float64[:]),
	('NUCF7_BALPO4', nb.float64[:]),
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
	('OXCF3_REAR', nb.float64[:]),
	('OXCF3_DEC', nb.float64[:]),
	('OXCF3_BENDO', nb.float64[:]),
	('OXCF3_NITR', nb.float64[:]),
	('OXCF3_PHYT', nb.float64[:]),
	('OXCF3_ZOO', nb.float64[:]),
	('OXCF3_BALG', nb.float64[:]),
	('OXCF3_TOTAL', nb.float64[:]),
	('OXCF4_DEC', nb.float64[:]),
	('OXCF4_BENR', nb.float64[:]),
	('OXCF4_SNK', nb.float64[:]),
	('OXCF4_PHYT', nb.float64[:]),
	('OXCF4_ZOO', nb.float64[:]),
	('OXCF4_BALG', nb.float64[:]),
	('OXCF4_TOTAL', nb.float64[:]),
	('OZOO', nb.float64[:,:]),
	('PH', nb.float64[:]),
	('PHFG', nb.int32),
	('PHYCLA', nb.float64[:]),
	('PHYTO', nb.float64[:]),
	('PHCF3_DECCO2', nb.float64[:]),
	('PHCF3_PHYCO2', nb.float64[:]),
	('PHCF3_ZOOCO2', nb.float64[:]),
	('PHCF3_BGRCO2', nb.float64[:]),
	('PHCF3_BRECO2', nb.float64[:]),
	('PHCF3_INVCO2', nb.float64[:]),
	('PHCF3_TOTCO2', nb.float64[:]),
	('PKCF5_SNKPHY', nb.float64[:]),
	('PKCF5_ZOOPHY', nb.float64[:]),
	('PKCF5_DTHPHY', nb.float64[:]),
	('PKCF5_GROPHY', nb.float64[:]),
	('PKCF5_TOTPHY', nb.float64[:]),
	('PKCF6_GROZOO', nb.float64[:]),
	('PKCF6_DTHZOO', nb.float64[:]),
	('PKCF6_TOTZOO', nb.float64[:]),
	('PKCF7_GROBEN', nb.float64[:]),
	('PKCF7_DTHBEN', nb.float64[:]),
	('PKCF8_SNKORN', nb.float64[:]),
	('PKCF8_DTPORN', nb.float64[:]),
	('PKCF8_DTZORN', nb.float64[:]),
	('PKCF8_DTBORN', nb.float64[:]),
	('PKCF8_TOTORN', nb.float64[:]),
	('PKCF9_SNKORP', nb.float64[:]),
	('PKCF9_DTPORP', nb.float64[:]),
	('PKCF9_DTZORP', nb.float64[:]),
	('PKCF9_DTBORP', nb.float64[:]),
	('PKCF9_TOTORP', nb.float64[:]),
	('PKCF10_SNKORC', nb.float64[:]),
	('PKCF10_DTPORC', nb.float64[:]),
	('PKCF10_DTZORC', nb.float64[:]),
	('PKCF10_DTBORC', nb.float64[:]),
	('PKCF10_TOTORC', nb.float64[:]),
	('PKIF1', nb.float64[:]),
	('PKIF2', nb.float64[:]),
	('PKIF3', nb.float64[:]),
	('PKIF4', nb.float64[:]),
	('PKIF5', nb.float64[:]),
    ('PLADDR1', nb.float64[:]),
	('PLADDR2', nb.float64[:]),
	('PLADDR3', nb.float64[:]),
    ('PLADWT1', nb.float64[:]),
	('PLADWT2', nb.float64[:]),
	('PLADWT3', nb.float64[:]),
    ('PLADEP1', nb.float64[:]),
	('PLADEP2', nb.float64[:]),
	('PLADEP3', nb.float64[:]),
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
	('ROTN', nb.float64[:]),
	('ROTORC', nb.float64[:]),
	('ROTORN', nb.float64[:]),
	('ROTORP', nb.float64[:]),
	('ROTP', nb.float64[:]),
	('ROZOO', nb.float64[:]),
	('RPO4', nb.float64[:]),
	('RTAM', nb.float64[:]),
	('SAREA', nb.float64[:]),
	('SATDO', nb.float64[:]),
	('SATCO2', nb.float64[:]),
	('SCRFAC', nb.float64[:]),
	('SEDFG', nb.int32),
	('simlen', nb.int32),
	('SNH41', nb.float64[:]),
	('SNH42', nb.float64[:]),
	('SNH43', nb.float64[:]),
	('SOLRAD', nb.float64[:]),
	('SOVOL', nb.float64[:,:]),
	('SPO41', nb.float64[:]),
	('SPO42', nb.float64[:]),
	('SPO43', nb.float64[:]),
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
	('TNUIF1', nb.float64[:]),
	('TNUIF2', nb.float64[:]),
	('TNUIF3', nb.float64[:]),
	('TNUIF4', nb.float64[:]),
	('TNUCF1_1', nb.float64[:]),
	('TNUCF1_2', nb.float64[:]),
	('TNUCF1_3', nb.float64[:]),
	('TNUCF1_4', nb.float64[:]),
	('TPKIF_1', nb.float64[:]),
	('TPKIF_2', nb.float64[:]),
	('TPKIF_3', nb.float64[:]),
	('TPKIF_4', nb.float64[:]),
	('TPKIF_5', nb.float64[:]),
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

		# get external time series
		self.PREC  = ts['PREC']
		if self.uunits == 2:
			self.PREC = self.PREC / 3.281
		else:
			self.PREC = self.PREC * .0833
		self.SAREA = ts['SAREA'] * self.AFACT

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
				if 'SCRVEL' in ui:
					scrvel = ui['SCRVEL']	# table-type scour-parms
				else:
					scrvel = 10.0
					if self.uunits == 2:
						scrvel = 3.05
				if 'SCRMUL' in ui:
					scrmul = ui['SCRMUL']	# table-type scour-parms
				else:
					scrmul = 2.0
				
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
		
		self.RODOX = ts['OXCF1_1'] = zeros(simlen)   # reach outflow of DOX
		self.ROBOD = ts['OXCF1_2'] = zeros(simlen)   # reach outflow of BOD

		self.OXCF3_REAR = ts['OXCF3_REAR'] = zeros(simlen)  # flux terms
		self.OXCF3_DEC = ts['OXCF3_DEC'] = zeros(simlen)
		self.OXCF3_BENDO = ts['OXCF3_BENDO'] = zeros(simlen)
		self.OXCF3_NITR = ts['OXCF3_NITR'] = zeros(simlen)
		self.OXCF3_PHYT = ts['OXCF3_PHYT'] = zeros(simlen)
		self.OXCF3_ZOO = ts['OXCF3_ZOO'] = zeros(simlen)
		self.OXCF3_BALG = ts['OXCF3_BALG'] = zeros(simlen)
		self.OXCF3_TOTAL = ts['OXCF3_TOTAL'] = zeros(simlen)
		self.OXCF4_DEC = ts['OXCF4_DEC'] = zeros(simlen)
		self.OXCF4_BENR = ts['OXCF4_BENR'] = zeros(simlen)
		self.OXCF4_SNK = ts['OXCF4_SNK'] = zeros(simlen)
		self.OXCF4_PHYT = ts['OXCF4_PHYT'] = zeros(simlen)
		self.OXCF4_ZOO = ts['OXCF4_ZOO'] = zeros(simlen)
		self.OXCF4_BALG = ts['OXCF4_BALG'] = zeros(simlen)
		self.OXCF4_TOTAL = ts['OXCF4_TOTAL'] = zeros(simlen)

		if 'OXIF1' not in ts:
			ts['OXIF1'] = zeros(simlen)
		ts['IDOX'] = ts['OXIF1']
		if 'OXIF2' not in ts:
			ts['OXIF2'] = zeros(simlen)
		ts['IBOD'] = ts['OXIF2']

		if nexits > 1:
			for i in range(nexits):
				ts['OXCF2_' + str(i + 1) + '1'] = zeros(simlen)	# DOX outflow by exit
				ts['OXCF2_' + str(i + 1) + '2'] = zeros(simlen)	# BOD outflow by exit

		#-------------------------------------------------------
		# NUTRX - initialize:
		#-------------------------------------------------------		
		if self.NUTFG == 1:

			# nutrient inflows:
			self.INO3 = zeros(simlen); self.INO2 = zeros(simlen)
			self.INH4 = zeros(simlen); self.IPO4 = zeros(simlen)

			if 'NUIF1_1' in ts:
				self.INO3 = ts['NUIF1_1']   # optional, input

			if 'NUIF1_2' in ts:
				self.INH4 = ts['NUIF1_2']   # optional, input
			
			if 'NUIF1_3' in ts:
				self.INO2 = ts['NUIF1_3']   # optional, input

			if 'NUIF1_4' in ts:
				self.IPO4 = ts['NUIF1_4']   # optional, input

			# sediment-adsorbed (NH4, PO4):
			self.ISNH41 = zeros(simlen)
			self.ISNH42 = zeros(simlen)
			self.ISNH43 = zeros(simlen)
			self.ISPO41 = zeros(simlen)
			self.ISPO42 = zeros(simlen)
			self.ISPO43 = zeros(simlen)

			if 'NUIF2_11' in ts:  self.ISNH41 = ts['NUIF2_11']
			if 'NUIF2_21' in ts:  self.ISNH42 = ts['NUIF2_21']
			if 'NUIF2_31' in ts:  self.ISNH43 = ts['NUIF2_31']

			if 'NUIF2_12' in ts:  self.ISPO41 = ts['NUIF2_12']
			if 'NUIF2_22' in ts:  self.ISPO42 = ts['NUIF2_22']
			if 'NUIF2_32' in ts:  self.ISPO43 = ts['NUIF2_32']

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
			self.TNUIF1  = ts['TNUIF1'] = zeros(simlen)  # total inflow
			self.TNUIF2  = ts['TNUIF2'] = zeros(simlen)  # total inflow
			self.TNUIF3  = ts['TNUIF3'] = zeros(simlen)  # total inflow
			self.TNUIF4  = ts['TNUIF4'] = zeros(simlen)  # total inflow

			self.NUADDR1 = ts['NUADDR1'] = zeros(simlen)
			self.NUADDR2 = ts['NUADDR2'] = zeros(simlen)
			self.NUADDR3 = ts['NUADDR3'] = zeros(simlen)
			self.NUADWT1 = ts['NUADWT1'] = zeros(simlen)
			self.NUADWT2 = ts['NUADWT2'] = zeros(simlen)
			self.NUADWT3 = ts['NUADWT3'] = zeros(simlen)
			self.NUADEP1 = ts['NUADEP1'] = zeros(simlen)
			self.NUADEP2 = ts['NUADEP2'] = zeros(simlen)
			self.NUADEP3 = ts['NUADEP3'] = zeros(simlen)

			self.NUCF4_NITNO3 = ts['NUCF4_NITNO3'] = zeros(simlen)  # flux terms
			self.NUCF4_DENNO3 = ts['NUCF4_DENNO3'] = zeros(simlen)
			self.NUCF4_BODNO3 = ts['NUCF4_BODNO3'] = zeros(simlen)
			self.NUCF4_TOTNO3 = ts['NUCF4_TOTNO3'] = zeros(simlen)
			self.NUCF4_PHYNO3 = ts['NUCF4_PHYNO3'] = zeros(simlen)
			self.NUCF4_ZOONO3 = ts['NUCF4_ZOONO3'] = zeros(simlen)
			self.NUCF4_BALNO3 = ts['NUCF4_BALNO3'] = zeros(simlen)
			self.NUCF5_NITTAM = ts['NUCF5_NITTAM'] = zeros(simlen)
			self.NUCF5_VOLNH3 = ts['NUCF5_VOLNH3'] = zeros(simlen)
			self.NUCF5_BNRTAM = ts['NUCF5_BNRTAM'] = zeros(simlen)
			self.NUCF5_BODTAM = ts['NUCF5_BODTAM'] = zeros(simlen)
			self.NUCF5_TOTTAM = ts['NUCF5_TOTTAM'] = zeros(simlen)
			self.NUCF5_PHYTAM = ts['NUCF5_PHYTAM'] = zeros(simlen)
			self.NUCF5_ZOOTAM = ts['NUCF5_ZOOTAM'] = zeros(simlen)
			self.NUCF5_BALTAM = ts['NUCF5_BALTAM'] = zeros(simlen)
			self.NUCF6_NITNO2 = ts['NUCF6_NITNO2'] = zeros(simlen)
			self.NUCF6_TOTNO2 = ts['NUCF6_TOTNO2'] = zeros(simlen)
			self.NUCF7_BNRPO4 = ts['NUCF7_BNRPO4'] = zeros(simlen)
			self.NUCF7_BODPO4 = ts['NUCF7_BODPO4'] = zeros(simlen)
			self.NUCF7_TOTPO4 = ts['NUCF7_TOTPO4'] = zeros(simlen)
			self.NUCF7_PHYPO4 = ts['NUCF7_PHYPO4'] = zeros(simlen)
			self.NUCF7_ZOOPO4 = ts['NUCF7_ZOOPO4'] = zeros(simlen)
			self.NUCF7_BALPO4 = ts['NUCF7_BALPO4'] = zeros(simlen)

			#	total outflows:
			self.RONO3 = ts['NUCF1_1'] = zeros(simlen)   # dissolved outflow
			self.ROTAM = ts['NUCF1_2'] = zeros(simlen)   # dissolved outflow
			self.RONO2 = ts['NUCF1_3'] = zeros(simlen)   # dissolved outflow
			self.ROPO4 = ts['NUCF1_4'] = zeros(simlen)   # dissolved outflow
			self.TNUCF1_1 = ts['TNUCF1_1'] = zeros(simlen)   # total outflow
			self.TNUCF1_2 = ts['TNUCF1_2'] = zeros(simlen)   # total outflow
			self.TNUCF1_3 = ts['TNUCF1_3'] = zeros(simlen)   # total outflow
			self.TNUCF1_4 = ts['TNUCF1_4'] = zeros(simlen)   # total outflow
			
			if self.NUTRX.ADNHFG > 0:
				self.SNH41 = ts['SNH41']  = zeros(simlen)	    # sand
				self.SNH42 = ts['SNH42']  = zeros(simlen)	    # silt
				self.SNH43 = ts['SNH43']  = zeros(simlen)	    # clay
				self.ROSNH41 = ts['NUCF2_11'] = zeros(simlen)	# sand
				self.ROSNH42 = ts['NUCF2_21'] = zeros(simlen)	# silt
				self.ROSNH43 = ts['NUCF2_31'] = zeros(simlen)	# clay

			if self.NUTRX.ADPOFG > 0:
				self.SPO41 = ts['SPO41']  = zeros(simlen)	    # sand
				self.SPO42 = ts['SPO42']  = zeros(simlen)	    # silt
				self.SPO43 = ts['SPO43']  = zeros(simlen)	    # clay
				self.ROSPO41 = ts['NUCF2_12'] = zeros(simlen)	# sand
				self.ROSPO42 = ts['NUCF2_22'] = zeros(simlen)	# silt
				self.ROSPO43 = ts['NUCF2_32'] = zeros(simlen)	# clay

			# exit outflows:
			if nexits > 1:
				for i in range(nexits):
					ts['NUCF9_' + str(i + 1) + '1'] = zeros(simlen)
					ts['NUCF9_' + str(i + 1) + '2'] = zeros(simlen)
					ts['NUCF9_' + str(i + 1) + '3'] = zeros(simlen)
					ts['NUCF9_' + str(i + 1) + '4'] = zeros(simlen)

					if self.NUTRX.ADNHFG > 0:
						ts['OSNH4_' + str(i + 1) + '1'] = zeros(simlen)	# sand
						ts['OSNH4_' + str(i + 1) + '2'] = zeros(simlen)	# silt
						ts['OSNH4_' + str(i + 1) + '3'] = zeros(simlen)	# clay
					
					if self.NUTRX.ADPOFG > 0:
						ts['OSPO4_' + str(i + 1) + '1'] = zeros(simlen)	# sand
						ts['OSPO4_' + str(i + 1) + '2'] = zeros(simlen)	# silt
						ts['OSPO4_' + str(i + 1) + '3'] = zeros(simlen)	# clay

			self.RNO3 = ts['RNO3'] = zeros(simlen)
			self.RTAM = ts['RTAM'] = zeros(simlen)
			self.RNO2 = ts['RNO2'] = zeros(simlen)
			self.RPO4 = ts['RPO4'] = zeros(simlen)
			self.RNH4 = ts['RNH4'] = zeros(simlen)
			self.RNH3 = ts['RNH3'] = zeros(simlen)

			if 'NUIF1_1' not in ts:
				ts['NUIF1_1'] = zeros(simlen)
			ts['INO3'] = ts['NUIF1_1']
			if 'NUIF1_2' not in ts:
				ts['NUIF1_2'] = zeros(simlen)
			ts['INH4'] = ts['NUIF1_2']
			if 'NUIF1_3' not in ts:
				ts['NUIF1_3'] = zeros(simlen)
			ts['INO2'] = ts['NUIF1_3']
			if 'NUIF1_4' not in ts:
				ts['NUIF1_4'] = zeros(simlen)
			ts['IPO4'] = ts['NUIF1_4']

			if 'NUIF2_11' not in ts:
				ts['NUIF2_11'] = zeros(simlen)
			ts['ISNH41'] = ts['NUIF2_11']
			if 'NUIF2_21' not in ts:
				ts['NUIF2_21'] = zeros(simlen)
			ts['ISNH42'] = ts['NUIF2_21']
			if 'NUIF2_31' not in ts:
				ts['NUIF2_31'] = zeros(simlen)
			ts['ISNH43'] = ts['NUIF2_31']

			if 'NUIF2_12' not in ts:
				ts['NUIF2_12'] = zeros(simlen)
			ts['ISPO41'] = ts['NUIF2_12']
			if 'NUIF2_22' not in ts:
				ts['NUIF2_22'] = zeros(simlen)
			ts['ISPO42'] = ts['NUIF2_22']
			if 'NUIF2_32' not in ts:
				ts['NUIF2_32'] = zeros(simlen)
			ts['ISPO43'] = ts['NUIF2_32']

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
				self.BENAL1 = ts['BENAL1']       = zeros(simlen)  # concentration
				self.TBENAL1= ts['TBENAL1']      = zeros(simlen)  # concentration
				self.TBENAL2= ts['TBENAL2']      = zeros(simlen)  # concentration
				self.PHYCLA = ts['PHYCLA']       = zeros(simlen)  # concentration
				self.BALCLA1 = ts['BALCLA1']     = zeros(simlen)  # concentration
			
				self.ORN    = ts['ORN']    = zeros(simlen)  # state variable
				self.ORP    = ts['ORP']    = zeros(simlen)  # state variable
				self.ORC    = ts['ORC']    = zeros(simlen)  # state variable
				self.TORN   = ts['TORN']   = zeros(simlen)  # state variable
				self.TORP   = ts['TORP']   = zeros(simlen)  # state variable
				self.TORC   = ts['TORC']   = zeros(simlen)  # state variable
				self.POTBOD = ts['POTBOD'] = zeros(simlen)  # state variable
				self.TN     = ts['TN']     = zeros(simlen)  # state variable
				self.TP     = ts['TP']     = zeros(simlen)  # state variable

				self.PKCF5_SNKPHY = ts['PKCF5_SNKPHY'] = zeros(simlen)  # flux terms
				self.PKCF5_ZOOPHY = ts['PKCF5_ZOOPHY'] = zeros(simlen)
				self.PKCF5_DTHPHY = ts['PKCF5_DTHPHY'] = zeros(simlen)
				self.PKCF5_GROPHY = ts['PKCF5_GROPHY'] = zeros(simlen)
				self.PKCF5_TOTPHY = ts['PKCF5_TOTPHY'] = zeros(simlen)
				self.PKCF6_GROZOO = ts['PKCF6_GROZOO'] = zeros(simlen)
				self.PKCF6_DTHZOO = ts['PKCF6_DTHZOO'] = zeros(simlen)
				self.PKCF6_TOTZOO = ts['PKCF6_TOTZOO'] = zeros(simlen)
				self.PKCF7_GROBEN = ts['PKCF7_GROBEN'] = zeros(simlen)
				self.PKCF7_DTHBEN = ts['PKCF7_DTHBEN'] = zeros(simlen)
				self.PKCF8_SNKORN = ts['PKCF8_SNKORN'] = zeros(simlen)
				self.PKCF8_DTPORN = ts['PKCF8_DTPORN'] = zeros(simlen)
				self.PKCF8_DTZORN = ts['PKCF8_DTZORN'] = zeros(simlen)
				self.PKCF8_DTBORN = ts['PKCF8_DTBORN'] = zeros(simlen)
				self.PKCF8_TOTORN = ts['PKCF8_TOTORN'] = zeros(simlen)
				self.PKCF9_SNKORP = ts['PKCF9_SNKORP'] = zeros(simlen)
				self.PKCF9_DTPORP = ts['PKCF9_DTPORP'] = zeros(simlen)
				self.PKCF9_DTZORP = ts['PKCF9_DTZORP'] = zeros(simlen)
				self.PKCF9_DTBORP = ts['PKCF9_DTBORP'] = zeros(simlen)
				self.PKCF9_TOTORP = ts['PKCF9_TOTORP'] = zeros(simlen)
				self.PKCF10_SNKORC = ts['PKCF10_SNKORC'] = zeros(simlen)
				self.PKCF10_DTPORC = ts['PKCF10_DTPORC'] = zeros(simlen)
				self.PKCF10_DTZORC = ts['PKCF10_DTZORC'] = zeros(simlen)
				self.PKCF10_DTBORC = ts['PKCF10_DTBORC'] = zeros(simlen)
				self.PKCF10_TOTORC = ts['PKCF10_TOTORC'] = zeros(simlen)

				#	inflows:
				self.PKIF1  = ts['PKIF_PHYT'] = zeros(simlen)  # total inflow
				self.PKIF2  = ts['PKIF_ZOO'] = zeros(simlen)  # total inflow
				self.PKIF3  = ts['PKIF_ORN'] = zeros(simlen)  # total inflow
				self.PKIF4  = ts['PKIF_ORP'] = zeros(simlen)  # total inflow
				self.PKIF5  = ts['PKIF_ORC'] = zeros(simlen)  # total inflow
				self.TPKIF_1  = ts['TPKIF_1'] = zeros(simlen)  # total inflow
				self.TPKIF_2  = ts['TPKIF_2'] = zeros(simlen)  # total inflow
				self.TPKIF_3  = ts['TPKIF_3'] = zeros(simlen)  # total inflow
				self.TPKIF_4  = ts['TPKIF_4'] = zeros(simlen)  # total inflow
				self.TPKIF_5  = ts['TPKIF_5'] = zeros(simlen)  # total inflow

				self.PLADDR1 = ts['PLADDR1'] = zeros(simlen)
				self.PLADDR2 = ts['PLADDR2'] = zeros(simlen)
				self.PLADDR3 = ts['PLADDR3'] = zeros(simlen)
				self.PLADWT1 = ts['PLADWT1'] = zeros(simlen)
				self.PLADWT2 = ts['PLADWT2'] = zeros(simlen)
				self.PLADWT3 = ts['PLADWT3'] = zeros(simlen)
				self.PLADEP1 = ts['PLADEP1'] = zeros(simlen)
				self.PLADEP2 = ts['PLADEP2'] = zeros(simlen)
				self.PLADEP3 = ts['PLADEP3'] = zeros(simlen)

				#	outflows:
				self.ROPHYT   = ts['PKCF1_1'] = zeros(simlen)  # total outflow
				self.ROZOO    = ts['PKCF1_2']  = zeros(simlen)  # total outflow
				self.ROORN    = ts['PKCF1_3']  = zeros(simlen)  # total outflow
				self.ROORP    = ts['PKCF1_4']  = zeros(simlen)  # total outflow
				self.ROORC    = ts['PKCF1_5']  = zeros(simlen)  # total outflow

				self.ROTORN   = ts['ROTORN'] = zeros(simlen)  # total outflow
				self.ROTORP   = ts['ROTORP'] = zeros(simlen)  # total outflow
				self.ROTORC   = ts['ROTORC'] = zeros(simlen)  # total outflow
				self.ROTN     = ts['ROTN']   = zeros(simlen)  # total outflow
				self.ROTP     = ts['ROTP']   = zeros(simlen)  # total outflow

				if nexits > 1:
					for i in range(nexits):
						ts['PKCF2_' + str(i + 1) + '1'] = zeros(simlen)	# OPHYT
						ts['PKCF2_' + str(i + 1) + '2'] = zeros(simlen)	# OZOO
						ts['PKCF2_' + str(i + 1) + '3'] = zeros(simlen)	# OORN
						ts['PKCF2_' + str(i + 1) + '4'] = zeros(simlen)	# OORP
						ts['PKCF2_' + str(i + 1) + '5'] = zeros(simlen)	# OORC
						ts['TPKCF2_' + str(i + 1) + '1'] = zeros(simlen)	# OTORN
						ts['TPKCF2_' + str(i + 1) + '2'] = zeros(simlen)	# OTORP
						ts['TPKCF2_' + str(i + 1) + '3'] = zeros(simlen)	# OTORC
						ts['TPKCF2_' + str(i + 1) + '4'] = zeros(simlen)	# OTOTN
						ts['TPKCF2_' + str(i + 1) + '5'] = zeros(simlen)	# OTOTP
				
				#-------------------------------------------------------
				# PHCARB - initialize:
				#-------------------------------------------------------
				if self.PHFG == 1:

					# get PHCARB() specific external time series
					self.ALK = zeros(simlen)
					self.ITIC = zeros(simlen)
					self.ICO2 = zeros(simlen)

					if 'PHIF1' in ts:    self.ITIC = ts['PHIF1']  # optional input
					if 'PHIF2' in ts:    self.ICO2 = ts['PHIF2']  # optional input

					# PHCARBN - instantiate class:
					self.PHCARB = PHCARB_Class(siminfo, self.nexits, self.vol, ui, ui_nutrx, ui_phcarb, ts)

					if 'PHIF1' not in ts:
						ts['PHIF1'] = zeros(simlen)
					ts['ITIC'] = ts['PHIF1']
					if 'PHIF2' not in ts:
						ts['PHIF2'] = zeros(simlen)
					ts['ICO2'] = ts['PHIF2']

					if not 'ALKCON' in ts:
						ts['ALKCON'] = zeros(simlen)
					if 'CONS' + str(int(self.PHCARB.alkcon)) + '_CON' in ts:
						self.ALK = ts['CONS' + str(int(self.PHCARB.alkcon)) + '_CON']

					# preallocate output arrays for speed
					self.PH     = ts['PH']    = zeros(simlen)            # state variable
					self.TIC    = ts['TIC']   = zeros(simlen)            # state variable
					self.CO2    = ts['CO2']   = zeros(simlen)            # state variable
					self.SATCO2 = ts['SATCO2']= zeros(simlen)            # state variable

					self.PHCF3_DECCO2 = ts['PHCF3_DECCO2'] = zeros(simlen)  # flux terms
					self.PHCF3_PHYCO2 = ts['PHCF3_PHYCO2'] = zeros(simlen)
					self.PHCF3_ZOOCO2 = ts['PHCF3_ZOOCO2'] = zeros(simlen)
					self.PHCF3_BGRCO2 = ts['PHCF3_BGRCO2'] = zeros(simlen)
					self.PHCF3_BRECO2 = ts['PHCF3_BRECO2'] = zeros(simlen)
					self.PHCF3_INVCO2 = ts['PHCF3_INVCO2'] = zeros(simlen)
					self.PHCF3_TOTCO2 = ts['PHCF3_TOTCO2'] = zeros(simlen)

					self.ROTIC  = ts['ROTIC'] = zeros(simlen)            # reach total outflow
					self.ROCO2  = ts['ROCO2'] = zeros(simlen)            # reach total outflow
					self.OTIC   = zeros((simlen, nexits))  # outflow by exit
					self.OCO2   = zeros((simlen, nexits))  # outflow by exit
					self.TOTCO2 = ts['TOTCO2'] = zeros(simlen)            #  ??? computed, but not returned???			

					for i in range(nexits):
						ts['OTIC' + str(i + 1)] = zeros(simlen)
						ts['OCO2' + str(i + 1)] = zeros(simlen)

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
				tw = (tw - 32.0) * (0.5555)

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

			# define initial pH concentration (for use in NUTRX):
			if self.NUTFG == 1:
				if self.PHFG == 1 and self.NUTRX.PHFLAG == 1:
					phval = self.PHCARB.ph
				if self.NUTRX.PHFLAG == 2:
					phval = self.NUTRX.phval
				elif self.NUTRX.PHFLAG == 3:
					phval = ts['PHVAL'][loop]

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

				# compute atmospheric deposition influx
				nuaddr1 = self.SAREA[loop] * ts['NUADFX1'][loop]  # dry deposition;
				nuadwt1 = self.PREC[loop] * self.SAREA[loop] * ts['NUADCN1'][loop]  # wet deposition;
				nuadep_no3 = nuaddr1 + nuadwt1
				nuaddr2 = self.SAREA[loop] * ts['NUADFX2'][loop]  # dry deposition;
				nuadwt2 = self.PREC[loop] * self.SAREA[loop] * ts['NUADCN2'][loop]  # wet deposition;
				nuadep_nh3 = nuaddr2 + nuadwt2
				nuaddr3 = self.SAREA[loop] * ts['NUADFX3'][loop]  # dry deposition;
				nuadwt3 = self.PREC[loop] * self.SAREA[loop] * ts['NUADCN3'][loop]  # wet deposition;
				nuadep_po4 = nuaddr3 + nuadwt3

				# simulate nutrients:
				self.OXRX = self.NUTRX.simulate(loop, tw, wind, phval, self.OXRX, 
								self.INO3[loop], self.INH4[loop], self.INO2[loop], self.IPO4[loop], isnh4, ispo4,
								scrfac, avdepe, depcor, depscr, rosed, osed,
								nuadep_no3, nuadep_nh3, nuadep_po4, advData)
				

				# update DO / BOD totals:
				nitdox = self.NUTRX.nitdox
				denbod = self.NUTRX.denbod

				#-------------------------------------------------------
				# PLANK - simulate plankton components & associated reactions
				#-------------------------------------------------------				
				if self.PLKFG == 1:
					
					co2 = -99999.0 # 0.0
					if self.PHFG == 1: co2 = self.PHCARB.co2

					# compute atmospheric deposition influx
					pladdr1 = self.SAREA[loop] * ts['PLADFX1'][loop]  # dry deposition;
					pladwt1 = self.PREC[loop] * self.SAREA[loop] * ts['PLADCN1'][loop]  # wet deposition;
					pladep_orn = pladdr1 + pladwt1
					pladdr2 = self.SAREA[loop] * ts['PLADFX2'][loop]  # dry deposition;
					pladwt2 = self.PREC[loop] * self.SAREA[loop] * ts['PLADCN2'][loop]  # wet deposition;
					pladep_orp = pladdr2 + pladwt2
					pladdr3 = self.SAREA[loop] * ts['PLADFX3'][loop]  # dry deposition;
					pladwt3 = self.PREC[loop] * self.SAREA[loop] * ts['PLADCN3'][loop]  # wet deposition;
					pladep_orc = pladdr3 + pladwt3

					# benthic invertebrates
					binv = ts['BINV'][loop]

					(self.OXRX, self.NUTRX) \
						=	self.PLANK.simulate(tw, phval, co2, self.SSED4[loop], self.OXRX, self.NUTRX,
										self.IPHYT[loop], self.IZOO[loop], 
										self.IORN[loop], self.IORP[loop], self.IORC[loop], 
										self.WASH[loop], self.SOLRAD[loop],
										avdepe, avvele, depcor, ro, binv,
										pladep_orn, pladep_orp, pladep_orc, advData)

					phydox = self.PLANK.phydox
					zoodox = self.PLANK.zoodox
					baldox = self.PLANK.baldox
					
					#-------------------------------------------------------
					# PHCARB - simulate ph, carbon dioxide, total inorganic carbon, and alkalinity
					#-------------------------------------------------------
					if self.PHFG == 1:
						self.PHCARB.simulate(tw, self.OXRX, self.NUTRX, self.PLANK,
											 self.ITIC[loop], self.ICO2[loop], self.ALK[loop],
											 avdepe, scrfac, depcor, advData)
						
						# update pH and CO2 concentration for use in NUTRX/PLANK:
						phval = self.PHCARB.ph
						co2 = self.PHCARB.co2

				self.NUTRX.update_mass()

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
					ts['OXCF2_' + str(i + 1) + '1'][loop] = self.OXRX.odox[i] * self.OXRX.conv
					ts['OXCF2_' + str(i + 1) + '2'][loop] = self.OXRX.obod[i] * self.OXRX.conv

			self.OXCF3_REAR[loop] = self.OXRX.readox * self.OXRX.conv  # flux terms
			self.OXCF3_DEC[loop] = self.OXRX.boddox * self.OXRX.conv
			self.OXCF3_BENDO[loop] = self.OXRX.bendox * self.OXRX.conv
			self.OXCF3_TOTAL[loop] = self.OXRX.totdox * self.OXRX.conv
			self.OXCF4_DEC[loop] = self.OXRX.decbod * self.OXRX.conv
			self.OXCF4_BENR[loop] = self.OXRX.bnrbod * self.OXRX.conv
			self.OXCF4_SNK[loop] = self.OXRX.snkbod * self.OXRX.conv
			self.OXCF4_TOTAL[loop] = self.OXRX.totbod * self.OXRX.conv

			# NUTRX results:
			if self.NUTFG == 1:
				self.NO3[loop] = self.NUTRX.no3
				self.TAM[loop] = self.NUTRX.tam
				self.NO2[loop] = self.NUTRX.no2
				self.PO4[loop] = self.NUTRX.po4
				self.NH4[loop] = self.NUTRX.nh4
				self.NH3[loop] = self.NUTRX.nh3

				conv = self.NUTRX.conv
				self.NUADDR1[loop] = nuaddr1 * conv
				self.NUADDR2[loop] = nuaddr2 * conv
				self.NUADDR3[loop] = nuaddr3 * conv
				self.NUADWT1[loop] = nuadwt1 * conv
				self.NUADWT2[loop] = nuadwt2 * conv
				self.NUADWT3[loop] = nuadwt3 * conv
				self.NUADEP1[loop] = nuadep_no3 * conv
				self.NUADEP2[loop] = nuadep_nh3 * conv
				self.NUADEP3[loop] = nuadep_po4 * conv

				#	inflows (lb/ivld or kg/ivld):
				self.TNUIF1[loop] = (self.NUTRX.tnuif[1] * conv) + self.NUADEP1[loop]  # no3
				self.TNUIF2[loop] = (self.NUTRX.tnuif[2] * conv) + self.NUADEP2[loop]  # tam
				self.TNUIF3[loop] = (self.NUTRX.tnuif[3] * conv)
				self.TNUIF4[loop] = (self.NUTRX.tnuif[4] * conv) + self.NUADEP3[loop]  # po4

				#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
				self.RONO3[loop] = self.NUTRX.rono3 * conv
				self.ROTAM[loop] = self.NUTRX.rotam * conv
				self.RONO2[loop] = self.NUTRX.rono2 * conv
				self.ROPO4[loop] = self.NUTRX.ropo4 * conv
				self.TNUCF1_1[loop] = self.NUTRX.tnucf1[1] * conv
				self.TNUCF1_2[loop] = self.NUTRX.tnucf1[2] * conv
				self.TNUCF1_3[loop] = self.NUTRX.tnucf1[3] * conv
				self.TNUCF1_4[loop] = self.NUTRX.tnucf1[4] * conv

				if self.NUTRX.ADNHFG > 0:
					self.SNH41[loop] = self.NUTRX.snh4[1] * 1.0e6
					self.SNH42[loop] = self.NUTRX.snh4[2] * 1.0e6
					self.SNH43[loop] = self.NUTRX.snh4[3] * 1.0e6
					self.ROSNH41[loop] = self.NUTRX.rosnh4[1] * conv
					self.ROSNH42[loop] = self.NUTRX.rosnh4[2] * conv
					self.ROSNH43[loop] = self.NUTRX.rosnh4[3] * conv

				if self.NUTRX.ADPOFG > 0:
					self.SPO41[loop] = self.NUTRX.spo4[1] * 1.0e6
					self.SPO42[loop] = self.NUTRX.spo4[2] * 1.0e6
					self.SPO43[loop] = self.NUTRX.spo4[3] * 1.0e6
					self.ROSPO41[loop] = self.NUTRX.rospo4[1] * conv
					self.ROSPO42[loop] = self.NUTRX.rospo4[2] * conv
					self.ROSPO43[loop] = self.NUTRX.rospo4[3] * conv

				# exit outflows:
				if self.nexits > 1:
					for i in range(self.nexits):
						ts['NUCF9_' + str(i + 1) + '1'][loop] = self.NUTRX.ono3[i] * conv
						ts['NUCF9_' + str(i + 1) + '2'][loop] = self.NUTRX.otam[i] * conv
						ts['NUCF9_' + str(i + 1) + '3'][loop] = self.NUTRX.ono2[i] * conv
						ts['NUCF9_' + str(i + 1) + '4'][loop] = self.NUTRX.opo4[i] * conv

						if self.NUTRX.ADNHFG > 0:
							ts['OSNH4_' + str(i + 1) + '1'][loop] = self.NUTRX.osnh4[i,1] * conv	# sand
							ts['OSNH4_' + str(i + 1) + '2'][loop] = self.NUTRX.osnh4[i,2] * conv	# silt
							ts['OSNH4_' + str(i + 1) + '3'][loop] = self.NUTRX.osnh4[i,3] * conv	# clay
						
						if self.NUTRX.ADPOFG > 0:
							ts['OSPO4_' + str(i + 1) + '1'][loop] = self.NUTRX.ospo4[i,1] * conv	# sand
							ts['OSPO4_' + str(i + 1) + '2'][loop] = self.NUTRX.ospo4[i,2] * conv	# silt
							ts['OSPO4_' + str(i + 1) + '3'][loop] = self.NUTRX.ospo4[i,3] * conv	# clay

				#	mass storages:
				self.RNO3[loop] = self.NUTRX.no3 * self.vol * conv
				self.RTAM[loop] = self.NUTRX.tam * self.vol * conv
				self.RNO2[loop] = self.NUTRX.no2 * self.vol * conv
				self.RPO4[loop] = self.NUTRX.po4 * self.vol * conv
				self.RNH4[loop] = self.NUTRX.nh4 * self.vol * conv
				self.RNH3[loop] = self.NUTRX.nh3 * self.vol * conv

				self.OXCF3_NITR[loop] = self.NUTRX.nitdox * self.OXRX.conv  # flux terms
				self.NUCF4_NITNO3[loop] = self.NUTRX.nitno3 * conv
				self.NUCF4_DENNO3[loop] = self.NUTRX.denno3 * conv
				self.NUCF4_BODNO3[loop] = self.NUTRX.bodno3 * conv
				self.NUCF4_TOTNO3[loop] = self.NUTRX.totno3 * conv
				self.NUCF5_NITTAM[loop] = self.NUTRX.nittam * conv
				self.NUCF5_VOLNH3[loop] = self.NUTRX.volnh3 * conv
				self.NUCF5_BNRTAM[loop] = self.NUTRX.bnrtam * conv
				self.NUCF5_BODTAM[loop] = self.NUTRX.bodtam * conv
				self.NUCF5_TOTTAM[loop] = self.NUTRX.tottam * conv
				self.NUCF6_NITNO2[loop] = self.NUTRX.nitno2 * conv
				self.NUCF6_TOTNO2[loop] = self.NUTRX.nitno2 * conv
				self.NUCF7_BNRPO4[loop] = self.NUTRX.bnrpo4 * conv
				self.NUCF7_BODPO4[loop] = self.NUTRX.bodpo4 * conv
				self.NUCF7_TOTPO4[loop] = self.NUTRX.totpo4 * conv

				# PLANK results:
				if self.PLKFG == 1:

					self.PHYTO[loop] = self.PLANK.phyto
					if self.PLANK.ZOOFG:
						self.ZOO[loop] = self.PLANK.zoo / self.PLANK.zomass
					if self.PLANK.BALFG:
						self.BENAL1[loop] = self.PLANK.benal[0]
						self.TBENAL1[loop] = self.PLANK.tbenal[1]
						self.TBENAL2[loop] = self.PLANK.tbenal[2]
						self.BALCLA1[loop] = self.PLANK.balcla[0]

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

					conv = self.PLANK.conv

					self.PLADDR1[loop] = pladdr1 * conv
					self.PLADDR2[loop] = pladdr2 * conv
					self.PLADDR3[loop] = pladdr3 * conv
					self.PLADWT1[loop] = pladwt1 * conv
					self.PLADWT2[loop] = pladwt2 * conv
					self.PLADWT3[loop] = pladwt3 * conv
					self.PLADEP1[loop] = pladep_orn * conv
					self.PLADEP2[loop] = pladep_orp * conv
					self.PLADEP3[loop] = pladep_orc * conv

					#	inflows (lb/ivld or kg/ivld):
					self.PKIF1[loop] = self.PLANK.iphyto * conv
					self.PKIF2[loop] = self.PLANK.izoo * conv
					self.PKIF3[loop] = self.PLANK.iorn * conv
					self.PKIF4[loop] = self.PLANK.iorp * conv
					self.PKIF5[loop] = self.PLANK.iorc * conv
					self.TPKIF_1[loop] = (self.PLANK.itorn * conv) + self.PLADEP1[loop]
					self.TPKIF_2[loop] = (self.PLANK.itorp * conv) + self.PLADEP2[loop]
					self.TPKIF_3[loop] = (self.PLANK.itorc * conv) + self.PLADEP3[loop]
					self.TPKIF_4[loop] = (self.PLANK.itotn * conv) + self.PLADEP1[loop] + self.NUADEP1[loop] + self.NUADEP2[loop]
					self.TPKIF_5[loop] = (self.PLANK.itotp * conv) + self.PLADEP2[loop] + self.NUADEP3[loop]  # po4

					#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
					self.ROPHYT[loop] = self.PLANK.rophyt * conv
					self.ROZOO[loop]  = self.PLANK.rozoo * conv
					self.ROORN[loop]  = self.PLANK.roorn * conv
					self.ROORP[loop]  = self.PLANK.roorp * conv
					self.ROORC[loop]  = self.PLANK.roorc * conv

					self.ROTORN[loop]  = self.PLANK.rotorn * conv
					self.ROTORP[loop]  = self.PLANK.rotorp * conv
					self.ROTORC[loop]  = self.PLANK.rotorc * conv
					self.ROTN[loop]  = self.PLANK.rototn * conv
					self.ROTP[loop]  = self.PLANK.rototp * conv

					#	exit outflows:
					if self.nexits > 1:
						for i in range(self.nexits):
							ts['PKCF2_' + str(i + 1) + '1'][loop] = self.PLANK.ophyt[i] * conv
							ts['PKCF2_' + str(i + 1) + '2'][loop] = self.PLANK.ozoo[i] * conv
							ts['PKCF2_' + str(i + 1) + '3'][loop] = self.PLANK.oorn[i] * conv
							ts['PKCF2_' + str(i + 1) + '4'][loop] = self.PLANK.oorp[i] * conv
							ts['PKCF2_' + str(i + 1) + '5'][loop] = self.PLANK.oorc[i] * conv
							ts['TPKCF2_' + str(i + 1) + '1'][loop] = self.PLANK.otorn[i] * conv
							ts['TPKCF2_' + str(i + 1) + '2'][loop] = self.PLANK.otorp[i] * conv
							ts['TPKCF2_' + str(i + 1) + '3'][loop] = self.PLANK.otorc[i] * conv
							ts['TPKCF2_' + str(i + 1) + '4'][loop] = self.PLANK.ototn[i] * conv
							ts['TPKCF2_' + str(i + 1) + '5'][loop] = self.PLANK.ototp[i] * conv

					self.OXCF3_PHYT[loop] = self.PLANK.phydox * self.OXRX.conv  # flux terms
					self.OXCF3_ZOO[loop] = self.PLANK.zoodox * self.OXRX.conv
					self.OXCF3_BALG[loop] = self.PLANK.baldox * self.OXRX.conv
					self.OXCF4_PHYT[loop] = self.PLANK.phybod * self.OXRX.conv
					self.OXCF4_ZOO[loop] = self.PLANK.zoobod * self.OXRX.conv
					self.OXCF4_BALG[loop] = self.PLANK.balbod  * self.OXRX.conv
					self.OXCF4_TOTAL[loop] = self.PLANK.totbod * self.OXRX.conv
					self.NUCF4_PHYNO3[loop] = self.PLANK.phyno3 * self.NUTRX.conv
					self.NUCF4_ZOONO3[loop] = self.PLANK.zoono3 * self.NUTRX.conv
					self.NUCF4_BALNO3[loop] = self.PLANK.balno3 * self.NUTRX.conv
					self.NUCF4_TOTNO3[loop] = self.PLANK.totno3 * conv
					self.NUCF5_PHYTAM[loop] = self.PLANK.phytam * self.NUTRX.conv
					self.NUCF5_ZOOTAM[loop] = self.PLANK.zootam * self.NUTRX.conv
					self.NUCF5_BALTAM[loop] = self.PLANK.baltam * self.NUTRX.conv
					self.NUCF5_TOTTAM[loop] = self.PLANK.tottam * conv
					self.NUCF7_PHYPO4[loop] = self.PLANK.phypo4 * self.NUTRX.conv
					self.NUCF7_ZOOPO4[loop] = self.PLANK.zoopo4 * self.NUTRX.conv
					self.NUCF7_BALPO4[loop] = self.PLANK.balpo4 * self.NUTRX.conv
					self.NUCF7_TOTPO4[loop] = self.PLANK.totpo4 * conv

					self.PKCF5_SNKPHY[loop] = self.PLANK.snkphy * conv # flux terms
					self.PKCF5_ZOOPHY[loop] = self.PLANK.zoophy * conv
					self.PKCF5_DTHPHY[loop] = self.PLANK.dthphy * conv
					self.PKCF5_GROPHY[loop] = self.PLANK.grophy * conv
					self.PKCF5_TOTPHY[loop] = self.PLANK.totphy * conv
					self.PKCF6_GROZOO[loop] = self.PLANK.grozoo * conv
					self.PKCF6_DTHZOO[loop] = self.PLANK.dthzoo * conv
					self.PKCF6_TOTZOO[loop] = self.PLANK.totzoo * conv
					if self.PLANK.BALFG > 0:
						self.PKCF7_GROBEN[loop] = self.PLANK.grobal[0]
						self.PKCF7_DTHBEN[loop] = self.PLANK.dthbal[0]
					self.PKCF8_SNKORN[loop] = self.PLANK.snkorn * conv
					self.PKCF8_DTPORN[loop] = self.PLANK.phyorn * conv
					self.PKCF8_DTZORN[loop] = self.PLANK.zooorn * conv
					self.PKCF8_DTBORN[loop] = self.PLANK.balorn * conv
					self.PKCF8_TOTORN[loop] = self.PLANK.totorn * conv
					self.PKCF9_SNKORP[loop] = self.PLANK.snkorp * conv
					self.PKCF9_DTPORP[loop] = self.PLANK.phyorp * conv
					self.PKCF9_DTZORP[loop] = self.PLANK.zooorp * conv
					self.PKCF9_DTBORP[loop] = self.PLANK.balorp * conv
					self.PKCF9_TOTORP[loop] = self.PLANK.totorp * conv
					self.PKCF10_SNKORC[loop] = self.PLANK.snkorc * conv
					self.PKCF10_DTPORC[loop] = self.PLANK.phyorc * conv
					self.PKCF10_DTZORC[loop] = self.PLANK.zooorc * conv
					self.PKCF10_DTBORC[loop] = self.PLANK.balorc * conv
					self.PKCF10_TOTORC[loop] = self.PLANK.totorc * conv

					# PHCARB results:
					if self.PHFG == 1:
						self.TIC[loop] = self.PHCARB.tic
						self.CO2[loop] = self.PHCARB.co2
						self.PH[loop] = self.PHCARB.ph
						self.SATCO2[loop] = self.PHCARB.satco2

						conv = self.PHCARB.conv

						#	inflows (lb/ivld or kg/ivld):
						self.ITIC[loop] = self.PHCARB.itic * conv
						self.ICO2[loop] = self.PHCARB.ico2 * conv

						#	outflows (convert to mass per interval (lb/ivld or kg/ivld))
						self.ROTIC[loop] = self.PHCARB.rotic * conv
						self.ROCO2[loop] = self.PHCARB.roco2 * conv

						#	exit outflows:
						if self.nexits > 1:
							for i in range(self.nexits):
								ts['OTIC' + str(i + 1)][loop] = self.PHCARB.otic[i] * conv
								ts['OCO2' + str(i + 1)][loop] = self.PHCARB.oco2[i] * conv

						self.PHCF3_DECCO2[loop] = self.NUTRX.decco2 * conv  # flux terms
						self.PHCF3_PHYCO2[loop] = self.PLANK.pyco2 * conv
						self.PHCF3_ZOOCO2[loop] = self.PLANK.zoco2 * conv
						self.PHCF3_BGRCO2[loop] = self.PLANK.baco2 * conv
						self.PHCF3_BRECO2[loop] = self.PHCARB.benco2 * conv
						self.PHCF3_INVCO2[loop] = self.PHCARB.invco2 * conv
						self.PHCF3_TOTCO2[loop] = self.PHCARB.totco2 * conv
			
		return