from numba import njit

#@njit(cache=True)
def benth (dox, anaer, BRCON, scrfac, depcor, conc):
	''' simulate benthal release of constituent'''
	# calculate benthal release of constituent; release is a step function of aerobic/anaerobic conditions, and stream velocity;
	# scrfac, the scouring factor dependent on stream velocity and depcor, the conversion factor from mg/m2 to mg/l,
	# both calculated in rqual; releas is expressed in mg/m2.ivl
	releas = BRCON[0] * scrfac * depcor  if dox > anaer else BRCON[1] * scrfac * depcor
	conc  += releas
	return conc, releas


#@njit(cache=True)
def decbal(TAMFG, PO4FG, decnit, decpo4, tam, no3, po4):
	''' perform materials balance for transformation from organic to inorganic material by decay in reach water'''
	if TAMFG:
		tam += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
	else:
		no3 += decnit   # add nitrogen transformed to inorganic nitrogen by biomass decomposition
	if PO4FG:   # add phosphorus transformed to inorganic phosphorus by biomass decomposition to po4 state variable
		po4 += decpo4
	return tam, no3, po4


@njit(cache=True)
def sink (vol, avdepe, kset, conc):
	''' calculate quantity of material settling out of the control volume; determine the change in concentration as a result of sinking'''
	snkmat = 0.0

	if kset > 0.0 and avdepe > 0.17:
		# calculate concentration change due to outgoing material; snkout is expressed in mass/liter/ivl; kset is expressed as ft/ivl and avdepe as feet
		snkout = conc * (kset / avdepe)  if kset < avdepe else conc  # calculate portion of material which settles out of the control volume during time step; snkout is expressed as mass/liter.ivl; conc is the concentration of material in the control volume
		conc  -= snkout        # calculate remaining concentration of material in the control volume
		snkmat = snkout * vol    # find quantity of material that sinks out; units are  mass.ft3/l.ivl in english system, and mass.m3/l.ivl in metric system
	else:
		snkout = 0.0
		snkmat = 0.0		
	return conc, snkmat