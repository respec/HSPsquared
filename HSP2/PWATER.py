''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of HSPF HPERWAT.FOR into Python
'''

from numpy import zeros, ones, sqrt, array, full, nan, argmax, int64
from math import log, exp
from numba import njit
from HSP2.utilities import initm, hourflag, hoursval, make_numba_dict

MAXLOOPS  = 100      # newton method max loops
TOLERANCE = 0.01     # newton method exit tolerance

ERRMSGS =('PWATER: Sum of irrtgt in not one',             #ERRMSG0
          'PWATER: UZRAA exceeds UZRA array bounds',      #ERRMSG1
          'PWATER: INTGB exceeds INTGRL array bounds',    #ERRMSG2
          'PWATER: Reduced AGWO value to available',      #ERRMSG3
          'PWATER: Reduced GVWS to AGWS',                 #ERRMSG4
          'PWATER: GWVS < -0.02, set to zero',            #ERRMSG5
          'PWATER: Proute runoff did not converge',       #ERRMSG6
          'PWATER: UZI highly negative',                  #ERRMSG7
          'PWATER: Reset AGWS to zero',                   #ERRMSG8
          'PWATER: High Water Table code not implemented', #ERRMSG9
          )

def pwater(io_manager, siminfo, uci, ts):
    ''' PERLND WATER module
    CALL: pwater(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level info (sim_start for example)
       ui is a dictionary with PLS specific HSPF UCI like data
       ts is a dictionary with PLS specific timeseries '''

    steps   = siminfo['steps']                # number of simulation points

    #if RTOPFG == 3 and 'SURTAB' in ui:
    #    surtab = typeT(ui['SURTAB'])   # FTable

    # missing flows are treated as zeros
    for name in ('AGWLI','IFWLI','LGTMP','LZLI','PETINP','PREC','SURLI','UZLI'):
        if name not in ts:
            ts[name] = zeros(steps)

    # insure defined, but not usable accidently
    for name in ('AIRTMP','PACKI','PETINP','PREC','RAINF','SNOCOV','WYIELD'):
        if name not in ts:
            ts[name] = full(steps, nan)

    # Replace fixed parameters with time series if not already present
    for name in ('AGWRC','DEEPFR','INFILT','KVARY','LZSN','PETMIN','PETMAX'):
        if name not in ts and name in uci['PARAMETERS']:
            ts[name] = full(steps, uci['PARAMETERS'][name])

    # process optional monthly arrays to return interpolated data or constant array
    u = uci['PARAMETERS']
    if 'VLEFG' in u:
        flag = (u['VLEFG'] == 1) or (u['VLEFG'] == 3)

        ts['LZETP'] = initm(siminfo, uci, flag,        'MONTHLY_LZETP', u['LZETP'])
        ts['CEPSC'] = initm(siminfo, uci, u['VCSFG'],  'MONTHLY_CEPSC', u['CEPSC'])
        ts['INTFW'] = initm(siminfo, uci, u['VIFWFG'], 'MONTHLY_INTFW', u['INTFW'])
        ts['IRC']   = initm(siminfo, uci, u['VIRCFG'], 'MONTHLY_IRC',   u['IRC'])
        ts['NSUR']  = initm(siminfo, uci, u['VNNFG'],  'MONTHLY_NSUR',  u['NSUR'])
        ts['UZSN']  = initm(siminfo, uci, u['VUZFG'],  'MONTHLY_UZSN',  u['UZSN'])
    else:
        ts['LZETP'] = full(steps, u['LZETP'])
        ts['CEPSC'] = full(steps, u['CEPSC'])
        ts['INTFW'] = full(steps, u['INTFW'])
        ts['IRC'] = full(steps, u['IRC'])
        ts['NSUR'] = full(steps, u['NSUR'])
        ts['UZSN'] = full(steps, u['UZSN'])

    # true the first time and at start of every day of simulation
    ts['DAYFG'] = hourflag(siminfo, 0, dofirst=True).astype(float)

    # true the first time and at every hour of simulation
    ts['HRFG'] = hoursval(siminfo, ones(24), dofirst=True).astype(float)

    ui = make_numba_dict(uci)  # Note: all values coverted to float automatically
    ui['steps']  = siminfo['steps']
    ui['delt']   = siminfo['delt']
    ui['errlen'] = len(ERRMSGS)
    ui['uunits'] = siminfo['units']

    # kludge to make ICEFG available from SNOW to PWATER
    ui['ICEFG']  = siminfo['ICEFG'] if 'ICEFG' in siminfo else 0.0

    CSNOFG = 0
    if 'CSNOFG' in ui:
        CSNOFG = int(ui['CSNOFG'])
    # make CSNOFG available to other sections
    u['CSNOFG'] = CSNOFG

    ############################################################################
    errors = _pwater_(ui, ts)      # traditional HSPF HPERWAT
    ############################################################################

    return errors, ERRMSGS


@njit(cache=True)
def _pwater_(ui, ts):
    ''' simulate the water budget for a pervious land segment'''
    errors = zeros(int(ui['errlen'])).astype(int64)

    if 'HWTFG' in ui:
        if int(ui['HWTFG']):
            errors[9] += 1
            return

    delt60 = ui['delt'] / 60.0      # simulation interval in hours
    steps  = int(ui['steps'])
    uunits = ui['uunits']
    DAYFG  = ts['DAYFG'].astype(int64)
    HRFG   = ts['HRFG'].astype(int64)

    # table of coordinates for functions used to evaluate upper zone behavior
    uzra   = array([0.0, 1.25, 1.50, 1.75, 2.00, 2.10, 2.20, 2.25, 2.5, 4.0])
    intgrl = array([0.0, 1.29, 1.58, 1.92, 2.36, 2.81, 3.41, 3.8,  7.1, 3478.])

    # like MATLAB, much faster to preinitialize variables. Store in ts Dict
    ts['AGWET']  = AGWET  = zeros(steps)
    ts['AGWI']   = AGWI   = zeros(steps)
    ts['AGWO']   = AGWO   = zeros(steps)
    ts['AGWS']   = AGWS   = zeros(steps)
    ts['BASET']  = BASET  = zeros(steps)
    ts['CEPE']   = CEPE   = zeros(steps)
    ts['CEPS']   = CEPS   = zeros(steps)
    ts['GWVS']   = GWVS   = zeros(steps)
    ts['IFWI']   = IFWI   = zeros(steps)
    ts['IFWO']   = IFWO   = zeros(steps)
    ts['IFWS']   = IFWS   = zeros(steps)
    ts['IGWI']   = IGWI   = zeros(steps)
    ts['INFIL']  = INFIL  = zeros(steps)
    ts['LZET']   = LZET   = zeros(steps)
    ts['LZI']    = LZI    = zeros(steps)
    ts['LZS']    = LZS    = zeros(steps)
    ts['PERC']   = PERC   = zeros(steps)
    ts['PERO']   = PERO   = zeros(steps)
    ts['PERS']   = PERS   = zeros(steps)
    ts['PET']    = PET    = zeros(steps)
    ts['PETADJ'] = PETADJ = zeros(steps)
    ts['SUPY']   = SUPY   = zeros(steps)
    ts['SURI']   = SURI   = zeros(steps)
    ts['SURO']   = SURO   = zeros(steps)
    ts['SURS']   = SURS   = zeros(steps)
    ts['TAET']   = TAET   = zeros(steps)
    ts['TGWS']   = TGWS   = zeros(steps)
    ts['UZET']   = UZET   = zeros(steps)
    ts['UZI']    = UZI    = zeros(steps)
    ts['UZS']    = UZS    = zeros(steps)
    ts['INFFAC'] = INFFAC = ones(steps)

    irrappV = zeros(7)
    irrcep = 0.0   # ????
    #irdraw = zeros(3)

    CSNOFG = 0
    ICEFG = 0
    IFFCFG = 1
    IFRDFG = 0
    RTOPFG = 0
    UZFG = 0
    VLEFG = 0

    if 'ICEFG' in ui:
        ICEFG = int(ui['ICEFG'])

    if 'CSNOFG' in ui:
        CSNOFG = int(ui['CSNOFG'])
        IFFCFG = int(ui['IFFCFG'])
        IFRDFG = int(ui['IFRDFG'])
        RTOPFG = int(ui['RTOPFG'])
        UZFG   = int(ui['UZFG'])
        VLEFG  = int(ui['VLEFG'])

    agwetp = ui['AGWETP']
    agws   = ui['AGWS']
    basetp = ui['BASETP']
    ceps   = ui['CEPS']
    gwvs   = ui['GWVS']
    ifws   = ui['IFWS']
    infexp = ui['INFEXP']
    infild = ui['INFILD']
    lsur   = ui['LSUR']
    lzs    = ui['LZS']
    slsur  = ui['SLSUR']
    surs   = ui['SURS']
    uzs    = ui['UZS']
    forest = ui['FOREST']

    if uunits == 2:
        lsur = lsur * 3.28
        ceps = ceps * 0.0394  # / 25.4
        surs = surs * 0.0394  # / 25.4
        uzs  = uzs * 0.0394  # / 25.4
        ifws = ifws * 0.0394  # / 25.4
        lzs  = lzs * 0.0394  # / 25.4
        agws = agws * 0.0394  # / 25.4
        gwvs = gwvs * 0.0394  # / 25.4

    if ICEFG:
        fzg  = ui['FZG']
        fzgl = ui['FZGL']
        if uunits == 2:
            fzg = fzg * 0.0394

    AGWLI  = ts['AGWLI']
    AGWRC  = ts['AGWRC']
    AIRTMP = ts['AIRTMP']
    CEPSC  = ts['CEPSC']
    DEEPFR = ts['DEEPFR']
    IFWLI  = ts['IFWLI']
    INFILT = ts['INFILT'] * delt60        # convert to internal units
    INTFW  = ts['INTFW']
    IRC    = ts['IRC']
    KVARY  = ts['KVARY']
    LGTMP  = ts['LGTMP']
    LZETP  = ts['LZETP']
    LZLI   = ts['LZLI']
    LZSN   = ts['LZSN']
    NSUR   = ts['NSUR']
    PACKI  = ts['PACKI']
    PETINP = ts['PETINP']
    PETMAX = ts['PETMAX']
    PETMIN = ts['PETMIN']
    PREC   = ts['PREC']
    RAINF  = ts['RAINF']
    SNOCOV = ts['SNOCOV']
    SURLI  = ts['SURLI']
    UZLI   = ts['UZLI']
    UZSN   = ts['UZSN']
    WYIELD = ts['WYIELD']

    if uunits == 2:
        CEPSC = CEPSC * 0.0394 # / 25.4
        UZSN  = UZSN * 0.0394  # / 25.4
        INFILT= INFILT * 0.0394  # / 25.4
        KVARY = KVARY / 0.0394
        LZSN  = LZSN * 0.0394  # / 25.4
        PETMAX = (ts['PETMAX'] * 9./5.) + 32.
        PETMIN = (ts['PETMIN'] * 9./5.) + 32.
        WYIELD = WYIELD * 0.0394  # / 25.4

    # initialize  variables
    kgwV = 1.0 - AGWRC**(delt60/24.0)    # groundwater recession parameter
    rlzrat = -1.0E30
    lzfrac = -1.0E30
    rparm  = -1.0E30
    if agws < 0.0:        # no gw storage is active
        agws = 0.0
    TGWS[0] = agws

    msupy = 0.0
    dec   = nan
    src   = nan
    kifw  = nan
    ifwk2 = nan
    ifwk1 = nan

    # MASTER lOOP
    for step in range(steps):
        oldmsupy = msupy

        dayfg  = int(DAYFG[step])
        hrfg   = int(HRFG[step])
        inffac = INFFAC[step]
        kgw    = kgwV[step]

        # These lines allow constant parameters to be replaced by timeseries
        lzetp  = LZETP[step]
        cepsc  = CEPSC[step]
        uzsn   = UZSN[step]
        infilt = INFILT[step]
        kvary  = KVARY[step]
        lzsn   = LZSN[step]

        # PWATRX
        petinp = PETINP[step]
        if CSNOFG:
            airtmp = AIRTMP[step]
            petmax = PETMAX[step]
            petmin = PETMIN[step]
            snocov = SNOCOV[step]
            SUPY[step] = RAINF[step] * (1.0 - snocov) + WYIELD[step]
            if hrfg:
                petadj = (1.0 - forest) * (1.0 - snocov) + forest
                if (airtmp < petmax) and (petadj > 0.5):
                    petadj = 0.5
                if airtmp < petmin:
                    petadj = 0.0
                PETADJ[step] = petadj
            PET[step] = petinp * petadj
            if ICEFG:   # calculate factor to reduce infiltration and percolation to account for frozen ground
                inffac = max(fzgl, 1.0 - fzg * PACKI[step])
        else:
            SUPY[step] = PREC[step]
            PET[step] = petinp

        # adjust inffac based on soil temperature
        if IFFCFG == 2:
            inffac = fzgl if LGTMP[step] <= 0.0 else 1.0

        # ICEPT
        ''' Simulate the interception of moisture by vegetal or other ground cover'''
        ceps = ceps + SUPY[step] + irrcep       # add to interception storage
        cepo = 0.0
        if ceps > cepsc:
            cepo = ceps - cepsc
            ceps = cepsc
        # END ICEPT

        # in PWATRX
        suri  = cepo + SURLI[step]                # surface inflow
        msupy = suri + surs + irrappV[2]
        lzrat = lzs / lzsn   # determine the current value of the lower zone storage ratio

        if msupy <= 0.0:
            surs  = 0.0
            suro  = 0.0
            ifwi  = 0.0
            infil = 0.0
            uzi   = 0.0
        else:
            # SURFAC
            ''' Distribute the water available for infiltration and runoff - units of fluxes are in./ivl'''
            ''' establish locations of sloping lines on infiltration/inflow/sur runoff
            figure.  prefix "i" refers to "infiltration" line, ibar is the mean
            infiltration capacity over the segment, internal units of infilt are inches/ivl'''

            ibar = infilt / (lzrat**infexp)
            if inffac < 1.0:
                ibar = ibar * inffac
            imax = ibar * infild   # infild is an input parameter - ratio of maximum to mean infiltration capacity
            imin = ibar - (imax - ibar)

            if dayfg or oldmsupy==0.0:
                dummy = NSUR[step] * lsur
                dec = 0.00982 * (dummy / sqrt(slsur))**0.6
                src = 1020.0  * (sqrt(slsur) / dummy)

            ratio = max(1.0001, INTFW[step] * 2.0**lzrat)
            # DISPOSE
            # DIVISN
            if msupy <= imin:       # msupy line is entirely below other line
                under = msupy
                over  = 0.0
            elif msupy > imax:      # msupy line is entirely above other line
                under = (imin + imax) * 0.5
                over = msupy - under
            else:                   # msupy  line crosses other line
                over = ((msupy - imin)**2) * 0.5 / (imax - imin)
                under = msupy - over
            # END DIVISN
            infil = under
            if over <= 0.0:
                surs = 0.0
                suro = 0.0
                ifwi = 0.0
                uzi  = 0.0
            else:  # there is some potential interflow inflow and maybe surface detention/outflow -- the sum of these is potential direct runoff
                pdro = over

                # determine how much of this potential direct runoff will be taken by the upper zone
                if UZFG:
                    # $UZINF2 -- HSPX, ARM, NPS type calculation
                    '''Compute inflow to upper zone during this interval, using "fully forward"
                        type algorithm  as used in HSPX,ARM and NPS.  Note:  although this method
                        should give results closer to those produced by HSPX, etc., its output will
                        be more sensitive to delt than that given by subroutine uzinf'''
                    uzrat = uzs / uzsn
                    if uzrat < 2.0:
                        k1 = 3.0 - uzrat
                        uzfrac = 1.0 - (uzrat * 0.5) * ((1.0/(1.0 + k1))**k1)
                    else:
                        k2 = (2.0 * uzrat) - 3.0
                        uzfrac = (1.0/(1.0 +  k2))**k2
                    uzi = pdro * uzfrac
                else:
                    # UZINF
                    ''' Compute the inflow to the upper zone during this time interval. Do this
                        using a table look-up to handle the non-analytic integral given in
                        supporting documentation.'''

                    # find the value of the integral at initial uzra
                    uzraa = uzs  / uzsn
                    kk = argmax(uzraa < uzra)-1     # uzra[kk] < uzraa <= uzra[kk+1]
                    if kk == -1:
                        kk = 8
                        errors[1] += 1   # ERRMSG1: UZRAA exceeds UZRA array bounds
                    intga = intgrl[kk] + (intgrl[kk+1] - intgrl[kk]) * (uzraa - uzra[kk]) / (uzra[kk+1] - uzra[kk])
                    intgb = (pdro / uzsn) + intga

                    kk = argmax(intgb < intgrl)-1   # intgrl[kk] <= intgb < intgrl[kk+1]
                    if kk == -1:
                         errors[2] += 1  # ERRMSG2: INTGB exceeds INTGRL array bounds
                         kk = 8

                    uzrab = uzra[kk] + (uzra[kk+1] - uzra[kk])  * (intgb - intgrl[kk]) / (intgrl[kk+1] - intgrl[kk])
                    uzi = (uzrab - uzraa) * uzsn
                    if uzi < -1.0e-3:
                        errors[7] += 1        # UZI highly negative
                    uzi = max(0.0, uzi)        # negative inflow shouldn't happen, but does for extremely small pdro

                if uzi > pdro:
                    uzi = pdro
                uzfrac = uzi / pdro

                # the prefix "ii" is used on variables on second divisn
                iimin = imin * ratio
                iimax = imax * ratio

                # DIVISN
                if msupy <= iimin:   # msupy line is entirely below other line
                    over2 = 0.0
                elif msupy > iimax: # msupy line is entirely above other line
                    over2 = msupy - (iimin + iimax) * 0.5
                else:                   # msupy  line crosses other line
                    over2 = ((msupy - iimin)**2) * 0.5 / (iimax - iimin)
                #END DIVISN

                # psur is potential surface detention/runoff
                psur = over2
                pifwi = pdro - psur # pifwi is potential interflow inflow
                ifwi  = pifwi * (1.0 - uzfrac)

                if psur <= 0.0:
                    surs = 0.0
                    suro = 0.0
                else:
                    # there will be something on or running off the surface reduce it to account for the upper zone's share
                    psur = psur * (1.0 - uzfrac)

                    # determine how much of this potential surface detention/outflow will run off in this time interval
                    suro, surs = proute(psur, RTOPFG, delt60, dec, src, surs, errors)
            # END DISPOS
        # END SURFAC

        # INTFLW  to simulate interflow, irc only daily interpolation????
        if dayfg:
            kifw  = -log(IRC[step]) / (24.0 / delt60)
            ifwk2 = 1.0 - exp(-kifw)
            ifwk1 = 1.0 - (ifwk2 / kifw)

        # surface and near-surface zones of the land segment have not  been subdivided into blocks
        inflo = ifwi  + IFWLI[step]
        value = inflo + ifws
        if value > 0.00002:
            ifwo = (ifwk1 * inflo) + (ifwk2 * ifws)
            ifws = value - ifwo
        else:
            ifwo = 0.0
            ifws = 0.0
            uzs = uzs + value     # nothing worth routing-dump back to uzs

        # UZONE
        uzrat = uzs / uzsn
        uzs   = uzs + uzi + UZLI[step] + irrappV[3]  # add inflow to uzs
        perc = 0.0
        if uzrat - lzrat > 0.01:
            # simulate percolation
            perc = 0.1 * infilt * inffac * uzsn * (uzrat - lzrat)**3
            if perc > uzs: # computed value is too high so merely empty storage
                perc = uzs
                uzs = 0.0
            else:
                uzs -= perc

        # back to pwatrx
        iperc = perc + infil + LZLI[step]   # collect inflows to lower zone and groundwater

        # LZONE
        lperc = iperc + irrappV[4]
        lzi = 0.0
        if lperc > 0.0:    #  if necessary, recalculate the fraction of infiltration plus percolation which will be taken by lower zone
            if abs(lzrat - rlzrat) > 0.02 or IFRDFG:    #  it is time to recalculate
                rlzrat = lzrat
                if lzrat <= 1.0:
                    indx = 2.5 - 1.5 * lzrat
                    lzfrac = 1.0 if IFRDFG  else 1.0 - lzrat  * (1.0 / (1.0 + indx))**indx
                else:
                    indx   = 1.5 * lzrat - 0.5
                    exfact = -1.0 * IFRDFG
                    lzfrac = exp(exfact * (lzrat-1.0)) if IFRDFG else (1.0 / (1.0 + indx))**indx
            lzi = lzfrac * lperc
            lzs += lzi

        # simulate groundwater behavior - first account for the fact that iperc doesn't include lzirr
        gwi = iperc + irrappV[4] - lzi

        # GWATER
        igwi = 0.0
        agwi = 0.0
        if gwi > 0.0:
            igwi  = DEEPFR[step] * gwi
            agwi = gwi - igwi
        ainflo = agwi + AGWLI[step] + irrappV[5]  # active groundwater total inflow includes lateral inflow #$3466
        agwo = 0.0

        # evaluate groundwater recharge parameter
        if kvary > 0.0:
            # update the index to variable groundwater slope
            gwvs += ainflo
            if dayfg:
                gwvs = gwvs * 0.97 if gwvs > 0.0001 else 0.0

            # groundwater outflow(baseflow)
            if agws > 1.0e-20:
                # enough water to have outflow
                agwo = kgw * (1.0 + kvary * gwvs) * agws
                avail = ainflo + agws
                if agwo > avail:
                    errors[3] += 1     # ERRMSG3: Reduced AGWO value to available
                    agwo = avail
        elif agws > 1.0e-20:
            agwo = kgw * agws  # enough water to have outflow

        if agwo < 1.0e-12:   # beyond simgle precision math, HSPF will have zero
            agwo = 0.0

        # no remaining water - this should happen only with hwtfg=1 it may
        # happen from lateral inflows, which is a bug, in which case negative
        # values for agws should show up inthe output timeseries
        agws = agws + (ainflo - agwo)
        if agws < 0.0:
            errors[8] += 1    #ERRMSG8: Reset AGWS to zero
            agws = 0.0

        ''' # check removed - now total PERLND agreement with HSPF
        if abs(kvary) > 0.0 and gwvs > agws:
            errors[4] += 1  # ERRMSG4: Reduced GWVS to AGWS
            gwvs = agws
        '''

        # back in #$PWATRX
        TGWS[step] = agws

        # EVAPT to simulate evapotranspiration
        rempet = PET[step]  # rempet is remaining potential et - inches/ivl
        taet  = 0.0  # taet is total actual et - inches/ivlc
        baset = 0.0
        if rempet > 0.0 and basetp > 0.0:
            # in section #$etbase  there is et from baseflow
            baspet = basetp * rempet
            if baspet > agwo:
                baset = agwo
                agwo  = 0.0
            else:
                baset = baspet
                agwo -= baset
            taet   += baset
            rempet -= baset

        cepe  = 0.0
        if rempet > 0.0 and ceps > 0.0:
            # EVICEP
            if rempet > ceps:
                cepe = ceps
                ceps = 0.0
            else:
                cepe  = rempet
                ceps -= cepe
            taet   += cepe
            rempet -= cepe

        uzet  = 0.0
        if rempet > 0.0:
            # ETUZON
            # ETUZS
            if uzs > 0.001:  # there is et from the upper zone estimate the uzet opportunity
                uzrat = uzs / uzsn
                uzpet = rempet  if uzrat > 2.0  else  0.5 * uzrat * rempet
                if uzpet > uzs:
                    uzet = uzs
                    uzs  = 0.0
                else:
                    uzet = uzpet
                    uzs -= uzet
            # END UTUZA
            taet   += uzet    # these lines return to ETUZON
            rempet -= uzet
            # END ETUZON

        agwet = 0.0
        if rempet > 0.0 and agwetp > 0.0:
            # ETAGW et from groundwater determine remaining capacity
            gwpet = rempet * agwetp
            if gwpet > agws:
                agwet = agws
                agws  = 0.0
            else:
                agwet = gwpet
                agws -= agwet

            if abs(kvary) > 0.0:
                gwvs -= agwet   # update variable storage
                # if gwvs < -0.02:   # this check is commented out in HSPF v12+
                #     errors[5] += 1.0   # ERRMSG5: GWVS < -0.02, set to zero
                #     gwvs = 0.0
            taet   += agwet
            rempet -= agwet

        # et from lower zone is handled here because it must be called every interval to make sure that seasonal variation in
        # parameter lzetp and recalculation of rparm are correctly done ; simulate et from the lower zone
        # note: thj made changes in some release to the original HSPF, check carefully
        # ETLZON
        if dayfg:
            lzrat = lzs / lzsn  # it is time to recalculate et opportunity parameter rparm is max et opportunity - inches/ivl
            rparm = 0.25/(1.0-lzetp)*lzrat*delt60/24.0 if lzetp <= 0.99999 else 1.0e10
        lzet  = 0.0
        if rempet > 0.0 and lzs > 0.02:         # assume et can take place
            if lzetp >= 0.99999:          # special case - will try to draw et from whole land segment at remaining potential rate
                lzpet = rempet * lzetp
            elif VLEFG <= 1:   # usual case - desired et will vary over the whole land seg
                lzpet = 0.5*rparm if rempet > rparm else rempet*(1.0-rempet/(2.0*rparm))
                if lzetp < 0.5:
                    lzpet = lzpet * 2.0 * lzetp # reduce the et to account for area devoid of vegetation
            else:    #  VLEFG >= 2:   # et constant over whole land seg
                lzpet = lzetp*lzrat*rempet if lzrat < 1.0 else lzetp*rempet
            lzet = lzpet if lzpet < (lzs - 0.02) else lzs - 0.02
            lzs    -= lzet
            taet   += lzet
            rempet -= lzet
        # END ETLZON
        # END EVAPT

        # back in PWATRX
        TGWS[step] = agws

        # return to PWATRX
        AGWET[step] = agwet
        AGWI[step]  = agwi
        AGWO[step]  = agwo
        AGWS[step]  = agws
        BASET[step] = baset
        CEPE[step]  = cepe
        CEPS[step]  = ceps
        GWVS[step]  = gwvs
        IFWI[step]  = ifwi
        IFWO[step]  = ifwo
        IFWS[step]  = ifws
        IGWI[step]  = igwi
        INFFAC[step]= inffac
        INFIL[step] = infil
        LZET[step]  = lzet
        LZI[step]   = lzi
        LZS[step]   = lzs
        PERC[step]  = perc
        PERO[step]  = suro + ifwo + agwo
        PERS[step]  = ceps + surs + ifws + uzs + lzs + TGWS[step]
        SURI[step]  = suri
        SURO[step]  = suro
        SURS[step]  = surs
        TAET[step]  = taet
        UZET[step]  = uzet
        UZI[step]   = uzi
        UZS[step]   = uzs

        if uunits == 2:
            AGWET[step]= agwet * 25.4
            AGWI[step] = agwi * 25.4
            AGWO[step] = agwo * 25.4
            AGWS[step] = agws * 25.4
            BASET[step]= baset * 25.4
            CEPE[step] = cepe * 25.4
            CEPS[step] = ceps * 25.4
            GWVS[step] = gwvs * 25.4
            IFWI[step] = ifwi * 25.4
            IFWO[step] = ifwo * 25.4
            IFWS[step] = ifws * 25.4
            IGWI[step] = igwi * 25.4
            INFIL[step]= infil * 25.4
            LZET[step] = lzet * 25.4
            LZI[step]  = lzi * 25.4
            LZS[step]  = lzs * 25.4
            PERC[step] = perc * 25.4
            PERO[step] = (suro + ifwo + agwo) * 25.4
            PERS[step] = (ceps + surs + ifws + uzs + lzs + TGWS[step]) * 25.4
            SURI[step] = suri * 25.4
            SURO[step] = suro * 25.4
            SURS[step] = surs * 25.4
            TAET[step] = taet * 25.4
            UZET[step] = uzet * 25.4
            UZI[step]  = uzi * 25.4
            UZS[step]  = uzs * 25.4
            SUPY[step] = SUPY[step] * 25.4
            PET[step]  = PET[step] * 25.4

    # done with MASTER step
    #WATIN  = SUPY + SURLI + UZLI + IFWLI + LZLI + AGWLI+ irrapp[6]   # total input of water to the pervious land segment
    #WATDIF = WATIN - (PERO + IGWI + TAET + irdraw[2])                # net input of water to the pervious land segment
    return errors


@njit(cache=True)
def proute(psur, RTOPFG, delt60, dec, src, surs, errors):
    ''' Determine how much potential surface detention (PSUR) runs off in one simulation interval.'''
    if psur > 0.0002:
        # something is worth routing on the surface
        if RTOPFG != 1:
            # do routing the new way, estimate the rate of supply to the overland flow surface - inches/hour
            ssupr = (psur - surs) / delt60
            surse = dec * ssupr**0.6 if ssupr > 0.0 else 0.0         # determine equilibrium depth for this supply rate

            # determine runoff by iteration - newton's method,  estimate the new surface storage
            sursnw = psur
            suro    = 0.0
            for count in range(MAXLOOPS):
                if ssupr > 0.0:
                    ratio = sursnw / surse
                    fact = 1.0 + 0.6 * ratio**3 if ratio <= 1.0 else 1.6
                else:
                    ratio =  1.0e30
                    fact  = 1.6

                # coefficient in outflow equation
                ffact  = (delt60 * src * fact**1.667) * (sursnw**1.667)
                fsuro  = ffact - suro
                dfact  = -1.667 * ffact

                dfsuro = dfact / sursnw - 1.0
                if ratio <= 1.0:       #  additional term required in derivative wrt suro
                    dterm  = dfact / (fact * surse) * 1.8 * ratio**2
                    dfsuro = dfsuro + dterm
                dsuro = fsuro / dfsuro

                suro  = suro - dsuro
                if suro <= 1.0e-10:    # boundary condition- don't let suro go negative
                    suro = 0.0

                sursnw = psur - suro
                change = 0.0
                if abs(suro) > 0.0:
                    change = abs(dsuro / suro)
                if change < 0.01:
                    break
            else:
                errors[6] += 1        # ERRMSG6: Proute runoff did not converge
            surs = sursnw
        else:
            # do routing the way it is done in arm, nps, and hspx estimate the rate of supply to the overland flow surface - inches/ivl
            ssupr = psur - surs
            sursm = (surs + psur) * 0.5  # estimate the mean surface detention

            # estimate the equilibrium detention depth for this supply rate - surse
            if ssupr > 0.0:
                # preliminary estimate of surse
                dummy = dec * ssupr**0.6
                if dummy > sursm:
                    surse = dummy              # flow is increasing
                    dummy = sursm * (1.0 + 0.6 * (sursm/surse)**3)
                else:
                    dummy = sursm * 1.6  # flow on surface is at equilibrium or receding
            else:
                dummy = sursm * 1.6  # flow on the surface is receding - equilibrium detention is assumed equal to actual detention

            # check the temporary calculation of surface outflow
            tsuro = delt60 * src * dummy**1.667
            suro  = psur  if tsuro > psur  else tsuro
            surs  = 0.0   if tsuro > psur  else psur - suro
    else:
        # send what is on the overland flow plane straight to the channel
        suro = psur
        surs = 0.0
    if suro <= 1.0e-10:
        suro = 0.0     # fix bug in on pc - underflow leads to "not a number"

    return suro, surs



