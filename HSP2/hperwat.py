''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HPERWAT module to Python'''                                  #$$HPERWAT.FOR


MAXLOOPS  = 100      # newton method max loops
TOLERANCE = 0.01     # newton method exit tolerance

from numpy import zeros, sqrt, array, full, argmax, nan, where
from math import log, exp
from numba import jit
from HSP2 import initm


ERRMSG = ['PWATER: Sum of irrtgt is not one',             #ERRMSG0
          'PWATER: UZRAA exceeds UZRA array bounds',      #ERRMSG1
          'PWATER: INTGB exceeds INTGRL array bounds',    #ERRMSG2
          'PWATER: Reduced AGWO value to available',      #ERRMSG3
          'PWATER: Reduced GWVS to AGWS',                 #ERRMSG4
          'PWATER: GWVS < -0.02, set to zero',            #ERRMSG5
          'PWATER: Proute runoff did not converge',       #ERRMSG6
          'PWATER: UZI highly negative',                  #ERRMSG7
          'PWATER: Reset AGWS to zero']                   #ERRMSG8


def pwater(store, general, ui, ts):
    ''' high level driver for PERLND WATER module
    CALL: pwater(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level info (sim_start for example)
       ui is a dictionary with PLS specific HSPF UCI like data
       ts is a dictionary with PLS specific timeseries '''

    simlen = general['sim_len']                # number of simulation timesteps
    tindex = general['tindex']

    for name in ['AGWLI','IFWLI','LZLI','SURLI','UZLI']:
        if name not in ts:
            ts[name] = zeros(simlen)  # missing flows are treated as zeros
    for name in ['AIRTMP','LGTMP','PACKI','PETINP','PREC','RAINF','SNOCOV','WYIELD']:
        if name not in ts:
            ts[name] = full(simlen, nan) # insure defined, but not usable accidently
    for name in ['AGWRC','DEEPFR','FOREST','INFILT','KVARY','LZSN','PETMIN','PETMAX']:
        if name not in ts and name in ui:
            ts[name] = full(simlen, ui[name]) # Replace fixed parameters with time series if not already present

    # process optional monthly arrays to return interpolated data or constant array
    ui['flag'] = ui['VLEFG'] == 1 or ui['VLEFG'] == 3
    initm(general, ui, ts, 'flag',   'LZETPM', 'LZETP')
    initm(general, ui, ts, 'VCSFG',  'CEPSCM', 'CEPSC')
    initm(general, ui, ts, 'VIFWFG', 'INTFWM', 'INTFW')
    initm(general, ui, ts, 'VIRCFG', 'IRCM',   'IRC')
    initm(general, ui, ts, 'VNNFG',  'NSURM',  'NSUR')
    initm(general, ui, ts, 'VUZFG',  'UZSNM',  'UZSN')

    ui['ICEFG'] = general['ICEFG']  if 'ICEFG' in general else 0   # not in PWATER tables, saved by SNOW

    # Boolean array is true the first time and at midnight very day of simulation
    DAYFG = where(tindex.hour==0, True, False)
    DAYFG[0] = True
    ts['DAYFG'] = DAYFG

    ############################################################################
    if ui['HWTFG']:
        pass
        # pwahwt(general, localui, localts)    # pwater modified for high water table, low surface gradient
    else:
        errorsV = pwatrx(general, ui, ts)      # traditional HSPF HPERWAT

    # These lines output the Numba text file for debugging Numba
    #with open('numba_pwater.txt', 'w') as fnumba:
    #   pwatrx.inspect_types(file= fnumba) # ??? numba testing
    #with open('numba_pwater_proute.txt', 'w') as fnumba:
    #    proute.inspect_types(file= fnumba)
    ############################################################################

    return errorsV, ERRMSG


def pwatrx(general, ui, ts):
    ''' simulate the water budget for a pervious land segment'''
    errorsV = zeros(len(ERRMSG), dtype=int)

    delt60 = general['sim_delt'] / 60.0      # simulation interval in hours
    simlen = general['sim_len']
    DAYFG = ts['DAYFG']

    irrappV = zeros(7)
    irrcep = 0.0   # ????
    #irdraw = zeros(3)

    # like MATLAB (R), much faster to preinitialize variables
    AGWET  = ts['AGWET']  = zeros(simlen)
    AGWI   = ts['AGWI']   = zeros(simlen)
    AGWO   = ts['AGWO']   = zeros(simlen)
    AGWS   = ts['AGWS']   = zeros(simlen)
    BASET  = ts['BASET']  = zeros(simlen)
    CEPE   = ts['CEPE']   = zeros(simlen)
    CEPS   = ts['CEPS']   = zeros(simlen)
    GWVS   = ts['GWVS']   = zeros(simlen)
    IFWI   = ts['IFWI']   = zeros(simlen)
    IFWO   = ts['IFWO']   = zeros(simlen)
    IFWS   = ts['IFWS']   = zeros(simlen)
    IGWI   = ts['IGWI']   = zeros(simlen)
    INFFAC = ts['INFFAC'] = full(simlen, 1.0)                                   #$1084
    INFIL  = ts['INFIL']  = zeros(simlen)
    LZET   = ts['LZET']   = zeros(simlen)
    LZI    = ts['LZI']    = zeros(simlen)
    LZS    = ts['LZS']    = zeros(simlen)
    PERC   = ts['PERC']   = zeros(simlen)
    RPARM  = ts['RPARM']  = zeros(simlen)
    SURI   = ts['SURI']   = zeros(simlen)
    SURO   = ts['SURO']   = zeros(simlen)
    SURS   = ts['SURS']   = zeros(simlen)
    TAET   = ts['TAET']   = zeros(simlen)
    TGWS   = ts['TGWS']   = zeros(simlen)
    UZET   = ts['UZET']   = zeros(simlen)
    UZI    = ts['UZI']    = zeros(simlen)
    UZS    = ts['UZS']    = zeros(simlen)

    PERO   = ts['PERO']   = zeros(simlen)
    PERS   = ts['PERS']   = zeros(simlen)
    ans    = zeros(3)

    # PPWATR
    # get lateral inflow timeseries (set to zero array if no real data)
    AGWLI = ts['AGWLI']                                                         #$1161-1167
    IFWLI = ts['IFWLI']                                                         #$1140-1146
    LZLI  = ts['LZLI']                                                          #$1154-1160
    SURLI = ts['SURLI']                                                         #$1133-1139
    UZLI  = ts['UZLI']                                                          #$1147-1153

    # table of coordinates for functions used to evaluate upper zone behavior
    uzra   = array([0.0, 1.25, 1.50, 1.75, 2.00, 2.10, 2.20, 2.25, 2.5, 4.0])   #$76-85
    intgrl = array([0.0, 1.29, 1.58, 1.92, 2.36, 2.81, 3.41, 3.8,  7.1, 3478.]) #$87-96

    # pwat-parm1                                                                #$110
    CSNOFG = ui['CSNOFG']
    RTOPFG = ui['RTOPFG']
    UZFG   = ui['UZFG']
    VLEFG  = ui['VLEFG']
    IFFCFG = ui['IFFCFG']
    IFRDFG = ui['IFRDFG']

    # pwat-parm2                                                                #$144
    FOREST = ts['FOREST']
    LZSN   = ts['LZSN']
    INFILT = ts['INFILT'] * delt60        # convert to internal units
    lsur   = ui['LSUR']
    slsur  = ui['SLSUR']
    KVARY  = ts['KVARY']
    kgwV   = 1.0 - ts['AGWRC']**(delt60/24.0) # groundwater recession parameter #$156

    # pwat-parm3                                                                #$158
    PETMAX = ts['PETMAX']
    PETMIN = ts['PETMIN']
    infexp = ui['INFEXP']
    infild = ui['INFILD']
    DEEPFR = ts['DEEPFR']
    basetp = ui['BASETP']
    agwetp = ui['AGWETP']

    # pwat-parm5                                                                #$185
    # frozen ground parameters for corps of engineers - chicago district 10/93
    fzg  = ui['FZG']
    fzgl = ui['FZGL']

    #if RTOPFG == 3 and 'SURTAB' in ui:                                         #$229
    #    surtab = ui['SURTAB']   # FTable                                #$231

    # pwat-state1    initial conditions
    ceps = ui['CEPS']
    surs = ui['SURS']
    uzs  = ui['UZS']
    ifws = ui['IFWS']
    lzs  = ui['LZS']
    agws = ui['AGWS']
    gwvs = ui['GWVS']
                                                                                #$118-124
    CEPSC = ts['CEPSC']                                                         #$239-246
    UZSN  = ts['UZSN']                                                          #$248-255
    NSUR  = ts['NSUR']                                                          #$257-262
    INTFW = ts['INTFW']                                                         #$266-271
    IRC   = ts['IRC']                                                           #$275-282
    LZETP = ts['LZETP']                                                         #$285-292

    # initialize  variables
    rlzrat = -1.0E30                                                            #$103
    lzfrac = -1.0E30                                                            #$104
    rparm  = nan

    if agws < 0.0:        # no gw storage is active
        agws = 0.0
    TGWS[0] = agws

    # PWATER
    PETINP = ts['PETINP']                                                       #$970
    if CSNOFG:                                                                  #$974
        """ snow is being considered - allow for it find the moisture supplied
        to interception storage.  rainf is rainfall in inches/ivl. adjust for
        fraction of land segment covered by snow. wyield is the water yielded by
        the snowpack in inches/ivl. it has already been adjusted to an effective
        yield over the entire land segment """

        AIRTMP = ts['AIRTMP']                                                   #$983-995
        RAINF  = ts['RAINF']                                                    #$1005
        SNOCOV = ts['SNOCOV']                                                   #$1010
        WYIELD = ts['WYIELD']                                                   #$1015

        SUPY = RAINF * (1.0 - SNOCOV) + WYIELD                                  #$1030

        PETADJ = (1.0 - FOREST) * (1.0 - SNOCOV) + FOREST                       #$1037
        PETADJ[(AIRTMP < PETMAX) & (PETADJ > 0.5)] = 0.5                        #$1039,1045-1048
        PETADJ[ AIRTMP < PETMIN] = 0.0                                          #$1041-1044
        ts['PETADJ'] = PETADJ

        if ui['ICEFG']:                                                         #$1017
            # calculate factor to reduce infiltration and percolation to account for frozen ground
            INFFAC[:] = 1.0 - fzg * ts['PACKI']                                    #$1022
            INFFAC[INFFAC < fzgl] = fzgl
        PET = PETINP * PETADJ  #  adjust input pet                              #$1071
    else:
        # snow is not being considered all precipitation is assumed to be rain
        SUPY = ts['PREC']                                                       #$1079
        PET  = PETINP                                                           #$1082
    ts['SUPY'] = SUPY
    ts['PET']  = PET


    # adjust inffac based on soil temperature
    if IFFCFG == 2:
        INFFAC[:] = fzgl if ts['LGTMP'] <= 0.0 else 1.0                         #$1104-1110

    # PWATRX
    # MASTER LOOP
    pwater_liftedloop(AGWET, AGWI, AGWLI, AGWO, AGWS, BASET, CEPE, CEPS, CEPSC,
     DAYFG, DEEPFR, GWVS, IFRDFG, IFWI, IFWLI, IFWO, IFWS, IGWI, INFFAC, INFIL,
     INFILT, INTFW, IRC, KVARY, LZET, LZETP, LZI, LZLI, LZS, LZSN, NSUR, PERC, PERO,
     PERS, PET, RPARM, RTOPFG, SUPY, SURI, SURLI, SURO, SURS, TAET, TGWS, UZET,
     UZFG, UZI, UZLI, UZS, UZSN, VLEFG, agwetp, agws, ans, basetp, ceps, delt60,
     errorsV, gwvs, ifws, infexp, infild, intgrl, irrappV, irrcep, kgwV, lsur,
     lzfrac, lzs, rlzrat, rparm, simlen, slsur, surs, uzra, uzs)

    #WATIN  = SUPY + SURLI + UZLI + IFWLI + LZLI + AGWLI+ irrapp[6]   # total input of water to the pervious land segment
    #WATDIF = WATIN - (PERO + IGWI + TAET + irdraw[2])                # net input of water to the pervious land segment
    return errorsV


@jit(nopython=True, cache=True)
def pwater_liftedloop(AGWET, AGWI, AGWLI, AGWO, AGWS, BASET, CEPE, CEPS, CEPSC,
  DAYFG, DEEPFR, GWVS, IFRDFG, IFWI, IFWLI, IFWO, IFWS, IGWI, INFFAC, INFIL,
  INFILT, INTFW, IRC, KVARY, LZET, LZETP, LZI, LZLI, LZS, LZSN, NSUR, PERC, PERO,
  PERS, PET, RPARM, RTOPFG, SUPY, SURI, SURLI, SURO, SURS, TAET, TGWS, UZET,
  UZFG, UZI, UZLI, UZS, UZSN, VLEFG, agwetp, agws, ans, basetp, ceps, delt60,
  errorsV, gwvs, ifws, infexp, infild, intgrl, irrappV, irrcep, kgwV, lsur,
  lzfrac, lzs, rlzrat, rparm, simlen, slsur, surs, uzra, uzs):

    msupy = 0.0
    dec = nan
    src = nan
    ifwk2 = nan
    ifwk1 = nan
    for loop in range(simlen):
        dayfg  = DAYFG[loop]
        inffac = INFFAC[loop]
        kgw    = kgwV[loop]

        # These lines allow constant parameters to be replaced by timeseries
        lzetp  = LZETP[loop]
        cepsc  = CEPSC[loop]
        uzsn   = UZSN[loop]
        infilt = INFILT[loop]
        kvary  = KVARY[loop]
        lzsn   = LZSN[loop]

        oldmsupy = msupy

        # ICEPT
        ''' Simulate the interception of moisture by vegetal or other ground cover'''
        ceps = ceps + SUPY[loop] + irrcep       # add to interception storage   #$3597
        cepo = 0.0                                                              #$3603
        if ceps > cepsc:                                                        #$3598
            cepo = ceps - cepsc                                                 #$3600
            ceps = cepsc                                                        #$3601
        #END ICEPT

        #PWATRX
        suri  = cepo + SURLI[loop]                # surface inflow              #$1233
        msupy = suri + surs + irrappV[2]                                        #$1240
        lzrat = lzs / lzsn   # determine the current value of the lower zone storage ratio  #$1266

        if msupy <= 0.0:                                                        #$1278
            surs  = 0.0                                                         #$1280
            suro  = 0.0                                                         #$1281
            ifwi  = 0.0                                                         #$1282
            infil = 0.0                                                         #$1283
            uzi   = 0.0                                                         #$1284
        else:                                                                   #$1268
            # SURFAC
            ''' Distribute the water available for infiltration and runoff - units of fluxes are in./ivl'''
            ''' establish locations of sloping lines on infiltration/inflow/sur runoff
            figure.  prefix "i" refers to "infiltration" line, ibar is the mean
            infiltration capacity over the segment, internal units of infilt are inches/ivl'''

            ibar = infilt / (lzrat**infexp)                                     #$4030
            if inffac < 1.0:                                                    #$4032
                ibar = ibar * inffac                                            #$4034
            imax = ibar * infild   # infild is an input parameter - ratio of maximum to mean infiltration capacity
            imin = ibar - (imax - ibar)

            if dayfg or oldmsupy==0.0:
                dummy = NSUR[loop] * lsur
                dec = 0.00982 * (dummy / sqrt(slsur))**0.6                           #$4057
                src = 1020.0  * (sqrt(slsur) / dummy)                              #$4058

            ratio = max(1.0001, INTFW[loop] * 2.0**lzrat)                      #$4074-4077
            # DISPOSE
            # DIVISN
            if msupy <= imin:       # msupy line is entirely below other line   #$2881
                under = msupy                                                   #$2883
                over  = 0.0                                                     #$2884
            elif msupy > imax:      # msupy line is entirely above other line   #$2885
                under = (imin + imax) * 0.5                                     #$2888
                over = msupy - under                                            #$2889
            else:                   # msupy  line crosses other line            #$2890
                over = ((msupy - imin)**2) * 0.5 / (imax - imin)                #$2892
                under = msupy - over                                            #$2893
            #END DIVISN
            infil = under                                                       #$2791
            if over <= 0.0:
                surs = 0.0                                                      #$2846
                suro = 0.0                                                      #$2847
                ifwi = 0.0                                                      #$2848
                uzi  = 0.0                                                      #$2849
            else:  # there is some potential interflow inflow and maybe surface detention/outflow -- the sum of these is potential direct runoff
                pdro = over

                # determine how much of this potential direct runoff will be taken by the upper zone
                if UZFG:
                    # UZINF2 -- HSPX, ARM, NPS type calculation                #$4211
                    '''Compute inflow to upper zone during this interval, using "fully forward"
                        type algorithm  as used in HSPX,ARM and NPS.  Note:  although this method
                        should give results closer to those produced by HSPX, etc., its output will
                        be more sensitive to delt than that given by subroutine uzinf'''
                    uzrat = uzs / uzsn                                          #$4236
                    if uzrat < 2.0:                                             #$4238
                        k1 = 3.0 - uzrat                                        #$4239
                        uzfrac = 1.0 - (uzrat * 0.5) * ((1.0/(1.0 + k1))**k1)   #$4240
                    else:                                                       #$4241
                        k2 = (2.0 * uzrat) - 3.0                                #$4243
                        uzfrac = (1.0/(1.0 +  k2))**k2                          #$4244
                    uzi = pdro * uzfrac                                         #$4347
                    # END UZINF2
                else:
                    # UZINF                                                    #$2800
                    ''' Compute the inflow to the upper zone during this time interval. Do this
                        using a table look-up to handle the non-analytic integral given in
                        supporting documentation.'''

                    # find the value of the integral at initial uzra            #$4131-4204
                    uzraa = uzs  / uzsn
                    kk = argmax(uzraa < uzra)-1     # uzra[kk] < uzraa <= uzra[kk+1]
                    if kk == -1:
                        kk = 8
                        errorsV[1] += 1   # ERRMSG1: UZRAA exceeds UZRA array bounds
                    intga = intgrl[kk] + (intgrl[kk+1] - intgrl[kk]) * (uzraa - uzra[kk]) / (uzra[kk+1] - uzra[kk])
                    intgb = (pdro / uzsn) + intga

                    kk = argmax(intgb < intgrl)-1   # intgrl[kk] <= intgb < intgrl[kk+1]
                    if kk == -1:
                         errorsV[2] += 1  # ERRMSG2: INTGB exceeds INTGRL array bounds
                         kk = 8

                    uzrab = uzra[kk] + (uzra[kk+1] - uzra[kk])  * (intgb - intgrl[kk]) / (intgrl[kk+1] - intgrl[kk])
                    uzi = (uzrab - uzraa) * uzsn
                    if uzi < 0.0:
                        uzi = 0.0
                    # END UZINF
                if uzi > pdro:
                    uzi = pdro                                                  #$2807,2808
                uzfrac = uzi / pdro                                             #$2810

                # the prefix "ii" is used on variables on second divisn         #$2816-2818
                iimin = imin * ratio
                iimax = imax * ratio

                # DIVISN
                if msupy <= iimin:   # msupy line is entirely below other line  #$2881
                    over2 = 0.0                                                 #$2884
                elif msupy > iimax: # msupy line is entirely above other line   #$2885
                    over2 = msupy - (iimin + iimax) * 0.5                       #$2889
                else:                   # msupy  line crosses other line        #$2890
                    over2 = ((msupy - iimin)**2) * 0.5 / (iimax - iimin)        #$2892
                # END DIVISN

                # psur is potential surface detention/runoff                    #$2822
                psur = over2
                pifwi = pdro - psur # pifwi is potential interflow inflow       #$2825
                ifwi  = pifwi * (1.0 - uzfrac)

                if psur <= 0.0:                                                 #$2838
                    surs = 0.0                                                  #$2840
                    suro = 0.0                                                  #$2841
                else:                                                           #$2827
                    # there will be something on or running off the surface reduce it to account for the upper zone's share
                    psur = psur * (1.0 - uzfrac)                                #$2830

                    # determine how much of this potential surface detention/outflow will run off in this time interval
                    proute(psur, RTOPFG, delt60, dec, src, surs, ans)  #$2834
                    suro = ans[0]
                    surs = ans[1]
                    errorsV[6] += ans[2]
            # END DISPOS
        # END SURFAC

        # INTFLW  to simulate interflow
        if dayfg:
            kifw  = -log(IRC[loop]) / (24.0 / delt60)                               #$3672-3673
            ifwk2 = 1.0 - exp(-kifw)                                                #$3674
            ifwk1 = 1.0 - (ifwk2 / kifw)                                            #$3675

        # surface and near-surface zones of the land segment have not  been subdivided into blocks
        inflo = ifwi  + IFWLI[loop]                                             #$3680
        value = inflo + ifws                                                    #$3681
        if value > 0.00002:                                                     #$3684
            ifwo = (ifwk1 * inflo) + (ifwk2 * ifws)                             #$3685
            ifws = value - ifwo
        else:                                                                   #$3686
            ifwo = 0.0                                                          #$3688
            ifws = 0.0                                                          #$3689
            uzs = uzs + value     #nothing worth routing-dump back to uzs       #$3690
        # END INTFLW

        # UZONE
        uzrat = uzs / uzsn                                                      #$4283
        uzs   = uzs + uzi + UZLI[loop] + irrappV[3]  # add inflow to uzs        #$4285
        perc = 0.0                                                              #$4304
        if uzrat - lzrat > 0.01:                                                #$4288
            # simulate percolation
            perc = 0.1 * infilt * inffac * uzsn * (uzrat - lzrat)**3            #$4291
            if perc > uzs: # computed value is too high so merely empty storage #$4293
                perc = uzs                                                      #$4295
                uzs = 0.0                                                       #$4296
            else:
                uzs = uzs - perc                                                #$4299
        # END UZONE

        iperc = perc + infil + LZLI[loop]   # collect inflows to lower zone and groundwater #$1299

        # LZONE simulate lower zone behavior
        lperc = iperc + irrappV[4]                                              #$3731
        lzi = 0.0                                                               #$3767
        if lperc > 0.0:    #  if necessary, recalculate the fraction of infiltration plus percolation which will be taken by lower zone
            if abs(lzrat - rlzrat) > 0.02 or IFRDFG:    #  it is time to recalculate   #$3735
                rlzrat = lzrat                                                  #$3737
                if lzrat <= 1.0:                                                #$3738
                    indx = 2.5 - 1.5 * lzrat                                    #$3741
                    lzfrac = 1.0 if IFRDFG  else 1.0 - lzrat*(1.0/(1.0+indx))**indx  #$3739,3742,3745
                else:
                    indx   = 1.5 * lzrat - 0.5
                    exfact = -1.0 * IFRDFG                                      #$3754
                    lzfrac = exp(exfact*(lzrat-1.0)) if IFRDFG else (1.0/(1.0+indx))**indx #$3748,3751,3755
            lzi = lzfrac * lperc                                                #$3763
            lzs = lzs + lzi                                                     #$3764
        # END LZONE

        # simulate groundwater behavior - first account for the fact that iperc doesn't include lzirr
        gwi = (iperc + irrappV[4]) - lzi                                          #$1308

        # GWATER
        igwi = 0.0                                                              #$3460
        agwi = 0.0                                                              #$3461
        if gwi > 0.0:                                                           #$3455
            igwi = DEEPFR[loop] * gwi                                          #$3457
            agwi = gwi - igwi                                                   #$3458
        ainflo = agwi + AGWLI[loop] + irrappV[5]  # active groundwater total inflow includes lateral inflow #$3466
        agwo = 0.0                                                              #$3467

        # evaluate groundwater recharge parameter
        if abs(kvary) > 0.0:                                                         #$3469
            # update the index to variable groundwater slope
            gwvs = gwvs + ainflo                                                #$3471
            if dayfg:                                                     #$3472
                gwvs = gwvs * 0.97 if gwvs > 0.0001 else 0.0                    #$3474-3478

            # groundwater outflow(baseflow)
            if agws > 1.0e-20:                                                  #$3482
                # enough water to have outflow
                agwo = kgw * (1.0 + kvary * gwvs) * agws                        #$3484
                avail = ainflo + agws                                           #$3494
                if agwo > avail:                                                #$3495
                    errorsV[3] += 1     # ERRMSG3: Reduced AGWO value to available
                    agwo = avail                                                #$3498
        elif agws > 1.0e-20:                                                    #$3503
            agwo = kgw * agws  # enough water to have outflow                   #$3505

        if agwo < 0.0:
            agwo = 0.0

        # no remaining water - this should happen only with hwtfg=1 it may
        # happen from lateral inflows, which is a bug, in which case negative
        # values for agws should show up in the output timeseries
        agws = agws + (ainflo - agwo)                                           #$3509,3513,3519

        # check removed to fix PERLND segments 101, 185
        if agws < 0.0:
            errorsV[8] += 1    #ERRMSG8: Reset AGWS to zero
            agws = 0.0

        ''' # check removed - now total PERLND agreement with HSPF
        if abs(kvary) > 0.0 and gwvs > agws:                                         #$3533,3534
            errorsV[4] += 1  # ERRMSG4: Reduced GWVS to AGWS
            gwvs = agws
        '''
        # END GWATER

        TGWS[loop] = agws                                                       #$1312

        # EVAPT to simulate evapotranspiration
        rempet = PET[loop]  # rempet is remaining potential et - inches/ivl     #$3318
        taet  = 0.0  # taet is total actual et - inches/ivlc                    #$3316

        baset = 0.0
        if rempet > 0.0 and basetp > 0.0:                                       #$3319
            # ETBASE   et from baseflow                    #$3002
            baspet = basetp * rempet                                            #$3004
            if baspet > agwo:                                                   #$3005
                baset = agwo                                                    #$3007
                agwo  = 0.0                                                     #$3008
            else:                                                               #$3009
                baset = baspet                                                  #$3011
                agwo  = agwo - baset                                            #$3012
            taet   = taet   + baset                                             #$3016
            rempet = rempet - baset                                             #$3017                                                   #$3020
            # END ETBASE

        cepe  = 0.0
        if rempet > 0.0 and ceps > 0.0:                                         #$3325
            # EVICEP                                                            #$3393
            if rempet > ceps:
                cepe = ceps                                                     #$3398
                ceps = 0.0                                                      #$3399
            else:
                cepe = rempet                                                   #$3403
                ceps = ceps - cepe                                              #$3404
            taet   = taet   + cepe                                              #$3403
            rempet = rempet - cepe                                              #$3404
            # END EVICEP

        uzet  = 0.0
        if rempet > 0.0:                                                        #$3331
            # ETUZON
            # ETUZS                             #$3211
            if uzs > 0.001:  # there is et from the upper zone estimate the uzet opportunity #$3233
                uzrat = uzs / uzsn
                uzpet = rempet  if uzrat > 2.0  else  0.5 * uzrat * rempet      #$3237-3240
                if uzpet > uzs:
                    uzet  = uzs                                                 #$3246
                    uzs   = 0.0                                                 #$3247
                else:
                    uzet  = uzpet                                               #$3251
                    uzs   = uzs - uzet                                          #$3252
            # END UTUZA
            taet   = taet   + uzet    # these lines return to ETUZON            #$3203
            rempet = rempet - uzet                                              #$3204
            # END ETUZON

        agwet = 0.0
        if rempet > 0.0 and agwetp > 0.0:                                       #$3337
            # ETAGW et from groundwater determine remaining capacity
            gwpet = rempet * agwetp                                             #$2934
            if gwpet > agws:
                agwet = agws                                                    #$2938
                agws  = 0.0                                                     #$2939
            else:
                agwet = gwpet                                                   #$2943
                agws  = agws - agwet                                            #$2944

            if abs(kvary) > 0.0:                                                #$2948
                gwvs = gwvs - agwet   # update variable storage                 #$2950
                if gwvs < -0.02:                                                #$2961
                    errorsV[5] += 1   # ERRMSG5: GWVS < -0.02, set to zero
                    gwvs = 0.0                                                  #$2964
            taet   = taet   + agwet                                             #$2967
            rempet = rempet - agwet                                             #$2968
            # END ETAGW

        # et from lower zone is handled here because it must be called every interval to make sure that seasonal variation in
        # parameter lzetp and recalculation of rparm are correctly done ; simulate et from the lower zone
        # note: thj made changes in some releae to the original HSPF, check carefully

        #ETLZON                                                                #$3365
        if dayfg:
            lzrat = lzs / lzsn  # it is time to recalculate et opportunity parameter rparm is max et opportunity - inches/ivl
            rparm = 0.25/(1.0-lzetp)*lzrat*delt60/24.0 if lzetp <= 0.99999 else 1.0e10  #$3096-3102
        lzet  = 0.0
        if rempet > 0.0 and lzs > 0.02:         # assume et can take place
            if lzetp >= 0.99999:          # special case - will try to draw et from whole land segment at remaining potential rate                                 #$3110
                lzpet = rempet * lzetp                                          #$3114
            elif VLEFG <= 1:   # usual case - desired et will vary over the whole land seg
                lzpet = 0.5*rparm if rempet > rparm else rempet*(1.0-rempet/(2.0*rparm))  #3121-3128
                if lzetp < 0.5:                                                 #$3130
                    lzpet = lzpet * 2.0 * lzetp # reduce the et to account for area devoid of vegetation
            else:    #  VLEFG >= 2:   # et constant over whole land seg
                lzpet = lzetp*lzrat*rempet if lzrat < 1.0 else lzetp*rempet     #$3137-3143
            lzet = lzpet if lzpet < lzs-0.02 else lzs-0.02                #$3148-3155
            lzs    = lzs    - lzet                                              #$3158
            taet   = taet   + lzet                                              #$3159
            rempet = rempet - lzet                                              #$3160
        # END ETLZON
        # END EVAPT                                                       #$3167
        TGWS[loop] = agws                                                       #$1319

        AGWET[loop] = agwet
        AGWI[loop]  = agwi
        AGWO[loop]  = agwo
        AGWS[loop]  = agws
        BASET[loop] = baset
        CEPE[loop]  = cepe
        CEPS[loop]  = ceps
        GWVS[loop]  = gwvs
        IFWI[loop]  = ifwi
        IFWO[loop]  = ifwo
        IFWS[loop]  = ifws
        IGWI[loop]  = igwi
        INFIL[loop] = infil
        LZET[loop]  = lzet
        LZI[loop]   = lzi
        LZS[loop]   = lzs
        PERC[loop]  = perc
        RPARM[loop] = rparm
        SURI[loop]  = suri
        SURO[loop]  = suro
        SURS[loop]  = surs
        TAET[loop]  = taet
        UZET[loop]  = uzet
        UZI[loop]   = uzi
        UZS[loop]   = uzs
        PERO[loop]  = suro + ifwo + agwo
        PERS[loop]  = ceps + surs + ifws + uzs + lzs + TGWS[loop]
    return


@jit(nopython=True, cache=True)
def proute(psur, RTOPFG, delt60, dec, src, surs, ans):                          #$3775
    ''' Determine how much potential surface detention (PSUR) runs off in one simulation interval.'''
    err = 0
    if psur > 0.0002:                                                           #$3816
        # something is worth routing on the surface
        if RTOPFG != 1:                                                          #$3818
            # do routing the new way, estimate the rate of supply to the overland flow surface - inches/hour
            ssupr = (psur - surs) / delt60                                      #$3822
            surse = dec * ssupr**0.6 if ssupr > 0.0 else 0.0                    #$3824-3827             # determine equilibrium depth for this supply rate

            # determine runoff by iteration - newton's method,  estimate the new surface storage
            sursnw = psur                                                       #$3830
            suro    = 0.0                                                       #$3831
            for count in range(MAXLOOPS):                                            #$3834,3852,3864
                if ssupr > 0.0:
                    ratio = sursnw / surse
                    fact = 1.0 + 0.6 * ratio**3 if ratio <= 1.0 else 1.6        #$3835,3837,3839,3841
                else:
                    ratio =  1.0e30
                    fact  = 1.6

                # coefficient in outflow equation                                         #$3851
                ffact  = (delt60 * src * fact**1.667 ) *(sursnw**1.667)                                             #$3853
                fsuro  = ffact - suro                                           #$3854
                dfact  = -1.667 * ffact                                         #$3855

                dfsuro = dfact / sursnw - 1.0                                   #$3856
                if ratio <= 1.0:       #  additional term required in derivative wrt suro
                    dterm  = dfact / (fact * surse) * 1.8 * ratio**2            #$3859
                    dfsuro = dfsuro + dterm                                     #$3860
                dsuro = fsuro / dfsuro                                          #$3862

                suro = suro - dsuro                                            #$3878
                if suro <= 1.0e-10:    # boundary condition- don't let suro go negative
                    suro = 0.0                                                  #$3881

                sursnw = psur - suro                                            #$3883
                change = 0.0                                                   #$3884
                if abs(suro) > 0.0:                                            #$3886
                    change = abs(dsuro / suro)                                 #$3887
                if change < 0.01:                                              #$3890
                    break                                                      #$3890
            else:
                err += 1          # ERRMSG6: Proute runoff did not converge    #$3874
            surs = sursnw                                                       #$3892
        else:
            # do routing the way it is done in arm, nps, and hspx estimate the rate of supply to the overland flow surface - inches/ivl
            ssupr = psur - surs                                                 #$3897
            sursm = (surs + psur) * 0.5  # estimate the mean surface detention  #$3900

            # estimate the equilibrium detention depth for this supply rate - surse
            if ssupr > 0.0:                                                     #$3903
                # preliminary estimate of surse
                dummy = dec * ssupr**0.6                                        #$3905
                if dummy > sursm:                                               #$3906
                    surse = dummy              # flow is increasing             #$3908
                    dummy = sursm * (1.0 + 0.6 * (sursm/surse)**3)              #$3909
                else:
                    dummy = sursm * 1.6  # flow on surface is at equilibrium or receding  #$3912
            else:
                dummy = sursm * 1.6  # flow on the surface is receding - equilibrium detention is assumed equal to actual detention

            # check the temporary calculation of surface outflow
            tsuro = delt60 * src * dummy**1.667                                 #$3920
            suro  = psur  if tsuro > psur  else tsuro                           #$3923,3925,3928
            surs  = 0.0   if tsuro > psur  else psur - suro                     #$3923,3926,3929
    else:
        # send what is on the overland flow plane straight to the channel
        suro = psur                                                             #$3935
        surs = 0.0                                                              #$3936

    if suro <= 1.0e-10:                                                         #$3939
        suro = 0.0     # fix bug in on pc - underflow leads to "not a number"   #$3941

    #return suro, surs, err
    ans[0] = suro
    ans[1] = surs
    ans[2] = err
    return
