''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HPERSNO module into Python'''                                #$$HPERSNO.FOR


from numpy import where, zeros, minimum, full, nan
from math import sqrt, floor
from numba import jit
from HSP2 import transform

'''
if icefg and delt > 360: print errormsg
vkmfg == 1 =? KMELTM available
airtfg == 0:  AIRTMP available
snopfg == 0: dtmpg available
snopfg == 0: WINMOV and SOLRAD available

'''

ERRMSG = ['snow simulation cannot function properly with delt> 360',            #ERRMSG0
 ]

def snow(store, general, ui, ts):
    ''' high level driver for SNOW module
    CALL: snow(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with segment specific HSPF UCI like data
       ts is a dictionary with segment specific tim
    '''
    simlen  = general['sim_len']                # number of simulation timesteps
    tindex  = general['tindex']

    # so these don't need to added to EXT_SOURCES
    ts['SaturatedVaporPressureTable'] = store['/TIMESERIES/SaturatedVaporPressureTable'].values
    ts['SEASONS12'] = store['TIMESERIES/SEASONS12']

    ui['CLOUDFG'] = 'CLOUD' in ts

    for name in ['AIRTMP', 'CLOUD', 'DTMPG', 'PREC', 'SOLRAD', 'WINMOV']:
        if name not in ts:
            ts[name] = full(simlen, nan)  # insure defined, but can't be used accidently
    for name in ['CCFACT','COVIND','MGMELT','MWATER','SHADE','SNOEVP','SNOWCF']:
        if name not in ts and name in ui:     # Replace fixed parameters in HSPF with time series
            ts[name] = full(simlen, ui[name])

    # Boolean array is true the first time and at 6am every day of simulation
    HR6FG = where(tindex.hour==6, True, False)
    ts['HR6FG'] = HR6FG

    general['ICEFG'] = ui['ICEFG'] if 'ICEFG' in ui else 0  # make ICEFG available to PWATER later.

    # kmelt code below converts                                                 #$117,118,395,2247,126,138-148,2254,389,2255-2258
    if 'KMELT' not in ts:
        if 'KMELTM' in ui:
            d1 = transform(ui['KMELTM'], tindex, 'DAYVAL').values
        ts['KMELT'] = d1  if 'KMELTM' in ui else full(simlen, ui['KMELT'])

    ############################################################################
    snow_(general, ui, ts)

    #with open('numba_snow.txt', 'w') as fnumba:
    #    snow_.inspect_types(file= fnumba) # numba testing
    #with open('numba_vapor.txt', 'w') as fnumba:
    #    vapor.inspect_types(file=fnumba)
    ############################################################################

    return [0], ERRMSG



def snow_(general, ui, ts):
    ''' SNOW processing '''
    delt    = general['sim_delt']               # simulation interval in minutes
    simlen  = general['sim_len']                # number of simulation points
    delt60  = delt / 60.0                       # hours in simulation interval
    tindex  = general['tindex']

    svp = ts['SaturatedVaporPressureTable']

    skyclr = ui['SKYCLR']
    dull   = ui['DULL']
    icefg  = ui['ICEFG']
    snopfg = ui['SNOPFG']
    packf  = ui['PACKF']  # inital df.PKSNOW += df.PKICE fixed in usiReader     #$157
    packi  = ui['PACKI']                                                        #$158
    packw  = ui['PACKW']                                                        #$159
    paktmp = ui['PAKTMP']                                                       #$162
    rdenpf = ui['RDENPF']                                                       #$160
    xlnmlt = ui['XLNMLT']                                                       #$173
    tsnow  = ui['TSNOW']
    rdcsn  = ui['RDCSN']
    melev  = ui['MELEV']
    tbase  = ui['TBASE']
    covinx = ui['COVINX']                                                       #$185-187

    # get required time series
    DTMPG  = ts['DTMPG']                                                        #$285-297
    PREC   = ts['PREC']                                                         #$268-272
    WINMOV = ts['WINMOV']                                                       #$307-312
    AIRTMP = ts['AIRTMP']                                                       #$275-283
    SOLRAD = ts['SOLRAD']

    cloudfg = ui['CLOUDFG']
    CLOUD = ts['CLOUD']

    COVIND  = ts['COVIND']
    MGMELT  = ts['MGMELT'] * delt/1440.0     # time conversion                  #$136
    MWATER  = ts['MWATER']
    SNOEVP  = ts['SNOEVP']
    SNOWCF  = ts['SNOWCF']
    CCFACT  = ts['CCFACT']
    SHADE   = ts['SHADE']
    HR6FG   = ts['HR6FG']
    KMELT   = ts['KMELT'] * delt/1440.0
    if 'SEASONS' in ts:
        SEASONS = transform(ts['SEASONS'].astype(bool), tindex, 'SAME').values
    else:
        SEASONS = transform(ts['SEASONS12'].astype(bool), tindex, 'MONTHLY12').values

    # like MATLAB, much faster to preallocate arrays!
    COVINX = ts['COVINX'] = zeros(simlen)
    MELT   = ts['MELT']   = zeros(simlen) # not initialized
    NEGHTS = ts['NEGHTS'] = zeros(simlen)
    PACKF  = ts['PACKF']  = zeros(simlen)
    PACKI  = ts['PACKI']  = zeros(simlen)
    PACKW  = ts['PACKW']  = zeros(simlen)
    PAKTMP = ts['PAKTMP'] = zeros(simlen)
    PDEPTH = ts['PDEPTH'] = zeros(simlen)
    PRAIN  = ts['PRAIN']  = zeros(simlen) # not initialized
    RDENPF = ts['RDENPF'] = zeros(simlen)
    SNOCOV = ts['SNOCOV'] = zeros(simlen)
    SNOWE  = ts['SNOWE']  = zeros(simlen) # not initialized
    WYIELD = ts['WYIELD'] = zeros(simlen) # not initialized
    XLNMLT = ts['XLNMLT'] = zeros(simlen)
    SNOWF  = ts['SNOWF']  = zeros(simlen)
    RAINF  = ts['RAINF']  = zeros(simlen)
    SKYCLR = ts['SKYCLR'] = zeros(simlen)
    DULL   = ts['DULL']   = zeros(simlen)
    SNOTMP = ts['SNOTMP'] = zeros(simlen)
    DEWTMP = ts['DEWTMP'] = zeros(simlen)
    ALBEDO = ts['ALBEDO'] = zeros(simlen)

    packwc = 0.0
    prec   = 0.0
    mneghs = 0.0
    snotmp = tsnow

    '''
    if tindex.hour[0] >= 7:
        hr6update = 0
    else:
        hr6update = 1
    '''

    if packf + packw <= 1.0e-5:             # reset state variables
        #NOPACK
        covinx = 0.1 * COVIND[0]
        paktmp = 32.0                                                           #$530
        #hr6update = 1
        packf  = 0.0
        packi  = 0.0
        packw  = 0.0
        pdepth = 0.0
        rdenpf = nan
        snocov = 0.0
        neghts = 0.0                                                            #$529
        dull   = 0.0
        albedo = 0.0
        snowep = 0.0
        #END NOPACK
    else:
        if covinx < 1.0e-5:
            covinx = 0.1 * COVIND[0]
        pdepth = packf / rdenpf                                                 #$189
        snocov = packf / covinx if packf < covinx else 1.0                      #$189-194
        neghts = (32.0 - paktmp) * 0.00695 * packf                              #$203

    # needed by Numba 0.31
    dewtmp = 0.0
    rdnsn = 0.0
    compct = 0.0
    snowep = 0.0
    mostht = 0.0
    vap = 0.0
    neght = 0.0
    gmeltr = 0.0
    albedo = 0.0

    # MAIN LOOP (this can't be done functionally, must be loop)
    snow_liftedloop(AIRTMP, ALBEDO, CCFACT, CLOUD, COVIND, COVINX, DEWTMP, DTMPG,
     DULL, HR6FG, KMELT, MELT, MGMELT, MWATER, NEGHTS, PACKF, PACKI, PACKW,
     PAKTMP, PDEPTH, PRAIN, PREC, RAINF, RDENPF, SEASONS, SHADE, SKYCLR, SNOCOV,
     SNOEVP, SNOTMP, SNOWCF, SNOWE, SNOWF, SOLRAD, WINMOV, WYIELD, XLNMLT,
     albedo, cloudfg, compct, covinx, delt, delt60, dewtmp, dull, gmeltr,
     #hr6update,
     icefg, melev, mneghs, mostht, neght, neghts, packf, packi, packw,
     packwc, pdepth, prec, rdcsn, rdnsn, simlen, skyclr, snocov, snopfg, snotmp,
     snowep, svp, tbase,
     #tindex,
     tsnow, vap, xlnmlt)

    # after masterloop - work completed
    # pack, pakin, pakdif not saved since trival recalulation from saved data
    #    pack = packf + packw                                                   #$163
    #    pakin = snowf + prain                                                  #$534
    #    pakdif = pakin - (snowe + wyield)                                      #$537
    return


@jit(nopython=True, cache=True)
def snow_liftedloop(AIRTMP, ALBEDO, CCFACT, CLOUD, COVIND, COVINX, DEWTMP, DTMPG,
 DULL, HR6FG, KMELT, MELT, MGMELT, MWATER, NEGHTS, PACKF, PACKI, PACKW, PAKTMP,
 PDEPTH, PRAIN, PREC, RAINF, RDENPF, SEASONS, SHADE, SKYCLR, SNOCOV, SNOEVP,
 SNOTMP, SNOWCF, SNOWE, SNOWF, SOLRAD, WINMOV, WYIELD, XLNMLT, albedo, cloudfg,
 compct, covinx, delt, delt60, dewtmp, dull, gmeltr,
 #hr6update,
 icefg, melev,
 mneghs, mostht, neght, neghts, packf, packi, packw, packwc, pdepth, prec, rdcsn,
 rdnsn, simlen, skyclr, snocov, snopfg, snotmp, snowep, svp, tbase,
 #tindex,
 tsnow, vap, xlnmlt):
    for loop in range(simlen):
        oldprec = prec

        # pay for loop indexing once
        airtmp = AIRTMP[loop]
        prec   = PREC[loop]
        dtmpg  = DTMPG[loop]
        winmov = WINMOV[loop]
        solrad = SOLRAD[loop]

        covind = COVIND[loop]
        mgmelt = MGMELT[loop]
        mwater = MWATER[loop]
        snowcf = SNOWCF[loop]
        shade  = SHADE[loop]
        hr6fg  = HR6FG[loop]

        reltmp = airtmp - 32.0            # needed in many places, compute once

        #METEOR
        if prec > 0.0:
            fprfg = oldprec == 0.0
        else:
            fprfg = False

        hrfg = True   #??? Need once an hour flag

        if hrfg:  # estimate the dewpoint                                       #$1207,1209,1213,1217,1212,1216,1219
            dewtmp = airtmp if (prec > 0.0 and airtmp > tsnow) or dtmpg > airtmp else dtmpg

        if prec > 0.0:
            # find the temperature which divides snow from rain, and compute snow or rain fall
            if hrfg or fprfg:
                dtsnow = (airtmp - dewtmp) * (0.12 + 0.008 * airtmp)            #$1234,1236
                snotmp = tsnow + minimum(1.0, dtsnow)                           #$1236-1240
            if snopfg == 0:
                skyclr = 0.15

            if airtmp < snotmp:
                snowf = prec * snowcf                                           #$1252
                rainf = 0.0
                if  hrfg or fprfg:
                    rdnsn = rdcsn + (airtmp/100.0)**2 if airtmp > 0.0 else rdcsn #$1257-1263
            else:
                rainf = prec
                snowf = 0.0
        else:
            rainf = 0.0
            snowf = 0.0
            if snopfg == 0 and skyclr < 1.0:
                skyclr += (0.0004 * delt)
                if skyclr > 1.0:
                    skyclr = 1.0

        if snopfg == 0 and cloudfg:
            skyclr = max(0.15, 1.0 - (CLOUD[loop] / 10.0))
        #END METEOR

        if packf == 0.0 and snowf == 0.0:
            prain = 0.0
            snowe = 0.0
            wyield = 0.0
            melt = 0.0
            RAINF[loop] = prec
            continue # no pack, no new snow - no need to do rest of loop        #$525,527-530

        if packf == 0.0:      # => snowf > 0.0
            iregfg = True
            if snopfg == 0:
                dull = 0.0
        else:
            iregfg = hrfg

        #EFFPRC
        if snowf > 0.0:
            packf  += snowf                                                     #$696
            pdepth += (snowf/rdnsn)                                             #$697
            if packf > covinx:                                                  #$700
                covinx = covind if packf > covind else packf                    #$700-708
            if snopfg == 0:
                dummy = 1000.0 * snowf
                dull = 0.0 if dummy >= dull else dull - dummy
            prain = 0.0
        else:
            prain = rainf * snocov if rainf > 0.0 else 0.0
        if snopfg == 0:
            if dull < 800:
                dull += delt60
        #END EFFPRC

        #COMPAC
        if iregfg:
            rdenpf = packf / pdepth                                             #$572
            dummy = 1.0 - (0.00002 * delt60 * pdepth * (0.55 - rdenpf))         #$573-577
            compct = dummy if rdenpf < 0.55 else 1.0                            #$573-577
        if compct < 1.0:                                                        #$581
            pdepth *= compct                                                    #$582
        #END COMPAC

        if snopfg == 0:
            #SNOWEV
            if iregfg:
                vap    = vapor(svp, dewtmp)
                satvap = vapor(svp, airtmp)
                dummy = SNOEVP[loop] * 0.0002 * winmov * (satvap - vap) * snocov
                snowep = 0.0 if vap >= 6.108 else  dummy                        #$1478-1486

            if snowep >= packf:
                snowe = packf
                pdepth = 0.0
                packi  = 0.0
                packf  = 0.0
            else:
                pdepth *=  (1.0 - snowep / packf)                               #$1495,1500
                packf -= snowep                                                 #$1497,1501
                snowe = snowep                                                  #$1496
                if packi > packf:                                               #$1503
                    packi = packf                                               #$1504
            #END SNOWEV
        else:
            snowe = 0.0

        if iregfg:
            if snopfg == 0:
                #HEXCHR
                factr = CCFACT[loop] * 0.00026 * winmov                         #$871
                dummy = 8.59 *  (vap - 6.108)                                   #$874-880
                condht = dummy * factr if vap > 6.108 else 0.0                  #$874-880

                dummy = reltmp * (1.0 - 0.3 * melev/10000.0) * factr            #$883-889
                convht = dummy if airtmp > 32.0 else 0.0

                #ALBEDO
                dummy = sqrt(dull/24.0)
                summer = float(max(0.45, 0.80 - 0.10*dummy))
                winter = float(max(0.60, 0.85 - 0.07*dummy))
                albedo = summer if SEASONS[loop] else winter                    #$2200-2210
                #END ALBEDO

                k = (1.0 - shade) * delt60
                long1 = (shade * 0.26 * reltmp) + k * (0.20 * reltmp - 6.6)     #$902
                long2 = (shade * 0.20 * reltmp) + k * (0.17 * reltmp - 6.6)     #$905
                long_ = long1 if reltmp > 0.0 else long2                        #$901-907
                if long_ < 0.0:                       # back radiation          #$911-913
                    long_ *= skyclr

                short = solrad  * (1.0 - albedo) * (1.0 - shade)                #$314-316,897
                mostht = (short + long_)/203.2 + convht + condht                #$917,920
                #END HEXCHR
            else:
                #DEGDAY
                mostht = KMELT[loop]* (airtmp - tbase)
                #END DEGDAY

        rnsht = reltmp * rainf / 144.0 if rainf > 0.0 else 0.0
        sumht = mostht + rnsht                                                  #$405-409,416
        if snocov < 1.0:                                                        #$417
            sumht = sumht * snocov                                              #$418
        paktmp = 32.0 if neghts == 0.0 else 32.0 - neghts / (0.00695 * packf)   #$425-429

        #COOLER
        if iregfg:
            mneghs = 0.0 if reltmp > 0.0 else -reltmp * 0.00695 * packf/2.0     #$622-627
            neght  = 0.0 if paktmp <= airtmp else 0.0007 * (paktmp - airtmp) * delt60 #$630-634

        if sumht < 0.0:
            if paktmp > airtmp:                                                 #$639
                neghts = min(mneghs, neghts + neght)                            #$641-643
            sumht = 0.0                                                         #$650
        #END COOLER

        if neghts > 0.0:                                                        #$440
            #WARMUP
            if sumht > 0.0:                                                     #$1607
                if sumht > neghts:                                              #$1609
                    sumht -= neghts                                             #$1611
                    neghts = 0.0                                                #$1612
                else:                                                           #$1613
                    neghts -= sumht                                             #$1615
                    sumht = 0.0                                                 #$1616

            if prain > 0.0:                                                     #$1625
                if prain > neghts:                                              #$1627
                    rnfrz  = neghts
                    packf += rnfrz                                              #$1631
                    neghts = 0.0                                                #$1632
                else:                                                           #$1633
                    rnfrz  = prain
                    neghts -= prain                                             #$1637
                    packf  += prain                                             #$1638

                if packf > pdepth:                                              #$1643
                    pdepth = packf                                              #$1645
            else:                                                               #$1646
                rnfrz = 0.0                                                     #$1647
            #END WARMUP
        else:
            rnfrz = 0.0                                                         #$449

        #MELTER
        if sumht >= packf:                                                      #$1112
            melt   = packf                                                      #$1114
            packf  = 0.0                                                        #$1115
            pdepth = 0.0                                                        #$1116
            packi  = 0.0                                                        #$1117
        elif sumht > 0.0:
            melt    = sumht                                                     #$1121
            pdepth *= (1.0 - melt / packf)                                      #$1122
            packf  -= melt                                                      #$1123
            if packi > packf:
                packi = packf
        else:                                                                   #$1128
            melt = 0.0                                                          #$1130
        #END MELTER

        #LIQUID
        if iregfg and packf > 0.0:                                              #$1050
            rdenpf = packf / pdepth                                             #$1053
            if rdenpf <= 0.6:
                packwc = mwater
            else:
                dummy = 3.0 - 3.33 * rdenpf                                     #$1059
                packwc = mwater * dummy if dummy >= 0.0 else 0.0                #$1054-1065

        pwsupy = packw + melt + prain - rnfrz                                   #$1072
        mpws = packwc * packf                                                   #$1074
        if (pwsupy - mpws) > (0.01 * delt60):                                   #$1075
            wyield = pwsupy - mpws                                              #$1078
            packw = mpws                                                        #$1079
        else:
            packw = pwsupy                                                      #$1083
            wyield = 0.0                                                        #$1084
        #END LIQUID

        if icefg:                                                               #$101,466
            #ICING
            '''
            if tindex.hour[loop] >= 7:
                if hr6update <> 0:
                    if snocov < 1.0:
                        xlnem = -reltmp * 0.01                                          #$963,965
                        if xlnem > xlnmlt:                                              #$969
                            xlnmlt = xlnem                                              #$970
                    hr6update = 0
            else:
                hr6update = 1
            '''
            if hr6fg and snocov < 1.0:
                xlnem = -reltmp * 0.01                                          #$963,965
                if xlnem > xlnmlt:                                              #$969
                    xlnmlt = xlnem                                              #$970

            if wyield > 0.0 and xlnmlt > 0.0:                                   #$987
                if wyield < xlnmlt:                                             #$990
                    freeze  = wyield
                    xlnmlt -= wyield                                            #$994
                    wyield  = 0.0                                               #$995
                else:
                    freeze  = xlnmlt
                    wyield -= xlnmlt                                            #$999
                    xlnmlt  = 0.0                                               #$1001
                packf  += freeze                                                #$1005
                packi  += freeze                                                #$1006
                pdepth += freeze                                                #$1008
            #END ICING

        #GMELT
        if iregfg:
            if paktmp >= 32.0:                                                  #$782
                gmeltr = mgmelt                                                 #$783
            elif paktmp > 5.0:                                                  #$784
                gmeltr = mgmelt * (1.0 - 0.03 * (32.0 - paktmp))                #$786
            else:                                                               #$787
                gmeltr = mgmelt * 0.19                                          #$789
        if packf <= gmeltr:                                                     #$797
            wyield = wyield + packf + packw                                     #$799
            packf = 0.0
            packi = 0.0
            packw = 0.0
            pdepth = 0.0
            neghts = 0.0
        else:
            dummy = 1.0 - (gmeltr / packf)                                      #$810
            packw  += gmeltr                                                    #$802,807
            pdepth *= dummy                                                     #$803,811
            neghts *= dummy                                                     #$804,812
            packf  -= gmeltr                                                    #$800,813
            packi = packi - gmeltr if packi > gmeltr else 0.0                   #$801
        #END GMELT

        if packf > 0.005:                                                       #$498
            rdenpf = packf / pdepth                                             #$500
            paktmp = 32.0 if neghts == 0.0 else 32.0 - neghts/(0.00695 * packf) #$504-508
            snocov = packf / covinx if packf < covinx else 1.0                  #$510-514
        else:
            melt   += packf                                                     #$517
            wyield += packf + packw                                             #$518

            #NOPACK
            covinx = 0.1 * covind                                               #$1356
            paktmp = 32.0                                                       #$1360
            rdenpf = nan                                                        #$1355
            #hr6update = 1
            packf  = 0.0                                                        #$1350
            packi  = 0.0                                                        #$1351
            packw  = 0.0                                                        #$1352
            xlnmlt = 0.0                                                        #$1358
            snocov = 0.0                                                        #$1357
            neghts = 0.0                                                        #$1361
            pdepth = 0.0                                                        #$1354
            prain  = 0.0                                                        #$527
            snowe  = 0.0                                                        #$528
            mneghs = nan
            #END NOPACK

        # save calculations
        COVINX[loop] = covinx
        MELT[loop]   = melt
        NEGHTS[loop] = neghts
        PACKF[loop]  = packf
        PACKI[loop]  = packi
        PACKW[loop]  = packw
        PAKTMP[loop] = paktmp
        PDEPTH[loop] = pdepth
        PRAIN[loop]  = prain
        RDENPF[loop] = rdenpf
        SNOCOV[loop] = snocov
        SNOWE[loop]  = snowe
        WYIELD[loop] = wyield
        XLNMLT[loop] = xlnmlt

        SNOTMP[loop] = snotmp
        RAINF[loop]  = rainf
        SNOWF[loop]  = snowf
        DEWTMP[loop] = dewtmp
        SKYCLR[loop] = skyclr
        DULL[loop]   = dull
        ALBEDO[loop] = albedo
    return


@jit(nopython=True, cache=True)
def vapor(svp, temp):
    indx = (temp + 100.0) * 0.2 - 1.0
    lower = int(floor(indx))
    if lower < 0:
        return 1.005
    upper = lower + 1
    if upper > 39:
        return 64.9
    return float(svp[lower] + (indx-lower) * (svp[upper] - svp[lower]))
