''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

no category version of HSPF HRCHHYD'''

''' Development Notes:
  Categories not implimented in this version
  Irregation only partially implimented in this version
'''

TOLERANCE = 0.001
MAXLOOPS  = 100

from numpy import zeros, any, full, nan
from math import sqrt, log10
from numba import jit
from HSP2  import initm

ERRMSG = ['HYDR: SOLVE equations are indeterminate',             #ERRMSG0
          'HYDR: extrapolation of rchtab will take place',       #ERRMSG1
          'HYDR: SOLVE trapped with an oscillating condition',   #ERRMSG2
          'HYDR: Solve did not converge',                        #ERRMSG3
          'HYDR: Solve converged to point outside valid range']  #ERRMSG4

# units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
VFACT  = 43560.0                                                                #$162
AFACT  = 43560.0                                                                #$164
VFACTA = 1.0/VFACT                                                              #$4076
LFACTA = 1.0                                                                    #$4078
AFACTA = 1.0/AFACT                                                              #$4081
SFACTA = 1.0                                                                    #$4082
TFACTA = 1.0                                                                    #$4083

# physical constants (English units)
GAM = 62.4
GRAV = 32.2
AKAPPA = 0.4              # von karmen constant             #$155

def hydr(store, general, ui, ts):
    ''' high level driver for RCHRES Hydr code
    CALL: hydr(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with RID specific HSPF UCI like data
       ts is a dictionary with RID specific timeseries'''

    simlen = general['sim_len']
    nexits = ui['NEXITS']

    # COLIND timeseries might come in as COLIND, COLIND0, etc. otherwise init table
    COLINDM = zeros((simlen, nexits))
    COLINDM[0:simlen,...] = ui['ODFVF'][0:nexits]
    COLINDM[0,...] = ui['COLIN'][0:nexits]
    colindkeys = [k for k in ts if 'COLIND' in k]
    for i,n in  [(0 if x=='COLIND' else int(x[6:])-1, x) for x in colindkeys]:
        COLINDM[:,i] = ts[n]
    ts['COLINDM'] = COLINDM

    # OUTDGT timeseries might come in as OUTDGT, OUTDGT0, etc. otherwise use init table
    OUTDGTM = zeros((simlen, nexits))
    OUTDGTM[0:simlen,...] = ui['OUTDG'][0:nexits]
    outdgtkeys = [k for k in ts if 'OUTDGT' in k]
    for i,n in  [(0 if x=='OUTDGT' else int(x[6:])-1, x) for x in outdgtkeys]:
        OUTDGTM[:,i] = ts[n]
    ts['OUTDGTM'] = OUTDGTM

    for name in ['SOLRAD','CLOUD','DEWTEMP','GATMP','WIND']:
        if name not in ts:
            ts[name] = full(simlen, nan)    # optional - defined, but can't used accidently

    for name in ['IVOL','POTEV','PREC']:
        if name not in ts:
            ts[name] = zeros(simlen)        # optional timesereis

    ui['CONVF'] = 1.0
    initm(general, ui, ts, 'VCONFG', 'CONVFM',  'CONVF')                   #$201,210-212

    ############################################################################
    errorsV = hydr_(general, ui, ts)              # run reaches simulation code

    # The following lines output the Numba text files for debugging Numba implementation
    #with open('numba_hydr.txt', 'w') as fnumba:
    #    hydr_.inspect_types(file= fnumba)
    #with open('numba_hydr_demand.txt', 'w') as fnumba:
    #    demand.inspect_types(file= fnumba)
    #with open('numba_hydr_auxil.txt', 'w') as fnumba:
    #    auxil.inspect_types(file= fnumba)
    #with open('numba_hydr_fndrow.txt', 'w') as fnumba:
    #    fndrow.inspect_types(file=fnumba)
    ############################################################################

    if ui['NEXITS'] == 1:
        del ts['O']
        del ts['OVOL']
    return errorsV, ERRMSG


def hydr_(general, ui, ts):
    '''find the state of the reach or reservoir at the end of the time interval
    and the outflows during the interval'''

    errorsV = zeros(len(ERRMSG), dtype=int) # array for error counts
    simlen  = general['sim_len']            # number of simulation steps
    delts   = general['sim_delt'] * 60.0    # seconds in simulation intervals
    nexits  = ui['NEXITS']

    # Get ready for main loop, #$$PHYDR
    # extract key columns of FTable for this segment for faster access (1d vs. 2d)
    rchtab   = ui['rchtab']                                                     #$184-192
    volumeFT = rchtab['Volume'].values * VFACT                                  #$1484
    depthFT  = rchtab['Depth'].values
    sareaFT  = rchtab['Area'].values   * AFACT                                  #$1482
    rowsFT   = rchtab.values
    nrows, ncols = rchtab.shape

    # hydr-parm1
    AUX1FG = ui['AUX1FG']           # True means DEP, SAREA will be computed
    ODFVF  = ui['ODFVF'][0:nexits]  # nexits length, volume dependent flow flags
    ODGTF  = ui['ODGTF'][0:nexits]
    nodfv  = bool(any(ODFVF))                                                    #$124,126-130

    # hydr-parm2, other values read at point of use
    CONVF  = ts['CONVF']                                                        #$201,210-212
    convf  = CONVF[0]
    ks     = ui['KS']
    coks   = 1 - ks                                                             #$198
    facta1 = 1.0 / (coks * delts)                                               #$199

    # Initializations prior to main loop (over simulation time)
    COLIND = ts['COLINDM']
    OUTDGT = ts['OUTDGTM']
    IVOL   = ts['IVOL']  * VFACT       # or sum civol, zeros if no inflow ???
    POTEV  = ts['POTEV'] / 12.0
    PREC   = ts['PREC']  / 12.0
    dep    = 0.0  #nan
    sarea  = 0.0  #nan
    avvel  = nan

    # faster to preallocate arrays - like MATLAB)
    o      = zeros(nexits)
    od1    = zeros(nexits)
    od2    = zeros(nexits)
    odz    = zeros(nexits)  # Numba 0.31 needed this
    ovol   = zeros(nexits)
    oseff  = zeros(nexits)
    outdgt = zeros(nexits)
    colind = zeros(nexits)

    colind[:] = COLIND[0,0:nexits]
    outdgt[:] = OUTDGT[0,0:nexits]

    PRSUPY = ts['PRSUPY'] = zeros(simlen)
    RO     = ts['RO']     = zeros(simlen)
    O      = ts['O']      = zeros((simlen, nexits))
    ROVOL  = ts['ROVOL']  = zeros(simlen)
    OVOL   = ts['OVOL']   = zeros((simlen, nexits))
    VOL    = ts['VOL']    = zeros(simlen)
    VOLEV  = ts['VOLEV']  = zeros(simlen)
    IRRDEM = ts['IRRDEM'] = zeros(simlen)

    if AUX1FG:                                                                  #$661
        DEP   = ts['DEP']   = zeros(simlen)                                                   #$671-673
        SAREA = ts['SAREA'] = zeros(simlen)                                                   #$671-673
        USTAR = ts['USTAR'] = zeros(simlen)
        TAU   = ts['TAU']   = zeros(simlen)

    funct = ui['FUNCT'][0:nexits]
    zeroindex = fndrow(0.0, volumeFT)                                           #$1126-1127
    topvolume = volumeFT[-1]

    # find row index that brackets the VOL per comment in lines #1130-1131  #$$DISCH
    vol = ui['VOL'] * VFACT   # hydr-init, initial volume of water              #$247
    if vol >= topvolume:
        errorsV[1] += 1      # ERRMSG1: extrapolation of rchtab will take place

    indx = fndrow(vol, volumeFT)
    if nodfv:  # simple interpolation, the hard way!!                            #$1124
        v1 = volumeFT[indx]                                                     #$1136
        v2 = volumeFT[indx+1]                                                   #$1137
        rod1,od1[:] = demand(v1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt, ODGTF)  #$1140-1142
        rod2,od2[:] = demand(v2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)  #$1145-1147
        a1 = (v2 - vol) / (v2 - v1)                                             #$1151
        o[:] = (a1 * od1)  + ((1.0 - a1) * od2)                                         #$1152-1156
        ro   = (a1 * rod1) + ((1.0 - a1) * rod2)                                         #$1152-1156
    else:        # no outflow demands have an f(vol) component
        ro,o[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)  #$1159-1160

    # back to PHYDR
    if AUX1FG >= 1:                         # initial depth and surface area
        dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errorsV)

    # hydr-irrig
    irexit = int(ui['IREXIT']) -1    # exit no for irrigation withdrawals, 0 based   #$221
    irrdemV = zeros(simlen)
    irminv = ui['IRMINV']                                                       #$222
    rirwdl = 0.0                                                                #$237
    #rirdem = 0.0                                                               #$238
    #rirsht = 0.0                                                               #$239
    irrdem = 0.0

    # NEEDED BY NUMBA 0.31 to avoid including ts,ui in lifted loop - which fails
    LEN    = ui['LEN']
    AUX2FG = ui['AUX2FG']
    AUX3FG = ui['AUX3FG']
    LKFG   = ui['LKFG']                # flag, 1:lake, 0:stream          #$3791
    DB50   = ui['DB50'] / 12.0     # mean diameter of bed material  #$152
    DELTH  = ui['DELTH']                                #$153

    # simulation main loop - HYDR (except where noted)      #$1841
    hydr_liftedloop(AUX1FG, AUX2FG, AUX3FG, COLIND, CONVF, DB50, DELTH, DEP, IVOL,
    LEN, LKFG, O, ODGTF, OUTDGT, OVOL, POTEV, PREC, PRSUPY, RO, ROVOL, SAREA, TAU,
    USTAR, VOL, VOLEV, avvel, coks, colind, delts, depthFT, errorsV, facta1,
    funct, indx, irexit, irminv, irrdem, irrdemV, ks, nexits, nodfv, nrows, o,
    od1, od2, odz, oseff, outdgt, ovol, rirwdl, ro, rowsFT, sarea, sareaFT,
    simlen, topvolume, vol, volumeFT, zeroindex)

    # END MAIN LOOP - save computed timeseries & convert to expected units
    if nexits > 1:
        O    *= SFACTA * LFACTA
        OVOL *= VFACTA
    PRSUPY *= AFACTA
    RO     *= SFACTA * LFACTA
    ROVOL  *= VFACTA
    VOL    *= VFACTA
    VOLEV  *= VFACTA

    if AUX1FG:   #save results except stage, avdep, hrad, and twid which can be trivially calculated
        SAREA *= AFACTA
        if AUX3FG:
            USTAR *= LFACTA
            TAU   *= TFACTA
    if irexit >= 0:
        IRRDEM = irrdemV
    return errorsV


@jit(nopython=True, cache=True)
def hydr_liftedloop(AUX1FG, AUX2FG, AUX3FG, COLIND, CONVF, DB50, DELTH, DEP, IVOL,
 LEN, LKFG, O, ODGTF, OUTDGT, OVOL, POTEV, PREC, PRSUPY, RO, ROVOL, SAREA, TAU,
 USTAR, VOL, VOLEV, avvel, coks, colind, delts, depthFT, errorsV, facta1, funct,
 indx, irexit, irminv, irrdem, irrdemV, ks, nexits, nodfv, nrows, o, od1, od2,
 odz, oseff, outdgt, ovol, rirwdl, ro, rowsFT, sarea, sareaFT, simlen, topvolume,
 vol, volumeFT, zeroindex):

    for loop in range(simlen):
        convf  = CONVF[loop]
        outdgt[:] = OUTDGT[loop,0:nexits]
        colind[:] = COLIND[loop,0:nexits]
        roseff = ro                                                             #$3275,1979
        oseff[:] = o                                                              #$3276

        # vols, sas variables and their initializations  not needed.            #$1977,1978
        if irexit >= 0:             # irrigation exit is set                    #$1910
            if rirwdl > 0.0:  # equivalent to OVOL for the irrigation exit      #$1913
                vol = irminv if irminv > vol - rirwdl else vol - rirwdl         #$1917-1921
                if vol >= volumeFT[-1]:
                    errorsV[1] += 1 # ERRMSG1: extrapolation of rchtab will take place

                # DISCH with hydrologic routing           #$1948
                # find row index that brackets the VOL per comment in lines     #1130-1131
                indx = fndrow(vol, volumeFT)
                vv1 = volumeFT[indx]
                rod1,od1[:] = demand(vv1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                vv2 = volumeFT[indx+1]                                          #$1137
                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                aa1 = (vv2 - vol) / (vv2 - vv1)                                 #$1151

                ro   = (aa1 * rod1) + ((1.0 - aa1) * rod2)                                         #$1152-1156
                o[:] = (aa1 * od1)  + ((1.0 - aa1) * od2)             #$1152,1155

                # back to HYDR
                if AUX1FG >= 1:     # recompute surface area and depth          #$1949
                    dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errorsV)  #$1951-1959,1962
            else:
                irrdem =  0.0                                                   #$1965
            #o[irexit] = 0.0                                                   #$1980-1992  #???? not used anywhere, check if o[irexit]

        prsupy = PREC[loop] * sarea                                            #$2038,2041,2044,2080,2082
        volt   = vol + IVOL[loop] + prsupy                                   #$2022,2039,2042, 2044
        volev = 0.0
        if AUX1FG:                  # subtract evaporation                  #$2085
            volpev = POTEV[loop] * sarea
            if volev >= volt:
                volev = volt
                volt = 0.0
            else:
                volev = volpev
                volt -= volev

        # ROUTE/NOROUT  calls                           #$2157, 3019-3022
        # common code
        volint = volt - (ks * roseff * delts)    # find intercept of eq 4 on vol axis #$3288
        if volint < (volt * 1.0e-5):                                       #$3289
            volint = 0.0                                                        #$3291
        if volint <= 0.0:  #  case 3 -- no solution to simultaneous equations   #$3294, 3025-3042
            indx  = zeroindex
            vol   = 0.0                                                         #$3298
            ro    = 0.0                                                         #$3299
            o[:]  = 0.0                                                         #$3300
            rovol = volt                                                        #$3302
            if roseff > 0.0:                                                    #$3203-3311
                ovol[:] = (rovol/roseff) * oseff
            else:
                ovol[:] = rovol / nexits

        else:   # case 1 or 2
            oint = volint * facta1      # == ointsp, so ointsp variable dropped                           #$3316,3317
            if nodfv:
                # ROUTE
                rodz,odz[:] = demand(0.0, rowsFT[zeroindex,:], funct, nexits, delts, convf, colind,  outdgt, ODGTF)   #$3320-3323
                if oint > rodz:
                    # SOLVE,  Solve the simultaneous equations for case 1-- outflow demands can be met in full
                    # premov will be used to check whether we are in a trap, arbitrary value
                    premov = -20                                                #$3471
                    move   = 10

                    vv1 = volumeFT[indx]                                       #$3478
                    rod1,od1[:] = demand(vv1, rowsFT[indx, :], funct, nexits, delts, convf,colind, outdgt, ODGTF) #$3479-3481
                    vv2 = volumeFT[indx+1]                                     #$3485
                    rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF) #$3486-3488
                    while move != 0:                                            #$3474
                        facta2 = rod1 - rod2                                    #$3496
                        factb2 = vv2 - vv1                                      #$3497
                        factc2 = vv2 * rod1 - vv1 * rod2                        #$3498
                        det = facta1 * factb2 - facta2                          #$3501
                        if det == 0.0:                                          #$3502
                            det = 0.0001                                        #$3513
                            errorsV[0] += 1  # ERRMSG0: SOLVE is indeterminate

                        vol = max(0.0, (oint * factb2 - factc2 ) / det)           #$3515-3520
                        if vol > vv2:                                             #$3527
                            if indx >= nrows-2:                                 #$3532
                                if vol > topvolume:
                                    errorsV[1] += 1 # ERRMSG1: extrapolation of rchtab will take place
                                move = 0
                            else:
                                move   = 1                                      #$3529
                                indx  += 1                                   #$3525
                                vv1    = vv2
                                od1[:] = od2
                                rod1   = rod2
                                vv2    = volumeFT[indx+1]                       #$3485
                                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF) #$3486-3488
                        elif vol < vv1:                                           #$3548
                            indx  -= 1                                          #$3551
                            move   = -1                                         #$3550
                            vv2    = vv1
                            od2[:] = od1
                            rod2   = rod1
                            vv1    = volumeFT[indx]                             #$3478
                            rod1,od1[:] = demand(vv1, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF) #$3479-3481
                        else:
                            move = 0                                            #$3551,3552,3554

                        # check whether algorithm is in a trap, yo-yoing back and forth
                        if move + premov == 0:                                  #$3558
                            errorsV[2] += 1      # ERRMSG2: oscillating trap
                            move = 0
                        premov = move                                           #$3568

                    ro = oint - facta1 * vol                                   #$3574
                    if  vol < 1.0e-5:                                           #$3578
                        ro  = oint                                              #$3583
                        vol = 0.0                                               #$3584
                    if ro < 1.0e-10:                                            #$3587
                        ro  = 0.0                                               #$3589
                    if ro <= 0.0:
                        o[:] = 0.0
                    else:                                                       #$3597
                        diff  = vol - vv1                                       #$3598
                        factr = 0.0 if diff < 0.01 else  diff / (vv2 - vv1)     #$3599-3605
                        o[:]  = od1 + (od2 - od1) * factr                       #$3606-3608

                else:
                    # case 2 -- outflow demands cannot be met in full           #$3336, 3043-3083
                    ro  = 0.0                                                                   #$3343
                    for i in range(nexits):                                                    #$3344
                        tro  = ro + odz[i]                                                      #$3345
                        if tro <= oint:
                            o[i] = odz[i]
                            ro = tro
                        else:
                            o[i] = oint - ro
                            ro = oint
                    vol = 0.0
                    indx = zeroindex

            else:
                # NOROUT
                rod1,od1[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF) #$3050-3051
                if oint >= rod1: #case 1 -outflow demands are met in full      #$3054
                    ro   = rod1                                                  #$3056
                    vol  = volint - coks * ro * delts
                    if vol < 1.0e-5:              #$3057,3058,3060
                        vol = 0.0
                    o[:] = od1                                                  #$3062-3064
                else:    # case 2 -outflow demands cannot be met in full        #$3065
                    ro  = 0.0                                                                   #$3343
                    for i in range(nexits):                                                    #$3344
                        tro  = ro + odz[i]                                                      #$3345
                        if tro <= oint:
                            o[i] = odz[i]
                            ro = tro
                        else:
                            o[i] = oint - ro
                            ro = oint
                    vol = 0.0
                    indx = zeroindex

            # common  ROUTE/NOROUT code
            #  an irrigation demand was made before routing                         #$3095-3100
            if  (irexit >= 0) and (irrdem > 0.0):    #  an irrigation demand was made before routing  #$3358
                oseff[irexit] = irrdem                                              #$3360
                o[irexit]     = irrdem                                              #$3361
                roseff       += irrdem                                     #$3362
                ro           += irrdem                                        #$3363
                irrdemV[loop] = irrdem

            # estimate the volumes of outflow                                       #$3105-3107
            ovol[:] = (ks * oseff   + coks * o)  * delts                            #$3368-3371
            rovol   = (ks * roseff  + coks * ro) * delts                            #$3367

        # HYDR
        if AUX1FG:   # compute final depth, surface area                        #$2247
            if vol >= topvolume:
                errorsV[1] += 1       # ERRMSG1: extrapolation of rchtab
            indx = fndrow(vol, volumeFT)
            dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errorsV)   #$2251-2253
            DEP[loop]   = dep
            SAREA[loop] = sarea
        PRSUPY[loop] = prsupy
        O[loop,:]    = o
        RO[loop]     = ro
        OVOL[loop,:] = ovol
        ROVOL[loop]  = rovol
        VOLEV[loop]  = volev
        VOL[loop]    = vol

        length = LEN * 5280.0  # length of reach, hydr-parm2

        if vol > 0.0 and sarea > 0.0:
            twid  = sarea / length
            avdep = vol / sarea
        elif AUX1FG == 2:
            twid = sarea / length
            avdep = 0.0
        else:
            twid = 0.0
            avdep = 0.0

        if AUX2FG:
            avvel = (length * ro / vol) if vol > 0.0 else 0.0
        if AUX3FG:
            if avdep > 0.0:
                # these lines replace SHEAR; ustar (bed shear velocity), tau (bed shear stress)
                if LKFG:              # flag, 1:lake, 0:stream          #$3791
                    ustar = avvel / (17.66 + log10(avdep/(96.5*DB50))) * 2.3/AKAPPA   #$3794,3799
                    tau   =  GAM/GRAV * ustar**2              #3796
                else:
                    hrad = (avdep*twid)/(2.0*avdep + twid)  if avdep > 0.0 else 0.0# hydraulic radius, manual eq (41
                    slope = DELTH / length                                #$153
                    ustar = sqrt(GRAV * slope * hrad)          #$3799
                    tau = (GAM * slope) * hrad                 #$3801
            else:
                ustar = 0.0
                tau = 0.0
            USTAR[loop] = ustar
            TAU[loop] = tau
    return


@jit(nopython=True, cache=True)
def fndrow(v, volFT):
    ''' finds highest index in FTable volume column whose volume  < v'''
    for indx,vol in enumerate(volFT):
        if v < vol:
            return indx-1
    return len(volFT) - 2


@jit(nopython=True, cache=True)
def demand(vol, rowFT, funct, nexits, delts, convf, colind, outdgt, ODGTF):            #$2283
    od = zeros(nexits)                                                          #$2421
    for i in range(nexits):
        col = colind[i]
        icol = int(col)
        if icol != 0:
            diff = col - float(icol)
            if diff >= 1.0e-6:                                                  #$2352
                _od1 = rowFT[icol-1]                                             #$2354
                od[i] = _od1 + diff * (_od1 - rowFT[icol]) * convf                #$2356
            else:                                                               #$2341 - no interpolation ???
                od[i] = rowFT[icol-1] * convf                                          #$2359

        icol = int(ODGTF[i])              #$2389          #$2390-2430
        if icol != 0:
            if col > 0.0:   # both f(time) and f(vol)
                a = od[i]
                b = outdgt[icol-1]
                c = (vol - b)  / delts  #???
                if   funct[i] == 1: od[i] = min(a,b)  # pbd fix in arguments to min function
                elif funct[i] == 2: od[i] = max(a,b)
                elif funct[i] == 3: od[i] = a+b
                elif funct[i] == 4: od[i] = max(a,c)
            else:
                od[i] = outdgt[icol-1]  # pbd added for f(time) only

    return od.sum(), od


@jit(nopython=True, cache=True)
def auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errorsV):
    ''' Compute depth and surface area'''
    if vol > 0.0:                                                               #$3747
        sa1  = sareaFT[indx]                                                    #$3674
        a    = sareaFT[indx+1]  - sa1                                                        #$3676
        b    = 2.0 * sa1
        vol1 = volumeFT[indx]                                                   #$3678
        c = -((vol - vol1) / (volumeFT[indx+1] - vol1)) * (b+a)          #$3680

        rdep2 = 0.5  # initial guess for the Newton's method
        for i in range(MAXLOOPS):
            rdep1 = rdep2
            rdep2 = rdep1 - (a*rdep1**2 + b*rdep1 + c)/(2.0 * a * rdep1 + b)
            if abs(rdep2-rdep1) < TOLERANCE:
                break
        else:
            errorsV[3] += 1          # convergence failure error message
        if rdep2 > 1.0 or rdep2 < 0.0:
            errorsV[4] += 1        # converged outside valid range error message

        dep1  = depthFT[indx]                                                   #$3726
        dep   = dep1 + rdep2 * (depthFT[indx+1] - dep1)    # manual eq (36)                #$3728
        sarea = sa1 + a * rdep2                                                 #$3731
    elif AUX1FG == 2:
        dep   = depthFT[indx]    # removed in HSPF 12.4                     #$3739
        sarea = sareaFT[indx]                                                   #$3738
    else:
        dep = 0.0
        sarea = 0.0
    return dep, sarea
