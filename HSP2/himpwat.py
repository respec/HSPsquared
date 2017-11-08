''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HIMPWAT.FOR module into Python'''                            #$$HIMPWAT.FOR


from numpy import zeros, where, full, nan
from math import sqrt
from numba import jit
from HSP2 import initm

MAXLOOPS  = 100      # newton method max loops
TOLERANCE = 0.01     # newton method exit tolerance

ERRMSG =  ['IWATER: IROUTE Newton Method did not converge']    #ERRMSG0


def iwater(store, general, ui, ts):
    ''' Driver for IMPLND IWATER code. CALL: iwater(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with ILS specific HSPF UCI like data
       ts is a dictionary with ILS specific timeseries'''

    simlen = general['sim_len']

    for name in ['AIRTMP', 'PETINP', 'PREC', 'RAINF', 'SNOCOV', 'WYIELD']:
        if name not in ts:
            ts[name] = full(simlen, nan)  # insure defined, but not usable
    for name in ['SURLI']:
        if name not in ts:
            ts[name] = zeros(simlen)      # treat missing flows as zero flow
    for name in ['PETMAX', 'PETMIN']:
        if name not in ts:
            ts[name] = full(simlen, ui[name]) # Replace fixed parameters in HSPF with timeseries

    # process optional monthly arrays to return interpolated data or constant array
    initm(general, ui, ts, 'VRSFG','RETSCM', 'RETSC')                           #$62,70,73,1003,1005,1010-1012
    initm(general, ui, ts, 'VNNFG', 'NSURM', 'NSUR')                            #$61,78,82

    tindex = general['tindex']
    HR1FG = ts['HR1FG'] = where(tindex.hour==1, True, False)   # array is true at 1am every day
    HR1FG[0] = True

    PETINP = ts['PETINP']                                                       #$132
    if ui['CSNOFG']:                                                            #$136
        SNOCOV = ts['SNOCOV']                                                   #$159,171,172,180
        ts['SUPY'] = ts['RAINF']  * (1.0 - SNOCOV) + ts['WYIELD']               #$183,159,166,167,176,177,180

        PET, PETADJ = adjust_pet(ts['AIRTMP'], SNOCOV, PETINP, ts['PETMAX'], ts['PETMIN']) #67,145,151,154-157,210
        ts['PET']    = PET
        ts['PETADJ'] = PETADJ
    else:                                                                       #$212
        ts['SUPY'] = ts['PREC']                                                 #$212,220,223
        ts['PET']  = PETINP                                                     #$132, 212,224

    ############################################################################
    errorsV = iwater_(general, ui, ts)               # run IWATER simulation code

    #with open('numba_iwater.txt', 'w') as fnumba:
    #    iwater_.inspect_types(file= fnumba) # numba testing
    #with open('numba_adjust_pet.txt', 'w') as fnumba:
    #    adjust_pet.inspect_types(file= fnumba) # numba testing
    ############################################################################

    return errorsV, ERRMSG


def iwater_(general, ui, ts):
    ''' Impervious Water module'''
    errorsV = zeros(len(ERRMSG), dtype=int)      # storage for error counts

    delt60 = general['sim_delt'] / 60.0          # simulation interval in hours
    simlen = general['sim_len']

    # this section replaces $$PIWATER
    lsur   = ui['LSUR']                                                         #$59
    slsur  = ui['SLSUR']                                                        #$60
    RTLIFG = ui['RTLIFG']                                                       #$50

    HR1FG = ts['HR1FG']
    RETSC = ts['RETSC']
    NSUR  = ts['NSUR']
    PET   = ts['PET']
    SUPY  = ts['SUPY']
    SURLI = ts['SURLI']                                                         #$229-232
    retiV = where(RTLIFG, SUPY + SURLI, SUPY)                                   #$235,237,246,248

    # create arrays for later - better performance just like MATLAB(R)
    IMPEV = ts['IMPEV'] = zeros(simlen)                                         #$354
    RETS  = ts['RETS']  = zeros(simlen)
    SURI  = ts['SURI']  = zeros(simlen)
    SURO  = ts['SURO']  = zeros(simlen)
    SURS  = ts['SURS']  = zeros(simlen)

    # initial conditions
    rets = ui['RETS']                                                           #$89
    surs = ui['SURS']                                                           #$89
    msupy = surs

    # MAIN LOOPS
    iwater_liftedloop(HR1FG, IMPEV, NSUR, PET, RETS, RETSC, RTLIFG, SURI, SURLI,
     SURO, SURS, delt60, errorsV, lsur, msupy, retiV, rets, simlen, slsur,
     surs, ui['RTOPFG'])

    # done with looping
    # WATIN, WATDIF, IMPS not saved since trival calculation from saved data
    #    WATIN  = SUPY + SURLI                                                  #$309
    #    WATDIF = WATIN - (SURO + IMPEV)                                        #$312
    #    IMPS   = RETS + SURS                                                   #$93,315
    return errorsV


@jit(nopython=True, cache=True)
def iwater_liftedloop(HR1FG, IMPEV, NSUR, PET, RETS, RETSC, RTLIFG, SURI, SURLI,
 SURO, SURS, delt60, errorsV, lsur, msupy, retiV, rets, simlen, slsur,
 surs, RTOPFG):
    dec   = nan
    src   = nan
    if RTOPFG:    # (cleaner to not fix interweave both code blocks together!)  #$442
        for loop in range(simlen):
            # save on loop lookup code - do once per loop
            pet   = PET[loop]
            retsc = RETSC[loop]
            oldmsupy = msupy

            # RETN
            rets = rets + retiV[loop]                                           #$1018
            reto = rets - retsc if rets > retsc else 0.0                        #$1019,1022-1024
            rets = min(rets, retsc)                                             #$1019,1022

            # IWATER
            suri  = reto if RTLIFG else reto + SURLI[loop]                      #$235,244,246,250
            msupy = suri + surs                                                 #$259
            surs = 0.0
            suro = 0.0
            if msupy > 0.0:                                                     #$265
                # IROUTE for RTOPFG==True, the way it is done in arm, nps, and hspx
                if oldmsupy == 0.0 or HR1FG[loop]:   # Time to recompute        #$421
                    dummy  = NSUR[loop] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6                    #$436
                    src = 1020.0 * sqrt(slsur)/dummy                            #$437
                if msupy <= 0.0002:                                             #$440
                    suro = msupy                                                #$553
                    surs = 0.0                                                  #$554
                else:
                    sursm = (surs + msupy) * 0.5                                #$516
                    dummy = sursm * 1.6
                    if suri > 0.0:
                        d = dec*suri**0.6
                        if d > sursm:
                            surse = d
                            dummy = sursm * (1.0 + 0.6 * (sursm / surse)**3)    #$526,527

                    tsuro = delt60 * src * dummy**1.67                              #$538
                    suro  = msupy if tsuro > msupy else tsuro                       #$541,543,546
                    surs  = 0.0   if tsuro > msupy else msupy - suro                #$541,544,547

            # EVRETN
            if rets > 0.0:                                                      #$340
                IMPEV[loop] = rets if pet > rets else pet                       #$342,345,347,349
                rets        = 0.0  if pet > rets else rets - IMPEV[loop]        #$342,346,347,350

            RETS[loop] = rets
            SURI[loop] = suri
            SURO[loop] = suro
            SURS[loop] = surs
    else:
        for loop in range(simlen):
            # save on loop lookup code - do once per loop
            retsc = RETSC[loop]
            oldmsupy = msupy

            # RETN
            rets = rets + retiV[loop]                                           #$1017
            reto = rets - retsc if rets > retsc else 0.0                        #$1019,1022-1024
            rets = min(rets, retsc)                                             #$1019,1022

            # IWATER
            suri  = reto if RTLIFG else reto + SURLI[loop]                      #$235,244,246,250
            msupy = suri + surs                                                 #$259

            surs = 0.0
            suro = 0.0
            if msupy > 0.0:                                                     #$265
                # IROUTE for RTOPFG==False
                if oldmsupy == 0.0 or HR1FG[loop]:   # Time to recompute        #$421
                    dummy = NSUR[loop] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6                    #$436
                    src = 1020.0 * sqrt(slsur)/dummy
                if msupy <= 0.0002:                                             #$440
                    suro = msupy                                                #$546
                    surs = 0.0                                                  #$547
                else:
                    ssupr  = suri / delt60                                      #$446
                    surse  = dec * ssupr**0.6 if ssupr > 0.0 else 0.0           #$448-451

                    sursnw = msupy                                                  #$454
                    suro   = 0.0                                                    #$455

                    for count in range(MAXLOOPS):                                   #$456,458,506
                        if ssupr > 0.0:                                             #$459
                            ratio = sursnw / surse                                  #$460
                            fact = 1.0 + 0.6 * ratio**3  if ratio <= 1.0 else 1.6   #$461,463
                        else:
                            fact  = 1.6                                             #$465,470
                            ratio = 1e30                                            #$469

                        ffact  = (delt60 * src * fact**1.667) * (sursnw**1.667)     #$474,475,477
                        fsuro  = ffact - suro                                       #$478
                        dfact  = -1.667 * ffact                                     #$479

                        dfsuro = dfact/sursnw - 1.0                                 #$480
                        if ratio <= 1.0:                                            #$481
                            dfsuro += (dfact/(fact * surse)) * 1.8 * ratio**2       #$483
                        dsuro = fsuro / dfsuro                                      #$486

                        suro = suro - dsuro                                         #$503
                        sursnw = msupy - suro                                       #$504

                        if abs(dsuro / suro) < TOLERANCE:
                            break
                    else:
                        errorsV[0] = errorsV[0] + 1  # IROUTE did not converge      #$448-499
                    surs = sursnw                                                   #$508

            # this section replaces EVRETN
            if rets > 0.0:                                                      #$340
                IMPEV[loop] = rets if PET[loop] > rets else PET[loop]           #$342,345,347,349
                rets = 0.0  if PET[loop] > rets else rets - IMPEV[loop]         #$342,346,347,350

            #save results
            RETS[loop] = rets
            SURI[loop] = suri
            SURS[loop] = surs
            SURO[loop] = suro
    return


@jit(nopython=True, cache=True)
def adjust_pet(AIRTMP, SNOCOV, PETINP, PETMAX, PETMIN):
    size = len(AIRTMP)
    petadj = 1.0 - SNOCOV[0]     # Numba needs varible to have an initial value
    PET = zeros(size)
    PETADJ = zeros(size)
    for loop in range(size):
        airtmp = AIRTMP[loop]        # pay for loop indexing only once
        petadj = 1.0 - SNOCOV[loop]                    #$189
        if (airtmp < PETMAX[loop])and petadj > 0.5:  #$191,193,196-198,204,205
            petadj = 0.5
        if airtmp < PETMIN[loop]:     #$191,193,195,204,205
            petadj = 0.0
        PETADJ[loop] = petadj
        PET[loop] = PETINP[loop] * petadj
    return PET, PETADJ
