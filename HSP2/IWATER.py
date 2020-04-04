''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HIMPWAT.FOR module into Python'''


from numpy import zeros, ones, full, nan, int64
from math import sqrt
from numba import njit
from utilities import hourflag, hoursval, initm, make_numba_dict

MAXLOOPS  = 100      # newton method max steps
TOLERANCE = 0.01     # newton method exit tolerance

ERRMSGS =  ('IWATER: IROUTE Newton Method did not converge',    #ERRMSG0
  )


def iwater(store, siminfo, uci, ts):
    ''' Driver for IMPLND IWATER code. CALL: iwater(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with ILS specific HSPF UCI like data
       ts is a dictionary with ILS specific timeseries'''

    steps   = siminfo['steps']                  # number of simulation points

    # insure defined, but not usable
    for name in ('AIRTMP', 'PETINP', 'PREC', 'RAINF', 'SNOCOV', 'WYIELD'):
        if name not in ts:
            ts[name] = full(steps, nan)

    # treat missing flows as zero flow
    for name in ['SURLI']:
        if name not in ts:
            ts[name] = zeros(steps)

    # Replace fixed parameters in HSPF with timeseries
    for name in ['PETMAX', 'PETMIN']:
        if name not in ts:
            ts[name] = full(steps, uci['PARAMETERS'][name])

    # process optional monthly arrays to return interpolated data or constant array
    u = uci['PARAMETERS']
    ts['RETSC'] = initm(siminfo, uci, u['VRSFG'], 'MONTHLY_RETSC', u['RETSC'])
    ts['NSUR']  = initm(siminfo, uci, u['VNNFG'], 'MONTHLY_NSUR',  u['NSUR'])

    # true the first time and at 1am every day of simulation
    ts['HR1FG'] = hourflag(siminfo, 1, dofirst=True).astype(float)  # numba Dict limitation

    # true the first time and at every hour of simulation
    ts['HRFG'] = hoursval(siminfo, ones(24), dofirst=True).astype(float)  # numba Dict limitation

    ui = make_numba_dict(uci)  # Note: all values coverted to float automatically
    ui['steps']  = steps
    ui['delt']   = siminfo['delt']
    ui['errlen'] = len(ERRMSGS)

    ############################################################################
    errors = _iwater_(ui, ts)                       # run IWATER simulation code
    ############################################################################

    return errors, ERRMSGS


@njit(cache=True)
def _iwater_(ui, ts):
    ''' Simulate the water budget for an impervious land segment. '''
    errors = zeros(int(ui['errlen'])).astype(int64)      # storage for error counts

    delt60 = ui['delt'] / 60.0          # simulation interval in hours
    steps  = int(ui['steps'])

    lsur   = ui['LSUR']
    slsur  = ui['SLSUR']
    RTLIFG = int(ui['RTLIFG'])

    HRFG   = ts['HRFG']
    HR1FG  = ts['HR1FG']
    RETSC  = ts['RETSC']
    NSUR   = ts['NSUR']
    AIRTMP = ts['AIRTMP']
    PETMAX = ts['PETMAX']
    PETMIN = ts['PETMIN']
    SNOCOV = ts['SNOCOV']
    SURLI  = ts['SURLI']
    PETINP = ts['PETINP']
    RAINF  = ts['RAINF']
    WYIELD = ts['WYIELD']
    PREC   = ts['PREC']

    # like MATLAB, much faster to preinitialize variables. Store in ts Dict
    ts['IMPEV'] = IMPEV = zeros(steps)
    ts['PET']   = PET   = zeros(steps)
    ts['PETADJ']= PETADJ= zeros(steps)
    ts['RETS']  = RETS  = zeros(steps)
    ts['SUPY']  = SUPY  = zeros(steps)
    ts['SURI']  = SURI  = zeros(steps)
    ts['SURO']  = SURO  = zeros(steps)
    ts['SURS']  = SURS  = zeros(steps)

    # initial conditions
    rets = ui['RETS']
    surs = ui['SURS']
    msupy = surs

    # Needed by Numba 0.31
    dec   = nan
    src   = nan
    surse = nan
    ssupr = nan
    dummy = nan
    d     = nan
    supy  = 0.0

    # MAIN LOOP (cleaner to not fix interweave both code blocks together!)
    if ui['RTOPFG']:
        for step in range(steps):
            # save on step lookup code - do once per step
            oldmsupy = msupy
            retsc = RETSC[step]
            petinp = PETINP[step]

            if int(ui['CSNOFG']):
                airtmp = AIRTMP[step]
                petmax = PETMAX[step]
                petmin = PETMIN[step]

                snocov = SNOCOV[step]
                SUPY[step] = RAINF[step] * (1.0 - snocov) + WYIELD[step]
                if int(HRFG[step]):
                    petadj = 1.0 - snocov
                    if (airtmp < petmax) and (petadj > 0.5):
                        petadj = 0.5
                    if airtmp < petmin:
                        petadj = 0.0
                    PETADJ[step] = petadj
                PET[step] = petinp * petadj
            else:
                SUPY[step] = PREC[step]
                PET[step] = petinp
            pet  = PET[step]
            supy = SUPY[step]

            # RETN
            reti = supy + SURLI[step] if RTLIFG else supy
            rets = rets + reti
            reto = rets - retsc if rets > retsc else 0.0
            rets = min(rets, retsc)

            # IWATER
            suri  = reto if RTLIFG else reto + SURLI[step]
            msupy = suri + surs
            surs = 0.0
            suro = 0.0
            if msupy > 0.0:
                # IROUTE for RTOPFG==True, the way it is done in arm, nps, and hspx
                if oldmsupy == 0.0 or HR1FG[step]:   # Time to recompute
                    dummy  = NSUR[step] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6
                    src = 1020.0 * sqrt(slsur)/dummy
                if msupy <= 0.0002:
                    suro = msupy
                    surs = 0.0
                else:
                    sursm = (surs + msupy) * 0.5
                    dummy = sursm * 1.6
                    if suri > 0.0:
                        d = dec*suri**0.6
                        if d > sursm:
                            surse = d
                            dummy = sursm * (1.0 + 0.6 * (sursm / surse)**3)
                tsuro = delt60 * src * dummy**1.67
                suro  = msupy if tsuro > msupy else tsuro
                surs  = 0.0   if tsuro > msupy else msupy - suro

            # EVRETN
            if rets > 0.0:
                IMPEV[step] = rets if pet > rets else pet
                rets        = 0.0  if pet > rets else rets - IMPEV[step]

            RETS[step] = rets
            SURI[step] = suri
            SURO[step] = suro
            SURS[step] = surs
    else:
        for step in range(steps):
            # save on step lookup code - do once per step
            retsc = RETSC[step]
            oldmsupy = msupy

            # RETN
            reti = supy + SURLI[step] if RTLIFG else supy
            rets = rets + reti
            reto = rets - retsc if rets > retsc else 0.0
            rets = min(rets, retsc)

            # IWATER
            suri  = reto if RTLIFG else reto + SURLI[step]
            msupy = suri + surs

            surs = 0.0
            suro = 0.0
            if msupy > 0.0:
                # IROUTE for RTOPFG==False
                if oldmsupy == 0.0 or HR1FG[step]:   # Time to recompute
                    dummy = NSUR[step] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6
                    src = 1020.0 * sqrt(slsur)/dummy
                if msupy <= 0.0002:
                    suro = msupy
                    surs = 0.0
                else:
                    ssupr  = suri / delt60
                    surse  = dec * ssupr**0.6 if ssupr > 0.0 else 0.0
                sursnw = msupy
                suro   = 0.0

                for count in range(MAXLOOPS):
                    if ssupr > 0.0:
                        ratio = sursnw / surse
                        fact = 1.0 + 0.6 * ratio**3  if ratio <= 1.0 else 1.6
                    else:
                        fact  = 1.6
                        ratio = 1e30

                    ffact  = (delt60 * src * fact**1.667) * (sursnw**1.667)
                    fsuro  = ffact - suro
                    dfact  = -1.667 * ffact

                    dfsuro = dfact/sursnw - 1.0
                    if ratio <= 1.0:
                        dfsuro += (dfact/(fact * surse)) * 1.8 * ratio**2
                    dsuro = fsuro / dfsuro

                    suro = suro - dsuro
                    sursnw = msupy - suro

                    if abs(dsuro / suro) < TOLERANCE:
                        break
                else:
                    errors[0] = errors[0] + 1  # IROUTE did not converge
                surs = sursnw

            # EVRETN
            if rets > 0.0:
                IMPEV[step] = rets if PET[step] > rets else PET[step]
                rets = 0.0  if PET[step] > rets else rets - IMPEV[step]

            #save results
            RETS[step] = rets
            SURI[step] = suri
            SURS[step] = surs
            SURO[step] = suro

    # done with steping
    # WATIN, WATDIF, IMPS not saved since trival calculation from saved data
    #    WATIN  = SUPY + SURLI
    #    WATDIF = WATIN - (SURO + IMPEV)
    #    IMPS   = RETS + SURS
    return errors



