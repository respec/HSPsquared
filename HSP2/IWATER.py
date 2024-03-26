''' Copyright 2017 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.

Conversion of HSPF HIMPWAT.FOR module into Python'''


from numpy import zeros, ones, full, nan, int64, float64
from math import sqrt
from numba import njit
from HSP2.utilities import hourflag, hoursval, initm, make_numba_dict

MAXLOOPS  = 100      # newton method max steps
TOLERANCE = 0.01     # newton method exit tolerance

ERRMSGS =  ('IWATER: IROUTE Newton Method did not converge',    #ERRMSG0
  )


def iwater(io_manager, siminfo, uci, ts):
    ''' Driver for IMPLND IWATER code. CALL: iwater(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation info (OP_SEQUENCE for example)
       ui is a dictionary with ILS specific HSPF UCI like data
       ts is a dictionary with ILS specific timeseries'''

    # WATIN, WATDIF, IMPS not saved since trival calculation from saved data
    #    WATIN  = SUPY + SURLI
    #    WATDIF = WATIN - (SURO + IMPEV)
    #    IMPS   = RETS + SURS

    steps   = siminfo['steps']                  # number of simulation points

    # missing flows are treated as zeros
    for name in ('PETINP','PREC','SURLI',):
        if name not in ts:
            ts[name] = zeros(steps)
       
    CSNOFG = int(uci.get('CSNOFG', 0))
    if CSNOFG:
        for name in ('RAINF', 'AIRTMP', 'SNOCOV', 'WYIELD'):
            if name not in ts:
                ts[name] = full(steps, nan, dtype=float64)
        # Replace fixed parameters in HSPF with timeseries
        for name in ('PETMAX', 'PETMIN'):
            if name not in ts:
                ts[name] = full(steps, uci['PARAMETERS'][name], dtype=float64)

    # process optional monthly arrays to return interpolated data or constant array
    u = uci.get('PARAMETERS', {})
    if 'VRSFG' in u:
        ts['RETSC'] = initm(siminfo, uci, u['VRSFG'], 'MONTHLY_RETSC', u['RETSC'])
        ts['NSUR']  = initm(siminfo, uci, u['VNNFG'], 'MONTHLY_NSUR',  u['NSUR'])
    else:
        ts['RETSC'] = full(steps, u['RETSC'])
        ts['NSUR']  = full(steps, u['NSUR'])

    # true the first time and at 1am every day of simulation
    ts['HR1FG'] = hourflag(siminfo, 1, dofirst=True).astype(float64)  # numba Dict limitation

    # true the first time and at every hour of simulation
    ts['HRFG'] = hoursval(siminfo, ones(24), dofirst=True).astype(float64)  # numba Dict limitation

    ui = make_numba_dict(uci)  # Note: all values coverted to float automatically
    ui['steps']  = steps
    ui['delt']   = siminfo['delt']
    ui['errlen'] = len(ERRMSGS)
    ui['uunits'] = siminfo['units']

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
    uunits = ui['uunits']

    lsur   = ui['LSUR']
    slsur  = ui['SLSUR']
    if uunits == 2:
        lsur = lsur * 3.28

    RTLIFG = int(ui.get('RTLIFG', 0))
    CSNOFG = int(ui.get('CSNOFG', 0))
    RTOPFG = int(ui.get('RTOPFG', 0))
    
    HRFG   = ts['HRFG'].astype(int64)
    HR1FG  = ts['HR1FG'].astype(int64)
    RETSC  = ts['RETSC']  # input parameter could be input monthly
    NSUR   = ts['NSUR']   # input parameter could be input monthly
    PETINP = ts['PETINP'] # ext
    PREC   = ts['PREC']   # ext
    SURLI  = ts['SURLI']
    if uunits == 2:
        RETSC = RETSC * 0.0394 # / 25.4
       
    if CSNOFG:
        AIRTMP = ts['AIRTMP'] # atemp # opt
        PETMAX = ts['PETMAX'] # input parameter # opt
        PETMIN = ts['PETMIN'] # input parameter # opt
        SNOCOV = ts['SNOCOV'] # snow # opt
        RAINF  = ts['RAINF']  # snow # opt
        WYIELD = ts['WYIELD'] # snow # opt
        PETADJ = ts['PETADJ'] = zeros(steps, dtype=float64)
        if uunits == 2:
            WYIELD = WYIELD * 0.0394 # / 25.4        ???  take to inches
            PETMAX = (PETMAX * 9./5.) + 32.
            PETMIN = (PETMIN * 9./5.) + 32.
    

    # like MATLAB, much faster to preinitialize variables. Store in ts Dict
    ts['IMPEV'] = IMPEV = zeros(steps, dtype=float64)
    ts['PET']   = PET   = zeros(steps, dtype=float64)
    ts['RETS']  = RETS  = zeros(steps, dtype=float64)
    ts['SUPY']  = SUPY  = zeros(steps, dtype=float64)
    ts['SURI']  = SURI  = zeros(steps, dtype=float64)
    ts['SURO']  = SURO  = zeros(steps, dtype=float64)
    ts['SURS']  = SURS  = zeros(steps, dtype=float64)

    # initial conditions
    rets = ui['RETS']
    surs = ui['SURS']
    if uunits == 2:
        rets = rets * 0.0394 # / 25.4
        surs = surs * 0.0394
    msupy = surs

    # Needed by Numba 0.31
    dec   = nan
    src   = nan
    surse = nan
    ssupr = nan
    dummy = nan
    d     = nan
    supy  = 0.0

    # MAIN LOOP
    for step in range(steps):
        # save on step lookup code - do once per step
        oldmsupy = msupy
        retsc = RETSC[step]
        petinp = PETINP[step]

        if CSNOFG:
            airtmp = AIRTMP[step]
            petmax = PETMAX[step]
            petmin = PETMIN[step]
            snocov = SNOCOV[step]
            supy   = RAINF[step] * (1.0 - snocov) + WYIELD[step]
            if HRFG[step]:
                petadj = 1.0 - snocov
                if airtmp < petmax:
                    if airtmp < petmin:
                        petadj = 0.0
                    if petadj > 0.5:
                        petadj = 0.5
                PETADJ[step] = petadj
            pet = petinp * petadj
        else:
            supy = PREC[step]
            pet  = petinp
        PET[step]  = pet
        SUPY[step] = supy

        surli = SURLI[step]
        retsc = RETSC[step]
        if RTLIFG: # surface lateral inflow (if any) is subject to retention
            reti = supy + surli

            # RETN
            rets += reti
            if rets > retsc:
                reto = rets - retsc
                rets = retsc
            else:
                reto = 0.0
            suri = reto
        else:
            reti = supy

            # RETN
            rets += reti
            if rets > retsc:
                reto = rets - retsc
                rets = retsc
            else:
                reto = 0.0
            suri = reto + surli
        # IWATER
        msupy = suri + surs

        suro = 0.0
        if msupy > 0.0002:
            if RTOPFG:
                # IROUTE for RTOPFG==True, the way it is done in arm, nps, and hspx
                if oldmsupy == 0.0 or HR1FG[step]:   # Time to recompute
                    dummy  = NSUR[step] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6
                    src = 1020.0 * sqrt(slsur)/dummy

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
            else:
                # IROUTE for RTOPFG==False
                if oldmsupy == 0.0 or HR1FG[step]:   # Time to recompute
                    dummy = NSUR[step] * lsur
                    dec = 0.00982 * (dummy/sqrt(slsur))**0.6
                    src = 1020.0 * sqrt(slsur)/dummy
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
        else:
            suro = msupy
            surs = 0.0

        # EVRETN
        if rets > 0.0:
            if pet > rets:
                impev = rets
                rets  = 0.0
            else:
                impev = pet
                rets -= impev
        else:
            impev = 0.0
        IMPEV[step] = impev

        RETS[step] = rets
        SURI[step] = suri
        SURO[step] = suro
        SURS[step] = surs

        if uunits == 2:
            IMPEV[step]= impev * 25.4
            RETS[step] = rets * 25.4
            SURI[step] = suri * 25.4
            SURO[step] = suro * 25.4
            SURS[step] = surs * 25.4
            SUPY[step] = supy * 25.4
            PET[step]  = pet * 25.4
    return errors



