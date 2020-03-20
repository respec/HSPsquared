''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of no category version of HSPF HRCHHYD.FOR into Python'''


''' Development Notes:
  Categories not implimented in this version
  Irregation only partially implimented in this version
  COLIND, OUTDGT wrong if multiple timeseries in one WDM array
  Only English units currently supported
'''


from numpy import zeros, any, full, nan, array, int64
from pandas import DataFrame
from math import sqrt, log10
from numba import njit, types
from numba.typed import Dict
from utilities import initm


ERRMSGS =('HYDR: SOLVE equations are indeterminate',             #ERRMSG0
          'HYDR: extrapolation of rchtab will take place',       #ERRMSG1
          'HYDR: SOLVE trapped with an oscillating condition',   #ERRMSG2
          'HYDR: Solve did not converge',                        #ERRMSG3
          'HYDR: Solve converged to point outside valid range')  #ERRMSG4

TOLERANCE = 0.001   # newton method max loops
MAXLOOPS  = 100     # newton method exit tolerance

# units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
VFACT  = 43560.0
AFACT  = 43560.0
VFACTA = 1.0/VFACT
LFACTA = 1.0
AFACTA = 1.0/AFACT
SFACTA = 1.0
TFACTA = 1.0

# physical constants (English units)
GAM    = 62.4             # density of water
GRAV   = 32.2             # gravitational acceleration
AKAPPA = 0.4              # von karmen constant


def hydr(store, siminfo, uic, ts):
    ''' find the state of the reach/reservoir at the end of the time interval
    and the outflows during the interval

    CALL: hydr(store, general, ui, ts)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with RID specific HSPF UCI like data
       ts is a dictionary with RID specific timeseries'''

    steps   = siminfo['steps']                # number of simulation points
    segment = siminfo['segment']
    nexits  = uic['PARAMETERS'][segment]['NEXITS']

    u = uic['STATES'][segment]
    COLIN = array([u[name] for name in u.keys() if name.startswith('COLIN')])[0:nexits]
    OUTDG = array([u[name] for name in u.keys() if name.startswith('OUTDG')])[0:nexits]

    u = uic['PARAMETERS'][segment]
    funct  = array([u[name] for name in u.keys() if name.startswith('FUNCT')]).astype(int)[0:nexits]
    ODGTF  = array([u[name] for name in u.keys() if name.startswith('ODGTF')]).astype(int)[0:nexits]
    ODFVF  = array([u[name] for name in u.keys() if name.startswith('ODFVF')]).astype(int)[0:nexits]

    # COLIND timeseries might come in as COLIND, COLIND0, etc. otherwise UCI default
    names = list(sorted([n for n in ts if n.startswith('COLIND')], reverse=True))
    df = DataFrame()
    for i,c in enumerate(ODFVF):
        df[i] = ts[names.pop()] if c < 0 else full(steps, c)
    COLIND = df.to_numpy()

    # OUTDGT timeseries might come in as OUTDGT, OUTDGT0, etc. otherwise UCI default
    names = list(sorted([n for n in ts if n.startswith('OUTDG')], reverse=True))
    df = DataFrame()
    for i,c in enumerate(ODGTF):
        df[i] = ts[names.pop()] if c > 0 else full(steps, c)
    OUTDGT = df.to_numpy()

    # generic SAVE table doesn't know nexits for output flows and rates
    if nexits > 1:
        u = uic['SAVE'][segment]
        for key in ('O', 'OVOL'):
            for i in range(nexits):
                u[f'{key}{i+1}'] = u[key]
            del u[key]

    # optional - defined, but can't used accidently
    for name in ('SOLRAD','CLOUD','DEWTEMP','GATMP','WIND'):
        if name not in ts:
            ts[name] = full(steps, nan)

    # optional timeseries
    for name in ('IVOL','POTEV','PREC'):
        if name not in ts:
            ts[name] = zeros(steps)
    ts['CONVF'] = initm(siminfo, uic, 'VCONFG', 'MONTHLY_CONVF', 1.0)

    # extract key columns of specified FTable for faster access (1d vs. 2d)
    rchtab = store[f"FTABLES/{uic['PARAMETERS'][segment]['FTBUCI']}"]
    ts['volumeFT'] = rchtab['Volume'].values * VFACT
    ts['depthFT']  = rchtab['Depth'].values
    ts['sareaFT']  = rchtab['Area'].values   * AFACT

    ui = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for name in set(uic.keys()) & {'FLAGS', 'PARAMETERS', 'STATES'}:
        for key, value in uic[name][segment].items():
            if type(value) in {int, float}:
                ui[key] = float(value)

    # These automatically get converted to float because of Numba
    ui['steps']  = siminfo['steps']           # number of simulation increments
    ui['delt']   = siminfo['delt']
    ui['nexits'] = nexits
    ui['errlen'] = len(ERRMSGS)
    ui['nrows'], _  = rchtab.shape

    ############################################################################
    errors = hydr_(ui, ts, COLIND, OUTDGT, rchtab.values, funct, ODGTF, ODFVF )                  # run reaches simulation code
    ############################################################################

    return errors, ERRMSGS


@njit(cache=True)
def hydr_(ui, ts, COLIND, OUTDGT, rowsFT, funct, ODGTF, ODFVF):
    errors = zeros(int(ui['errlen'])).astype(int64)

    steps  = int(ui['steps'])            # number of simulation steps
    delts  = ui['delt'] * 60.0           # seconds in simulation interval
    nrows  = int(ui['nrows'])
    nexits = int(ui['nexits'])
    AUX1FG = int(ui['AUX1FG'])         # True means DEP, SAREA will be computed
    AUX2FG = int(ui['AUX2FG'])
    AUX3FG = int(ui['AUX3FG'])
    LKFG   = int(ui['LKFG'])           # flag, 1:lake, 0:stream
    length    = ui['LEN'] * 5280.0                # length of reach
    DB50   = ui['DB50'] / 12.0         # mean diameter of bed material
    DELTH  = ui['DELTH']

    volumeFT = ts['volumeFT']
    depthFT  = ts['depthFT']
    sareaFT  = ts['sareaFT']

    nodfv  = bool(any(ODFVF))
    ks     = ui['KS']
    coks   = 1 - ks
    facta1 = 1.0 / (coks * delts)

    # MAIN loop Initialization
    IVOL   = ts['IVOL']  * VFACT           # or sum civol, zeros if no inflow ???
    POTEV  = ts['POTEV'] / 12.0
    PREC   = ts['PREC']  / 12.0
    CONVF  = ts['CONVF']
    convf  = CONVF[0]

    # faster to preallocate arrays - like MATLAB)
    o      = zeros(nexits)
    odz    = zeros(nexits)
    ovol   = zeros(nexits)
    oseff  = zeros(nexits)
    od1    = zeros(nexits)
    od2    = zeros(nexits)
    outdgt = zeros(nexits)
    colind = zeros(nexits)

    outdgt[:] = OUTDGT[0,:]
    colind[:] = COLIND[0,:]

    # numba limitation, ts can't have both 1-d and 2-d arrays in save Dict
    O      = zeros((steps, nexits))
    OVOL   = zeros((steps, nexits))

    ts['PRSUPY'] = PRSUPY = zeros(steps)
    ts['RO']     = RO     = zeros(steps)
    ts['ROVOL']  = ROVOL  = zeros(steps)
    ts['VOL']    = VOL    = zeros(steps)
    ts['VOLEV']  = VOLEV  = zeros(steps)
    ts['IRRDEM'] = IRRDEM = zeros(steps)
    if AUX1FG:
        ts['DEP']   = DEP   = zeros(steps)
        ts['SAREA'] = SAREA = zeros(steps)
        ts['USTAR'] = USTAR = zeros(steps)
        ts['TAU']   = TAU   = zeros(steps)

    zeroindex = fndrow(0.0, volumeFT)                                           #$1126-1127
    topvolume = volumeFT[-1]

    vol = ui['VOL'] * VFACT   # hydr-init, initial volume of water
    if vol >= topvolume:
        errors[1] += 1      # ERRMSG1: extrapolation of rchtab will take place

    # find row index that brackets the VOL
    indx = fndrow(vol, volumeFT)
    if nodfv:  # simple interpolation, the hard way!!
        v1 = volumeFT[indx]
        v2 = volumeFT[indx+1]
        rod1,od1[:] = demand(v1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt, ODGTF)
        rod2,od2[:] = demand(v2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
        a1 = (v2 - vol) / (v2 - v1)
        o[:] = a1 * od1[:] + (1.0 - a1) * od2[:]
        ro   = (a1 * rod1) + ((1.0 - a1) * rod2)
    else:
        ro,o[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)  #$1159-1160

    # back to PHYDR
    if AUX1FG >= 1:
        dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errors) # initial depth and surface area

    # hydr-irrig
    irexit = int(ui['IREXIT']) -1    # irexit - exit number for irrigation withdrawals, 0 based ???
    #if irexit >= 1:
    irminv = ui['IRMINV']
    rirwdl = 0.0
    #rirdem = 0.0
    #rirsht = 0.0
    irrdem = 0.0

    # HYDR (except where noted)
    for step in range(steps):
        convf  = CONVF[step]
        outdgt[:] = OUTDGT[step, :]
        colind[:] = COLIND[step, :]
        roseff = ro
        oseff[:] = o

        # vols, sas variables and their initializations  not needed.
        if irexit >= 0:             # irrigation exit is set, zero based number
            if rirwdl > 0.0:  # equivalent to OVOL for the irrigation exit
                vol = irminv if irminv > vol - rirwdl else vol - rirwdl
                if vol >= volumeFT[-1]:
                    errors[1] += 1 # ERRMSG1: extrapolation of rchtab will take place

                # DISCH with hydrologic routing
                indx = fndrow(vol, volumeFT)                 # find row index that brackets the VOL
                vv1 = volumeFT[indx]
                rod1,od1[:] = demand(vv1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                vv2 = volumeFT[indx+1]
                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                aa1 = (vv2 - vol) / (vv2 - vv1)
                ro   = (aa1 * rod1)    + ((1.0 - aa1) * rod2)
                o[:] = (aa1 * od1[:])  + ((1.0 - aa1) * od2[:])

                # back to HYDR
                if AUX1FG >= 1:     # recompute surface area and depth
                    dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errors)
            else:
                irrdem =  0.0
            #o[irexit] = 0.0                                                   #???? not used anywhere, check if o[irexit]

        prsupy = PREC[step] * sarea
        volt   = vol + IVOL[step] + prsupy
        volev = 0.0
        if AUX1FG:                  # subtract evaporation
            volpev = POTEV[step] * sarea
            if volev >= volt:
                volev = volt
                volt = 0.0
            else:
                volev = volpev
                volt -= volev

        # ROUTE/NOROUT  calls
        # common code
        volint = volt - (ks * roseff * delts)    # find intercept of eq 4 on vol axis
        if volint < (volt * 1.0e-5):
            volint = 0.0
        if volint <= 0.0:  #  case 3 -- no solution to simultaneous equations
            indx  = zeroindex
            vol   = 0.0
            ro    = 0.0
            o[:]  = 0.0
            rovol = volt

            if roseff > 0.0:
                ovol[:] = (rovol/roseff) * oseff[:]
            else:
                ovol[:] = rovol / nexits

        else:   # case 1 or 2
            oint = volint * facta1      # == ointsp, so ointsp variable dropped
            if nodfv:
                # ROUTE
                rodz,odz[:] = demand(0.0, rowsFT[zeroindex,:], funct, nexits, delts, convf, colind,  outdgt, ODGTF)
                if oint > rodz:
                    # SOLVE,  Solve the simultaneous equations for case 1-- outflow demands can be met in full
                    # premov will be used to check whether we are in a trap, arbitrary value
                    premov = -20
                    move   = 10

                    vv1 = volumeFT[indx]
                    rod1,od1[:] = demand(vv1, rowsFT[indx, :], funct, nexits, delts, convf,colind, outdgt, ODGTF)
                    vv2 = volumeFT[indx+1]
                    rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)

                    while move != 0:
                        facta2 = rod1 - rod2
                        factb2 = vv2 - vv1
                        factc2 = vv2 * rod1 - vv1 * rod2
                        det = facta1 * factb2 - facta2
                        if det == 0.0:
                            det = 0.0001
                            errors[0] += 1  # ERRMSG0: SOLVE is indeterminate

                        vol = max(0.0, (oint * factb2 - factc2 ) / det)
                        if vol > vv2:
                            if indx >= nrows-2:
                                if vol > topvolume:
                                    errors[1] += 1 # ERRMSG1: extrapolation of rchtab will take place
                                move = 0
                            else:
                                move   = 1
                                indx  += 1
                                vv1    = vv2
                                od1[:] = od2
                                rod1   = rod2
                                vv2    = volumeFT[indx+1]
                                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                        elif vol < vv1:
                            indx  -= 1
                            move   = -1
                            vv2    = vv1
                            od2[:] = od1
                            rod2   = rod1
                            vv1    = volumeFT[indx]
                            rod1,od1[:] = demand(vv1, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                        else:
                            move = 0

                        # check whether algorithm is in a trap, yo-yoing back and forth
                        if move + premov == 0:
                            errors[2] += 1      # ERRMSG2: oscillating trap
                            move = 0
                        premov = move

                    ro = oint - facta1 * vol
                    if  vol < 1.0e-5:
                        ro  = oint
                        vol = 0.0
                    if ro < 1.0e-10:
                        ro  = 0.0
                    if ro <= 0.0:
                        o[:] = 0.0
                    else:
                        diff  = vol - vv1
                        factr = 0.0 if diff < 0.01 else  diff / (vv2 - vv1)
                        o[:]  = od1[:] + (od2[:] - od1[:]) * factr
                else:
                    # case 2 -- outflow demands cannot be met in full
                    ro  = 0.0
                    for i in range(nexits):
                        tro  = ro + odz[i]
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
                rod1,od1[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt, ODGTF)
                if oint >= rod1: #case 1 -outflow demands are met in full
                    ro   = rod1
                    vol  = volint - coks * ro * delts
                    if vol < 1.0e-5:
                        vol = 0.0
                    o[:] = od1[:]
                else:    # case 2 -outflow demands cannot be met in full
                    ro  = 0.0
                    for i in range(nexits):
                        tro  = ro + odz[i]
                        if tro <= oint:
                            o[i] = odz[i]
                            ro = tro
                        else:
                            o[i] = oint - ro
                            ro = oint
                    vol = 0.0
                    indx = zeroindex

            # common  ROUTE/NOROUT code
            #  an irrigation demand was made before routing
            if  (irexit >= 0) and (irrdem > 0.0):    #  an irrigation demand was made before routing
                oseff[irexit] = irrdem
                o[irexit]     = irrdem
                roseff       += irrdem
                ro           += irrdem
                IRRDEM[step] = irrdem

            # estimate the volumes of outflow
            ovol[:] = (ks * oseff[:] + coks * o[:]) * delts
            rovol   = (ks * roseff   + coks * ro)   * delts

        # HYDR
        if nexits > 1:
            O[step,:]    = o[:]    * SFACTA * LFACTA
            OVOL[step,:] = ovol[:] * SFACTA * LFACTA
        PRSUPY[step] = prsupy * AFACTA
        RO[step]     = ro     * SFACTA * LFACTA
        ROVOL[step]  = rovol  * VFACTA
        VOLEV[step]  = volev  * VFACTA
        VOL[step]    = vol    * VFACTA

        if AUX1FG:   # compute final depth, surface area
            if vol >= topvolume:
                errors[1] += 1       # ERRMSG1: extrapolation of rchtab
            indx = fndrow(vol, volumeFT)
            dep, sarea = auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errors)
            DEP[step]   = dep
            SAREA[step] = sarea * AFACTA

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
                    if LKFG:              # flag, 1:lake, 0:stream
                        ustar = (2.3/AKAPPA) * avvel / (17.66 + log10(avdep/(96.5*DB50)))
                        tau   =  GAM/GRAV * ustar**2              #3796
                    else:
                        hrad = (avdep*twid)/(2.0*avdep + twid)  if avdep > 0.0 else 0.0 # hydraulic radius, manual eq (41
                        slope = DELTH / length
                        ustar = sqrt(GRAV * slope * hrad)
                        tau = (GAM * slope) * hrad
                else:
                    ustar = 0.0
                    tau = 0.0
                USTAR[step] = ustar * LFACTA
                TAU[step]   = tau   * TFACTA
    # END MAIN LOOP

    # NUMBA limitation for ts, and saving to HDF5 file is in individual columns
    if nexits > 1:
        # Numba can't do 'O' + str(i) stuff yet, so save more than enough strings
        On = ('O1','O2','O3','O4','O5','O6','O7','O8','O9')
        OVOLn = ('OVOL1','OVOL2','OVOL3','OVOL4','OVOL5','OVOL6','OVOL7','OVOL8','OVOL9')
        for i in range(nexits):
            ts[On[i]]    = O[:,i]
            ts[OVOLn[i]] = OVOL[:,i]
    return errors


@njit(cache=True)
def fndrow(v, volFT):
    ''' finds highest index in FTable volume column whose volume  < v'''
    for indx,vol in enumerate(volFT):
        if v < vol:
            return indx-1
    return len(volFT) - 2


@njit(cache=True)
def demand(vol, rowFT, funct, nexits, delts, convf, colind, outdgt, ODGTF):
    od = zeros(nexits)
    for i in range(nexits):
        col = colind[i]
        icol = int(col)
        if icol != 0:
            diff = col - float(icol)
            if diff >= 1.0e-6:
                _od1 = rowFT[icol-1]
                od[i] = _od1 + diff * (_od1 - rowFT[icol]) * convf
            else:                                        # no interpolation ???
                od[i] = rowFT[icol-1] * convf

        icol = int(ODGTF[i])
        if icol != 0:
            a = od[i]
            b = outdgt[icol-1]
            c = (vol - b)  / delts                     #???
            if   funct[i] == 1: od[i] = min(a,c)
            elif funct[i] == 2: od[i] = max(a,b)
            elif funct[i] == 3: od[i] = a+b
            elif funct[i] == 4: od[i] = max(a,c)
    return od.sum(), od


@njit(cache=True)
def auxil(volumeFT, depthFT, sareaFT, indx, vol, AUX1FG, errors):
    ''' Compute depth and surface area'''
    if vol > 0.0:
        sa1  = sareaFT[indx]
        a    = sareaFT[indx+1] - sa1
        b    = 2.0 * sa1
        vol1 = volumeFT[indx]
        c = -((vol - vol1) / (volumeFT[indx+1] - vol1)) * (b+a)

        rdep2 = 0.5  # initial guess for the Newton's method
        for i in range(MAXLOOPS):
            rdep1 = rdep2
            rdep2 = rdep1 - (a*rdep1**2 + b*rdep1 + c)/(2.0 * a * rdep1 + b)
            if abs(rdep2-rdep1) < TOLERANCE:
                break
        else:
            errors[3] += 1          # convergence failure error message
        if rdep2 > 1.0 or rdep2 < 0.0:
            errors[4] += 1        # converged outside valid range error message

        dep1  = depthFT[indx]
        dep   = dep1 + rdep2 * (depthFT[indx+1] - dep1)    # manual eq (36)
        sarea = sa1 + a * rdep2
    elif AUX1FG == 2:
        dep   = depthFT[indx]    # removed in HSPF 12.4
        sarea = sareaFT[indx]
    else:
        dep   = 0.0
        sarea = 0.0
    return dep, sarea
