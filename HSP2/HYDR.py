''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
Conversion of no category version of HSPF HRCHHYD.FOR into Python'''


''' Development Notes:
  Categories not implimented in this version
  Irregation only partially implimented in this version
  Only English units currently supported
  FTABLE can come from WDM or UCI file based on FTBDSN 1 or 0
'''


from numpy import zeros, any, full, nan, array, int64, arange
from pandas import DataFrame
from math import sqrt, log10
from numba import njit
from numba.typed import List
from HSP2.utilities import initm, make_numba_dict

# the following imports added by rb to handle dynamic code and special actions
from HSP2.state import *
from HSP2.SPECL import specl
from HSP2.om import *
from HSP2.om_model_object import *
from HSP2.om_sim_timer import *
from HSP2.om_special_action import *
#from HSP2.om_equation import *
from HSP2.om_model_linkage import *
#from HSP2.om_data_matrix import *
#from HSP2.om_model_broadcast import *


ERRMSGS =('HYDR: SOLVE equations are indeterminate',             #ERRMSG0
          'HYDR: extrapolation of rchtab will take place',       #ERRMSG1
          'HYDR: SOLVE trapped with an oscillating condition',   #ERRMSG2
          'HYDR: Solve did not converge',                        #ERRMSG3
          'HYDR: Solve converged to point outside valid range')  #ERRMSG4

TOLERANCE = 0.001   # newton method max loops
MAXLOOPS  = 100     # newton method exit tolerance


def hydr(io_manager, siminfo, uci, ts, ftables, state):
    ''' find the state of the reach/reservoir at the end of the time interval
    and the outflows during the interval

    CALL: hydr(store, general, ui, ts, state)
       store is the Pandas/PyTable open store
       general is a dictionary with simulation level infor (OP_SEQUENCE for example)
       ui is a dictionary with RID specific HSPF UCI like data
       ts is a dictionary with RID specific timeseries
       state is a dictionary that contains all dynamic code dictionaries such as: 
       - specactions is a dictionary with all special actions
    '''

    steps   = siminfo['steps']                # number of simulation points
    uunits  = siminfo['units']
    nexits  = int(uci['PARAMETERS']['NEXITS'])

    # units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
    VFACT = 43560.0
    AFACT = 43560.0
    if uunits == 2:
        # si units conversion constants, 1 hectare is 10000 sq m, assumes area input in hectares, vol in Mm3
        VFACT = 1.0e6
        AFACT = 10000.0

    u = uci['PARAMETERS']
    funct  = array([u[name] for name in u.keys() if name.startswith('FUNCT')]).astype(int)[0:nexits]
    ODGTF  = array([u[name] for name in u.keys() if name.startswith('ODGTF')]).astype(int)[0:nexits]
    ODFVF  = array([u[name] for name in u.keys() if name.startswith('ODFVF')]).astype(int)[0:nexits]

    u = uci['STATES']
    colin = array([u[name] for name in u.keys() if name.startswith('COLIN')]).astype(float)[0:nexits]
    outdg = array([u[name] for name in u.keys() if name.startswith('OUTDG')]).astype(float)[0:nexits]

    # COLIND timeseries might come in as COLIND, COLIND0, etc. otherwise UCI default
    names = list(sorted([n for n in ts if n.startswith('COLIND')], reverse=True))
    df = DataFrame()
    for i,c in enumerate(ODFVF):
        df[i] = ts[names.pop()][0:steps] if c < 0 else full(steps, c)
    COLIND = df.to_numpy()

    # OUTDGT timeseries might come in as OUTDGT, OUTDGT0, etc. otherwise UCI default
    names = list(sorted([n for n in ts if n.startswith('OUTDG')], reverse=True))
    df = DataFrame()
    for i,c in enumerate(ODGTF):
        df[i] = ts[names.pop()][0:steps] if c > 0 else zeros(steps)
    OUTDGT = df.to_numpy()

    # generic SAVE table doesn't know nexits for output flows and rates
    if nexits > 1:
        u = uci['SAVE']
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
    ts['CONVF'] = initm(siminfo, uci, 'VCONFG', 'MONTHLY_CONVF', 1.0)

    # extract key columns of specified FTable for faster access (1d vs. 2d)
    rchtab = ftables[f"{uci['PARAMETERS']['FTBUCI']}"]
    #rchtab = store[f"FTABLES/{uci['PARAMETERS']['FTBUCI']}"]
    ts['volumeFT'] = rchtab['Volume'].to_numpy() * VFACT
    ts['depthFT']  = rchtab['Depth'].to_numpy()
    ts['sareaFT']  = rchtab['Area'].to_numpy()   * AFACT
    rchtab = rchtab.to_numpy()

    ui = make_numba_dict(uci) # Note: all values coverted to float automatically
    ui['steps']  = steps
    ui['delt']   = siminfo['delt']
    ui['nexits'] = nexits
    ui['errlen'] = len(ERRMSGS)
    ui['nrows']  = rchtab.shape[0]
    ui['nodfv']  = any(ODFVF)
    ui['uunits'] = uunits

    # Numba can't do 'O' + str(i) stuff yet, so do it here. Also need new style lists
    Olabels = List()
    OVOLlabels = List()
    for i in range(nexits):
        Olabels.append(f'O{i+1}')
        OVOLlabels.append(f'OVOL{i+1}')

    #######################################################################################
    # the following section (1 of 3) added to HYDR by rb to handle dynamic code and special actions
    #######################################################################################
    # state_info is some generic things about the simulation
    # must be numba safe, so we don't just pass the whole state which is not
    state_info = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type)
    state_info['operation'], state_info['segment'], state_info['activity'] = state['operation'], state['segment'], state['activity']
    state_info['domain'], state_info['state_step_hydr'], state_info['state_step_om'] = state['domain'], state['state_step_hydr'], state['state_step_om']
    hsp2_local_py = state['hsp2_local_py']
    # It appears necessary to load this here, instead of from main.py, otherwise,
    # _hydr_() does not recognize the function state_step_hydr()? 
    if (hsp2_local_py != False):
        from hsp2_local_py import state_step_hydr
    else:
        from HSP2.state_fn_defaults import state_step_hydr
    # initialize the hydr paths in case they don't already reside here
    hydr_init_ix(state, state['domain'])
    # must split dicts out of state Dict since numba cannot handle mixed-type nested Dicts
    state_ix, dict_ix, ts_ix = state['state_ix'], state['dict_ix'], state['ts_ix']
    state_paths = state['state_paths']
    model_exec_list = state['model_exec_list'] # order of special actions and other dynamic ops
    op_tokens = state['op_tokens']
    #######################################################################################

    # Do the simulation with _hydr_   (ie run reaches simulation code)
    errors = _hydr_(ui, ts, COLIND, OUTDGT, rchtab, funct, Olabels, OVOLlabels,
                    state_info, state_paths, state_ix, dict_ix, ts_ix, state_step_hydr, op_tokens, model_exec_list)

    if 'O'    in ts:  del ts['O']
    if 'OVOL' in ts:  del ts['OVOL']

    # save initial outflow(s) from reach:
    uci['PARAMETERS']['ROS'] = ui['ROS']
    for i in range(nexits):
        uci['PARAMETERS']['OS'+str(i+1)] = ui['OS'+str(i+1)]
    
    return errors, ERRMSGS


@njit(cache=True)
def _hydr_(ui, ts, COLIND, OUTDGT, rowsFT, funct, Olabels, OVOLlabels, state_info, state_paths, state_ix, dict_ix, ts_ix, state_step_hydr, op_tokens, model_exec_list):
    errors = zeros(int(ui['errlen'])).astype(int64)

    steps  = int(ui['steps'])            # number of simulation steps
    delts  = ui['delt'] * 60.0           # seconds in simulation interval
    uunits = ui['uunits']
    nrows  = int(ui['nrows'])
    nexits = int(ui['nexits'])
    AUX1FG = int(ui['AUX1FG'])         # True means DEP, SAREA will be computed
    AUX2FG = int(ui['AUX2FG'])
    AUX3FG = int(ui['AUX3FG'])
    LKFG   = int(ui['LKFG'])           # flag, 1:lake, 0:stream
    length = ui['LEN'] * 5280.0        # length of reach, in feet
    if uunits == 2:
        length = ui['LEN'] * 1000.0    # length of reach, in meters
    DB50   = ui['DB50'] / 12.0         # mean diameter of bed material
    if uunits == 2:
        DB50 = ui['DB50'] / 40.0       # mean diameter of bed material
    DELTH  = ui['DELTH']
    stcor  = ui['STCOR']

    # units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
    VFACT = 43560.0
    AFACT = 43560.0
    LFACTA = 1.0
    SFACTA = 1.0
    TFACTA = 1.0
    # physical constants (English units)
    GAM = 62.4  # density of water
    GRAV = 32.2  # gravitational acceleration
    AKAPPA = 0.4  # von karmen constant
    if uunits == 2:
        # si units conversion constants, 1 hectare is 10000 sq m, assumes area input in hectares, vol in Mm3
        VFACT = 1.0e6
        AFACT = 10000.0
        # physical constants (English units)
        GAM = 9806.  # density of water
        GRAV = 9.81  # gravitational acceleration

    volumeFT = ts['volumeFT']
    depthFT  = ts['depthFT']
    sareaFT  = ts['sareaFT']

    nodfv  = ui['nodfv']
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
        ts['AVDEP'] = AVDEP = zeros(steps)
        ts['AVVEL'] = AVVEL = zeros(steps)
        ts['HRAD']  = HRAD  = zeros(steps)
        ts['TWID']  = TWID  = zeros(steps)

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
        rod1,od1[:] = demand(v1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt)
        rod2,od2[:] = demand(v2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt)
        a1 = (v2 - vol) / (v2 - v1)
        o[:] = a1 * od1[:] + (1.0 - a1) * od2[:]
        ro   = (a1 * rod1) + ((1.0 - a1) * rod2)
    else:
        ro,o[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt)  #$1159-1160

    # back to PHYDR
    if AUX1FG >= 1:
        dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, length, stcor, AUX1FG, errors) # initial

    # hydr-irrig
    irexit = int(ui['IREXIT']) -1    # irexit - exit number for irrigation withdrawals, 0 based ???
    #if irexit >= 1:
    irminv = ui['IRMINV']
    rirwdl = 0.0
    #rirdem = 0.0
    #rirsht = 0.0
    irrdem = 0.0

    # store initial outflow from reach:
    ui['ROS'] = ro
    for index in range(nexits):
        ui['OS' + str(index + 1)] = o[index]

    # other initial vars
    rovol = 0.0
    volev = 0.0
    IVOL0   = ts['IVOL']                   # the actual inflow in simulation native units

    #######################################################################################
    # the following section (2 of 3) added by rb to HYDR, this one to prepare for dynamic state including special actions
    #######################################################################################
    hydr_ix = hydr_get_ix(state_ix, state_paths, state_info['domain'])
    # these are integer placeholders faster than calling the array look each timestep
    o1_ix, o2_ix, o3_ix, ivol_ix = hydr_ix['O1'], hydr_ix['O2'], hydr_ix['O3'], hydr_ix['IVOL']
    ro_ix, rovol_ix, volev_ix, vol_ix = hydr_ix['RO'], hydr_ix['ROVOL'], hydr_ix['VOLEV'], hydr_ix['VOL']
    # handle varying length outdgt
    out_ix = arange(nexits)
    if nexits > 0:
        out_ix[0] = o1_ix
    if nexits > 1:
        out_ix[1] = o2_ix
    if nexits > 2:
        out_ix[2] = o3_ix
    #######################################################################################

    # HYDR (except where noted)
    for step in range(steps):
        convf  = CONVF[step]
        outdgt[:] = OUTDGT[step, :]
        colind[:] = COLIND[step, :]
        roseff = ro
        oseff[:] = o[:]

        #######################################################################################
        # the following section (3 of 3) added by rb to accommodate dynamic code, operations models, and special actions
        #######################################################################################
        # set state_ix with value of local state variables and/or needed vars
        # Note: we pass IVOL0, not IVOL here since IVOL has been converted to different units
        state_ix[ro_ix], state_ix[rovol_ix] = ro, rovol
        di = 0
        for oi in range(nexits):
            state_ix[out_ix[oi]] = outdgt[oi] 
        state_ix[vol_ix], state_ix[ivol_ix] = vol, IVOL0[step]
        state_ix[volev_ix] = volev
        # - these if statements may be irrelevant if default functions simply return
        #   when no objects are defined.
        if (state_info['state_step_om'] == 'enabled'):
            pre_step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
        if (state_info['state_step_hydr'] == 'enabled'):
            state_step_hydr(state_info, state_paths, state_ix, dict_ix, ts_ix, hydr_ix, step)
        if (state_info['state_step_om'] == 'enabled'):
            #print("trying to execute state_step_om()")
            # model_exec_list contains the model exec list in dependency order
            # now these are all executed at once, but we need to make them only for domain end points
            step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)   # traditional 'ACTIONS' done in here
        if ( (state_info['state_step_hydr'] == 'enabled')
            or (state_info['state_step_om'] == 'enabled') ):
            # Do write-backs for editable STATE variables
            # OUTDGT is writeable
            for oi in range(nexits):
                outdgt[oi] = state_ix[out_ix[oi]]
            # IVOL is writeable.
            # Note: we must convert IVOL to the units expected in _hydr_
            # maybe routines should do this, and this is not needed (but pass VFACT in state)
            IVOL[step] = state_ix[ivol_ix] * VFACT
        # End dynamic code step()
        #######################################################################################

        # vols, sas variables and their initializations  not needed.
        if irexit >= 0:             # irrigation exit is set, zero based number
            if rirwdl > 0.0:  # equivalent to OVOL for the irrigation exit
                vol = irminv if irminv > vol - rirwdl else vol - rirwdl
                if vol >= volumeFT[-1]:
                    errors[1] += 1 # ERRMSG1: extrapolation of rchtab will take place

                # DISCH with hydrologic routing
                indx = fndrow(vol, volumeFT)                 # find row index that brackets the VOL
                vv1 = volumeFT[indx]
                rod1,od1[:] = demand(vv1, rowsFT[indx,  :], funct, nexits, delts, convf, colind, outdgt)
                vv2 = volumeFT[indx+1]
                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt)
                aa1 = (vv2 - vol) / (vv2 - vv1)
                ro   = (aa1 * rod1)    + ((1.0 - aa1) * rod2)
                o[:] = (aa1 * od1[:])  + ((1.0 - aa1) * od2[:])

                # back to HYDR
                if AUX1FG >= 1:     # recompute surface area and depth
                    dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, length, stcor,
                                                                 AUX1FG, errors)
            else:
                irrdem =  0.0
            #o[irexit] = 0.0                                                   #???? not used anywhere, check if o[irexit]

        prsupy = PREC[step] * sarea
        if uunits == 2:
            prsupy = PREC[step] * sarea / 3.281
        volt   = vol + IVOL[step] + prsupy
        volev = 0.0
        if AUX1FG:                  # subtract evaporation
            volpev = POTEV[step] * sarea
            if uunits == 2:
                volpev = POTEV[step] * sarea / 3.281
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

            if roseff > 0.0: # numba limitation, cant combine into one line
                ovol[:] = (rovol/roseff) * oseff[:]
            else:
                ovol[:] = rovol / nexits

        else:   # case 1 or 2
            oint = volint * facta1      # == ointsp, so ointsp variable dropped
            if nodfv:
                # ROUTE
                rodz,odz[:] = demand(0.0, rowsFT[zeroindex,:], funct, nexits, delts, convf, colind,  outdgt)
                if oint > rodz:
                    # SOLVE - case 1-- outflow demands can be met in full
                    # premov will be used to check whether we are in a trap, arbitrary value
                    premov = -20
                    move   = 10

                    vv1 = volumeFT[indx]
                    rod1,od1[:] = demand(vv1, rowsFT[indx, :], funct, nexits, delts, convf,colind, outdgt)
                    vv2 = volumeFT[indx+1]
                    rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt)

                    while move != 0:
                        facta2 = rod1 - rod2
                        factb2 = vv2 - vv1
                        factc2 = vv2 * rod1 - vv1 * rod2
                        det = facta1 * factb2 - facta2
                        if det <= 0.0:
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
                                od1[:] = od2[:]
                                rod1   = rod2
                                vv2    = volumeFT[indx+1]
                                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, nexits, delts, convf, colind, outdgt)
                        elif vol < vv1:
                            indx  -= 1
                            move   = -1
                            vv2    = vv1
                            od2[:] = od1[:]
                            rod2   = rod1
                            vv1    = volumeFT[indx]
                            rod1,od1[:] = demand(vv1, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt)
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
                rod1,od1[:] = demand(vol, rowsFT[indx,:], funct, nexits, delts, convf, colind, outdgt)
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
            OVOL[step,:] = ovol[:] / VFACT
        PRSUPY[step] = prsupy / VFACT
        RO[step]     = ro     * SFACTA * LFACTA
        ROVOL[step]  = rovol  / VFACT
        VOLEV[step]  = volev  / VFACT
        VOL[step]    = vol    / VFACT

        if AUX1FG:   # compute final depth, surface area
            if vol >= topvolume:
                errors[1] += 1       # ERRMSG1: extrapolation of rchtab
            indx = fndrow(vol, volumeFT)
            dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, length, stcor, AUX1FG, errors)
            DEP[step]   = dep
            SAREA[step] = sarea / AFACT

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
                    # SHEAR; ustar (bed shear velocity), tau (bed shear stress)
                    if LKFG:              # flag, 1:lake, 0:stream
                        ustar = avvel / (17.66 + (log10(avdep / (96.5 * DB50))) * 2.3 / AKAPPA)
                        tau   =  GAM/GRAV * ustar**2              #3796
                    else:
                        hrad = (avdep*twid)/(2.0*avdep + twid) # hydraulic radius, manual eq 41
                        slope = DELTH / length
                        ustar = sqrt(GRAV * slope * hrad)
                        tau = (GAM * slope) * hrad
                else:
                    ustar = 0.0
                    tau   = 0.0
                    hrad  = 0.0
                USTAR[step] = ustar * LFACTA
                TAU[step]   = tau   * TFACTA

            AVDEP[step] = avdep
            AVVEL[step] = avvel
            HRAD[step]  = hrad
            TWID[step]  = twid
    # END MAIN LOOP

    # NUMBA limitation for ts, and saving to HDF5 file is in individual columns
    if nexits > 1:
        for i in range(nexits):
            ts[Olabels[i]]    = O[:,i]
            ts[OVOLlabels[i]] = OVOL[:,i]
    return errors


@njit(cache=True)
def fndrow(v, volFT):
    ''' finds highest index in FTable volume column whose volume  < v'''
    for indx,vol in enumerate(volFT):
        if v < vol:
            return indx-1
    return len(volFT) - 2


@njit(cache=True)
def demand(vol, rowFT, funct, nexits, delts, convf, colind, outdgt):
    od = zeros(nexits)
    for i in range(nexits):
        col = colind[i]
        icol = int(col)
        if icol != 0:
            diff = col - float(icol)
            if diff >= 1.0e-6:
                _od1 = rowFT[icol-1]
                odfv = _od1 + diff * (_od1 - rowFT[icol]) * convf
            else:
                odfv = rowFT[icol-1] * convf
        else:
            odfv = 0.0
        odgt = outdgt[i]

        if   odfv == 0.0 and odgt == 0.0:
            od[i] = 0.0
        elif odfv != 0.0 and odgt == 0.0:
            od[i] = odfv
        elif odfv == 0.0 and odgt != 0.0:
            od[i] = odgt
        else:
            if   funct[i] == 1: od[i] = min(odfv,odgt)
            elif funct[i] == 2: od[i] = max(odfv,odgt)
            elif funct[i] == 3: od[i] = odfv + odgt
            elif funct[i] == 4: od[i] = max(odfv, (vol - odgt) / delts)
    return od.sum(), od


@njit(cache=True)
def auxil(volumeFT, depthFT, sareaFT, indx, vol, length, stcor, AUX1FG, errors):
    '''Compute depth, stage, surface area, average depth, topwidth and hydraulic radius'''
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

        avdep = vol / sarea                           # average depth calculation, manual eq (39)
        twid = sarea / length                         # top-width calculation, manual eq (40)
        hrad = (avdep * twid) / (2.0 * avdep + twid)  # hydraulic radius, manual eq (41)
    elif AUX1FG == 2:
        dep   = depthFT[indx]    # removed in HSPF 12.4
        sarea = sareaFT[indx]
        avdep = 0.0
        twid = sarea / length
        hrad = 0.0
    else:
        dep   = 0.0
        sarea = 0.0
        avdep = 0.0
        twid  = 0.0
        hrad  = 0.0

    stage = dep + stcor    # stage calculation and output, manual eq (37)

    return dep, stage, sarea, avdep, twid, hrad

def expand_HYDR_masslinks(flags, uci, dat, recs):
    if flags['HYDR']:
        # IVOL
        rec = {}
        rec['MFACTOR'] = dat.MFACTOR
        rec['SGRPN'] = 'HYDR'
        if dat.SGRPN == "ROFLOW":
            rec['SMEMN'] = 'ROVOL'
        else:
            rec['SMEMN'] = 'OVOL'
        rec['SMEMSB1'] = dat.SMEMSB1
        rec['SMEMSB2'] = dat.SMEMSB2
        rec['TMEMN'] = 'IVOL'
        rec['TMEMSB1'] = dat.TMEMSB1
        rec['TMEMSB2'] = dat.TMEMSB2
        rec['SVOL'] = dat.SVOL
        recs.append(rec)
    return recs
    