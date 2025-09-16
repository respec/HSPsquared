from hsp2.hsp2.classes.rchres.handler import HandlerRCHRES

# Define the spec for the ModelRCHRES class
# note: we store the float, int, array props on the HandlerRCHRES object for convenience
#       but we could just as easily opt to keep it in a separate location loaded via include
model_rchres_spec = model_base + model_make_spec(HandlerRCHRES.float_props, float64) + model_make_spec(HandlerRCHRES.farray_props, float64[:]) + model_make_spec(HandlerRCHRES.carray_props, types.unicode_type) + model_make_spec(HandlerRCHRES.int_props, int32)

@jitclass(model_rchres_spec )
class ModelRCHRES:
    def __init__(self):
        # Note: no inheritance in jitclass as of py 3.12, so, the following cannot work:
        # super(ModelRCHRES, self).__init__()
        # must copy any base method stuff from ModelBase
        self.path = '' # must initialize
        self.value = 0
        self.ix = 0 # this is the pointer to the current value for this object in state_ix
        self.state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
        self.state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        self.ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        return
    
    def step(self, step):
        self.state_read()
        self.step_HYDR(step)
        self.step_RQUAL(step)
        self.step_SEDTRN(step)
    
    # state_read_vars: must declare class props that are mutable in state 
    # state_write_vars: all props to expose for reading
    def state_read(self):
        # If a particular variable should be shared, but *not* mutable 
        #     it would NOT be in state_read_vars but would be in state_write_vars
        self.IVOL = self.state[self.path,'/IVOL']
        for i in range(nexits):
            self.outdgt[i] = self.state[self.path,'/O',i]
        return
    
    def state_write(self):
        state_write_vars = ['AUX1FG', 'AUX2FG', 'AUX3FG', 'AVDEP', 'AVVEL', 'DEP', 'HRAD', 'IRRDEM', 'LEN', 'LKFG', 'PRSUPY', 'RO', 'ROVOL', 'SAREA', 'STCOR', 'TAU', 'TWID', 'USTAR', 'VOL', 'VOLEV']
    
    def step_HYDR(self, step):
        # options for modes right now (not selectable ATM)
        self.step_HYDRix(self, step)
        #self.step_HYDRstate(self, step)
        return
    
    def step_SEDTRN(self, step):
        # options for modes right now (not selectable ATM)
        return
    
    def step_HYDRix(self, step):
        # self.state_paths[self.path,'/IVOL']
        ix = 1
        self.ovol[0] = self.IVOL * self.ROVOL / self.RO
        # versus alternate call external jitted function
        self.ovol[0] = self.IVOL * self.ROVOL / self.RO
        return
    
    def step_HYDRstate(self, step):
        ipath = self.path + '/IVOL'
        if (step < 2):
            print("ipath = ", ipath)
        self.ovol[0] = self.state_paths[ipath] * self.ROVOL / self.RO
        return
    
    def step_RQUAL(self, step):
        return

@njit
def fn_hydr_step(rchres, step):
    # will need to pass in dt and get all others from the object state memory
    rchres.ovol[0] = rchres.state_ix[ix] * rchres.ROVOL / rchres.RO


# May be a method of the reach, but with many support functions to reduce size
@njit(cache=True)
def hydr_step(rchres, state, ts, step):
    
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
    if rchres.uunits == 2:
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
    facta1 = 1.0 / (coks * rchres.delts)

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
        rod1,od1[:] = demand(v1, rowsFT[indx,  :], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
        rod2,od2[:] = demand(v2, rowsFT[indx+1,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
        a1 = (v2 - vol) / (v2 - v1)
        o[:] = a1 * od1[:] + (1.0 - a1) * od2[:]
        ro   = (a1 * rod1) + ((1.0 - a1) * rod2)
    else:
        ro,o[:] = demand(vol, rowsFT[indx,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)  #$1159-1160

    # back to PHYDR
    if rchres.AUX1FG >= 1:
        dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, rchres.length, stcor, rchres.AUX1FG, errors) # initial

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
    for index in range(rchres.nexits):
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
    out_ix = arange(rchres.nexits)
    if rchres.nexits > 0:
        out_ix[0] = o1_ix
    if rchres.nexits > 1:
        out_ix[1] = o2_ix
    if rchres.nexits > 2:
        out_ix[2] = o3_ix
    #######################################################################################
    
    # HYDR (except where noted)
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
    for oi in range(rchres.nexits):
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
        for oi in range(rchres.nexits):
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
            rod1,od1[:] = demand(vv1, rowsFT[indx,  :], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
            vv2 = volumeFT[indx+1]
            rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
            aa1 = (vv2 - vol) / (vv2 - vv1)
            ro   = (aa1 * rod1)    + ((1.0 - aa1) * rod2)
            o[:] = (aa1 * od1[:])  + ((1.0 - aa1) * od2[:])

            # back to HYDR
            if rchres.AUX1FG >= 1:     # recompute surface area and depth
                dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, rchres.length, stcor,
                                                                rchres.AUX1FG, errors)
        else:
            irrdem =  0.0
        #o[irexit] = 0.0                                                   #???? not used anywhere, check if o[irexit]

    prsupy = PREC[step] * sarea
    if rchres.uunits == 2:
        prsupy = PREC[step] * sarea / 3.281
    volt   = vol + IVOL[step] + prsupy
    volev = 0.0
    if rchres.AUX1FG:                  # subtract evaporation
        volpev = POTEV[step] * sarea
        if rchres.uunits == 2:
            volpev = POTEV[step] * sarea / 3.281
        if volev >= volt:
            volev = volt
            volt = 0.0
        else:
            volev = volpev
            volt -= volev

    # ROUTE/NOROUT  calls
    # common code
    volint = volt - (ks * roseff * rchres.delts)    # find intercept of eq 4 on vol axis
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
            ovol[:] = rovol / rchres.nexits

    else:   # case 1 or 2
        oint = volint * facta1      # == ointsp, so ointsp variable dropped
        if nodfv:
            # ROUTE
            rodz,odz[:] = demand(0.0, rowsFT[zeroindex,:], funct, rchres.nexits, rchres.delts, convf, colind,  outdgt)
            if oint > rodz:
                # SOLVE - case 1-- outflow demands can be met in full
                # premov will be used to check whether we are in a trap, arbitrary value
                premov = -20
                move   = 10

                vv1 = volumeFT[indx]
                rod1,od1[:] = demand(vv1, rowsFT[indx, :], funct, rchres.nexits, rchres.delts, convf,colind, outdgt)
                vv2 = volumeFT[indx+1]
                rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)

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
                        if indx >= rchres.nrows-2:
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
                            rod2,od2[:] = demand(vv2, rowsFT[indx+1,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
                    elif vol < vv1:
                        indx  -= 1
                        move   = -1
                        vv2    = vv1
                        od2[:] = od1[:]
                        rod2   = rod1
                        vv1    = volumeFT[indx]
                        rod1,od1[:] = demand(vv1, rowsFT[indx,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
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
                for i in range(rchres.nexits):
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
            rod1,od1[:] = demand(vol, rowsFT[indx,:], funct, rchres.nexits, rchres.delts, convf, colind, outdgt)
            if oint >= rod1: #case 1 -outflow demands are met in full
                ro   = rod1
                vol  = volint - coks * ro * rchres.delts
                if vol < 1.0e-5:
                    vol = 0.0
                o[:] = od1[:]
            else:    # case 2 -outflow demands cannot be met in full
                ro  = 0.0
                for i in range(rchres.nexits):
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
        ovol[:] = (ks * oseff[:] + coks * o[:]) * rchres.delts
        rovol   = (ks * roseff   + coks * ro)   * rchres.delts

    # HYDR
    if rchres.nexits > 1:
        O[step,:]    = o[:]    * SFACTA * LFACTA
        OVOL[step,:] = ovol[:] / VFACT
    PRSUPY[step] = prsupy / VFACT
    RO[step]     = ro     * SFACTA * LFACTA
    ROVOL[step]  = rovol  / VFACT
    VOLEV[step]  = volev  / VFACT
    VOL[step]    = vol    / VFACT

    if rchres.AUX1FG:   # compute final depth, surface area
        if vol >= topvolume:
            errors[1] += 1       # ERRMSG1: extrapolation of rchtab
        indx = fndrow(vol, volumeFT)
        dep, stage, sarea, avdep, twid, hrad = auxil(volumeFT, depthFT, sareaFT, indx, vol, rchres.length, stcor, rchres.AUX1FG, errors)
        DEP[step]   = dep
        SAREA[step] = sarea / AFACT

        if vol > 0.0 and sarea > 0.0:
            twid  = sarea / rchres.length
            avdep = vol / sarea
        elif rchres.AUX1FG == 2:
            twid = sarea / rchres.length
            avdep = 0.0
        else:
            twid = 0.0
            avdep = 0.0

        if rchres.AUX2FG:
            avvel = (rchres.length * ro / vol) if vol > 0.0 else 0.0
        if rchres.AUX3FG:
            if avdep > 0.0:
                # SHEAR; ustar (bed shear velocity), tau (bed shear stress)
                if rchres.LKFG:              # flag, 1:lake, 0:stream
                    ustar = avvel / (17.66 + (log10(avdep / (96.5 * rchres.db50u))) * 2.3 / AKAPPA)
                    tau   =  GAM/GRAV * ustar**2              #3796
                else:
                    hrad = (avdep*twid)/(2.0*avdep + twid) # hydraulic radius, manual eq 41
                    slope = rchres.DELTH / rchres.length
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

    return errors

# THIS IS THE END OF THE HYDR LOOP, NOT YET IMPLEMENTED
def hydr_finish(rchres):
    # NUMBA limitation for ts, and saving to HDF5 file is in individual columns
    if rchres.nexits > 1:
        for i in range(rchres.nexits):
            rchres.ts[Olabels[i]]    = rchres.O[:,i]
            rchres.ts[OVOLlabels[i]] = rchres.OVOL[:,i]