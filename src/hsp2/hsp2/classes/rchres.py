#from hsp2.hsp2.classes.base import HandlerBase

class HandlerRCHRES(HandlerBase):
    float_props = ['DB50', 'AUX1FG', 'AUX2FG', 'AUX3FG', 'AVDEP', 'AVVEL', 'DB50', 'DELTH', 'DEP', 'HRAD', 'IRRDEM', 'LEN', 'LKFG', 'PRSUPY', 'RO', 'ROVOL', 'SAREA', 'STCOR', 'TAU', 'TWID', 'USTAR', 'VOL', 'VOLEV']
    int_props = ['nrows', 'nexits', 'AUX1FG', 'AUX2FG', 'AUX3FG', 'LKFG', 'LEN', 'DB50','DELTH','STCOR', 'UUNITS']
    farray_props = ['o', 'odz', 'ovol', 'oseff', 'od1', 'od2', 'outdgt', 'colind']
    carray_props = ['state_read_vars', 'state_write_vars']
    def __init__(self, props = None):
        super(HandlerRCHRES, self).__init__(props)
        return
    
    def make_model(self):
        # Create an empty model
        model = ModelRCHRES()
        print("Creating ModelRCHRES with props:", self.model_props)
        # Populate model props and return
        self.set_props(model, self.model_props)
        self.init_nexits(model)
        return model
    
    def init_nexits(self, model):
        # faster to preallocate arrays - like MATLAB)
        nexprops = ['o', 'odz', 'ovol', 'oseff', 'outdgt', 'od1', 'od2', 'colind']
        for i in nexprops:
            setattr(model, i, zeros(model.nexits))
        return

# Define the spec for the ModelRCHRES class
# note: we store the float, int, array props on the HandlerRCHRES object for convenience
#       but we could just as easily opt to keep it in a separate location loaded via include
model_rchres_spec = model_base + model_make_spec(HandlerRCHRES.float_props, float64) + model_make_spec(HandlerRCHRES.farray_props, float64[:]) + model_make_spec(HandlerRCHRES.carray_props, types.unicode_type) + model_make_spec(HandlerRCHRES.int_props, int32)

@jitclass(model_rchres_spec )
class ModelRCHRES:
    def __init__(self, props = None):
        # calls shared external method to avoid duplication due to lack of inheritance in jitclass
        self.path = '' # must initialize
        return
    
    def step(self, step):
        self.state_read()
        self.step_HYDR(step)
        self.step_RQUAL(step)
    
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
        state_write_vars = ['AUX1FG', 'AUX2FG', 'AUX3FG', 'AVDEP', 'AVVEL', 'DB50', 'DELTH', 'DEP', 'HRAD', 'IRRDEM', 'LEN', 'LKFG', 'PRSUPY', 'RO', 'ROVOL', 'SAREA', 'STCOR', 'TAU', 'TWID', 'USTAR', 'VOL', 'VOLEV']
    
    def step_HYDR(self, step):
        # options for modes right now (not selectable ATM)
        self.step_HYDRix(self, step)
        #self.step_HYDRstate(self, step)
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
    convf  = CONVF[step]
    outdgt[:] = OUTDGT[step, :]
    colind[:] = COLIND[step, :]
    roseff = ro
    oseff[:] = o[:]

    #######################################################################################
    # the following section (3 of 3) added by rb to accommodate dynamic code, operations models, and special actions
    #######################################################################################
    # set state.state_ix with value of local state variables and/or needed vars
    # Note: we pass IVOL0, not IVOL here since IVOL has been converted to different units
    state.state_ix[ro_ix], state.state_ix[rovol_ix] = ro, rovol
    di = 0
    for oi in range(nexits):
        state.state_ix[out_ix[oi]] = outdgt[oi] 
    state.state_ix[vol_ix], state.state_ix[ivol_ix] = vol, IVOL0[step]
    state.state_ix[volev_ix] = volev
    # - these if statements may be irrelevant if default functions simply return
    #   when no objects are defined.
    if (state_info['state_step_om'] == 'enabled'):
        pre_step_model(model_exec_list, op_tokens, state.state_ix, dict_ix, ts_ix, step)
    if (state_info['state_step_hydr'] == 'enabled'):
        state_step_hydr(state_info, state_paths, state.state_ix, dict_ix, ts_ix, hydr_ix, step)
    if (state_info['state_step_om'] == 'enabled'):
        #print("trying to execute state_step_om()")
        # model_exec_list contains the model exec list in dependency order
        # now these are all executed at once, but we need to make them only for domain end points
        step_model(model_exec_list, op_tokens, state.state_ix, dict_ix, ts_ix, step)   # traditional 'ACTIONS' done in here
    if ( (state_info['state_step_hydr'] == 'enabled')
        or (state_info['state_step_om'] == 'enabled') ):
        # Do write-backs for editable STATE variables
        # OUTDGT is writeable
        for oi in range(nexits):
            outdgt[oi] = state.state_ix[out_ix[oi]]
        # IVOL is writeable.
        # Note: we must convert IVOL to the units expected in _hydr_
        # maybe routines should do this, and this is not needed (but pass VFACT in state)
        IVOL[step] = state.state_ix[ivol_ix] * VFACT
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