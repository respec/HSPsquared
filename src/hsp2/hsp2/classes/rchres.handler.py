from hsp2.hsp2.classes.base import HandlerBase

class HandlerRCHRES(HandlerBase):
    float_props = ['DB50', 'db50u', 'AUX1FG', 'AUX2FG', 'AUX3FG', 'AVDEP', 'AVVEL', 'DELTH', 'DEP', 'HRAD', 'IRRDEM', 'LEN', 
                   'LKFG', 'PRSUPY', 'RO', 'ROVOL', 'SAREA', 'LEN', 'length', 'STCOR', 'TAU', 'TWID', 'USTAR', 'VOL', 'VOLEV', 'delts']
    int_props = ['nrows', 'nexits', 'AUX1FG', 'AUX2FG', 'AUX3FG', 'LKFG', 'DELTH','STCOR', 'uunits']
    farray_props = ['o', 'odz', 'ovol', 'oseff', 'od1', 'od2', 'outdgt', 'colind', 'CONVF']
    carray_props = ['state_read_vars', 'state_write_vars']
    # props with number of exits
    nexprops = ['o', 'odz', 'ovol', 'oseff', 'outdgt', 'od1', 'od2', 'colind']
    def __init__(self, model_props = None):
        super(HandlerRCHRES, self).__init__(model_props)
        return
    
    def prep_run(model):
        # insure that all local timeseries linkages are correct, inputs are sound
        # later we will test if this is advantageous or if the notation
        # self.inputs['PREC'] will work as well in equations
        model.POTEV = model.ts['POTEV']
        model.PREC = model.ts['PREC']
        model.CONVF = model.ts['CONVF']
        model.convf = model.CONVF[0]
        model.volumeFT = model.ts['volumeFT']
        model.depthFT = model.ts['depthFT']
        model.sareaFT = model.ts['sareaFT']
        # units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
        model.VFACT = 43560.0
        model.AFACT = 43560.0
        model.LFACTA = 1.0
        model.SFACTA = 1.0
        model.TFACTA = 1.0
        # physical constants (English units)
        model.GAM = 62.4  # density of water
        model.GRAV = 32.2  # gravitational acceleration
        model.length = model.LEN * 5280.0 # length of reach, in feet
        AKAPPA = 0.4  # von karmen constant
        if model.uunits == 2:
            # si units conversion constants, 1 hectare is 10000 sq m, assumes area input in hectares, vol in Mm3
            model.VFACT = 1.0e6
            model.AFACT = 10000.0
            # physical constants (English units)
            model.GAM = 9806.  # density of water
            model.GRAV = 9.81  # gravitational acceleration
        model.IVOL = model.ts['IVOL']  * VFACT # or sum civol, zeros if no inflow ???
        model.CONVF = model.ts['CONVF']
        model.CONVF = model.ts['CONVF']
        if model.AUX1FG:
            model.ts['DEP']   = DEP   = zeros(steps)
            model.ts['SAREA'] = SAREA = zeros(steps)
            model.ts['USTAR'] = USTAR = zeros(steps)
            model.ts['TAU']   = TAU   = zeros(steps)
            model.ts['AVDEP'] = AVDEP = zeros(steps)
            model.ts['AVVEL'] = AVVEL = zeros(steps)
            model.ts['HRAD']  = HRAD  = zeros(steps)
            model.ts['TWID']  = TWID  = zeros(steps)
        
        
        if model.uunits == 2:
            model.db50u = model.DB50 / 40.0 # mean diameter of bed material
        else:
            model.db50u   = model.DB50 / 12.0 # mean diameter of bed material
        
        return
    
    def set_props(self, model, strict = False ):
        # sub classes can check to see if this is in the right format, or change to numba compatible types etc.
        super().set_props(model, strict)
        # Handle special props
        if 'delt' in self.model_props:
            model.delts = self.model_props['delt'] * 60.0
    
    def init_nexits(self, model):
        # faster to preallocate arrays - like MATLAB)
        for i in self.nexprops:
            setattr(model, i, zeros(model.nexits))
        return
