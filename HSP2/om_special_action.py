"""
The class SpecialAction is used to support original HSPF ACTIONS.
"""
from HSP2.state import *
from HSP2.om import *
from HSP2.om_model_object import ModelObject
from numba import njit
class SpecialAction(ModelObject):
    def __init__(self, name, container = False, model_props = {}):
        super(SpecialAction, self).__init__(name, container, model_props)

        self.optype = 100 # Special Actions start indexing at 100 
    
    def parse_model_props(self, model_props, strict=False):
        super().parse_model_props(model_props, strict)
        # comes in as row from special ACTIONS table
        # ex: {
        #   'OPTYP': 'RCHRES', 'RANGE1': '1', 'RANGE2': '', 'DC': 'DY', 'DS': '', 
        #   'YR': '1986', 'MO': '3', 'DA': '1', 'HR': '12', 'MN': '', 
        #   'D': '2', 'T': 3, 'VARI': 'IVOL', 'S1': '', 'S2': '', 
        #   'AC': '+=', 'VALUE': 30.0, 'TC': '', 'TS': '', 'NUM': '', 'CURLVL': 0, 
        # defined by:
        # - operand1, i.e. variable to access + update, path = /STATE/[OPTYP]_[op_abbrev][RANGE1]/[VARI]
        # - action(operation) to perform = AC
        # - operand2, a numeric value for simple ACTION = [VALUE]
        # note: [op_abbrev] is *maybe* the first letter of the OPTYP?  Not a very good idea to have a coded convention like that
        print("Creating ACTION with props", model_props) 
        self.op_type = self.handle_prop(model_props, 'OPTYP')
        self.range1 = self.handle_prop(model_props, 'RANGE1')
        self.range2 = self.handle_prop(model_props, 'RANGE2')
        self.ac = '=' # set the default, and also adds a property for later testing.
        self.ac = self.handle_prop(model_props, 'AC') # must handle this before we handle the operand VALUE to check for DIV by Zero
        self.vari = self.handle_prop(model_props, 'VARI')
        self.op2_val = self.handle_prop(model_props, 'VALUE')
        self.op2_ix = self.constant_or_path('op_val', self.op2_val) # constant values must be added to STATE and thus are referenced by their state_ix number
        self.num = self.handle_prop(model_props, 'NUM', False, 1) # number of times to perform action
        self.timer_ix = self.handle_prop(model_props, 'when', False, 1) # when to begin the first attempt at action
        self.ctr_ix = self.constant_or_path('ctr', 0) # this initializes the counter for how many times an action has been performed
        # now add the state value that we are operating on (the target) as an input, so that this gets executed AFTER this is set initially
        self.add_input('op1', ('/STATE/' + self.op_type + '_' + self.op_type[0] + str(self.range1).zfill(3) + "/" + self.vari ), 2, True )
        # @tbd: support time enable/disable
        #       - check if time ops have been set and add as inputs like "year", or "month", etc could give explicit path /STATE/year ...
        #       - add the time values to match as constants i.e. self.constant_or_path()
    
    def handle_prop(self, model_props, prop_name, strict = False, default_value = None ):
        # Insure all values are legal ex: no DIV by Zero
        prop_val = super().handle_prop(model_props, prop_name, strict, default_value )
        if (prop_name == 'VALUE') and (self.ac == '/='):
            if (prop_val == 0) or (prop_val == None):
                raise Exception("Error: in properties passed to "+ self.name + " AC must be non-zero or non-Null .  Object creation halted. Path to object with error is " + self.state_path)
        if (prop_name == 'AC'):
           self.handle_ac(prop_val)
        if (prop_name == 'when'):
           # when to perform this?  timestamp or time-step index
           prop_val = 0
           si = ModelObject.model_object_cache['/STATE/timer']
           if len(model_props['YR']) > 0:
               # translate date to equivalent model step
               datestring = model_props['YR'] + '-' + model_props['MO'] + '-' + \
                            model_props['DA'] + ' ' + model_props['HR'] + ':' + \
                            model_props['MN'] + ':00'
               if datestring in si.model_props_parsed['tindex']:
                   prop_val = si.model_props_parsed['tindex'].get_loc(datestring)
        if (prop_name == 'NUM') and (prop_val == ''):
            prop_val = default_value
        return prop_val
    
    def handle_ac(self, ac):
        # cop_code 0: =/eq, 1: </lt, 2: >/gt, 3: <=/le, 4: >=/ge, 5: <>/ne 
        cop_codes = {
            '=': 1,
            '+=': 2,
            '-=': 3,
            '*=': 4,
            '/=': 5,
            'MIN': 6
        }
        # From HSPF UCI docs:
        # 1 = T= A 
        # 2 += T= T+ A
        # 3 -= T= T- A 
        # 4 *= T= T*A
        # 5 /= T= T/A 
        # 6 MIN T= Min(T,A)
        # 7 MAX T= Max(T,A) 
        # 8 ABS T= Abs(A)
        # 9 INT T= Int(A) 
        # 10 ^= T= T^A
        # 11 LN T= Ln(A) 
        # 12 LOG T= Log10(A)
        # 13 MOD T= Mod(T,A)
        if not (is_float_digit(ac)):
            if not (ac in cop_codes.keys()):
               raise Exception("Error: in "+ self.name + " AC (" + ac + ") not supported.  Object creation halted. Path to object with error is " + self.state_path)
            opid = cop_codes[ac]
            self.ac = ac
        else:
            # this will fail catastrophically if the requested function is not supported
            # which is a good thing
            if not (ac in cop_codes.values()):
               raise Exception("Error: in "+ self.name + "numeric AC (" + ac + ") not supported.  Object creation halted. Path to object with error is " + self.state_path)
            opid = ac
            self.ac = list(cop_codes.keys())[list(cop_codes.values()).index(ac) ]
        self.opid = opid

    def tokenize(self):
        # call parent method to set basic ops common to all 
        super().tokenize() # sets self.ops = op_type, op_ix
        self.ops = self.ops + [self.inputs_ix['op1'], self.opid, self.op2_ix, self.timer_ix, self.ctr_ix, self.num]
        # @tbd: check if time ops have been set and tokenize accordingly
        print("Specl", self.name, "tokens", self.ops)
    
    def add_op_tokens(self):
        # this puts the tokens into the global simulation queue 
        # can be customized by subclasses to add multiple lines if needed.
        super().add_op_tokens()
    
    @staticmethod
    def hdf5_load_all(hdf_source):
       specla=hdf_source['/SPEC_ACTIONS/ACTIONS/table']
       for idx, x in np.ndenumerate(specla):
           print(x[1].decode("utf-8"),x[2].decode("utf-8"), x[13].decode("utf-8"), x[16].decode("utf-8"), x[17])


# njit functions for runtime

@njit(cache=True)
def step_special_action(op, state_ix, dict_ix, step):
    ix = op[1] # ID of this op
    # these indices must be adjusted to reflect the number of common op tokens
    # SpecialAction has:
    # - type of condition (+=, -=, ...)
    # - operand 1 (left side)
    # - operand 2 (right side) 
    # @tbd: check if time ops have been set and enable/disable accordingly
    #     - 2 ops will be added for each time matching switch, the state_ix of the time element (year, month, ...) and the state_ix of the constant to match
    #     - matching should be as simple as if (state_ix[tix1] <> state_ix[vtix1]): return state_ix[ix1] (don't modify the value)
    #     - alternative: save the integer timestamp or timestep of the start, and if step/stamp > value, enable
    # @tbd: add number of repeats, and save the value of repeats in a register
    ix1 = op[2] # ID of source of data and destination of data
    sop = op[3]
    ix2 = op[4]
    tix = op[5] # which slot is the time comparison in?
    if (tix in state_ix and step < state_ix[tix]):
        return
    ctr_ix = op[6] # id of the counter variable
    num_ix = op[7] # max times to complete
    num_done = state_ix[ctr_ix]
    num = state_ix[num_ix] # num to complete
    if (tix in state_ix and num_done >= num):
       return
    else:
        if sop == 1:
            result = state_ix[ix2]
        elif sop == 2:
            result = state_ix[ix1] + state_ix[ix2]
        elif sop == 3:
            result = state_ix[ix1] - state_ix[ix2]
        elif sop == 4:
            result = state_ix[ix1] * state_ix[ix2]
        elif sop == 5:
            result = state_ix[ix1] / state_ix[ix2]
    
    # set value in target
    # tbd: handle this with a model linkage? cons: this makes a loop since the ix1 is source and destination
    
    state_ix[ix1] = result
    state_ix[op[1]] = result
    return result

