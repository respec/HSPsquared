"""
The class SpecialAction is used to support original HSPF ACTIONS.
"""
from HSP2.state import *
from HSP2.om import *
from HSP2.om_model_object import ModelObject
from numba import njit
class SpecialAction(ModelObject):
    def __init__(self, name, container = False, model_props = []):
        super(SpecialAction, self).__init__(name, container, model_props)

        self.optype = 100 # Special Actions start indexing at 100 
    
    def parse_model_props(self, model_props, strict=False):
        print("SpecialAction.parse_model_props() called")
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
        self.op_type = self.handle_prop(model_props, 'OPTYP')
        self.range1 = self.handle_prop(model_props, 'RANGE1')
        self.range2 = self.handle_prop(model_props, 'RANGE2')
        self.ac = self.handle_prop(model_props, 'AC') # must handle this before we handle the operand to check for DIV by Zero
        self.vari = self.handle_prop(model_props, 'VARI')
        self.op2_val = self.handle_prop(model_props, 'VALUE')
        self.op2_ix = self.constant_or_path('op_val', self.op2_val) # constant values must be added to STATE and thus are referenced by their state_ix number
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
        return prop_val
    
    def tokenize(self):
        # call parent method to set basic ops common to all 
        super().tokenize() # sets self.ops = op_type, op_ix
        # cop_code 0: =/eq, 1: </lt, 2: >/gt, 3: <=/le, 4: >=/ge, 5: <>/ne 
        cop_codes = {
            '+=': 0,
            '-=': 1,
            '*=': 2,
            '/=': 3
        }
        self.ops = self.ops + [self.inputs_ix['op1'], cop_codes[self.ac], self.op2_ix]
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

@njit
def step_saction(op, state_ix, dict_ix, step):
    ix = op[1] # ID of this op
    # these indices must be adjusted to reflect the number of common op tokens
    # SpecialAction has:
    # - type of condition (+=, -=, ...)
    # - operand 1 (left side)
    # - operand 2 (right side) 
    # @tbd: check if time ops have been set and enable/disable accordingly
    #     - 2 ops will be added for each time matching switch, the state_ix of the time element (year, month, ...) and the state_ix of the constant to match
    #     - matching should be as simple as if (state_ix[tix1] <> state_ix[vtix1]): return state_ix[ix1] (don't modify the value)
    # 
    ix1 = op[2] # ID of source of data and destination of data
    sop = op[3]
    ix2 = op[4]
    if sop == 0:
      result = state_ix[ix1] + state_ix[ix2]
    if sop == 1:
      result = state_ix[ix1] - state_ix[ix2]
    if sop == 2:
      result = state_ix[ix1] * state_ix[ix2]
    if sop == 3:
      result = state_ix[ix1] / state_ix[ix2]
    # set value in target
    # tbd: handle this with a link?
    state_ix[ix1] = result
    return result

