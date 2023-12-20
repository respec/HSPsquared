"""
The class SpecialAction is used to support original HSPF ACTIONS.
"""
from HSP2.state import *
from HSP2.om import *
from HSP2.om_model_object import ModelObject
from numba import njit
class SpecialAction(ModelObject):
    def __init__(self, name, container = False, model_props = []):
        super(ModelObject, self).__init__(name, container)
        self.src_op = self.handle_prop(model_props, 'input')
        self.dest_op = self.handle_prop(model_props, 'target')
        self.cop = self.handle_prop(model_props, 'op')

        self.optype = 100 # Special Actions start indexing at 100 
    
    def tokenize(self):
        # call parent method to set basic ops common to all 
        super().tokenize()
        # cop_code 0: =/eq, 1: </lt, 2: >/gt, 3: <=/le, 4: >=/ge, 5: <>/ne 
        cop_codes = [
            '+=': 0,
            '-=': 1,
            '*=': 2,
            '/=': 3,
        ]
        self.ops = self.ops + [self.left_ix, cop_codes[self.cop], self.right_ix]
    
    def add_op_tokens(self):
        # this puts the tokens into the global simulation queue 
        # can be customized by subclasses to add multiple lines if needed.
        super().add_op_tokens()

# njit functions for runtime

@njit
def exec_saction(op, state_ix, dict_ix):
    ix = op[1] # ID of this op
    dix = op[2] # ID of place to store data
    # these indices must be adjusted to reflect the number of common op tokens
    # SpecialAction has:
    # - type of condition (+=, -=, ...)
    # - operand 1 (left side)
    # - operand 2 (right side) 
    op = op[3]  
    ix1 = op[4]
    ix2 = op[5]
    if op == 0:
      result = state_ix[ix1] + state_ix[ix2]
    if op == 1:
      result = state_ix[ix1] - state_ix[ix2]
        
    return result

