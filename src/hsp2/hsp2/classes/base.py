from typing import List
import numba
import time
from numpy import zeros
from numba.types import string as numba_str
from numba import njit, types, int32, float32, float64, typeof   # import the types
from numba.experimental import jitclass
from numba.typed import Dict, List as NumbaList

def model_make_spec(prop_names, prop_type):
    new_spec = [(x, prop_type) for x in prop_names]
    return new_spec

# this is temporatry, these will be merged with state soon
# state vars - two options to see which is fastest
state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

state_paths_ty = ('state_paths', typeof(state_paths))
state_ix_ty = ('state_ix', typeof(state_ix))

model_num_type = float32
model_str_type = numba_str # Imported from numba.types.string
model_str_props = ['name' , 'path']
model_num_props = ['value']
model_base = [state_paths_ty, state_ix_ty]+ model_make_spec(model_str_props,model_str_type ) + model_make_spec(model_num_props, model_num_type )

@jitclass(model_base)
class ModelBase:
    def __init__(self):
        self.value = 0
        self.state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
        self.state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    
    def step(self, step):
        ret = self.value
        self.value += 1
    
    def pre_step(self):
        # get remote inputs if needed, load timeseries
        self.step_TSGET()
        return
    
    def post_step(self):
        # perform logging actions (timeseries writes, etc)
        return
    
    def step_TSGET(self):
        # TSGET: get timeseries if need be
        return

class HandlerBase:
    def __init__(self, props = None):
        self.model_props = props
        return
    
    def make_model(self):
        # Create an empty model
        model = ModelBase()
        print("Creating ModelBase with props:", self.model_props)
        # Populate model props and return
        self.set_props(model, self.model_props)
        return model
    
    def set_props(self, model, model_props, strict = False ):
        if model_props == None:
            return
        for prop in model_props:
            if hasattr(model, prop):
                propval = self.handle_propval(model, prop, model_props[prop], strict)
                setattr(model,prop,propval)
    
    def handle_propval(self, model, prop, propval, strict = False ):
        # sub classes can check to see if this is in the right format, or change to numba compatible types etc.
        return propval
    
    def make_numba_dict(props):
        ret = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        for i in props.keys():
            ret[i] = props[i]
        return ret

@njit
def iteration_test(it_ops, it_nums):
    ctr = 0
    for n in range(it_nums):
        for i in range(len(it_ops)):
            it_ops[i].step(n)
        ctr=ctr+1
    print("Completed ", ctr, " loops")

