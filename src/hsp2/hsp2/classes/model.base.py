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

# this is temporary, these will be merged with state soon
# state vars - two options to see which is fastest
# and ts will be gained from the hsp2 libs
state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

state_paths_ty = ('state_paths', typeof(state_paths))
state_ix_ty = ('state_ix', typeof(state_ix))
ts_ty = ('ts', typeof(ts))

model_num_type = float32
model_str_type = numba_str # Imported from numba.types.string
model_str_props = ['name' , 'path']
model_num_props = ['value']
model_base = [state_paths_ty, state_ix_ty, ts_ty]+ model_make_spec(model_str_props,model_str_type ) + model_make_spec(model_num_props, model_num_type )

@jitclass(model_base)
class ModelBase:
    def __init__(self):
        self.path = '' # must initialize
        self.value = 0
        self.state_ix = Dict.empty(key_type=types.int64, value_type=types.float64)
        self.state_paths = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        self.ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
        return
    
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
    def __init__(self, model_props = None):
        self.model_props = model_props
        self.model_props_parsed = {} # after handling
        self.parse_model_props(model_props)
        return
    
    def parse_model_props(self, model_props, strict = False ):
        # sub-classes will allow an create argument "model_props" and handle them here.
        #  - subclasses should insure that they call super().parse_model_props() or include all code below
        # see also: handle_prop(), which will be called y parse_model_props 
        #           for all attributes supported by the class
        # this base object only handles inputs
        #self.handle_inputs(model_props)
        if model_props is None:
            return False
        self.model_props_parsed = model_props
        return True
    
    def set_props(self, model, strict = False ):
        if self.model_props == None:
            raise Exception("Model properties are empty, Process terminated.")
            return False # maybe return exception?
        for prop in self.model_props:
            if hasattr(model, prop):
                propval = self.handle_propval(prop, self.model_props[prop], strict)
                setattr(model,prop,propval)
    
    def handle_propval(self, propname, propval, strict = False ):
        # sub classes can check to see if this is in the right format, or change to numba compatible types etc.
        if not propname:
            return False # MAYBE should throw an exception here
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

