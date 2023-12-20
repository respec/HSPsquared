"""
The class ModelObject is the base class upon which all other dynamic model objects are built on.
It handles all Dict management functions, but provides for no runtime execution of it's own.
All runtime exec is done by child classes.
"""
from HSP2.om import *
from HSP2.state import *
from pandas import Series, DataFrame, concat, HDFStore, set_option, to_numeric
from pandas import Timestamp, Timedelta, read_hdf, read_csv

class ModelObject:
    state_ix = {} # Shared Dict with the numerical state of each object 
    state_paths = {} # Shared Dict with the hdf5 path of each object 
    dict_ix = {} # Shared Dict with the hdf5 path of each object 
    ts_ix = {} # Shared Dict with the hdf5 path of each object 
    op_tokens = {} # Shared Dict with the tokenized representation of each object 
    model_object_cache = {} # Shared with actual objects, keyed by their path 
    model_exec_list = {} # Shared with actual objects, keyed by their path 
    
    def __init__(self, name, container = False):
        self.name = name
        self.container = container # will be a link to another object
        self.log_path = "" # Ex: "/RESULTS/RCHRES_001/SPECL" 
        self.attribute_path = "" # 
        self.model_props_parsed = {} # a place to stash parse record for debugging
        if (hasattr(self,'state_path') == False):
            # if the state_path has already been set, we accept it. 
            # this allows sub-classes to override the standard path guessing approach.
            self.state_path = "" # Ex: "/STATE/RCHRES_001" # the pointer to this object state
        self.inputs = {} # associative array with key=local_variable_name, value=hdf5_path Ex: [ 'Qin' : '/STATE/RCHRES_001/IVOL' ]
        self.inputs_ix = {} # associative array with key=local_variable_name, value=state_ix integer key
        self.ix = False
        self.paths_found = False # this should be False at start
        self.default_value = 0.0
        self.ops = []
        self.optype = 0 # 0 - shell object, 1 - equation, 2 - datamatrix, 3 - input/ModelLinkage, 4 - broadcastChannel, 5 - SimTimer, 6 - Conditional, 7 - ModelConstant (numeric), 8 - matrix accessor, 9 - MicroWatershedModel, 10 - MicroWatershedNetwork, 11 - ModelTimeseries, 12 - ModelRegister, 13 - SimpleChannel, 14 - SimpleImpoundment
        # this is replaceable. to replace state_path/re-register the index :
        # - remove the old PATH from state_paths: del state_paths[self.state_path]
        # you should never create an object without knowing its container, but if you do
        # you can TRY to do the following:
        # - set this objects new path based on containment and call:
        #         [my_object].make_paths()
        # - add this manually to state_paths:
        #   state_paths[[my_object].state_path] = [my_object].ix 
        # - call [my_object].register_path()   
        self.register_path()
    
    @staticmethod
    def required_properties():
        # returns a list or minimum properties to create.
        # see ModelConstant below for how to call this in a sub-class 
        # note: 
        # req_props = super(DataMatrix, DataMatrix).required_properties()
        req_props = ['name']
        return req_props
    
    @classmethod
    def check_properties(cls, model_props):
        # this is for pre-screening properties for validity in model creation routines 
        # returns True or False and can be as simple as checking the list of required_properties
        # or a more detailed examination of suitability of what those properties contain 
        req_props = cls.required_properties()
        matching_props = set(model_props).intersection(set(req_props))
        if len(matching_props) < len(req_props):
            return False 
        return True
    
    def handle_prop(self, model_props, prop_name, strict = False, default_value = None ):
        # this checks to see if the prop is in dict with value form, or just a value 
        # strict = True causes an exception if property is missing from model_props dict 
        prop_val = model_props.get(prop_name)
        if type(prop_val) == list:
            prop_val = prop_val.get('value')
        elif type(prop_val) == dict:
            prop_val = prop_val.get('value')
        if strict and (prop_val == None):
            raise Exception("Cannot find property " + prop_name + " in properties passed to "+ self.name + " and strict = True.  Object creation halted. Path to object with error is " + self.state_path)
        return prop_val
    
    def parse_model_props(self, model_props, strict = False ):
        # sub-classes will allow an create argument "model_props" and handle them here.
        # see also: handle_prop(), which will be called y parse_model_props 
        #           for all attributes supported by the class
        self.model_props_parsed = model_props
        return True
    
    def set_state(self, set_value):
        var_ix = set_state(self.state_ix, self.state_paths, self.state_path, set_value)
        return var_ix
    
    def load_state_dicts(self, op_tokens, state_paths, state_ix, dict_ix):
        self.op_tokens = op_tokens
        self.state_paths = state_paths
        self.state_ix = state_ix
        self.dict_ix = dict_ix
    
    def save_object_hdf(self, hdfname, overwrite = False ):
        # save the object in the full hdf5 path
        # if overwrite = True replace this and all children, otherwise, just save this.     
        # note: "with" statement helps prevent unclosed resources, see: https://www.geeksforgeeks.org/with-statement-in-python/
        with HDFStore(hdfname, mode = 'a') as store:
            dummy_var = True
    
    def make_paths(self, base_path = False):
        if base_path == False: # we are NOT forcing paths
            if not (self.container == False):
                self.state_path = self.container.state_path + "/" + self.name
                self.attribute_path = self.container.attribute_path + "/" + self.name
            elif self.name == "":
                self.state_path = "/STATE" 
                self.attribute_path = "/OBJECTS" 
            else:
                self.state_path = "/STATE/" + self.name
                self.attribute_path = "/OBJECTS/" + self.name
        else:
            # base_path is a Dict with state_path and attribute_path set 
            self.state_path = base_path['STATE'] + self.name
            self.attribute_path = base_path['OBJECTS'] + self.name
        return self.state_path
    
    def get_state(self, var_name = False):
        if var_name == False:
            return self.state_ix[self.ix]
        else:
            var_path = self.find_var_path(var_name)
            var_ix = get_state_ix(self.state_ix, self.state_paths, var_path)
        if (var_ix == False):
            return False
        return self.state_ix[var_ix]
    
    def get_exec_order(self, var_name = False):
        if var_name == False:
            var_ix = self.ix
        else:
            var_path = self.find_var_path(var_name)
            var_ix = get_state_ix(self.state_ix, self.state_paths, var_path)
        exec_order = get_exec_order(self.model_exec_list,var_ix)
        return exec_order
    
    def get_object(self, var_name = False):
        if var_name == False:
            return self.model_object_cache[self.state_path]
        else:
            var_path = self.find_var_path(var_name)
            return self.model_object_cache[var_path]
        
        
    def find_var_path(self, var_name, local_only = False):
        # check local inputs for name
        if var_name in self.inputs.keys():
            #print("Found", var_name, "on ", self.name, "path=", self.inputs[var_name])
            return self.inputs[var_name]
        if local_only:
            return False # we are limiting the scope, so just return
        # check parent for name
        if not (self.container == False):
            #print(self.name,"looking to parent", self.container.name, "for", var_name)
            return self.container.find_var_path(var_name)
        # check for root state vars STATE + var_name
        if ("/STATE/" + var_name) in self.state_paths.keys():
            #return self.state_paths[("/STATE/" + var_name)]
            return ("/STATE/" + var_name)
        # check for root state vars
        if var_name in self.state_paths.keys():
            #return self.state_paths[var_name]
            return var_name
        #print(self.name, "could not find", var_name)
        return False
    
    def constant_or_path(self, keyname, keyval, trust = False):
        #print("Called constant_or_path with", keyname, " = ", keyval)
        if is_float_digit(keyval):
            # we are given a constant value, not a variable reference 
            #print("Creating constant ", keyname, " = ", keyval)
            k = ModelConstant(keyname, self, float(keyval))
            kix = k.ix
        else:
            #print("Adding input ", keyname, " = ", keyval)
            kix = self.add_input(keyname, keyval, 2, trust)
        return kix
    
    def register_path(self):
        # initialize the path variable if not already set
        #print(self.name,"called register_path()")
        if self.state_path == '':
            self.make_paths()
        #print("Setting ", self.name, "state to", self.default_value)
        self.ix = set_state(self.state_ix, self.state_paths, self.state_path, self.default_value)
        # store object in model_object_cache
        if not (self.state_path in self.model_object_cache.keys()):
            self.model_object_cache[self.state_path] = self 
        # this should check to see if this object has a parent, and if so, register the name on the parent 
        # default is as a child object. 
        if not (self.container == False):
            #print("Adding", self.name,"as input to", self.container.name)
            # since this is a request to actually create a new path, we instruct trust = True as last argument
            return self.container.add_input(self.name, self.state_path, 1, True)
        return self.ix
    
    def add_input(self, var_name, var_path, input_type = 1, trust = False):
        # this will add to the inputs, but also insure that this 
        # requested path gets added to the state/exec stack via an input object if it does 
        # not already exist.
        # - var_name = the local name for this linked entity/attribute 
        # - var_path = the full path of the entity/attribute we are linking to 
        # - input types: 1: parent-child link, 2: state property link, 3: timeseries object property link 
        # - trust = False means fail if the path does not already exist, True means assume it will be OK which is bad policy, except for the case where the path points to an existing location
        # do we have a path here already or can we find on the parent?
        # how do we check if this is a path already, in which case we trust it?
        # todo: we should be able to alias a var_name to a var_path, for example 
        #       calling add_input('movar', 'month', 1, True)
        #       this *should* search for month and find the STATE/month variable 
        #       BUT this only works if both var_name and var_path are month 
        #       so add_input('month', 'month', 1, True) works.
        found_path = self.find_var_path(var_path)
        #print("Searched", var_name, "with path", var_path,"found", found_path)
        var_ix = get_state_ix(self.state_ix, self.state_paths, found_path)
        if var_ix == False:
            if (trust == False):
                raise Exception("Cannot find variable path: " + var_path + " when adding input to object " + self.name + " as input named " + var_name + " ... process terminated. Path to object with error is " + self.state_path)
            var_ix = self.insure_path(var_path)
        else:
            # if we are to trust the path, this might be a child property just added,
            # and therefore, we don't look further than this 
            # otherwise, we use found_path, whichever it is, as 
            # we know that this path is better, as we may have been given a simple variable name
            # and so found_path will look more like /STATE/RCHRES_001/...
            if trust == False:
                var_path = found_path
        self.inputs[var_name] = var_path
        self.inputs_ix[var_name] = var_ix
        return self.inputs_ix[var_name]
    
    def add_object_input(self, var_name, var_object, link_type = 1):
        # See above for details.
        # this adds an object as a link to another object 
        self.inputs[var_name] = var_object.state_path
        self.inputs_ix[var_name] = var_object.ix
        return self.inputs_ix[var_name]
    
    def create_parent_var(self, parent_var_name, source_object):
        # see decision points: https://github.com/HARPgroup/HSPsquared/issues/78
        # This is used when an object sets an additional property on its parent
        # Like in simple_channel sets [channel prop name]_Qout on its parent 
        # Generally, this should have 2 components.  
        # 1 - a state variable on the child (this could be an implicit sub-comp, or a constant sub-comp, the child handles the setup of this) see constant_or_path()
        # 2 - an input link 
        self.container.add_object_input(parent_var_name, source_object, 1)
    
    def insure_path(self, var_path):
        # if this path can be found in the hdf5 make sure that it is registered in state
        # and that it has needed object class to render it at runtime (some are automatic)
        # RIGHT NOW THIS DOES NOTHING TO CHECK IF THE VAR EXISTS THIS MUST BE FIXED
        var_ix = set_state(self.state_ix, self.state_paths, var_path, 0.0)
        return var_ix 
    
    def get_dict_state(self, ix = -1):
        if ix >= 0:
            return self.dict_ix[ix]
        return self.dict_ix[self.ix]
    
    def find_paths(self):
        # Note: every single piece of data used by objects, even constants, are resolved to a PATH in the hdf5
        # find_paths() is called to insure that all of these can be found, and then, are added to inputs/inputs_ix
        # - We wait to find the index values for those variables after all things have been loaded
        # - base ModelObject does not have any "implicit" inputs, since all of its inputs are 
        #   explicitly added children objects, thus we default to True
        self.paths_found = True
        # - But children such as Equation and DataMatrix, etc
        #   so they mark paths_found = False and then 
        #   should go through their own locally defined data 
        #   and call add_input() for any data variables encountered
        # - add_input() will handle searching for the paths and ix values 
        #   and should also handle deciding if this is a constant, like a numeric value 
        #   or a variable data and should handle them accordingly  
        return True
        
    def tokenize(self):
        # renders tokens for high speed execution
        if (self.paths_found == False):
            raise Exception("path_found False for object" + self.name + "(" + self.state_path + "). " + "Tokens cannot be generated until method '.find_paths()' is run for all model objects ... process terminated. (see function `model_path_loader(model_object_cache)`)")
        self.ops = [self.optype, self.ix]
    
    def add_op_tokens(self):
        # this puts the tokens into the global simulation queue 
        # can be customized by subclasses to add multiple lines if needed.
        if self.ops == []:
            self.tokenize()
        #print(self.name, "tokens", self.ops)
        self.op_tokens[self.ix] = np.asarray(self.ops, dtype="i8")
    
    def step(self, step):
        # this tests the model for a single timestep.
        # this is not the method that is used for high-speed runs, but can theoretically be used for 
        # easier to understand demonstrations
        step_one(self.op_tokens, self.op_tokens[self.ix], self.state_ix, self.dict_ix, self.ts_ix, step)
        #step_model({self.op_tokens[self.ix]}, self.state_ix, self.dict_ix, self.ts_ix, step)
    
    def dddstep_model(op_tokens, state_ix, dict_ix, ts_ix, step):
        for i in op_tokens.keys():
            if op_tokens[i][0] == 1:
                state_ix[i] = step_equation(op_tokens[i], state_ix)
            elif op_tokens[i][0] == 2:
                state_ix[i] = exec_tbl_eval(op_tokens[i], state_ix, dict_ix)
            elif op_tokens[i][0] == 3:
                step_model_link(op_tokens[i], state_ix, ts_ix, step)
            elif op_tokens[i][0] == 4:
                return False
            elif op_tokens[i][0] == 5:
                step_sim_timer(op_tokens[i], state_ix, dict_ix, ts_ix, step)
        return 

"""
The class ModelConstant is for storing constants.  It must be loaded here because ModelObject calls it.
Is this useful or just clutter?  Useful I think since there are numerical constants...
"""
class ModelConstant(ModelObject):
    def __init__(self, name, container = False, value = 0.0, state_path = False):
        if (state_path != False):
            # this allows us to mandate the location. useful for placeholders, broadcasts, etc.
            self.state_path = state_path
        super(ModelConstant, self).__init__(name, container)
        self.default_value = float(value) 
        self.optype = 7 # 0 - shell object, 1 - equation, 2 - datamatrix, 3 - input, 4 - broadcastChannel, 5 - SimTimer, 6 - Conditional, 7 - ModelConstant (numeric)
        #print("ModelConstant named",self.name, "with path", self.state_path,"and ix", self.ix, "value", value)
        var_ix = self.set_state(float(value))
        self.paths_found = True
        # self.state_ix[self.ix] = self.default_value
    
    def required_properties():
        req_props = super(ModelConstant, ModelConstant).required_properties()
        req_props.extend(['value'])
        return req_props

# njit functions for runtime

@njit
def exec_model_object( op, state_ix, dict_ix):
    ix = op[1]
    return 0.0