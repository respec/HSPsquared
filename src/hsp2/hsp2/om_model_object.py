"""
The class ModelObject is the base class upon which all other dynamic model objects are built on.
It handles all Dict management functions, but provides for no runtime execution of it's own.
All runtime exec is done by child classes.
"""

from hsp2.state.state import set_state, get_state_ix
from numba.typed import Dict
from hsp2.hsp2.om import is_float_digit
from pandas import HDFStore
from numpy import pad, asarray, zeros, int32
from numba import njit, types


class ModelObject:
    max_token_length = 64  # limit on complexity of tokenized objects since op_tokens must be fixed dimensions for numba
    runnables = [
        1,
        2,
        3,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        100,
    ]  # runnable components important for optimization
    ops_data_type = "ndarray"  # options are ndarray or Dict - Dict appears slower, but unsure of the cause, so keep as option.
    
    def __init__(self, name, container=False, model_props=None, state=None, model_object_cache=None):
        self.name = name
        self.container = container  # will be a link to another object
        if state is None:
            # we must verify that we have a properly formatted state Dictionary, or that our parent does.
            if self.container == False:
                raise Exception(
                    "Error: State object must be passed to root object. ",
                    + name
                    + " cannot be created.  See state::init_state_dicts()",
                )
            else:
                state = self.container.state
        if model_object_cache is None:
            # we must verify that we have a properly formatted state Dictionary, or that our parent does.
            if self.container == False:
                raise Exception(
                    "Error: model_object_cache object must be available on to root object. ",
                    + name
                    + " cannot be created.  See state::init_state_dicts()",
                )
            else:
                model_object_cache = self.container.model_object_cache
        self.state = state  # make a copy here. is this efficient?
        self.model_object_cache = model_object_cache  # make a copy here. is this efficient?
        self.handle_deprecated_args(name, container, model_props, state)
        # END - handle deprecated
        if model_props is None:
            model_props = {}
        self.state_path = self.handle_prop(model_props, "state_path", False, False)
        # Local properties
        self.model_props_parsed = {}  # a place to stash parse record for debugging
        self.log_path = ""  # Ex: "/RESULTS/RCHRES_001/SPECL"
        self.attribute_path = ""  #
        if hasattr(self, "state_path") == False:
            # if the state_path has already been set, we accept it.
            # this allows sub-classes to override the standard path guessing approach.
            self.state_path = (
                ""  # Ex: "/STATE/RCHRES_001" # the pointer to this object state
            )
        self.inputs = {}  # associative array with key=local_variable_name, value=hdf5_path Ex: [ 'Qin' : '/STATE/RCHRES_001/IVOL' ]
        self.inputs_ix = {}  # associative array with key=local_variable_name, value=state_ix integer key
        self.ix = False
        self.paths_found = False  # this should be False at start
        self.default_value = 0.0
        self.ops = []
        self.optype = 0  # OpTypes are as follows:
        #                 0 - model object, 1 - equation, 2 - datamatrix, 3 - input/ModelLinkage,
        #                 4 - broadcastChannel, 5 - SimTimer, 6 - Conditional, 7 - ModelVariable (numeric),
        #                 8 - matrix accessor, 9 - MicroWatershedModel, 10 - MicroWatershedNetwork, 11 - ModelTimeseries,
        #                 12 - ModelRegister, 13 - SimpleChannel, 14 - SimpleImpoundment, 15 - FlowBy
        #                 16 - ModelConstant
        #                 100 - SpecialAction, 101 - UVNAME, 102 - UVQUAN, 103 - DISTRB
        self.register_path()  # note this registers the path AND stores the object in model_object_cache
        self.parse_model_props(model_props)
    
    def handle_deprecated_args(self, name, container, model_props, state):
        # Handle old deprecated format for Register, Constant and Variable
        state_path = False
        if type(model_props) != dict:
            value = model_props
            model_props = {"name": name, "value": value}
            print(
                "WARNING: deprecated 3rd argument for ModelObject. Use: [Model class](name, container, model_props, state)"
            )
        if type(state) == str:
            model_props["state_path"] = state
            state = None
            if container != None:
                state = container.state
            print(
                "WARNING: deprecated 4th argument for ModelObject. Use: [Model class](name, container, model_props, state)"
            )
    
    @staticmethod
    def required_properties():
        # returns a list or minimum properties to create.
        # see ModelVariable below for how to call this in a sub-class
        # note:
        # req_props = super(DataMatrix, DataMatrix).required_properties()
        req_props = ["name"]
        return req_props
    
    @staticmethod
    def make_op_tokens(num_ops=5000):
        if ModelObject.ops_data_type == "ndarray":
            op_tokens = int32(
                zeros((num_ops, 64))
            )  # was Dict.empty(key_type=types.int64, value_type=types.i8[:])
        else:
            op_tokens = Dict.empty(key_type=types.int64, value_type=types.i8[:])
        return op_tokens
    
    @staticmethod
    def runnable_op_list(op_tokens, meo, debug=False):
        # only return those objects that do something at runtime
        rmeo = []
        run_ops = {}
        for ops in op_tokens:
            # the base class defines the type of objects that are runnable (i.e. have a step() method)
            if ops[0] in ModelObject.runnables:
                run_ops[ops[1]] = ops
                if debug == True:
                    print("Found runnable", ops[1], "type", ops[0])
        for ix in meo:
            if ix in run_ops.keys():
                rmeo.append(ix)
        rmeo = asarray(rmeo, dtype="i8")
        return rmeo
    
    @staticmethod
    def model_format_ops(ops):
        if ModelObject.ops_data_type == "ndarray":
            ops = pad(ops, (0, ModelObject.max_token_length))[
                0 : ModelObject.max_token_length
            ]
        else:
            ops = asarray(ops, dtype="i8")
        return ops
    
    def format_ops(self):
        # this can be sub-classed if needed, but should not be since it is based on the ops_data_type
        # See ModelObject.model_format_ops()
        return ModelObject.model_format_ops(self.ops)
    
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
    
    def handle_path_aliases(self, object_path):
        if not (self.container == False):
            object_path = object_path.replace("[parent]", self.container.state_path)
        object_path = object_path.replace("[self]", self.state_path)
        return object_path
    
    def handle_inputs(self, model_props):
        if "inputs" in model_props.keys():
            for i_pair in model_props["inputs"]:
                i_name = i_pair[0]
                i_target = i_pair[1]
                i_target = handle_path_aliases(i_target)
                self.add_input(i_name, i_target)
    
    def handle_prop(self, model_props, prop_name, strict=False, default_value=None):
        # this checks to see if the prop is in dict with value form, or just a value
        # strict = True causes an exception if property is missing from model_props dict
        prop_val = model_props.get(prop_name)
        if (
            type(prop_val) == list
        ):  # this doesn't work, but nothing gets passed in like this? Except broadcast params, but they are handled in the sub-class
            prop_val = prop_val
        elif type(prop_val) == dict:
            prop_val = prop_val.get("value")
        if strict and (prop_val == None):
            raise Exception(
                "Cannot find property "
                + prop_name
                + " in properties passed to "
                + self.name
                + " and strict = True.  Object creation halted. Path to object with error is "
                + self.state_path
            )
        if (prop_val == None) and not (default_value == None):
            prop_val = default_value
        return prop_val
    
    def parse_model_props(self, model_props, strict=False):
        # sub-classes will allow an create argument "model_props" and handle them here.
        #  - subclasses should insure that they call super().parse_model_props() or include all code below
        # see also: handle_prop(), which will be called y parse_model_props
        #           for all attributes supported by the class
        # this base object only handles inputs
        self.handle_inputs(model_props)
        self.model_props_parsed = model_props
        return True
    
    def set_state(self, set_value):
        var_ix = self.state.set_state(
            self.state_path,
            set_value,
        )
        return var_ix
    
    def save_object_hdf(self, hdfname, overwrite=False):
        # save the object in the full hdf5 path
        # if overwrite = True replace this and all children, otherwise, just save this.
        # note: "with" statement helps prevent unclosed resources, see: https://www.geeksforgeeks.org/with-statement-in-python/
        with HDFStore(hdfname, mode="a") as store:
            dummy_var = True
    
    def make_paths(self, base_path=False):
        print("calling make_paths from", self.name, "with base path", base_path)
        if base_path == False:  # we are NOT forcing paths
            if not (self.container == False):
                print("Using container path as base:", self.container.state_path + "/" + str(self.name))
                self.state_path = self.container.state_path + "/" + str(self.name)
                self.attribute_path = (
                    self.container.attribute_path + "/" + str(self.name)
                )
            elif self.name == "":
                self.state_path = "/STATE"
                self.attribute_path = "/OBJECTS"
            else:
                self.state_path = "/STATE/" + str(self.name)
                self.attribute_path = "/OBJECTS/" + str(self.name)
        else:
            # base_path is a Dict with state_path and attribute_path set
            self.state_path = base_path["STATE"] + self.name
            self.attribute_path = base_path["OBJECTS"] + self.name
        return self.state_path
    
    def get_state(self, var_name=False):
        if var_name == False:
            return self.state_ix[self.ix]
        else:
            var_path = self.find_var_path(var_name)
            print("Looking for state ix of:", var_path)
            var_ix = self.state.get_state_ix(var_path)
        if var_ix == False:
            return False
        return self.state_ix[var_ix]
    
    def get_tindex(self):
        timer = self.get_object('timer')
        tindex = self.state.dict_ix[timer.ix]
        return(tindex)
    
    def get_object(self, var_name=False):
        if var_name == False:
            return self.model_object_cache[self.state_path]
        else:
            var_path = self.find_var_path(var_name)
            return self.model_object_cache[var_path]
    
    def find_var_path(self, var_name, local_only=False):
        # check local inputs for name
        if type(var_name) == str:
            # print("Expanding aliases for", var_name)
            var_name = self.handle_path_aliases(var_name)  # sub out any wildcards
        # print(self.name, "called", "find_var_path(self, ", var_name, ", local_only = False)")
        if var_name in self.inputs.keys():
            return self.inputs[var_name]
        if local_only:
            return False  # we are limiting the scope, so just return
        # check parent for name
        if not (self.container == False):
            return self.container.find_var_path(var_name)
        # check for root state vars STATE + var_name
        if (state.model_root_object + "/" + var_name) in self.state.state_paths:
            # return self.state['state_paths'][("/STATE/" + var_name)]
            return state.model_root_object + "/" + var_name
        # check for full paths
        if var_name in self.state.state_paths:
            # return self.state['state_paths'][var_name]
            return var_name
        return False
    def constant_or_path(self, keyname, keyval, trust=False):
        if is_float_digit(keyval):
            # we are given a constant value, not a variable reference
            # print("Creating ModelVariable", keyname, "on", self.name, "with default", keyval)
            k = ModelVariable(
                keyname, self, {"name": keyname, "value": float(keyval)}, self.state
            )
            kix = k.ix
        else:
            kix = self.add_input(keyname, keyval, 2, trust)
        return kix
    def register_path(self):
        # initialize the path variable if not already set
        print("register_path called for", self.name, "with state_path", self.state_path)
        if self.state_path == "" or self.state_path == False:
            self.make_paths()
        self.ix = self.state.set_state(self.state_path, self.default_value)
        # store object in model_object_cache - always, if we have reached this point we need to overwrite
        print("Adding ", self.name, "with state_path", self.state_path, "to model_object_cache")
        self.model_object_cache[self.state_path] = self
        # this should check to see if this object has a parent, and if so, register the name on the parent
        # default is as a child object.
        if not (self.container == False):
            # since this is a request to actually create a new path, we instruct trust = True as last argument
            # print("Adding", self.name, "as input to", self.container.name)
            return self.container.add_input(self.name, self.state_path, 1, True)
        return self.ix
    def add_input(self, var_name, var_path, input_type=1, trust=False):
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
        print("Searched", var_name, "with path", var_path,"found", found_path)
        var_ix = self.state.get_state_ix(found_path)
        if var_ix == False:
            if trust == False:
                raise Exception(
                    "Cannot find variable path: "
                    + var_path
                    + " when adding input to object "
                    + self.name
                    + " as input named "
                    + var_name
                    + " ... process terminated. Path to object with error is "
                    + self.state_path
                )
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
        # print(self.name, "added input", var_name, var_ix)
        # Should we create a register for the input to be reported here?
        # i.e., if we have an input named Qin on RCHRES_R001, shouldn't we be able
        # to find the data in /STATE/RCHRES_R001/Qin ???  It is redundant data and writing
        # but matches a complete data model and prevents stale data?
        return self.inputs_ix[var_name]
    def add_object_input(self, var_name, var_object, link_type=1):
        # See above for details.
        # this adds an object as a link to another object
        self.inputs[var_name] = var_object.state_path
        self.inputs_ix[var_name] = var_object.ix
        return self.inputs_ix[var_name]
    def create_parent_var(self, parent_var_name, source):
        # see decision points: https://github.com/HARPgroup/HSPsquared/issues/78
        # This is used when an object sets an additional property on its parent
        # Like in simple_channel sets [channel prop name]_Qout on its parent
        # Generally, this should have 2 components.
        # 1 - a state variable on the child (this could be an implicit sub-comp, or a constant sub-comp, the child handles the setup of this) see constant_or_path()
        # 2 - an input link
        # the beauty of this is that the parent object and any of it's children will find the variable "[source_object]_varname"
        if type(source) == str:
            self.container.add_input(parent_var_name, source, 1, False)
        elif isinstance(source, ModelObject):
            self.container.add_object_input(parent_var_name, source, 1)
    def insure_path(self, var_path):
        # if this path can be found in the hdf5 make sure that it is registered in state
        # and that it has needed object class to render it at runtime (some are automatic)
        # RIGHT NOW THIS DOES NOTHING TO CHECK IF THE VAR EXISTS THIS MUST BE FIXED
        var_ix = self.state.set_state(var_path, 0.0)
        return var_ix
    def get_dict_state(self, ix=-1):
        if ix >= 0:
            return self.state.dict_ix[ix]
        return self.state.dict_ix[self.ix]
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
    def insure_register(
        self,
        var_name,
        default_value,
        register_container,
        register_path=False,
        is_accumulator=True,
    ):
        # we send with local_only = True so it won't go upstream
        if register_path == False:
            register_path = register_container.find_var_path(var_name, True)
        if (register_path == False) or (
            register_path not in self.model_object_cache.keys()
        ):
            # create a register as a placeholder for the data at the hub path
            # in case there are no senders, or in the case of a timeseries logger, we need to register it so that its path can be set to hold data
            # print("Creating a register for data for hub ", register_container.name, "(", register_container.state_path, ")", " var name ",var_name)
            if is_accumulator == True:
                var_register = ModelRegister(
                    var_name, register_container, default_value, register_path
                )
            else:
                # this is just a standard numerical data holder so set up a constant
                reg_props = {
                    "name": var_name,
                    "value": default_value,
                    "state_path": register_path,
                }
                var_register = ModelVariable(
                    var_name, register_container, reg_props, self.state
                )
        else:
            var_register = self.model_object_cache[register_path]
        return var_register
    def tokenize(self):
        # renders tokens for high speed execution
        if self.paths_found == False:
            raise Exception(
                "path_found False for object"
                + self.name
                + "("
                + self.state_path
                + "). "
                + "Tokens cannot be generated until method '.find_paths()' is run for all model objects ... process terminated. (see function `model_path_loader(model_object_cache)`)"
            )
        self.ops = [self.optype, self.ix]
    def add_op_tokens(self):
        # this puts the tokens into the global simulation queue
        # can be customized by subclasses to add multiple lines if needed.
        if self.ops == []:
            self.tokenize()
        if len(self.ops) > self.max_token_length:
            raise Exception(
                "op tokens cannot exceed max length of"
                + self.max_token_length
                + "("
                + self.state_path
                + "). "
            )
        self.state.set_token(self.ix, self.format_ops())
    def step(self, step):
        # this tests the model for a single timestep.
        # this is not the method that is used for high-speed runs, but can theoretically be used for
        # easier to understand demonstrations
        # this has not been tested since changes to the state from array to object
        step_one(
            self.state.op_tokens,
            self.state.op_tokens[self.ix],
            self.state_ix,
            self.state.dict_ix,
            self.state.state_ix,
            self.state.ts_ix,
            step,
        )
        # step_model({self.state['op_tokens'][self.ix]}, self.state['state_ix'], self.state['dict_ix'], self.state['ts_ix'], step)
    def finish(self):
        # Do end of model activities here
        # we do this in the object context if it involves IO and other complex actions
        return True


"""
The class ModelVariable is a base cass for storing numerical values.  Used for UVQUAN and misc numerical constants...
"""


class ModelVariable(ModelObject):
    def __init__(self, name, container=False, model_props=None, state=None):
        super(ModelVariable, self).__init__(name, container, model_props, state)
        # print("ModelVariable named", name, "with path", self.state_path,"beginning")
        value = self.handle_prop(model_props, "value")
        self.default_value = float(value)
        self.optype = 7  # 0 - shell object, 1 - equation, 2 - datamatrix, 3 - input, 4 - broadcastChannel, 5 - SimTimer, 6 - Conditional, 7 - ModelVariable (numeric)
        # print("ModelVariable named",self.name, "with path", self.state_path,"and ix", self.ix, "value", value)
        var_ix = self.set_state(float(value))
        self.paths_found = True
        # self.state['state_ix'][self.ix] = self.default_value
    def required_properties():
        req_props = super(ModelVariable, ModelVariable).required_properties()
        req_props.extend(["value"])
        return req_props


"""
The class ModelConstant is for storing non-changing values.
"""


class ModelConstant(ModelVariable):
    def __init__(self, name, container=False, model_props=None, state=None):
        super(ModelConstant, self).__init__(name, container, model_props, state)
        self.optype = 16  # 0 - shell object, 1 - equation, 2 - datamatrix, 3 - input, 4 - broadcastChannel, 5 - SimTimer, 6 - Conditional, 7 - ModelVariable (numeric)
        # print("ModelVariable named",self.name, "with path", self.state_path,"and ix", self.ix, "value", value)


"""
The class ModelRegister is for storing push values.
Behavior is to zero each timestep.  This could be amended later.
Maybe combined with stack behavior?  Or accumulator?
"""


class ModelRegister(ModelVariable):
    def __init__(self, name, container=None, model_props=False, state=False):
        super(ModelRegister, self).__init__(name, container, model_props, state)
        self.optype = 12  #
        # self.state['state_ix'][self.ix] = self.default_value
    def required_properties():
        req_props = super(ModelVariable, ModelVariable).required_properties()
        req_props.extend(["value"])
        return req_props


# njit functions for runtime
@njit
def pre_step_register(op, state_ix):
    ix = op[1]
    # print("Resetting register", ix,"to zero")
    state_ix[ix] = 0.0
    return


# Note: ModelVariable has no runtime execution


# njit functions for end of model run
@njit
def finish_model_object(op_token, state_ix, ts_ix):
    return


@njit
def finish_register(op_token, state_ix, ts_ix):
    # todo: push the values of ts_ix back to the hdf5? or does this happen in larger simulation as it is external to OM?
    return
