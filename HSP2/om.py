# set up libraries to import for the load_sim_dicts function
# later, this will be drawing from the hdf5, but for now we 
# are hard-wiring a set of components for testing.
# Note: these import calls must be done down here AFTER the helper functions
#       defined aove that are called by the object classes
import random # this is only used for a demo so may be deprecated
import json
import requests
from requests.auth import HTTPBasicAuth
import csv
import pandas as pd
import numpy as np
import time
from numba.typed import Dict
from numpy import zeros, int32
from numba import int8, float32, njit, types, typed # import the types
import random # this is only used for a demo so may be deprecated
from HSP2.state import *


def get_exec_order(model_exec_list, var_ix):
    """
    Find the integer key of a variable name in state_ix 
    """
    model_exec_list = dict(enumerate(model_exec_list.flatten(), 1))
    for exec_order, ix in model_exec_list.items():
        if var_ix == ix:
            # we need to add this to the state 
            return exec_order 
    return False

def init_op_tokens(op_tokens, tops, eq_ix):
    """
    Iinitialize the op_tokens Dict
    This contains the runtime op code for every dynamic operation to be used
    """
    for j in range(len(tops)):
        if isinstance(tops[j], str):
            # must add this to the state array as a constant
            s_ix = append_state(state_ix, float(tops[j]))
            tops[j] = s_ix
    
    op_tokens[eq_ix] = np.asarray(tops, dtype="i8")

def is_float_digit(n: str) -> bool:
    """
    Helper Function to determine if a variable is numeric
    """
    try:
        float(n)
        return True
    except ValueError:
        return False

def model_element_paths(mel, state):
    """
    Informational. If given a list of state_ix keys, shows the operators local name and state_path
    """
    ixn = 1
    for ix in mel:
        ip = get_ix_path(state['state_paths'], ix)
        im = state['model_object_cache'][ip]
        print(ixn, ":", im.name, "->", im.state_path, '=', im.get_state())
        ixn = ixn + 1
    return


# Import Code Classes
from HSP2.om_model_object import *
from HSP2.om_sim_timer import *
#from HSP2.om_equation import *
from HSP2.om_model_linkage import *
from HSP2.om_special_action import *
#from HSP2.om_data_matrix import *
#from HSP2.om_model_broadcast import *
#from HSP2.om_simple_channel import *
#from HSP2.om_impoundment import *
from HSP2.utilities import versions, get_timeseries, expand_timeseries_names, save_timeseries, get_gener_timeseries

def init_om_dicts():
    """
    The base dictionaries used to store model object info 
    """
    op_tokens = ModelObject.make_op_tokens() # this is just to start, layer it is resized to the object needs
    # Was
    #op_tokens = Dict.empty(key_type=types.int64, value_type=types.i8[:])
    model_object_cache = {} # this does not need to be a special Dict as it is not used in numba 
    return op_tokens, model_object_cache


def state_load_om_json(state, io_manager, siminfo):
    # - model objects defined in file named '[model h5 base].json -- this will populate an array of object definitions that will 
    #   be loadable by "model_loader_recursive()"
    # JSON file would be in same path as hdf5
    hdf5_path = io_manager._input.file_path
    (fbase, fext) = os.path.splitext(hdf5_path)
    # see if there is custom json
    fjson = fbase + ".json"
    print("Looking for custom om json ", fjson)
    if (os.path.isfile(fjson)):
        print("Found local json file", fjson)
        jfile = open(fjson)
        json_data = json.load(jfile)
        # dict.update() combines the arg dict with the base
        state['model_data'].update(json_data)
    # merge in the json siminfo data
    if 'siminfo' in state['model_data'].keys():
        siminfo.update(state['model_data']['siminfo'])
    return

def state_load_om_python(state, io_manager, siminfo):
    # Look for a [hdf5 file base].py file with specific named functions
    # - function "om_init_model": This function can be defined in the [model h5 base].py file containing things to be done 
    #   early in the model loading, like setting up model objects.  This file will already have been loaded by the state module, 
    #   and will be present in the module variable hsp2_local_py (we should rename to state_local_py?)
    # - this file may also contain other dynamically redefined functions such as state_step_hydr()
    #   which can contain code that is executed every timestep inside the _hydr_() function
    #   and can literally supply hooks for any desired user customizable code
    hdf5_path = io_manager._input.file_path
    (fbase, fext) = os.path.splitext(hdf5_path)
    # see if there is a code module with custom python 
    print("Looking for custom om loader in python code ", (fbase + ".py"))
    hsp2_local_py = state['hsp2_local_py']
    # Load a function from code if it exists 
    if 'om_init_model' in dir(hsp2_local_py):
        hsp2_local_py.om_init_model(io_manager, siminfo, state['op_tokens'], state['state_paths'], state['state_ix'], state['dict_ix'], state['ts_ix'], state['model_object_cache'])

def state_initialize_om(state):
    # this function will check to see if any of the multiple paths to loading
    # dynamic operational model objects has been supplied for the model.
    # Grab globals from state for easy handling
    op_tokens, model_object_cache = init_om_dicts()
    state_paths, state_ix, dict_ix, ts_ix = state['state_paths'], state['state_ix'], state['dict_ix'], state['ts_ix']
    # set globals on ModelObject, this makes them persistent throughout all subsequent object instantiation and use
    ModelObject.op_tokens, ModelObject.state_paths, ModelObject.state_ix, ModelObject.dict_ix, ModelObject.model_object_cache, ModelObject.ts_ix = (
        op_tokens, state_paths, state_ix, dict_ix, model_object_cache, ts_ix
    )
    state['op_tokens'], state['model_object_cache'] = op_tokens, model_object_cache 


def state_load_dynamics_om(state, io_manager, siminfo):
    # this function will check to see if any of the multiple paths to loading
    # dynamic operational model objects has been supplied for the model.
    # state_initialize_om(state) must have been called already
    # load dynamic coding libraries if defined by user
    # note: this used to be inside this function, I think that the loaded module should be no problem 
    #       occuring within this function call, since this function is also called from another runtime engine
    #       but if things fail post develop-specact-1 pull requests we may investigate here
    # also, it may be that this should be loaded elsewhere?
    # comment state_load_om_python() to disable dynamic python
    state_load_om_python(state, io_manager, siminfo)
    state_load_om_json(state, io_manager, siminfo)
    return

def state_om_model_root_object(state, siminfo):
    # Create the base that everything is added to. this object does nothing except host the rest.
    if 'model_root_object' not in state.keys():
        model_root_object = ModelObject("") # we give this no name so that it does not interfer with child paths like timer, year, etc (i.e. /STATE/year, ...)
        state['model_root_object'] = model_root_object
        # set up the timer as the first element 
    if '/STATE/timer' not in state['state_paths'].keys():
        timer = SimTimer('timer', model_root_object, siminfo)
    # add base object for the HSP2 domains and other things already added to state so they can be influenced
    for (seg_name,seg_path) in state['hsp_segments'].items():
        if (seg_path not in state['model_object_cache'].keys()):
            # Create an object shell for this
            new_object = ModelObject(seg_name)


def state_om_model_run_prep(state, io_manager, siminfo):
    # insure model base is set
    state_om_model_root_object(state, siminfo)
    # now instantiate and link objects
    # state['model_data'] has alread been prepopulated from json, .py files, hdf5, etc.
    model_root_object = state['model_root_object']
    model_loader_recursive(state['model_data'], model_root_object)
    print("Loaded objects & paths: insures all paths are valid, connects models as inputs")
    # both state['model_object_cache'] and the model_object_cache property of the ModelObject class def 
    # will hold a global repo for this data this may be redundant?  They DO point to the same datset?
    # since this is a function that accepts state as an argument and these were both set in state_load_dynamics_om
    # we can assume they are there and functioning
    if 'model_object_cache' in state.keys():
        model_object_cache = state['model_object_cache']
    else:
        model_object_cache = ModelObject.model_object_cache
    model_path_loader(model_object_cache)
    # len() will be 1 if we only have a simtimer, but > 1 if we have a river being added
    model_exec_list = []
    # put all objects in token form for fast runtime execution and sort according to dependency order
    print("Tokenizing models")
    if 'ops_data_type' in siminfo.keys():
        ModelObject.ops_data_type = siminfo['ops_data_type'] # allow override of dat astructure settings
    ModelObject.op_tokens = ModelObject.make_op_tokens(max(ModelObject.state_ix.keys()) + 1)
    model_tokenizer_recursive(model_root_object, model_object_cache, model_exec_list)
    op_tokens = ModelObject.op_tokens
    # model_exec_list is the ordered list of component operations
    #print("model_exec_list(", len(model_exec_list),"items):", model_exec_list)
    # This is used to stash the model_exec_list in the dict_ix, this might be slow, need to verify.
    # the resulting set of objects is returned.
    state['state_step_om'] = 'disabled'
    state['model_object_cache'] = model_object_cache
    state['model_exec_list'] = np.asarray(model_exec_list, dtype="i8") 
    if ModelObject.ops_data_type == 'ndarray':
        state_keyvals = np.asarray(zeros(max(ModelObject.state_ix.keys()) + 1), dtype="float32")
        for ix, val in ModelObject.state_ix.items():
            state_keyvals[ix] = val
        state['state_ix'] = state_keyvals
    else:
        state['state_ix'] = ModelObject.state_ix
    state['op_tokens'] = op_tokens 
    if len(op_tokens) > 0:
        state['state_step_om'] = 'enabled' 
    print("op_tokens is type", type(op_tokens))
    print("state_ix is type", type(state['state_ix']))
    print("op_tokens has", len(op_tokens),"elements, with ", len(model_exec_list),"executable elements")
    return

# model class reader
# get model class  to guess object type in this lib 
# the parent object must be known
def model_class_loader(model_name, model_props, container = False):
    # todo: check first to see if the model_name is an attribute on the container
    # Use: if hasattr(container, model_name):
    # if so, we set the value on the container, if not, we create a new subcomp on the container 
    if model_props == None:
        return False
    if type(model_props) is str:
        if is_float_digit(model_props):
            model_object = ModelConstant(model_name, container, float(model_props) )
            return model_object
        else:
            return False
    elif type(model_props) is dict:
      object_class = model_props.get('object_class')
      if object_class == None:
          # return as this is likely an attribute that is used for the containing class as attribute 
          # and is handled by the container 
          # todo: we may want to handle this here?  Or should this be a method on the class?
          # Use: if hasattr(container, model_name):
          return False
      model_object = False
      # Note: this routine uses the ".get()" method of the dict class type 
      #       for attributes to pass in. 
      #       ".get()" will return NoValue if it does not exist or the value. 
      if object_class == 'Equation':
          model_object = Equation(model_props.get('name'), container, model_props )
          #remove_used_keys(model_props, 
      elif object_class == 'SimpleChannel':
          model_object = SimpleChannel(model_props.get('name'), container, model_props )
      elif object_class == 'Impoundment':
          model_object = Impoundment(model_props.get('name'), container, model_props )
      elif object_class == 'Constant':
          model_object = ModelConstant(model_props.get('name'), container, model_props.get('value') )
      elif ( object_class.lower() == 'datamatrix'):
          # add a matrix with the data, then add a matrix accessor for each required variable 
          has_props = DataMatrix.check_properties(model_props)
          if has_props == False:
              print("Matrix object must have", DataMatrix.required_properties())
              return False
          # create it
          model_object = DataMatrix(model_props.get('name'), container, model_props)
      elif object_class == 'ModelBroadcast':
          # add a matrix with the data, then add a matrix accessor for each required variable 
          has_props = ModelBroadcast.check_properties(model_props)
          if has_props == False:
              print("ModelBroadcast object must have", ModelBroadcast.required_properties())
              return False
          # create it
          model_object = ModelBroadcast(model_props.get('name'), container, model_props)
      elif object_class == 'MicroWatershedModel':
          # add a matrix with the data, then add a matrix accessor for each required variable 
          has_props = MicroWatershedModel.check_properties(model_props)
          if has_props == False:
              print("MicroWatershedModel object must have", MicroWatershedModel.required_properties())
              return False
          # create it
          model_object = DataMatrix(model_props.get('name'), container, model_props)
      elif object_class == 'ModelLinkage':
          model_object = ModelLinkage(model_props.get('name'), container, model_props)
      elif object_class == 'SpecialAction':
          model_object = SpecialAction(model_props.get('name'), container, model_props)
      else:
          #print("Loading", model_props.get('name'), "with object_class", object_class,"as ModelObject")
          model_object = ModelObject(model_props.get('name'), container, model_props)
    # one way to insure no class attributes get parsed as sub-comps is:
    # model_object.remove_used_keys() 
    if len(model_object.model_props_parsed) == 0:
        # attach these to the object for posterity
        model_object.model_props_parsed = model_props
    # better yet to just NOT send those attributes as typed object_class arrays, instead just name : value
    return model_object

def model_class_translate(model_props, object_class):
    # make adjustments to non-standard items 
    # this might better be moved to methods on the class handlers
    if object_class == 'hydroImpoundment':
        # special handling of matrix/storage_stage_area column
        # we need to test to see if the storage table has been renamed 
        # make table from matrix or storage_stage_area
        # then make accessors from 
        storage_stage_area = model_props.get('storage_stage_area')
        matrix = model_props.get('matrix')
        if ( (storage_stage_area == None) and (matrix != None)): 
            model_props['storage_stage_area'] = matrix
            del model_props['matrix']
    if object_class == 'broadCastObject':
        model_props['object_class'] = 'ModelBroadcast'
        model_props['broadcast_channel'] = model_props['broadcast_class']
    if object_class == 'USGSChannelGeomObject_sub':
        model_props['object_class'] = 'SimpleChannel'
        print("Handling USGSChannelGeomObject_sub as SimpleChannel")
    if object_class == 'hydroImpoundment':
        model_props['object_class'] = 'Impoundment'
        print("Handling hydroImpoundment as Impoundment")
    if object_class == 'hydroImpSmall':
        model_props['object_class'] = 'Impoundment'
        print("Handling hydroImpSmall as Impoundment")
    # now handle disabled classes - this is temporary to prevent having to comment and uncomment
    disabled_classes = {'SimpleChannel', 'Impoundment', 'DataMatrix', 'dataMatrix'}
    if model_props['object_class'] in disabled_classes:
        print("Disabling class", model_props['object_class'], 'rendering as ModelObject')
        model_props['object_class'] = 'ModelObject'

def model_loader_recursive(model_data, container):
    k_list = model_data.keys()
    object_names = dict.fromkeys(k_list , 1)
    if type(object_names) is not dict:
        return False 
    for object_name in object_names:
        #print("Handling", object_name)
        if object_name in {'name', 'object_class', 'id', 'value', 'default'}:
            # we should ask the class what properties are part of the class and also skips these
            # therefore, we can assume that anything else must be a child object that needs to 
            # be handled first -- but how to do this?
            continue
        model_props = model_data[object_name]
        if type(model_props) is not dict:
            # this is a constant, the loader  is built to handle this, but this causes errors with 
            # properties on the class that are expected so we just skip and trust that all constants
            # are formally declared as type Constant
            continue
        if type(model_props) is dict:
            if not ('object_class' in model_props):
                # this is either a class attribute or an un-handleable meta-data 
                # if the class atttribute exists, we should pass it to container to load 
                #print("Skipping un-typed", object_name)
                continue
            #print("Translating", object_name)
            # this is a kludge, but can be important 
            object_class = model_props['object_class']
            model_class_translate(model_props, object_class)
        # now we either have a constant (key and value), or a 
        # fully defined object.  Either one should work OK.
        #print("Trying to load", object_name)
        model_object = model_class_loader(object_name, model_props, container)
        if model_object == False:
            print("Could not load", object_name)
            continue # not handled, but for now we will continue, tho later we should bail?
        # now for container type objects, go through its properties and handle
        if type(model_props) is dict:
            model_loader_recursive(model_props, model_object)

def model_path_loader(model_object_cache):
    k_list = model_object_cache.keys()
    model_names = dict.fromkeys(k_list , 1)
    for model_name in model_names:
        #print("Loading paths for", model_name)
        model_object = model_object_cache[model_name]
        model_object.find_paths()


def model_tokenizer_recursive(model_object, model_object_cache, model_exec_list, model_touch_list = []):
    """
    Given a root model_object, trace the inputs to load things in order
    Store this order in model_exec_list
    Note: All ordering is as-needed organic, except Broadcasts
          - read from children is completed after all other inputs 
          - read from parent is completed before all other inputs 
          - could this be accomplished by more sophisticated handling of read 
            broadcasts?  
            - When loading a read broadcast, can we iterate through items 
            that are sending to that broadcast? 
            - Or is it better to let it as it is, 
    """
    #print("Handling", model_object.name, " ", model_object.state_path)
    if model_object.ix in model_exec_list:
        return
    if model_object.ix in model_touch_list:
        #print("Already touched", model_object.name, model_object.ix, model_object.state_path)
        return
    # record as having been called, and will ultimately return, to prevent recursions
    model_touch_list.append(model_object.ix)
    k_list = model_object.inputs.keys()
    input_names = dict.fromkeys(k_list , 1)
    if type(input_names) is not dict:
        return 
    # isolate broadcasts, and sort out -- what happens if an equation references a broadcast var?
    # is this a limitation of treating all children as inputs? 
    # alternative, leave broadcasts organic, but load children first?
    # children first, then local sub-comps is old method? old method:
    #   - read parent broadcasts
    #   - get inputs (essentially, linked vars)
    #   - send child broadcasts (will send current step parent reads, last step local proc data)
    #   - execute children
    #   - execute local sub-comps
    for input_name in input_names:
        #print("Checking input", input_name)
        input_path = model_object.inputs[input_name]
        if input_path in model_object_cache.keys():
            input_object = model_object_cache[input_path]
            model_tokenizer_recursive(input_object, model_object_cache, model_exec_list, model_touch_list)
        else:
            if input_path in model_object.state_paths.keys():
                # this is a valid state reference without an object 
                # thus, it is likely part of internals that are manually added 
                # which should be fine.  tho perhaps we should have an object for these too.
                continue
            print("Problem loading input", input_name, "input_path", input_path, "not in model_object_cache.keys()")
            return
    # now after tokenizing all inputs this should be OK to tokenize
    model_object.add_op_tokens()
    if model_object.optype in ModelObject.runnables:
        model_exec_list.append(model_object.ix)


def model_order_recursive(model_object, model_object_cache, model_exec_list, model_touch_list = []):
    """
    Given a root model_object, trace the inputs to load things in order
    Store this order in model_exec_list
    Note: All ordering is as-needed organic, except Broadcasts
          - read from children is completed after all other inputs 
          - read from parent is completed before all other inputs 
          - could this be accomplished by more sophisticated handling of read 
            broadcasts?  
            - When loading a read broadcast, can we iterate through items 
            that are sending to that broadcast? 
            - Or is it better to let it as it is, 
    """
    if model_object.ix in model_exec_list:
        return
    if model_object.ix in model_touch_list:
        #print("Already touched", model_object.name, model_object.ix, model_object.state_path)
        return
    # record as having been called, and will ultimately return, to prevent recursions
    model_touch_list.append(model_object.ix)
    k_list = model_object.inputs.keys()
    input_names = dict.fromkeys(k_list , 1)
    if type(input_names) is not dict:
        return 
    # isolate broadcasts, and sort out -- what happens if an equation references a broadcast var?
    # is this a limitation of treating all children as inputs? 
    # alternative, leave broadcasts organic, but load children first?
    # children first, then local sub-comps is old method? old method:
    #   - read parent broadcasts
    #   - get inputs (essentially, linked vars)
    #   - send child broadcasts (will send current step parent reads, last step local proc data)
    #   - execute children
    #   - execute local sub-comps
    for input_name in input_names:
        #print("Checking input", input_name)
        input_path = model_object.inputs[input_name]
        if input_path in model_object_cache.keys():
            input_object = model_object_cache[input_path]
            model_order_recursive(input_object, model_object_cache, model_exec_list, model_touch_list)
        else:
            if input_path in model_object.state_paths.keys():
                # this is a valid state reference without an object 
                # thus, it is likely part of internals that are manually added 
                # which should be fine.  tho perhaps we should have an object for these too.
                continue
            print("Problem loading input", input_name, "input_path", input_path, "not in model_object_cache.keys()")
            return
    # now after loading input dependencies, add this to list
    model_exec_list.append(model_object.ix)

def model_domain_dependencies(state, domain, ep_list):
    """
    Given an hdf5 style path to a domain, and a list of variable endpoints in that domain, 
    Find all model elements that influence the endpoints state
    Returns them as a sorted list of index values suitable as a model_exec_list
    """
    mello = []
    for ep in ep_list:
        mel = []
        mtl = []
        # if the given element is NOT in model_object_cache, then nothing is acting on it, so we return empty list
        if (domain + '/' + ep) in state['model_object_cache'].keys():
            endpoint = state['model_object_cache'][domain + '/' + ep]
            model_order_recursive(endpoint, state['model_object_cache'], mel, mtl)
            mello = mello + mel
    
    return mello

def save_object_ts(io_manager, siminfo, op_tokens, ts_ix, ts):
    # Decide on using from utilities.py:
    # - save_timeseries(io_manager, ts, savedict, siminfo, saveall, operation, segment, activity, compress=True)
    # Or, skip the save_timeseries wrapper and call write_ts() directly in io.py:
    #  write_ts(self, data_frame:pd.DataFrame, save_columns: List[str], category:Category, operation:Union[str,None]=None, segment:Union[str,None]=None, activity:Union[str,None]=None)
    # see line 317 in utilities.py for use example of write_ts()
    x = 0 # dummy
    return

@njit
def iterate_models(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, steps, dstep = -1):
    checksum = 0.0
    for step in range(steps):
        pre_step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
        step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
    #print("Steps completed", step)
    return checksum

@njit
def pre_step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step):
    for i in model_exec_list:
        if op_tokens[i][0] == 12:
            # register type data (like broadcast accumulators) 
            pre_step_register(op_tokens[i], state_ix)
    return

@njit
def step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step):
    for i in model_exec_list:
        step_one(op_tokens, op_tokens[i], state_ix, dict_ix, ts_ix, step, 0)
    return 



@njit
def step_one(op_tokens, ops, state_ix, dict_ix, ts_ix, step, debug = 0):
    # op_tokens is passed in for ops like matrices that have lookups from other 
    # locations.  All others rely only on ops 
    # todo: decide if all step_[class() functions should set value in state_ix instead of returning value?
    val = 0
    if debug == 1:
        print("DEBUG: Operator ID", ops[1], "is op type", ops[0])
    if ops[0] == 1:
        pass #step_equation(ops, state_ix)
    elif ops[0] == 2:
        # todo: this should be moved into a single function, 
        # with the conforming name step_matrix(op_tokens, ops, state_ix, dict_ix)
        if (ops[1] == ops[2]):
            if debug == 1:
                print("DEBUG: Calling exec_tbl_values", ops)
            # this insures a matrix with variables in it is up to date 
            # only need to do this if the matrix data and matrix config are on same object
            # otherwise, the matrix data is an input and has already been evaluated
            pass #exec_tbl_values(ops, state_ix, dict_ix)
        if (ops[3] > 0):
            # this evaluates a single value from a matrix if the matrix is configured to do so.
            if debug == 1:
                print("DEBUG: Calling exec_tbl_eval", ops)
            pass #exec_tbl_eval(op_tokens, ops, state_ix, dict_ix)
    elif ops[0] == 3:
        step_model_link(ops, state_ix, ts_ix, step)
    elif ops[0] == 4:
        val = 0
    elif ops[0] == 5:
        step_sim_timer(ops, state_ix, dict_ix, ts_ix, step)
    elif ops[0] == 9:
        val = 0 
    elif ops[0] == 13:
        pass #step_simple_channel(ops, state_ix, dict_ix, step)
    # Op 100 is Basic ACTION in Special Actions
    elif ops[0] == 100:
        step_special_action(ops, state_ix, dict_ix, step)
    return 

@njit
def step_model_test(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step, debug_step = -1):
    for i in model_exec_list:
        ops = op_tokens[i]
        val = 0
        if (step == debug_step):
            print("Exec'ing step ", step, " model ID", i)
        # op_tokens is passed in for ops like matrices that have lookups from other 
        # locations.  All others rely only on ops 
        step_one(op_tokens, op_tokens[i], state_ix, dict_ix, ts_ix, step, 0)
    return 

@njit
def step_model_pcode(model_exec_list, op_tokens, state_info, state_paths, state_ix, dict_ix, ts_ix, step):
    '''
    This routine includes support for dynamically loaded python code which is powerful but slow
    This is not yet implemented anywhere, just an idea. But in theory it would allow easy switching between
    the faster runtime without dynamic code if the user did not request it.
    At minimum, this could be used to more efficiently enable/disable this feature for testing by simply calling
    a separate routine.
    - to do so we would need to add state_paths to the variables passed to step_model which should be OK?
    '''
    hydr_ix = hydr_get_ix(state_ix, state_paths, state_info['domain']) # could be done more efficiently, once per model run
    state_step_hydr(state_info, state_paths, state_ix, dict_ix, ts_ix, hydr_ix, step)
    val = 0
    for i in model_exec_list:
        step_one(op_tokens, op_tokens[i], state_ix, dict_ix, ts_ix, step, 0)
    return 

@njit
def post_step_model(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step):
    return 

def step_object(thisobject, step):
    # this calls the step for a given model object and timestep
    # this is a workaround since the object method ModelObject.step() fails to find the step_one() function ?
    step_one(thisobject.op_tokens, thisobject.op_tokens[thisobject.ix], thisobject.state_ix, thisobject.dict_ix, thisobject.ts_ix, step)


@njit
def pre_step_test(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step):
    for i in model_exec_list:
        ops = op_tokens[i]
    #for i in model_exec_list:
    #    op = op_tokens[i]
        if ops[0] == 12:
            # register type data (like broadcast accumulators) 
            pre_step_register(ops, state_ix)
            #continue
        #elif ops[0] == 1:
        #    # register type data (like broadcast accumulators) 
        #    continue
    return

@njit
def iterate_perf(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, steps, debug_step = -1):
    checksum = 0.0
    for step in range(steps):
        pre_step_test(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step)
        #step_model_test(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, step, debug_step)
    #print("Steps completed", step)
    return checksum

def time_perf(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, steps):
    start = time.time()       
    iterate_perf(model_exec_list, op_tokens,  state_ix, dict_ix, ts_ix, steps)
    end = time.time()
    print(len(model_exec_list), "components iterated over", siminfo['steps'], "time steps took" , end - start, "seconds")
