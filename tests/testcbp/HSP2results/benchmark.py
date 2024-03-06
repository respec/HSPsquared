# bare bones tester
import os
os.chdir("C:/usr/local/home/git/HSPsquared")
from HSP2.main import *
from HSP2.om import *
#from HSP2.om_equation import *

state = init_state_dicts()
# set up info and timer
siminfo = {}
siminfo['delt'] = 60
siminfo['tindex'] = date_range("1984-01-01", "2020-12-31", freq=Minute(siminfo['delt']))[1:]
steps = siminfo['steps'] = len(siminfo['tindex'])
# get any pre-loaded objects 
model_data = state['model_data']
( ModelObject.op_tokens, ModelObject.model_object_cache) = init_om_dicts()
ModelObject.state_paths, ModelObject.state_ix, ModelObject.dict_ix, ModelObject.ts_ix = state['state_paths'], state['state_ix'], state['dict_ix'], state['ts_ix']
( op_tokens, state_paths, state_ix, dict_ix, model_object_cache, ts_ix) = ( ModelObject.op_tokens, ModelObject.state_paths, ModelObject.state_ix, ModelObject.dict_ix, ModelObject.model_object_cache, ModelObject.ts_ix )
state_context_hsp2(state, 'RCHRES', 'R001', 'HYDR')
print("Init HYDR state context for domain", state['domain'])
hydr_init_ix(state['state_ix'], state['state_paths'], state['domain'])
# Now, assemble a test dataset
container = False 
model_root_object = ModelObject("")
# set up the timer as the first element 
timer = SimTimer('timer', model_root_object, siminfo)

facility = ModelObject('facility', model_root_object)
for k in range(1000):
    #eqn = str(25*random.random()) + " * " + c[round((2*random.random()))]
    #newq = Equation('eq' + str(k), facility, {'equation':eqn} )
    conval = 50.0*random.random()
    newq = ModelConstant('con' + str(k), facility, conval)
    speca = SpecialAction('specl' + str(k), facility, {'OPTYP': 'RCHRES', 'RANGE1': 1, 'RANGE2':'', 'AC':'+=', 'VARI':'IVOL', 'VALUE':10.0, 'YR':'2000', 'DA':'1', 'MO':'1', 'HR':'1','MN':''})

# adjust op_tokens length to insure capacity
op_tokens = ModelObject.op_tokens = ModelObject.make_op_tokens(max(ModelObject.state_ix.keys()) + 1)
model_loader_recursive(model_data, model_root_object) 
# Parse, load and order all objects
model_path_loader(ModelObject.model_object_cache)
model_exec_list = []
model_touch_list = []
# put all objects in token form for fast runtime execution and sort according to dependency order
print("Tokenizing models")
model_tokenizer_recursive(model_root_object, ModelObject.model_object_cache, model_exec_list, model_touch_list )
op_tokens = ModelObject.op_tokens 
model_exec_list = np.asarray(model_exec_list, dtype="i8") 
# the resulting set of objects is returned.
state['model_object_cache'] = ModelObject.model_object_cache
state['op_tokens'] = ModelObject.op_tokens
state['state_step_om'] = 'disabled'

# using only these runnables cuts runtime by over 40%
# Test and time the run
start = time.time()
iterate_models(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, siminfo['steps'], -1)
end = time.time()
print(len(model_exec_list), "components iterated over state_ix", siminfo['steps'], "time steps took" , end - start, "seconds")


# test with np.array state 
#np_state_ix = np.asarray(list(state_ix.values()), dtype="float32")
np_state_ix = zeros(max(state_ix.keys()) + 1, dtype="float32")
for ix, iv in state_ix.items():
    np_state_ix[ix] = iv

start = time.time()
iterate_models(model_exec_list, op_tokens, np_state_ix, dict_ix, ts_ix, siminfo['steps'], -1)
end = time.time()
print(len(model_exec_list), "components iterated over np_state_ix(", type(np_state_ix),")", siminfo['steps'], "time steps took" , end - start, "seconds")

