# bare bones tester
# tests special actions and constants.
import os
# disabled for auto testing, but may use at command prompt if needed
#os.chdir("C:/usr/local/home/git/HSPsquared")
from HSP2.main import *
from HSP2.om import *
#from HSP2.om_equation import *
import pytest

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
state_om_model_root_object(state, siminfo)
model_root_object = state['model_root_object']

facility = ModelObject('facility', model_root_object)
for k in range(1000):
    #eqn = str(25*random.random()) + " * " + c[round((2*random.random()))]
    #newq = Equation('eq' + str(k), facility, {'equation':eqn} )
    conval = 50.0*random.random()
    newq = ModelConstant('con' + str(k), facility, conval)
    speca = SpecialAction('specl' + str(k), facility, {'OPTYP': 'RCHRES', 'RANGE1': 1, 'RANGE2':'', 'AC':'+=', 'VARI':'IVOL', 'VALUE':10.0, 'YR':'2000', 'DA':'1', 'MO':'1', 'HR':'1','MN':''})

# create a register to test TS
ts1 = facility.insure_register('/TIMESERIES/facility/con1', 0.0, facility)
# do all linking and tokenizing, 2nd arg "io_manager" is False as we do not have an hdf5 here
# set ops_data_type = Dict to test the Dict performance for state_ix
# override ops_data_type for testing:
ModelObject.ops_data_type = 'Dict' 
# - this forces state_om_model_run_prep() to use Dict type for op_tokens and state_ix
# - this can also be overridden by setting "siminfo" : {"ops_data_type" : "Dict" } in the json file
# now initialize model data sets (sorts, tokenizes, etc.)
state_om_model_run_prep(state, False, siminfo) 
op_tokens = state['op_tokens']

# run 1 time to compile all if anything is changed
model_exec_list = state['model_exec_list']
iterate_models(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, 1, -1)

# test with np.array state_ix
#np_state_ix = np.asarray(list(state_ix.values()), dtype="float32")
np_state_ix = zeros(max(state_ix.keys()) + 1, dtype="float32")
# this insures that the keys in the nparray version of state match 
for ix, iv in state_ix.items():
    np_state_ix[ix] = iv


# Test and time the run with Dict version of state_ix
start = time.time()
iterate_models(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, siminfo['steps'], -1)
end = time.time()
print(len(model_exec_list), "state_ix components iterated with full execution via iterate_models()", siminfo['steps'], "time steps took" , end - start, "seconds")

start = time.time()
iterate_models(model_exec_list, op_tokens, np_state_ix, dict_ix, ts_ix, siminfo['steps'], -1)
end = time.time()
print(len(model_exec_list), "np_state_ix components iterated with full execution via iterate_models()", siminfo['steps'], "time steps took" , end - start, "seconds")


@njit
def iteration_test_dat(op_order, it_ops, state_ix, it_nums):
    ctr = 0
    ttr = 0.0
    for n in range(it_nums):
        for i in op_order:
            op = it_ops[i]
            getsx = state_ix[i] 
            state_ix[i] = getsx * 1.0 
            #ttr = ttr + getsx
            ctr = ctr + 1
    print("Completed ", ctr, " loops")


# Now test just the data structures with no actual calculation from primitives
start = time.time()
iteration_test_dat(model_exec_list, op_tokens, state_ix, siminfo['steps'])
end = time.time()
print(len(model_exec_list), "components iterated over test data setter state_ix", siminfo['steps'], "time steps took" , end - start, "seconds")

start = time.time()
iteration_test_dat(model_exec_list, op_tokens, np_state_ix, siminfo['steps'])
end = time.time()
print(len(model_exec_list), "components iterated over test data setter np_state_ix", siminfo['steps'], "time steps took" , end - start, "seconds")

start = time.time()
iterate_perf(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, siminfo['steps'])
end = time.time()
print(len(model_exec_list), "state_ix components iterated over iterate_perf (selective component) ", siminfo['steps'], "time steps took" , end - start, "seconds")


start = time.time()
iterate_perf(model_exec_list, op_tokens, np_state_ix, dict_ix, ts_ix, siminfo['steps'])
end = time.time()
print(len(model_exec_list), "np_state_ix components iterated over iterate_perf (selective component)", siminfo['steps'], "time steps took" , end - start, "seconds")

def test_benchmark(model_exec_list, start, end):
    print(len(model_exec_list), "np_state_ix components iterated over iterate_perf (selective component)", siminfo['steps'], "time steps took" , end - start, "seconds")
