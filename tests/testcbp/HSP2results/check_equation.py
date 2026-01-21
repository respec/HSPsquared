# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory

import os
import numpy
from hsp2.hsp2.main import *
from hsp2.state.state import *
from hsp2.hsp2.om import *
from hsp2.hsp2.SPECL import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.state.state import *
from hsp2.hsp2.om_timer import timer_class

fpath = "./tests/testcbp/HSP2results/PL3_5250_0001.h5"
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
timer = timer_class()
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)

parameter_obj = io_manager.read_parameters()
opseq = parameter_obj.opseq
ddlinks = parameter_obj.ddlinks
ddmasslinks = parameter_obj.ddmasslinks
ddext_sources = parameter_obj.ddext_sources
ddgener = parameter_obj.ddgener
model = parameter_obj.model
siminfo = parameter_obj.siminfo
ftables = parameter_obj.ftables
specactions = parameter_obj.specactions
monthdata = parameter_obj.monthdata

start, stop = siminfo["start"], siminfo["stop"]

copy_instances = {}
gener_instances = {}
print("io_manager.read_parameters() call and config", timer.split(), "seconds")
#######################################################################################
# initialize STATE dicts
#######################################################################################
# Set up Things in state that will be used in all modular activities like SPECL
state = state_class(
    state_empty["state_ix"], state_empty["op_tokens"], state_empty["state_paths"], 
    state_empty["op_exec_lists"], state_empty["model_exec_list"], state_empty["dict_ix"], 
    state_empty["ts_ix"], state_empty["hsp_segments"]
)
print("init_state_dicts()", timer.split(), "seconds")
om_operations = om_init_state()  # set up operational model specific containers
state_siminfo_hsp2(state, parameter_obj, siminfo, io_manager)
state_om_model_root_object(state, om_operations, siminfo)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
print("state_load_dynamics_hsp2() call and config", timer.split(), "seconds")
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities, timer)
print("state_init_hsp2() call and config", timer.split(), "seconds")
hsp2_domain_dependencies(state, opseq, activities, om_operations, False)
print("hsp2_domain_dependencies() call and config", timer.split(), "seconds")
specl_load_om(om_operations, specactions)  # load traditional special actions
print("specl_load_om() call and config", timer.split(), "seconds")
state_load_dynamics_om(
    state, io_manager, siminfo, om_operations
) # operational model for custom python
print("state_load_dynamics_om() call and config", timer.split(), "seconds")
# finalize all dynamically loaded components and prepare to run the model
state_om_model_run_prep(opseq, activities, state, om_operations, siminfo)
print("state_om_model_run_prep() call and config", timer.split(), "seconds")
#######################################################################################

# debug loading:
# mtl = []
# mel = []
# model_order_recursive(endpoint, om_operations["model_object_cache"], mel, mtl, True)

RCHRES = om_operations["model_object_cache"]["/STATE/PL3_5250_0001/RCHRES_R001"]
O2 = om_operations["model_object_cache"]["/STATE/PL3_5250_0001eq/RCHRES_R001/O2"]
wd_500 = om_operations["model_object_cache"][RCHRES.find_var_path('wd_500')]

wd_cfs = om_operations["model_object_cache"]["/STATE/PL3_5250_0001eq/RCHRES_R001/wd_cfs"]

state.get_ix_path(wd_cfs.ops[6]) 
state.get_ix_path(wd_cfs.ops[7]) 

wd_cfs.find_var_path("O2")


##### Check exec list
ep_list = hydr_init_ix(state, RCHRES.state_path)
op_exec_list = model_domain_dependencies(
    om_operations, state, RCHRES.state_path, ep_list, True, False
)
op_exec_list = np.asarray(op_exec_list)

# state['model_root_object'].find_var_path('RCHRES_R001')
# Get the timeseries naked, without an object
Rlocal = om_operations["model_object_cache"]["/STATE/RCHRES_R001/Rlocal"]
Rlocal_ts = Rlocal.read_ts()
rchres1 = om_operations["model_object_cache"]["/STATE/RCHRES_R001"]
Rlocal_check = ModelLinkage(
    "Rlocal1", rchres1, {"right_path": "/TIMESERIES/TS010", "link_type": 3}
)
# Calls:
# - ts = Rlocal.io_manager.read_ts(Category.INPUTS, None, Rlocal.ts_name)
# - ts = transform(ts, Rlocal.ts_name, 'SAME', Rlocal.siminfo)
Rlocal.io_manager._output._store.keys()
# write it back.  We can give an arbitrary name or it will default to write back to the source path in right_path variable
ts1 = (
    precip_ts.read_ts()
)  # same as precip_ts.ts_ix[precip_ts.ix], same as state['ts_ix'][precip_ts.ix]
# we can specify a custom path to write this TS to
precip_ts.write_path = "/RESULTS/test_TS039"
precip_ts.write_ts()
# precip_ts.write_ts is same as:
#     ts4 = precip_ts.format_ts(ts1, ['tsvalue'], siminfo['tindex'])
#     ts4.to_hdf(precip_ts.io_manager._output._store, precip_ts.write_path, format='t', data_columns=True, complevel=precip_ts.complevel)

start = time.time()
iterate_models(
    model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, siminfo["steps"], -1
)
end = time.time()
print(
    len(model_exec_list),
    "components iterated over state_ix",
    siminfo["steps"],
    "time steps took",
    end - start,
    "seconds",
)


# try also:
# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
# sometimes when testing you may need to close the file, so try:
# import h5py;f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
import os
import numpy
import h5py
from hsp2.hsp2.main import *
from hsp2.state.state import *
from hsp2.hsp2.om import *
from hsp2.hsp2.SPECL import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.hsp2tools.readUCI import *
from src.hsp2.hsp2tools.commands import import_uci, run
from pandas import read_hdf

fpath = "./tests/testcbp/HSP2results/PL3_5250_0001.h5"
# to run use this:
# run(fpath, saveall=True, compress=False)
dstore_hydr = pd.HDFStore(str(fpath), mode='r')
hsp2_hydr = read_hdf(dstore_hydr, '/RESULTS/RCHRES_R001/HYDR')
np.quantile(hsp2_hydr[:]['O2'], [0,0.25,0.5,0.75,1.0])
# To re-run:
dstore_hydr.close()

fpath = "./tests/testcbp/HSP2results/PL3_5250_0001wd.h5"
# to run use this:
# run(fpath, saveall=True, compress=False)
dstore_hydr = pd.HDFStore(str(fpath), mode='r')
hsp2_wd_hydr = read_hdf(dstore_hydr, '/RESULTS/RCHRES_R001/HYDR')
dstore_hydr.close()
np.quantile(hsp2_wd_hydr[:]['O2'], [0,0.25,0.5,0.75,1.0])
# To re-run:


fpath = "./tests/testcbp/HSP2results/PL3_5250_0001eq.h5"
# to run use this:
# run(fpath, saveall=True, compress=False)
dstore_hydreq = pd.HDFStore(str(fpath), mode='r')
hsp2_eq_hydr = read_hdf(dstore_hydreq, '/RESULTS/RCHRES_R001/HYDR')
dstore_hydreq.close()
np.quantile(hsp2_eq_hydr[:]['O2'], [0,0.25,0.5,0.75,1.0])
# To re-run:

np.mean(hsp2_hydr[:]['IVOL'])
np.mean(hsp2_eq_hydr[:]['IVOL'])
np.mean(hsp2_hydr[:]['OVOL3'])
np.mean(hsp2_eq_hydr[:]['OVOL3'])
