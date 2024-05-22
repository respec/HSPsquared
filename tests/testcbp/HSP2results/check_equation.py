# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
import os
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
import numpy
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)
uci_obj = io_manager.read_uci()
siminfo = uci_obj.siminfo
opseq = uci_obj.opseq
# Note: now that the UCI is read in and hdf5 loaded, you can see things like:
# - hdf5_instance._store.keys() - all the paths in the UCI/hdf5
# - finally stash specactions in state, not domain (segment) dependent so do it once
# now load state and the special actions
state = init_state_dicts()
state_initialize_om(state)
state['specactions'] = uci_obj.specactions # stash the specaction dict in state

state_siminfo_hsp2(uci_obj, siminfo)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities)
state_load_dynamics_specl(state, io_manager, siminfo) # traditional special actions
state_load_dynamics_om(state, io_manager, siminfo) # operational model for custom python
state_om_model_run_prep(state, io_manager, siminfo) # this creates all objects from the UCI and previous loads
# state['model_root_object'].find_var_path('RCHRES_R001')
# Get the timeseries naked, without an object
Rlocal = state['model_object_cache']['/STATE/RCHRES_R001/Rlocal']
Rlocal_ts = Rlocal.read_ts()
rchres1 = state['model_object_cache']['/STATE/RCHRES_R001']
Rlocal_check = ModelLinkage('Rlocal1', rchres1, {'right_path':'/TIMESERIES/TS010', 'link_type':3})
# Calls:
# - ts = Rlocal.io_manager.read_ts(Category.INPUTS, None, Rlocal.ts_name)
# - ts = transform(ts, Rlocal.ts_name, 'SAME', Rlocal.siminfo)
Rlocal.io_manager._output._store.keys()
# write it back.  We can give an arbitrary name or it will default to write back to the source path in right_path variable
ts1 = precip_ts.read_ts() # same as precip_ts.ts_ix[precip_ts.ix], same as state['ts_ix'][precip_ts.ix]
# we can specify a custom path to write this TS to
precip_ts.write_path = '/RESULTS/test_TS039'
precip_ts.write_ts()
# precip_ts.write_ts is same as:
#     ts4 = precip_ts.format_ts(ts1, ['tsvalue'], siminfo['tindex'])
#     ts4.to_hdf(precip_ts.io_manager._output._store, precip_ts.write_path, format='t', data_columns=True, complevel=precip_ts.complevel)

start = time.time()
iterate_models(model_exec_list, op_tokens, state_ix, dict_ix, ts_ix, siminfo['steps'], -1)
end = time.time()
print(len(model_exec_list), "components iterated over state_ix", siminfo['steps'], "time steps took" , end - start, "seconds")

