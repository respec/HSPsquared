# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
import os

import numpy
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.state.state import *

fpath = "./tests/test10specl/HSP2results/test10specl.h5"
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)
parameter_obj = io_manager.read_parameters()
ddext_sources = parameter_obj.ddext_sources
siminfo = parameter_obj.siminfo
opseq = parameter_obj.opseq
# Note: now that the UCI is read in and hdf5 loaded, you can see things like:
(state, om_operations, statenb) = om_state_hsp2_run_setup(parameter_obj, io_manager, activities)

# state['model_root_object'].find_var_path('RCHRES_R001')
# Get the timeseries naked, without an object
rchres1 = om_operations["model_object_cache"]["/STATE/test10specl/RCHRES_R001"]
Rlocal = rchres1.get_object('Rlocal')
Rlocal_ts = Rlocal.read_ts()
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
