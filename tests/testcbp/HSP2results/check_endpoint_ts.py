# Must be run from the HSPsquared source directory, the h5 file has already been setup with hsp import_uci test10.uci
# bare bones tester - must be run from the HSPsquared source directory
import os
from hsp2.hsp2.main import *
from hsp2.hsp2.om import *
from hsp2 import hsp2iO
import numpy
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
fpath = './tests/test10/HSP2results/test10.h5'
# try also:
# fpath = './tests/testcbp/HSP2results/PL3_5250_0001.h5'
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
rchres1 = state['model_object_cache']['/STATE/RCHRES_R001']
precip_ts = ModelLinkage('PRCP', rchres1, {'right_path':'/TIMESERIES/TS039', 'SVOLNO':'TS039', 'link_type':3})
ts1 = precip_ts.read_ts() # same as precip_ts.ts_ix[precip_ts.ix], same as state['ts_ix'][precip_ts.ix]

# NOTE: THIS SEEMS TO FAIL RIGHT NOW DUE TO WRITING THE TS VALUES BACK WITH A NAME OF tsvalue_0, and is now embeded


# is the same as:
# - ts = precip_ts.io_manager.read_ts(Category.INPUTS, None, precip_ts.ts_name)
# - ts = transform(ts, precip_ts.ts_name, 'SAME', precip_ts.siminfo)
# - ts = precip_ts.io_manager.read_ts(Category.INPUTS, None, precip_ts.ts_name).columns
# - ts = np.transpose(ts)[0]
# precip_ts.io_manager._input._store
# write it back.  We can give an arbitrary name or it will default to write back to the source path in right_path variable
# we can specify a custom path to write this TS to
precip_ts.write_path = '/RESULTS/test_TS039'
precip_ts.write_ts()
# precip_ts.write_ts is same as:
#     ts4 = precip_ts.format_ts(ts1, ['tsvalue'], siminfo['tindex'])
#     ts4.to_hdf(precip_ts.io_manager._output._store, precip_ts.write_path, format='t', data_columns=True, complevel=precip_ts.complevel)

tsdf = pd.DataFrame(data=ts1, index=siminfo['tindex'],columns=None)
# verify
ts1 = precip_ts.read_ts() # same as precip_ts.ts_ix[precip_ts.ix], same as state['ts_ix'][precip_ts.ix]
# Calls:
# - ts = precip_ts.io_manager.read_ts(Category.INPUTS, None, precip_ts.ts_name)
# - ts = transform(ts, self.ts_name, 'SAME', self.siminfo)
# - ts = np.transpose(ts)[0]
# should yield equivalent of:
ts2 = hdf5_instance._store[precip_ts.ts_path]
# data_frame.to_hdf(self._store, path, format='t', data_columns=True, complevel=complevel)
ts3 = hdf5_instance._store[precip_ts.write_path]
# and is same as
ts4 = precip_ts.io_manager._output._store[precip_ts.write_path]

exdd = defaultdict(list)
exdd[(precip_ts.SVOL, precip_ts.SVOLNO)].append(pd.DataFrame({'SVOL':precip_ts.SVOLNO, 'SVOLNO':precip_ts.SVOLNO, 'MFACTOR':precip_ts.MFACTOR, 'TMEMN':precip_ts.TMEMN, 'TMEMSB':precip_ts.TMEMSB}))
exdd[(precip_ts.SVOL, precip_ts.SVOLNO)].append({'SVOL':precip_ts.SVOLNO, 'SVOLNO':precip_ts.SVOLNO, 'MFACTOR':precip_ts.MFACTOR, 'TMEMN':precip_ts.TMEMN, 'TMEMSB':precip_ts.TMEMSB})

exdd = defaultdict(list)
df = pd.DataFrame({'SVOL':[precip_ts.SVOLNO], 'SVOLNO':[precip_ts.SVOLNO], 'MFACTOR':[precip_ts.MFACTOR], 'TMEMN':[precip_ts.TMEMN], 'TMEMSB':[precip_ts.TMEMSB]})
for row in df.itertuples():
    exdd[(precip_ts.SVOL, precip_ts.SVOLNO)].append(row)
for row in exdd:
    data_frame = precip_ts.io_manager.read_ts(category=Category.INPUTS,segment=row.SVOLNO)

xdr = list(df.itertuples())
xdr = precip_ts.ext_sourcesdd
for row in xdr:
    data_frame = precip_ts.io_manager.read_ts(category=Category.INPUTS,segment=row.SVOLNO)
    clean_name(row.TMEMN,row.TMEMSB)

ext_sourcesdd = XDR
ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
data_frame = precip_ts.io_manager.read_ts(category=Category.INPUTS,segment=row.SVOLNO)

if row.MFACTOR != 1.0:
    data_frame *= row.MFACTOR

t = transform(data_frame, row.TMEMN, row.TRAN, siminfo)

tname = clean_name(row.TMEMN,row.TMEMSB)
if tname in ts:
    ts[tname] += t
else:
    ts[tname]  = t


# Debug: def read_ts(self, category:Category, operation:Union[str,None]=None,  segment:Union[str,None]=None,  activity:Union[str,None]=None, ...)
ts = get_timeseries(precip_ts.io_manager, precip_ts.ext_sourcesdd, precip_ts.siminfo)
for row in precip_ts.ext_sourcesdd[(precip_ts.SVOL, precip_ts.SVOLNO)]:
    print(str(row.SVOLNO))
precip_ts.io_manager.read_ts(Category.INPUTS, None, precip_ts.ts_name)
precip_ts.io_manager.read_ts(category=Category.INPUTS,segment=row.SVOLNO)
precip_ts.io_manager.read_ts(category=Category.INPUTS,segment=str(row.SVOLNO))
precip_ts.io_manager.read_ts(category=Category.INPUTS, operation=None,segment=row.SVOLNO)
