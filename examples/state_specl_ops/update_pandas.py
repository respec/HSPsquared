
from numpy import float64, ones
from pandas import DataFrame, date_range
from pandas.tseries.offsets import Minute
from datetime import datetime as dt
from typing import Union
import os
# must import Category, Modelfirst to prevent circular error with HDF5
# this does not happen in real runtime...
from hsp2.hsp2.model import Model
from hsp2.hsp2io.protocols import Category
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2.utilities import (
    versions,
    get_timeseries,
    expand_timeseries_names,
    save_timeseries,
    get_gener_timeseries,
    hoursval
)
from hsp2.hsp2.configuration import activities, noop, expand_masslinks
from hsp2.state.state import (
    init_state_dicts,
    state_siminfo_hsp2,
    state_load_dynamics_hsp2,
    state_init_hsp2,
    state_context_hsp2,
)
from hsp2.hsp2.om import (
    om_init_state,
    state_om_model_run_prep,
    state_load_dynamics_om,
    state_om_model_run_finish,
)
from hsp2.hsp2.SPECL import specl_load_state

from hsp2.hsp2io.io import IOManager, SupportsReadTS, Category


fpath = "./tests/test10/HSP2results/test10.h5"
# try also:
# fpath = './tests/testcbp/HSP2results/JL1_6562_6560.h5'
# sometimes when testing you may need to close the file, so try:
# f = h5py.File(fpath,'a') # use mode 'a' which allows read, write, modify
# # f.close()
hdf5_instance = HDF5(fpath)
io_manager = IOManager(hdf5_instance)

# Begin code from main.py

# read user control, parameters, states, and flags parameters and map to local variables
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

#######################################################################################
# initialize STATE dicts
#######################################################################################
# Set up Things in state that will be used in all modular activities like SPECL
state = init_state_dicts()
state_siminfo_hsp2(parameter_obj, siminfo, io_manager, state)
# Add support for dynamic functions to operate on STATE
# - Load any dynamic components if present, and store variables on objects
state_load_dynamics_hsp2(state, io_manager, siminfo)
# Iterate through all segments and add crucial paths to state
# before loading dynamic components that may reference them
state_init_hsp2(state, opseq, activities)
# - finally stash specactions in state, not domain (segment) dependent so do it once
state["specactions"] = specactions  # stash the specaction dict in state
om_init_state(state)  # set up operational model specific state entries
specl_load_state(state, io_manager, siminfo)  # traditional special actions
state_load_dynamics_om(
    state, io_manager, siminfo
)  # operational model for custom python
# finalize all dynamically loaded components and prepare to run the model
state_om_model_run_prep(state, io_manager, siminfo)
#######################################################################################

# main processing loop
print(1, f"Simulation Start: {start}, Stop: {stop}")

# Test functions pandas
# this is from PWATER, should be easy?
ts_HRFG = hoursval(siminfo, ones(24), dofirst=True).astype(float)
# do import to allow dev of hoursval3 - temporary
from numpy import float64, full, tile, zeros
ts_HRFG3 = hoursval3(siminfo, ones(24), dofirst=True).astype(float)
from hsp2.hsp2.utilities import LAPSE
ts_LAPSE = hoursval(siminfo, LAPSE, lapselike=True)
ts_LAPSE3 = hoursval3(siminfo, LAPSE, lapselike=True)

# check these transform() calls inside of get_timeseries()
(operation, segment) = ('PERLND', 'P001')
psrc = ddext_sources[('PERLND', 'P001')]

ts = get_timeseries(
    io_manager, psrc, siminfo
)
prec_orig = ts['PREC']

row = psrc[0] # 0 is PRCP
data_frame = io_manager.read_ts(
    category=Category.INPUTS, segment=row.SVOLNO
)
# are they the same?
(prec_orig == data_frame).all()
# True - so we don't expect transform to do anything?
precip_pandas2 = transform(data_frame, row.TMEMN, row.TRAN, siminfo)
precip_pandas3 = transform3(data_frame, row.TMEMN, row.TRAN, siminfo)

# precip was fine, so iterate through them all and check the difference
for row in psrc:


# replicate with new code
data_frame = io_manager.read_ts(
    category=Category.INPUTS, segment=segment
)
tsfreq = ts.index.freq
freq = Minute(siminfo["delt"])
stop = siminfo["stop"]