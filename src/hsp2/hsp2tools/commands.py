"""
Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
"""

import pandas as pd

from hsp2.hsp2.main import main
from hsp2.hsp2io.control_file_readers.read_uci_parameters import (
    read_uci_parameters as readUCI,
)
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import IOManager
from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts as readWDM

_WDM_OFFSET = {"WDM": 0, "WDM1": 100000, "WDM2": 200000, "WDM3": 300000, "WDM4": 400000}


def run(h5file, saveall=True, compress=True):
    """
    Run a HSPsquared model.

    Parameters
    ----------
    h5file: str
        HDF5 (path) filename used for both input and output.
    saveall: bool
        [optional] Default is True.
        Saves all calculated data ignoring SAVE tables.
    compression: bool
        [optional] Default is True.
        use compression on the save h5 file.
    """
    with HDF5(h5file) as hdf5_instance:
        io_manager = IOManager(hdf5_instance)
        main(io_manager, saveall=saveall, jupyterlab=compress)


def import_uci(ucifile, h5file):
    """
    Import UCI and WDM files into HDF5 file.

    Parameters
    ----------
    ucifile: str
        The UCI file to import into HDF file.
    h5file: str
        The destination HDF5 file.
    """
    # Read parameters.
    ddf = readUCI(ucifile)

    # Write parameters to HDF5 file.
    with pd.HDFStore(h5file, mode="a") as store:
        for path, df in ddf.items():
            df.to_hdf(store, key=path, data_columns=True)

    uci_dir = "/".join(ucifile.split("/")[:-1])

    wdmfiles = pd.read_hdf(h5file, key="/FILES/FILES").query(
        "FTYPE.str.startswith('WDM')"
    )

    for row in wdmfiles.itertuples():
        time_series = readWDM(
            f"{uci_dir}/{row.FNAME}", ts_number_shift=_WDM_OFFSET[row.FTYPE]
        )
        with pd.HDFStore(h5file) as store:
            for path, df in time_series.items():
                df.to_hdf(
                    store, key=f"/TIMESERIES/{path}", data_columns=True, format="table"
                )


def update_uci(ucifile, h5file):
    """
    Import parameters from User Control Interface file into HDF5 file.

    Note: this will NOT update time-series from WDM files.

    Parameters
    ----------
    ucifile: str
        The UCI file to import into HDF file.
    h5file: str
        The destination HDF5 file.

    See Also
    --------
    import_uci : Import parameters from UCI and time-series from WDM files into
    HDF5 file.
    """
    # Read parameters.
    ddf = readUCI(ucifile)

    # Write parameters to HDF5 file.
    with pd.HDFStore(h5file, mode="a") as store:
        for path, df in ddf.items():
            df.to_hdf(store, key=path, data_columns=True)
