"""
Copyright 2020 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
"""

import os.path

import pandas as pd

from hsp2.hsp2.utilities import deprecated
from hsp2.hsp2io.control_file_readers.read_uci_parameters import read_uci_parameters

pd.set_option("io.hdf.default_format", "table")


@deprecated("""

readUCI is deprecated and will be removed in a future release.
Use hsp2.hsp2io.control_file_readers.read_uci_parameters to read the parameters.

""")
def readUCI(uciname, hdfname, overwrite=True):
    """
    Read data from a UCI file and create an HDF file with the data.

    Parameters
    ----------
    uciname : str
        The name of the UCI file to read.
    hdfname : str
        The name of the HDF file to store the data.
    overwrite : bool, optional
        Whether to overwrite existing data in the HDF file. Defaults to True.

    Returns
    -------
    None
    """
    if overwrite is True and os.path.exists(hdfname):
        os.remove(hdfname)

    parameters = read_uci_parameters(uciname)

    with pd.HDFStore(hdfname, mode="a") as store:
        for path, data in parameters.items():
            data.to_hdf(store, key=path, data_columns=True)
