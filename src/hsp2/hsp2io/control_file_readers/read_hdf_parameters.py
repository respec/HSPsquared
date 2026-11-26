"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

License: LGPL2
"""

# Future imports
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

# Local folder imports
from ..utilities import default_and_validation

_dirname = Path(__file__).parent.parent.parent
_tables = pd.read_csv(_dirname / "data" / "Tables.csv")
_hdf_tables = _tables["HDF_TABLE"].to_list()


def read_hdf_parameters(hdfname: str | Path) -> dict:
    """
    Reads HSPsquared parameter tables from an HDF5 file.

    Parameters
    ----------
    hdfname
        Name/path of HDF5 file to read.

    Returns
    -------
    df
        Returns a dictionary of DataFrames containing the HSPsquared parameter
        dataframes read from the HDF5 file.
    """
    collect = {}
    with pd.HDFStore(hdfname, mode="r") as hdf_store:
        for key in hdf_store:
            if key in _hdf_tables or key.startswith("/FTABLES/"):
                print(f"Reading {key} from HDF5...")
                collect[key] = pd.read_hdf(
                    hdf_store, key=key, data_columns=True, format="table"
                )

    return default_and_validation(collect)
