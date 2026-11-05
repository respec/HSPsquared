"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.

Based on MATLAB program by Seth Kenner, RESPEC
License: LGPL2
"""

# Future imports
from __future__ import annotations

# Standard library imports
from threading import Lock
from typing import Literal

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.protocols import Category


def read_hdf_ts(
    hdffile,
    category: Category | Literal["RESULT", "INPUT"],
    operation: str | None = None,
    segment: str | None = None,
    activity: str | None = None,
) -> pd.DataFrame:
    """
    Read time series from HDF5 file.

    If category is Category.RESULTS or "RESULT", the HDF file key is

    "/RESULTS/{operation}_{segment}/{activity}"

    For example: "/RESULTS/IMPLND_I001/IQUAL"

    If category is Category.INPUTS or "INPUT", the HDF file key is

    "/TIMESERIES/{segment}"

    For example: "/TIMESERIES/TS100046"

    Parameters
    ----------
    hdffile : str or h5py.File
        HDF5 file name or open file object
    category : Category
        Category of data to read.

        +------------------+------------+-----------------------------+
        | Enum             | String     | Description                 |
        +==================+============+=============================+
        | category.RESULTS | "RESULT"   | Output data that needs to   |
        |                  |            | be read in (e.g., IMPLND,   |
        |                  |            | IQUAL)                      |
        +------------------+------------+-----------------------------+
        | category.INPUTS  | "INPUT"    | Input time series (e.g.,    |
        |                  |            | TS100046)                   |
        +------------------+------------+-----------------------------+

    operation : str, optional
        Operation name (required if category is RESULTS).

        Available operations in HSPsquared are "COPY", "GENER", "IMPLND",
        "PERLND", and "RCHRES".

        Operations that are implemented in HSPF, but not currently implemented
        in HSPsquared but may be in the future are "BMPRAC", "DISPLY",
        "DURANL", "MUTSIN", "PLTGEN", and "REPORT".

    segment : str, optional
        Segment name (required)
    activity : str, optional
        Activity name (required if category is RESULTS)

    Returns
    -------
    pd.DataFrame
        Time series data as a DataFrame
    """
    if category in [Category.INPUTS, "INPUT"]:
        path = f"/TIMESERIES/{segment}"
    elif category in [Category.RESULTS, "RESULT"]:
        path = f"/RESULTS/{operation}_{segment}/{activity}"

    return pd.read_hdf(hdffile, path)


class HDF5Reader:
    """Read HSP2 result series using HSPF names and constituent mappings."""

    REQUIRES_MAPPING = {"GQUAL", "CONS", "IQUAL", "PQUAL"}

    def __init__(
        self,
        file_name: str,
        aliases: dict[tuple[str, str, str], str] | None = None,
    ) -> None:
        """Create a reader, optionally using an HSPF-to-HSP2 alias mapping."""
        self.file_name = file_name
        self.aliases = {} if aliases is None else aliases
        self.data: dict[tuple[str, str, str], pd.DataFrame] = {}
        self.lock = Lock()

        self.gqual_prefixes = self._read_nqual_mapping("RCHRES/GQUAL/GQUAL", "GQID", 7)
        self.cons_prefixes = self._read_nqual_mapping("RCHRES/CONS/CONS", "CONID", 7)
        self.iqual_prefixes = self._read_nqual_mapping("IMPLND/IQUAL/IQUAL", "QUALID")
        self.pqual_prefixes = self._read_nqual_mapping("PERLND/PQUAL/PQUAL", "QUALID")

    def _read_nqual_mapping(
        self, key: str, target_col: str, nquals: int = 10
    ) -> dict[str, str]:
        """Map constituent identifiers to their numbered output groups."""
        mappings = {}
        for i in range(1, nquals):
            try:
                with pd.HDFStore(self.file_name, "r") as store:
                    if key.endswith(("IQUAL", "PQUAL")):
                        df = pd.read_hdf(store, f"{key}{i}/FLAGS")
                    else:
                        df = pd.read_hdf(store, f"{key}{i}")
                mappings[df.iloc[0][target_col]] = str(i)
            except KeyError:
                continue
        return mappings

    def get_time_series(
        self, operation: str, id: str, constituent: str, activity: str
    ) -> pd.Series | None:
        """Read a requested HSP2 time series, resolving HSPF names as needed."""
        operation = operation.upper()
        constituent = constituent.upper()
        activity = activity.upper()

        constituent_prefix = ""
        if activity in self.REQUIRES_MAPPING:
            prefix_dict = getattr(self, f"{activity.lower()}_prefixes")
            for key, value in prefix_dict.items():
                if (
                    activity in {"PQUAL", "IQUAL"}
                    and constituent.endswith(key)
                    or activity not in {"PQUAL", "IQUAL"}
                    and constituent.startswith(key)
                ):
                    constituent_prefix = f"{activity}{value}_"
                    constituent = constituent.replace(key, "")

        cache_key = (operation, id, activity)
        try:
            with self.lock:
                if cache_key not in self.data:
                    self.data[cache_key] = self._read_table(operation, id, activity)

            data_frame = self.data[cache_key]
            if constituent_prefix + constituent in data_frame.columns:
                return data_frame[constituent_prefix + constituent]
            constituent_alias = self.aliases[(operation, activity, constituent)]
            return data_frame[constituent_prefix + constituent_alias]
        except KeyError:
            return None

    def _read_table(self, operation: str, id: str, activity: str) -> pd.DataFrame:
        segment = f"{operation[0]}{id}"
        try:
            return read_hdf_ts(
                self.file_name,
                Category.RESULTS,
                operation=operation,
                segment=segment,
                activity=activity,
            )
        except KeyError:
            return pd.DataFrame()
