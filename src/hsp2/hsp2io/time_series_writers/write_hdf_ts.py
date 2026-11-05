# Future imports
from __future__ import annotations

# Standard library imports
import contextlib
import textwrap
from pathlib import Path
from typing import Any

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.protocols import Category


def write_hdf_ts(
    hdffile: str,
    data_frame: pd.DataFrame | None = None,
    category: Category = "",
    operation: str = "",
    segment: str = "",
    activity: str = "",
    mode: str = "r+",
    **kwargs: Any,
) -> pd.DataFrame | None:
    """data_frame.empty
    Write or read time series data to/from an HDF5 file.

    If `data_frame` is None will read from the HDF5 file and return a pandas
    DataFrame.  Some parts of hsp2 need to read results from the output file.

    Parameters
    ----------
    hdffile
        The name of the HDF5 file to write to.
    data_frame
        The time series data to write.
    category
        The category of the time series (e.g., Category.INPUT or
        Category.RESULT).
    operation
        The operation associated with the time series.
    segment
        The segment associated with the time series.
    activity
        The activity associated with the time series.
    """
    if category == Category.INPUTS:
        path = f"/TIMESERIES/{segment}"
        if not Path(hdffile).exists():
            raise ValueError(
                textwrap.dedent(f"""\
                With "category" equal to "Category.INPUTS", time series should
                be read from "{hdffile}", but "{hdffile}" doesn't exist.
                """)
            )
    elif category == Category.RESULTS:
        path = f"/RESULTS/{operation}_{segment}/{activity}"

    if data_frame.empty:
        with contextlib.suppress(KeyError):
            return pd.read_hdf(hdffile, path)
    complevel = 9 if kwargs.get("compress") else None

    data_frame.to_hdf(
        hdffile,
        key=path,
        format="table",
        data_columns=True,
        complevel=complevel,
        mode=mode,
    )

    return None
