# Future imports
from __future__ import annotations

# Standard library imports
import textwrap
from collections.abc import Callable
from pathlib import Path

# First party imports
from hsp2.hsp2io.time_series_readers.read_hbn_ts import read_hbn_ts
from hsp2.hsp2io.time_series_readers.read_hdf_ts import read_hdf_ts
from hsp2.hsp2io.time_series_readers.read_mutsin_ts import read_mutsin_ts
from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts
from hsp2.hsp2io.time_series_writers.write_hdf_ts import write_hdf_ts

wrapper = textwrap.TextWrapper()


def get_ts_reader(source: bytes | str | Path) -> Callable:
    """
    Get the correct time series reader for the given source.

    Parameters
    ----------
    source : str or bytes
        The source to read from.  Can be a file path or URL.

    Returns
    -------
    dataframe
        A dataframe of time series with names in the columns and time in the
        index.
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    inext = str(source).rsplit(".", maxsplit=1)[-1].lower()

    if inext == "wdm":
        return read_wdm_ts
    if inext == "hbn":
        return read_hbn_ts
    if inext == "plt":
        return read_mutsin_ts
    if inext in ["h5", "hdf5"]:
        return read_hdf_ts

    raise ValueError(
        textwrap.dedent(f"""\

        The extension "{inext}" does not have a time series reader.

        """)
    )


def get_ts_writer(target: bytes | str | Path) -> Callable:
    """
    Get the correct time series writer for the given target.

    Parameters
    ----------
    target : str or bytes
        The source to write time series to.  Can be a file path or URL.

    Returns
    -------
    function
        A function that takes a pandas dataframe of time series with names in
        the columns and time in the index, an name and writes to a file.
    """
    if isinstance(target, bytes):
        target = target.decode("utf-8")
    inext = str(target).rsplit(".", maxsplit=1)[-1].lower()

    if inext in ["h5", "hdf5"]:
        return write_hdf_ts

    raise ValueError(
        textwrap.dedent(f"""\

        The extension "{inext}" does not have a time series writer.

        """)
    )


def convert_time_series(in_source, out_source):
    """
    Convert time series from one file to another.

    Parameters
    ----------
    in_source
        Input parameter source.
    out_source
        Output parameter target.
    """
    indfs = get_ts_reader(in_source)(in_source)
    get_ts_writer(out_source)(out_source, indfs)
