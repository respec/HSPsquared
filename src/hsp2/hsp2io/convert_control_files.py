# Future imports
from __future__ import annotations

# Standard library imports
import textwrap
from collections.abc import Callable
from pathlib import Path

# First party imports
from hsp2.hsp2io.control_file_readers.read_csv_parameters import read_csv_parameters
from hsp2.hsp2io.control_file_readers.read_hdf_parameters import read_hdf_parameters
from hsp2.hsp2io.control_file_readers.read_uci_parameters import read_uci_parameters
from hsp2.hsp2io.control_file_writers.write_csv_parameters import write_csv_parameters
from hsp2.hsp2io.control_file_writers.write_hdf_parameters import write_hdf_parameters
from hsp2.hsp2io.control_file_writers.write_uci_parameters import write_uci_parameters

wrapper = textwrap.TextWrapper()


def get_parameter_reader(source: bytes | str | Path | None) -> Callable:
    """
    Get the correct parameter reader for the given source.

    Parameters
    ----------
    source : str or bytes
        The source to read from.  Can be a file path or URL.

    Returns
    -------
    function
        A function that takes a source and returns a dictionary of parameter
        DataFrames keyed by HDF_TABLE name.
    """
    if not source:
        raise ValueError(
            textwrap.dedent("""\
            The source is None or empty.
            """)
        )
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    inext = str(source).rsplit(".", maxsplit=1)[-1].lower()

    if inext == "csv":
        return read_csv_parameters
    if inext == "uci":
        return read_uci_parameters
    if inext in ["h5", "hdf5"]:
        return read_hdf_parameters

    raise ValueError(
        textwrap.dedent(f"""\

        The extension "{inext}" does not have a parser.

        """)
    )


def get_parameter_writer(target: bytes | Path | str) -> Callable:
    """
    Get the correct parameter writer for the given target.

    Parameters
    ----------
    target : str or bytes
        The source to write to.

    Returns
    -------
    function
        A function that takes a target and a dictionary of parameter DataFrames
        keyed by HDF_TABLE name and writes to the target.
    """
    if isinstance(target, bytes):
        target = target.decode("utf-8")
    inext = str(target).rsplit(".", maxsplit=1)[-1].lower()

    if inext == "csv":
        return write_csv_parameters
    if inext in ["h5", "hdf5"]:
        return write_hdf_parameters
    if inext == "uci":
        return write_uci_parameters

    raise ValueError(
        textwrap.dedent(f"""\

        The extension "{inext}" does not have a parser.

        """)
    )


def convert_control_files(in_source, out_source):
    """
    Convert parameters.

    Parameters
    ----------
    in_source
        Input parameter source.
    out_source
        Output parameter target.
    """
    indfs = get_parameter_reader(in_source)(in_source)
    get_parameter_writer(out_source)(out_source, indfs)
