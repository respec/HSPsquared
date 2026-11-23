# Future imports
from __future__ import annotations

# Standard library imports
import textwrap
from collections.abc import Callable

# First party imports
from hsp2.hsp2io.log_file_writers.log_csv import write_csv_log
from hsp2.hsp2io.log_file_writers.log_hdf import write_hdf_log

wrapper = textwrap.TextWrapper()


def get_log_writer(target: bytes | str) -> Callable:
    """
    Get the correct log writer for the given target.

    logs
    ----------
    target : str or bytes
        The log file to write to. The extension of the file will determine the
        log writer used.

    Returns
    -------
    function
        A function that to write logs to the given target.
    """
    if isinstance(target, bytes):
        target = target.decode("utf-8")
    inext = str(target).rsplit(".", maxsplit=1)[-1].lower()

    if inext == "csv":
        return write_csv_log
    if inext in ["h5", "hdf5"]:
        return write_hdf_log

    raise ValueError(
        textwrap.dedent(f"""\

        The extension "{inext}" does not have a parser.

        """)
    )
