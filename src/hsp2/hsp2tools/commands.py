"""
Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
"""

# Standard library imports
from pathlib import Path

# First party imports
from hsp2.hsp2.main import main
from hsp2.hsp2io.control_file_readers.read_uci_parameters import read_uci_parameters
from hsp2.hsp2io.control_file_writers.write_hdf_parameters import write_hdf_parameters
from hsp2.hsp2io.convert_control_files import convert_control_files as convpars
from hsp2.hsp2io.io import IOManager
from hsp2.hsp2io.protocols import Category
from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts
from hsp2.hsp2io.time_series_writers.write_hdf_ts import write_hdf_ts
from hsp2.hsp2io.utilities import is_valid_url


def run(
    io_combined=None,
    parameter_source=None,
    ts_source=None,
    ts_target=None,
    log=None,
    saveall=True,
    compress=True,
):
    """
    Run a HSPsquared model.

    Parameters
    ----------
    io_combined: str
        HDF5 (path) filename used for both parameters and time-series data. If
        provided, this will be used instead of the individual parameter_source,
        ts_source, ts_target, and log parameters.
    parameter_source: str
        UCI or HDF5 source for parameters. Ignored if io_combined is provided.
    ts_source: str
        WDM or HDF5 filename for time-series data. Ignored if io_combined is
        provided.
    ts_target: str
        HDF5 (path) filename for time-series output. Ignored if io_combined is
        provided.
    log: str
        HDF5 (path) filename for log output. Ignored if io_combined is provided.
    saveall: bool
        [optional] Default is True.
        Saves all calculated data ignoring SAVE tables.
    compress: bool
        [optional] Default is True.
        Use compression on the tables in the output HDF5 file.
    """
    main(
        IOManager(
            io_combined=io_combined,
            parameter_source=parameter_source,
            ts_source=ts_source,
            ts_target=ts_target,
            log=log,
        ),
        saveall=saveall,
        jupyterlab=compress,
    )


def run_cli(
    parameter_source=None,
    ts_source=None,
    ts_target=None,
    log=None,
    saveall=True,
    compress=True,
    *io_combined,
):
    """
    Run a HSPsquared model.

    Use either a single combined HDF5 file, "hsp2 run fname.h5", or separate
    files, "hsp2 run --parameter_source=fname.uci --ts_source=met.h5
    --ts_target=results.h5".

    Parameters
    ----------
    parameter_source: str
        UCI, CSV, or HDF5 source for parameters.  Required if the combined
        HDF5 file is not given.
    ts_source: str
        WDM or HDF5 filename for time-series data.  Required if the combined
        HDF5 file is not given.
    ts_target: str
        HDF5 (path) filename for time-series output.  Required if the combined
        HDF5 file is not given.
    log: str
        HDF5 (path) filename for log output.
    saveall: bool
        [optional] Default is True.
        Saves all calculated data ignoring SAVE tables.
    compress: bool
        [optional] Default is True.
        Use compression on the tables in the output HDF5 file.
    io_combined: str
        [optional positional] HDF5 (path) filename used for both parameters and
        time-series data.  If provided, the other file options are ignored.
    """
    if len(io_combined) > 1:
        raise ValueError("Only one combined HDF5 file can be given.")
    run(
        io_combined=io_combined[0] if io_combined else None,
        parameter_source=parameter_source,
        ts_source=ts_source,
        ts_target=ts_target,
        log=log,
        saveall=saveall,
        compress=compress,
    )


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
    pars = read_uci_parameters(ucifile)
    write_hdf_parameters(h5file, pars, mode="w")

    files_block = pars["/FILES/FILES"]

    ts_number_shift = {
        "WDM": 100_000,
        "WDM1": 100_000,
        "WDM2": 200_000,
        "WDM3": 300_000,
        "WDM4": 400_000,
    }
    for _, row in files_block.iterrows():
        if row.iloc[0] in ["WDM", "WDM1", "WDM2", "WDM3", "WDM4"]:
            wdmfile = row.iloc[2].strip()
            if Path(wdmfile).is_absolute():
                wdmfile = Path(wdmfile)
            elif is_valid_url(ucifile):
                wdmfile = ucifile.split("/")[:-1] + [wdmfile]
                wdmfile = "/".join(wdmfile)
            else:
                wdmfile = Path(ucifile).parent / wdmfile
            time_series = read_wdm_ts(
                wdmfile, ts_number_shift=ts_number_shift[row.iloc[0].strip()]
            )
            for key, series in time_series.items():
                write_hdf_ts(
                    h5file, series, category=Category.INPUTS, segment=key, mode="r+"
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
    print("Updating parameters in h5 file from UCI.", ucifile)
    print("Note: this will NOT update external data such as WDM, mutsin etc.")
    print("To update all data, use the command 'hsp2 import_uci ...' ")
    pars = read_uci_parameters(ucifile)
    write_hdf_parameters(h5file, pars, mode="r+")


def convert_control_files(in_source, out_target):
    """
    Convert parameters storage.

    Parameters
    ----------
    in_source
        UCI, CSV, or HDF5 source for parameters.

        UCI File
        --------
        The path to the UCI file containing the parameters to import.

        CSV Files
        ---------
        The path and optional base name of the RUN.csv that lists the location
        of the parameter files.  The basename must end in "RUN.csv" and can
        contain a directory path.  The CSV files listed in the *RUN.csv file
        will be named "{HSPF_BLOCK}_{HSPF_TABLE}.csv" and
        "{HSPF_BLOCK}_{HSPF_TABLE}_{table_number}.csv" for FTABLES and MASS-LINK
        CSV files.

        HDF5 File
        ---------
        The path to the HDF5 file containing the pandas Dataframes of parameters
        keyed to HDF_TABLE to import.

    out_target
        UCI, CSV, or HDF5 target for parameters.

        UCI File
        --------
        The path to the UCI file to create or update with the converted
        parameters.

        CSV Files
        ---------
        If basename is set to "path/to/RUN.csv" will create CSV files in the
        "path/to" directory and a RUN.csv file listing them.

        If basename is set to "path/to/calibrated_RUN.csv", will create
        "calibrated_*.csv" files in the "path/to" directory and a
        calibrated_RUN.csv file listing them.

        HDF5 File
        ---------
        The path to the HDF5 file to create or update with the converted
        parameters.
    """
    convpars(in_source, out_target)
