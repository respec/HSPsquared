"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
"""

# Standard library imports
from pathlib import Path
from urllib.error import URLError

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.utilities import (
    _tables,
    default_and_validation,
    hspf_tables_to_hdf_tables,
    is_valid_url,
)


def process_source(ifile, run_csvname=""):
    ncollected = {}

    if ifile.lower().endswith("run.csv"):
        # Similar code to later on in the file, but thought this is the one
        # time where Don't Repeat Yourself doesn't fit.
        try:
            rec_ifiles = (
                pd.read_csv(ifile, header=None, sep=">", comment="#")
                .iloc[:, 0]
                .tolist()
            )
        except (FileNotFoundError, URLError):
            # This might mean that the file path is relative to the
            # `run_csvname` file, so try that.
            if is_valid_url(run_csvname):
                enclosing_dir = "/".join(run_csvname.split("/")[:-1])
            else:
                enclosing_dir = Path(run_csvname).parent

            try:
                rec_ifiles = (
                    pd.read_csv(
                        str(enclosing_dir / ifile), header=None, sep=">", comment="#"
                    )
                    .iloc[:, 0]
                    .tolist()
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"""
                    '{ifile}' listed in '{run_csvname}' not found.
                    Checked '{ifile}' and '{enclosing_dir / ifile}'.
                    Check the path and file name.""") from e

        for rec_ifile in rec_ifiles:
            rec_append_dict = process_source(rec_ifile, run_csvname=ifile)
            for key, val in rec_append_dict.items():
                ncollected[key] = ncollected.get(key, pd.DataFrame()).append(val)
        return ncollected

    # cheat that works for both local files or URLs
    words = Path(ifile).stem.split("_")
    nwords = []
    if (
        len(words) >= 3
        and (
            (_tables["HSPF_BLOCK"] == words[0])
            & (_tables["HSPF_TABLE_EXT"] == "_".join(words[1:]))
        ).any()
    ):
        # HSPF_TABLE_EXT of a repeated table that contains an underscore
        nwords = (words[0], "_".join(words[1:]), None)
    elif len(words) == 3:
        # HSPF_BLOCK, HSPF_TABLE, table_num
        nwords = (words[0], words[1], words[2])
    elif len(words) == 2:
        # HSPF_BLOCK, HSPF_TABLE, None
        nwords = (words[0], words[1], None)
    elif len(words) == 1:
        # HSPF_BLOCK, None, None
        nwords = (words[0], None, None)
    else:
        raise ValueError(f"""
            {ifile} does not follow the naming convention
            "{{HSPF_block}}_{{HSPF_table}}_{{HSPF_table_number}}.csv"
            or "*RUN.csv".
            """)

    try:
        val = pd.read_csv(ifile, sep=",", index_col=0)
        if nwords not in ncollected:
            ncollected[nwords] = val
        else:
            ncollected[nwords] = ncollected[nwords].join(val, how="outer")
    except (FileNotFoundError, URLError):
        # This might mean that the file path is relative to the
        # `run_csvname` file, so try that.
        if is_valid_url(run_csvname):
            enclosing_dir = "/".join(run_csvname.split("/")[:-1])
        else:
            enclosing_dir = Path(run_csvname).parent

        try:
            val = pd.read_csv(str(enclosing_dir / ifile), sep=",", index_col=0)
            if nwords not in ncollected:
                ncollected[nwords] = val
            else:
                ncollected[nwords] = ncollected[nwords].join(val, how="outer")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"""
                '{ifile}' listed in '{run_csvname}' not found.
                Check the path and file name.""") from exc

    return ncollected


def read_csv_parameters(run_csvname):
    """
    Reads a set of CSV files into a dictionary of DataFrames.

    Parameters
    ----------
    run_csvname : str
        The `run_csvname` file is a file that contains a list of the CSV files
        to read, one per line.  The file names in the *RUN.csv file must be
        named in the form using
        *{HSPFblock}_{HSPFtable}_{HSPFtablenumber}.csv.

        The "_" is the delimiter between the "HSPFblock", "HSPFtable", and
        "HSPFtablenumber". The "HSPFtablenumber" is only used for
        "FTABLES_FTABLE_*.csv" and "MASS-LINKS_MASS-LINKS_*.csv" files.  Some
        file names may not have "HSPFtable" or "HSPFtablenumber" in the name,
        and if so there should be no trailing underscores in the file name.

        You can organize your CSV files in any directory structure that makes
        sense to you, including reading from remote URLs.  For local files the
        path must be absolute or relative to the location of the *RUN.csv file.

        Comments start with "#" and are ignored.  Blank lines are also ignored.

        Example 1, if `run_csvname` is '/path/to/Black_Creek/calibrationRUN.csv' ::

            # Comments and blank lines will be ignored.  Following lines are
            # absolute paths to CSV files.
            /path/to/Black_Creek/EXT-SOURCES.csv
            /path/to/Black_Creek/EXT-TARGETS.csv
            /path/to/Black_Creek/FTABLES_FTABLE_1.csv
            /path/to/Black_Creek/FTABLES_FTABLE_2.csv
            /path/to/Black_Creek/MASS-LINK_MASS-LINK_1.csv

            # Then some relative paths...
            PERLND_PWAT-PARM1.csv
            PERLND_PWAT-PARM2.csv
            RCHRES_HYDR-PARM1.csv

            # Then a URL
            https://blah.org/path/to/Black_Creek/RCHRES_HYDR-PARM2.csv

            # Can even include *RUN.csv files, for example, that would
            # represent upstream sub-watersheds.
            https://blah.org/path/to/Black_Creek/SW1/RUN.csv
            https://blah.org/path/to/Black_Creek/SW2/RUN.csv

        Note that HSPF UCI block names "EXT SOURCES" and "EXT TARGETS" must
        replace the space with a "-".

        The `run_csvname` file can include entries to other '*/RUN.csv' files
        and duplicate table names are appended together.  This means that if
        you are careful about having unique operation IDs across
        sub-watersheds, each sub-watershed can be a stand-alone model, and then
        easily included into the basin model.

    Returns
    -------
    dict
        A dictionary of DataFrames, keyed on HSP2 table names.
    """
    # Use pd.read_csv to take advantage of its ability to read URLs or files.
    # Since reading in only one column, chose a ">" separator because it is not
    # likely to be used in the file names, and shouldn't be used in URLs.
    ifiles = (
        pd.read_csv(run_csvname, header=None, sep=">", comment="#").iloc[:, 0].tolist()
    )

    ncollected = {}
    for ifile in ifiles:
        append_dict = process_source(ifile, run_csvname=run_csvname)
        for key, val in append_dict.items():
            if key not in ncollected:
                ncollected[key] = val
            else:
                ncollected[key] = ncollected[key].join(val, how="outer")

    ncollected = hspf_tables_to_hdf_tables(ncollected)
    return default_and_validation(ncollected)
