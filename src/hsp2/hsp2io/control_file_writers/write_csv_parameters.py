# Standard library imports
from pathlib import Path

# Local folder imports
from ..utilities import _parameters, _tables

_parameters = _parameters.fillna("")


def write_csv_parameters(basename, parameters):
    """
    Write a set of CSV files from a parameters dictionary.

    The CSV files will be named based on the HSPF block and table names, and
    the RUN.csv file will list all the generated CSV files.

    This will only write to a local directory.  If you want to store the files
    at a URL, you should move them after they are written and edit the *RUN.csv
    file to point to the URL location.

    Parameters
    ----------
    basename : str
        The path and optional base name of the RUN.csv that identifies the
        location of the parameter files.  The basename must end in "RUN.csv" and
        can contain a directory path.

        If basename is set to "path/to/RUN.csv" will create CSV files in the
        "path/to" directory and a RUN.csv file listing them.

        If basename is set to "path/to/calibrated_RUN.csv", will create
        "calibrated_*.csv" files in the "path/to" directory and a
        calibrated_RUN.csv file listing them.

    parameters : dict
        A dictionary of pandas DataFrames.  The keys should correspond to the
        HDF_TABLE names, and the values should be the DataFrames containing the
        parameters for each table.
    """
    # Ensure the directory exists
    directory = Path(basename).parent
    if directory and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    basename = str(basename)[: -len("run.csv")]
    newdict = {}
    for key, df in parameters.items():
        tabpars = _parameters.loc[_parameters["HDF_TABLE"] == key, :]
        if key.startswith("/FTABLES"):
            hspf_keys = {("FTABLES", "FTABLE")}
        else:
            hspf_keys = set(
                zip(
                    tabpars["HSPF_BLOCK"].fillna("").to_list(),
                    tabpars["HSPF_TABLE"].fillna("").to_list(),
                )
            )

        # Repeated tables (CURV-DATA1, CONS-DATA1, SILT-CLAY-PMsilt, ...) share
        # one HSPF_TABLE, so write each to its own file named with the
        # HSPF_TABLE_EXT.
        ext_rows = _tables.loc[
            (_tables["HDF_TABLE"] == key)
            & (_tables["HSPF_TABLE_EXT"].notna())
            & (_tables["HSPF_TABLE_EXT"] != _tables["HSPF_TABLE"]),
            ["HSPF_BLOCK", "HSPF_TABLE_EXT"],
        ]
        if not key.startswith("/FTABLES") and not ext_rows.empty:
            for block, ext in zip(ext_rows["HSPF_BLOCK"], ext_rows["HSPF_TABLE_EXT"]):
                cols = [c for c in df.columns if c not in ("OPNID", "OPNIDLAST")]
                newdict[f"{block}_{ext}.csv"] = df[cols]
            continue

        for block, table in hspf_keys:
            pars = []
            fname = [block]
            if table:
                fname.append(table)
                table_num = ""
                if block in ["FTABLES"]:
                    table_num = str(int(key.split("/")[-1][2:]))
                if table_num:
                    fname.append(table_num)
            fname = f"{'_'.join(fname)}.csv"

            if block in ["SCHEMATIC", "NETWORK"]:
                # The "SCHEMATIC" and "NETWORK" blocks are combined into
                # the HDF_TABLE "/CONTROL/LINKS" and have overlapping
                # parameters and have to be separated.  All (probably) other
                # HDF_parameters will not have overlapping parameters and can be
                # easily separated into their component (HSPF_BLOCK,
                # HSPF_TABLE) tables.  Depending on the order in the original
                # "_parameters" and "_parameters" could mean all the
                # "/CONTROL/LINKS" records are under "SCHEMATIC" or "NETWORK"
                # so need to check both.
                if "MLNO" in df.columns:
                    # Possibly only SCHEMATIC block entries or mixture of
                    # SCHEMATIC and NETWORK entries in /CONTROL/LINKS
                    schematic = df.loc[df["MLNO"] != "", :]
                    if not schematic.empty:
                        newdict["SCHEMATIC.csv"] = schematic
                    network = df.loc[df["MLNO"] == "", :]
                    if not network.empty:
                        newdict["NETWORK.csv"] = network
                else:
                    # Only NETWORK block entries in /CONTROL/LINKS
                    newdict["NETWORK.csv"] = df
            elif block in ["GLOBAL", "FTABLES", "EXT SOURCES", "EXT TARGETS"]:
                newdict[fname] = df
            else:
                pars = _parameters.loc[
                    (_parameters["HSPF_BLOCK"] == block)
                    & (_parameters["HSPF_TABLE"] == table),
                    "PARAMETER_NAME",
                ].to_list()
                pars = [i for i in pars if i not in ("OPNID", "OPNIDLAST")]
                try:
                    newdict[fname] = df[pars]
                except KeyError:
                    continue

    with open(f"{basename}RUN.csv", "w") as fp:
        for nfname, outframe in newdict.items():
            outframe.to_csv(basename + nfname)
            lbasename = "" if basename.endswith("/") else Path(basename).name
            fp.write(f"{lbasename}{nfname}\n")
