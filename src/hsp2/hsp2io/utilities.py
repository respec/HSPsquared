# Future imports
from __future__ import annotations

# Standard library imports
import contextlib
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Third party imports
import numpy as np
import pandas as pd

with contextlib.suppress(Exception):
    pd.options.future.infer_string = True

wrapper = textwrap.TextWrapper()

_dirname = Path(__file__).parent.parent
fill_table_name = {"HSPF_TABLE": ""}
_tables = pd.read_csv(_dirname / "data" / "Tables.csv").fillna(fill_table_name)
_parameters = pd.read_csv(_dirname / "data" / "Parameters.csv").fillna(fill_table_name)
_save_tables = pd.read_csv(_dirname / "data" / "SaveTable.csv")

# Bring over HDF_TABLE from _tables to _parameters.
_parameters = pd.merge(
    _tables[
        [
            "HSPF_BLOCK",
            "HSPF_TABLE",
            "HSPF_TABLE_EXT",
            "HDF_TABLE",
        ]
    ],
    _parameters,
    on=["HSPF_BLOCK", "HSPF_TABLE"],
    how="left",
)


# There are several tables that appear multiple times in the UCI file, and the
# order matters.  For example, the SILT-CLAY-PM table appears twice, the first
# table are the parameters for silt and the second table the parameters for
# clay. Use the following mappings to adjust the table name when reading or writing.
# The position (key) is the order in the UCI file and the value is the suffix
# appended to the table name to make the HSPF_TABLE_EXT in Tables.csv.
_gq_daughter_map = {
    0: "_hydrolysis",
    1: "_oxidation",
    2: "_photolysis",
    3: "_reserved",
    4: "_biodegradation",
    5: "_first_order",
}
_silt_clay_pm_map = {0: "silt", 1: "clay"}
_two_layer_map = {0: "surface", 1: "upper"}
_four_layer_map = {0: "surface", 1: "upper", 2: "lower", 3: "groundwater"}

_10_map = {i: f"{i + 1}" for i in range(10)}
_03_map = {i: f"{i + 1}" for i in range(3)}
_119_map = {i: f"{i + 1}" for i in range(119)}

_four_layer_03_map = {
    key + j: f"{val}_{j}" for key, val in _four_layer_map.items() for j in range(3)
}

_multiple_table_names = {
    ("IMPLND", "MON-ACCUM"): _10_map,
    ("IMPLND", "MON-POTFW"): _10_map,
    ("IMPLND", "MON-SQOLIM"): _10_map,
    ("IMPLND", "QUAL-INPUT"): _10_map,
    ("IMPLND", "QUAL-PROPS"): _10_map,
    ("PERLND", "MON-ACCUM"): _10_map,
    ("PERLND", "MON-GRND-CONC"): _10_map,
    ("PERLND", "MON-IFLW-CONC"): _10_map,
    ("PERLND", "MON-NITAGUTF"): _four_layer_map,
    ("PERLND", "MON-NITIMAM"): _four_layer_map,
    ("PERLND", "MON-NITIMNI"): _four_layer_map,
    ("PERLND", "MON-NITUPAM"): _four_layer_map,
    ("PERLND", "MON-NITUPNI"): _four_layer_map,
    ("PERLND", "MON-NITUPT"): _four_layer_map,
    ("PERLND", "MON-NPRETBG"): _four_layer_map,
    ("PERLND", "MON-NPRETLI"): _two_layer_map,
    ("PERLND", "MON-NUPT-FR2"): _four_layer_map,
    ("PERLND", "MON-PUPT-FR2"): _four_layer_map,
    ("PERLND", "MON-PHOSUPT"): _four_layer_map,
    ("PERLND", "MON-POTFS"): _10_map,
    ("PERLND", "MON-POTFW"): _10_map,
    ("PERLND", "MON-SQOLIM"): _10_map,
    ("PERLND", "NIT-FSTPM"): _four_layer_map,
    ("PERLND", "NIT-ORGPM"): _four_layer_map,
    ("PERLND", "NIT-STOR1"): _four_layer_map,
    ("PERLND", "NIT-SVALPM"): _four_layer_map,
    ("PERLND", "NIT-UPIMCSAT"): _four_layer_map,
    ("PERLND", "NIT-UPIMKMAX"): _four_layer_map,
    ("PERLND", "PEST-FIRSTPM"): _four_layer_03_map,
    ("PERLND", "PEST-NONSVPM"): _four_layer_03_map,
    ("PERLND", "PEST-STOR1"): _four_layer_03_map,
    ("PERLND", "PEST-SVALPM"): _four_layer_03_map,
    ("PERLND", "PHOS-FSTPM"): _four_layer_map,
    ("PERLND", "PHOS-STOR1"): _four_layer_map,
    ("PERLND", "PHOS-SVALPM"): _four_layer_map,
    ("PERLND", "QUAL-INPUT"): _10_map,
    ("PERLND", "QUAL-PROPS"): _10_map,
    ("PLTGEN", "CURV-DATA"): _119_map,
    ("RCHRES", "CONS-DATA"): _10_map,
    ("RCHRES", "GQ-DAUGHTER"): _gq_daughter_map,
    ("RCHRES", "SILT-CLAY-PM"): _silt_clay_pm_map,
    ("RCHRES", "GQ-QALDATA"): _03_map,
    ("RCHRES", "GQ-QALFG"): _03_map,
    ("RCHRES", "GQ-HYDPM"): _03_map,
    ("RCHRES", "GQ-ROXPM"): _03_map,
    ("RCHRES", "GQ-PHOTPM"): _03_map,
    ("RCHRES", "GQ-CFGAS"): _03_map,
    ("RCHRES", "GQ-BIOPM"): _03_map,
    ("RCHRES", "GQ-GENDECAY"): _03_map,
    ("RCHRES", "GQ-SEDDECAY"): _03_map,
    ("RCHRES", "GQ-KD"): _03_map,
    ("RCHRES", "GQ-ADRATE"): _03_map,
    ("RCHRES", "GQ-SEDCONC"): _03_map,
}

# Tables with parameters spread over continuation lines, keyed by (HSPF_BLOCK,
# HSPF_TABLE) with the number of 80 character lines.  The reader joins the
# lines into one long line and the writer splits them again.  Parameters.csv
# START and STOP columns reflect the joined line.
_continuation_lines = {
    ("GLOBAL", ""): 4,
    ("RCHRES", "GQ-PHOTPM"): 3,
    ("RCHRES", "GQ-ALPHA"): 3,
    ("RCHRES", "GQ-GAMMA"): 3,
    ("RCHRES", "GQ-DAUGHTER"): 6,
    ("RCHRES", "GQ-DELTA"): 3,
    ("RCHRES", "GQ-CLDFACT"): 3,
    ("DURANL", "LCONC"): 2,
    ("DURANL", "LEVELS"): 2,
}


def _table_ext_names(block: str, table: str) -> list[str]:
    """HSPF_TABLE_EXT names of an HSPF_TABLE, expanded for repeated tables."""
    suffixes = _multiple_table_names.get((block, table))
    if suffixes is None:
        return [table]
    return [f"{table}{suffix}" for suffix in suffixes.values()]


# Same as _continuation_lines, but keyed by the collected (HSPF_BLOCK,
# HSPF_TABLE_EXT, table_num) used when reading a UCI file.
_continuation_keys = [
    ((block, ext, None), n_lines)
    for (block, table), n_lines in _continuation_lines.items()
    for ext in _table_ext_names(block, table)
]

# HSP2 combines the operation type and number in some tables.  So ("PERLND",
# 101) is "P101", ("IMPLND", 67) is "I067", ...etc.
#
# Currently HSP2 only uses COPY, GENER, IMPLND, PERLND, and RCHRES.  The others
# are used by HSPF and are kept to move parameters between UCI and HDF5 for all
# tables.
_operation_prefix_map = {
    "COPY": "C",
    "GENER": "G",
    "IMPLND": "I",
    "PERLND": "P",
    "RCHRES": "R",
    "BMPRAC": "B",
    "DISPLY": "D",
    "DURANL": "U",
    "MUTSIN": "M",
    "PLTGEN": "L",
    "REPORT": "E",
}

# (HDF_TABLE, column with the number, column with the operation type) that get
# the operation prefix in HDF5, but only a number in the UCI file.
_operation_prefix_columns = [
    ("/CONTROL/OP_SEQUENCE", "SEGMENT", "OPERATION"),
    ("/CONTROL/LINKS", "SVOLNO", "SVOL"),
    ("/CONTROL/LINKS", "TVOLNO", "TVOL"),
    ("/CONTROL/EXT_SOURCES", "TVOLNO", "TVOL"),
]

# (HDF_TABLE, column, prefix) for MASS-LINK numbers and FTABLE numbers.
_number_prefix_columns = [
    ("/CONTROL/MASS_LINKS", "MLNO", "ML"),
    ("/RCHRES/HYDR/PARAMETERS", "FTBUCI", "FT"),
    ("/CONTROL/LINKS", "MLNO", "ML"),
]


def hspf_tables_to_hdf_tables(ncollected):
    """
    Merge the collected HSPF tables into HDF tables.

    Used to merge {(HSFP_block, HSPF_table, table_num): dataframe} to be
    {HDF_TABLE: merged_dataframes}.

    Parameters
    ----------
    ncollected : dict of (HSPF_block, HSPF_table, table_num) to pd.DataFrame
        The collected HSPF tables.

    Returns
    -------
    nhdf_collected : dict of HDF_TABLE to pd.DataFrame
        The merged DataFrames keyed by HDF_TABLE name.
    """
    # Add all tables in ncollected and all defaultable tables from tables in
    # ncollected to xcollected since shouldn't modify ncollected within the
    # loop.
    xcollected = {}
    for (hspf_block, hspf_table, table_num), df in ncollected.items():
        xcollected[(hspf_block, hspf_table, table_num)] = df

        # Find all defaultable tables that require the (hspf_block, hspf_table,
        # table_num) table.
        # At this point table_num will always be None since there are no
        # defaultable tables that require FTABLES or MASS-LINK tables (the only
        # table_num numbered tables)
        defaultable = _tables.loc[
            (_tables["HSPF_BLOCK"] == hspf_block)
            & ((_tables["REQ_TABLE"] == hspf_table) | (_tables["REQ_TABLE"] == "NONE"))
        ]

        # Create empty Dataframes for each defaultable table that require
        # (hspf_block, hspf_table, None).
        for _, row in defaultable.iterrows():
            if (row["HSPF_BLOCK"], row["HSPF_TABLE"], None) in ncollected:
                continue
            param_names = _parameters.loc[
                (_parameters["HSPF_BLOCK"] == row["HSPF_BLOCK"])
                & (_parameters["HSPF_TABLE"] == row["HSPF_TABLE"]),
                "PARAMETER_NAME",
            ].tolist()
            param_names = [p for p in param_names if p not in ("OPNID", "OPNIDLAST")]
            xcollected[(hspf_block, row["HSPF_TABLE"], None)] = pd.DataFrame(
                index=df.index, columns=param_names
            )

    # Collect the (hspf_block, hspf_table, table_num) Dataframes into a list in
    # the dictionary hdf_collected keyed to the HDF_TABLE.
    hdf_collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    collect_hspf_tables = defaultdict(list)
    hspf_blocks = set()
    for (hspf_block, hspf_table, table_num), df in xcollected.items():
        record = _tables.loc[
            (_tables["HSPF_BLOCK"] == hspf_block)
            & (
                (_tables["HSPF_TABLE_EXT"] == hspf_table)
                | (_tables["HSPF_TABLE_EXT"].isnull())
            ),
        ]
        if record.empty:
            continue

        for lhdf_table in record["HDF_TABLE"]:
            # MASS-LINK, FTABLE, and MONTH-DATA have numbered tables in the UCI
            # file.
            # MASS-LINK is different where the table_num is stored in the pandas
            # Dataframe as MLNO, so we don't append the table_num to the HDF_TABLE
            # name for MASS-LINK.
            # So just append the table number to 'hdf_table' for the FTABLE and
            # MONTH-DATA tables.
            hdf_table = lhdf_table
            if hspf_table in ["FTABLE", "MONTH-DATA"]:
                hdf_table = f"{hdf_table}{int(table_num):03d}"
            hdf_collected[hdf_table].append(df)
            collect_hspf_tables[hdf_table].append((hspf_block, hspf_table))
            hspf_blocks.add(hspf_block)

    # Merge DataFrames for (HSPF_block, HSPF_table) into single DataFrame for
    # each HDF_TABLE.
    nhdf_collected: dict[str, pd.DataFrame | pd.Series] = {}
    for hdf_table, list_dfs in hdf_collected.items():
        dfs = pd.DataFrame()
        for df in list_dfs:
            if df.empty:
                continue
            if dfs.empty:
                dfs = df
            elif hdf_table in ["/CONTROL/LINKS", "/CONTROL/MASS_LINKS"]:
                dfs = pd.concat((dfs, df), ignore_index=True, sort=True)
            else:
                dfs = dfs.merge(
                    df,
                    left_index=True,
                    right_index=True,
                    how="outer",
                    suffixes=("", "_DROP"),
                )
                dfs = dfs.drop(dfs.filter(regex="_DROP$").columns, axis=1)
        nhdf_collected[hdf_table] = dfs.loc[:, ~dfs.columns.duplicated(keep="first")]

    return nhdf_collected


def default_and_validation(parameters):
    """
    Fill in default values and validate parameter values.

    The only required DataFrame is keyed to "/CONTROL/GLOBAL" and contains the
    parameters from the "GLOBAL" block within a UCI file.

    Parameters
    ----------
    parameters : dict of pd.DataFrame
        Dictionary of HDF table names to DataFrames of parameters.

    Returns
    -------
    nparameters : dict of pd.DataFrame
        Dictionary of HDF table names to DataFrames of parameters with
        defaults filled in and validated.
    """
    # Need to find the units to extract default, min, and max for float and
    # integer parameters.
    info = parameters["/CONTROL/GLOBAL"].loc[:, "Info"]
    parameters["/CONTROL/GLOBAL"].loc["Units", "Info"] = info.get("Units", "1")
    parameters["/CONTROL/GLOBAL"].loc["LEV", "Info"] = info.get("LEV", "0")
    parameters["/CONTROL/GLOBAL"].loc["SPA", "Info"] = info.get("SPA", "2")
    parameters["/CONTROL/GLOBAL"].loc["RES", "Info"] = info.get("RES", "0")
    parameters["/CONTROL/GLOBAL"].loc["RUN", "Info"] = info.get("RUN", "0")
    units = {"1": "ENGL", "2": "METR"}[
        parameters["/CONTROL/GLOBAL"].loc["Units", "Info"]
    ]
    parameters["/CONTROL/GLOBAL"] = (
        parameters["/CONTROL/GLOBAL"]
        .T.fillna(
            {
                "Units": "1",
                "LEV": "0",
                "SPA": "2",
                "RES": "0",
                "RUN": "0",
            }
        )
        .T
    )

    nparameters = {}
    for hdf_table, parms in parameters.items():
        if hdf_table == "/CONTROL/GLOBAL":
            nparameters[hdf_table] = parms
            continue

        # Ignore Series...
        if isinstance(parms, pd.Series):
            nparameters[hdf_table] = parms
            continue

        # Ignore /*/SAVE tables.  They will be recreated later.
        if hdf_table.endswith("/SAVE"):
            continue

        if "/RESULTS" in hdf_table:
            continue

        if "/RUN_INFO" in hdf_table:
            continue

        # The FTABLE parameter tables have special names like "/FTABLES/FT{XXX}"
        # where XXX is the FTABLE number.  Set the prefix "/FTABLES/FT" and the
        # number will be added later.
        table_name = "/FTABLES/FT" if hdf_table.startswith("/FTABLES/FT") else hdf_table

        parameter_data = _parameters[_parameters["HDF_TABLE"] == table_name]
        parameter_data = parameter_data.set_index("PARAMETER_NAME")

        cols = parms.columns.drop(
            ["OPNIDLAST", "TVOLNO", "TOPFST", "TOPLST", "Unnamed: 0"], errors="ignore"
        )
        parameter_data = parameter_data.loc[parameter_data.index.isin(cols), :]

        # Set the numerical defaults from DEFAULT_ENGL or DEFAULT_METR from the
        # Parameters.csv file and then use fillna.
        defaults = parameter_data[
            [f"DEFAULT_{units}", f"MIN_{units}", f"MAX_{units}"]
        ].astype(float)
        defaults.columns = ["DEFAULT", "MIN", "MAX"]

        # Fill in the numerical defaults.
        nparms = parms.infer_objects().fillna(
            dict(zip(defaults.index, defaults["DEFAULT"].values))
        )

        # Fill in the text defaults.
        defaults_text = parameter_data[["DEFAULT_TEXT"]].dropna().astype(str)
        nparms = nparms.fillna(
            dict(zip(defaults_text.index, defaults_text["DEFAULT_TEXT"].values))
        )

        # Set the types.
        nparms = nparms.astype(
            dict(zip(parameter_data.index, parameter_data["TYPE"].values)),
            errors="ignore",
        )

        # Only fill string-like columns with empty string.  Avoid filling
        # numeric or other dtypes which could coerce types or mask missing
        # numeric values. Operate on both 'object' and pandas 'string' dtypes.
        str_cols = nparms.select_dtypes(include=["object", "string"]).columns
        if not str_cols.empty:
            nparms[str_cols] = nparms[str_cols].fillna("")
            nparms[str_cols] = nparms[str_cols].replace("nan", "")
            nparms[str_cols] = nparms[str_cols].astype(
                object if pd.__version__ < "3" else "str"
            )

        for name, series in nparms.items():
            loop_series = series.dropna()
            if loop_series.empty:
                continue

            if pd.api.types.is_string_dtype(loop_series):
                loop_series = loop_series.str.replace("[a-zA-Z]", "", regex=True)
                loop_series = pd.to_numeric(loop_series, errors="coerce").dropna()

            # Check that values are within the allowed range.
            left = defaults.loc[parameter_data.index == name, "MIN"]
            left = np.nan if left.empty else left.values[0]
            if np.isfinite(left):
                lmask = loop_series >= left
            else:
                lmask = pd.Series([True] * len(loop_series), index=loop_series.index)

            right = defaults.loc[parameter_data.index == name, "MAX"]
            right = np.nan if right.empty else right.values[0]
            if np.isfinite(right):
                rmask = loop_series <= right
            else:
                rmask = pd.Series([True] * len(loop_series), index=loop_series.index)

            testdf = lmask & rmask
            if not testdf.all():
                raise ValueError(
                    wrapper.wrap(
                        textwrap.dedent(f"""\
                            Values for table_name={table_name}, and
                            parameter={name} are outside of the allowed range.
                            min={left}, max={right}, series={loop_series.values}.
                            """)
                    )
                )

        for name, series in nparms.items():
            # Check that string values are VALID.
            allowed = parameter_data.loc[parameter_data.index == name, "VALID"]
            if len(allowed) == 0:
                continue
            allowed = allowed.values[0]
            if isinstance(allowed, str):
                allowed = eval(allowed)
            else:
                continue
            lseries = series.dropna()
            if not lseries.isin(allowed).all():
                raise ValueError(
                    wrapper.wrap(
                        textwrap.dedent(f"""\
                            Values for hdf_table={hdf_table}, and
                            parameter={name} are outside of the allowed set.
                            allowed={allowed}, values={lseries.values}.
                            """)
                    )
                )

        nparms = nparms.drop(columns=["Unnamed: 0"], errors="ignore")

        if "OPNID" in nparms.columns:
            nparms["OPNID"] = nparms["OPNID"].astype("string")

        if not nparms.empty:
            nparameters[hdf_table] = pd.DataFrame(nparms)

    # Create the SAVE tables.  These identify the time-series that are saved.
    savetable = defaultdict(dict)
    for row in _save_tables.itertuples():
        savetable[row.OPERATION, row.ACTIVITY][row.NAME] = row.VALUE
    operations_activities = set()
    for key in nparameters:
        words = key.split("/")
        if len(words) < 4:
            continue
        operation = words[1]
        activity = words[2]
        if activity == "GENERAL":
            continue
        operations_activities.add((operation, activity))
    for op in _save_tables["OPERATION"].unique():
        for operation, activity in operations_activities:
            if operation != op:
                continue
            savedf = pd.DataFrame(
                index=sorted(nparameters[f"/{op}/GENERAL/INFO"].index)
            )
            for name, value in savetable[op, activity].items():
                savedf[name] = int(value)
            if savedf.empty:
                continue
            nparameters[f"/{op}/{activity}/SAVE"] = savedf

    return nparameters


def is_valid_url(
    url: bytes | str | os.PathLike, qualifying: Any | None = ("scheme", "netloc")
) -> bool:
    """
    Return whether "url" is valid.

    Parameters
    ----------
    url : str or bytes
        The URL to check.
    qualifying : iterable of str, optional
        The attributes of the parsed URL that must be non-empty for the URL to
        be considered valid.  The default is ("scheme", "netloc").

    Returns
    -------
    bool
        True if the URL is valid, False otherwise.
    """
    if isinstance(url, bytes):
        token = urlparse(url.decode("utf-8"))
    else:
        token = urlparse(str(url))

    return all(getattr(token, qualifying_attr) for qualifying_attr in qualifying)
