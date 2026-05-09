import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

wrapper = textwrap.TextWrapper()

_dirname = Path(__file__).parent.parent.parent
fill_table_name = {"HSPF_TABLE": ""}
_tables = pd.read_csv(_dirname / "data" / "Tables.csv").fillna(fill_table_name)
_parameters = pd.read_csv(_dirname / "data" / "Parameters.csv").fillna(fill_table_name)
_save_tables = pd.read_csv(_dirname / "data" / "SaveTable.csv")
_defaultable_tables = pd.read_csv(_dirname / "data" / "DefaultableTables.csv").fillna(
    fill_table_name
)

# Bring over HDF_TABLE from _tables to _parameters.
_parameters = pd.merge(
    _parameters,
    _tables[
        [
            "HSPF_BLOCK",
            "HSPF_TABLE",
            "HDF_TABLE",
        ]
    ],
    on=["HSPF_BLOCK", "HSPF_TABLE"],
    how="left",
)


def hspf_tables_to_hdf_tables(ncollected):
    # Now need to merge {(HSFP_block, HSPF_table, table_num): dataframe} to
    # be {HDF_TABLE: dataframe} where HDF_TABLE is looked up from
    # _tables.

    # Collect a list of (HSPF_block, HSPF_table) DataFrames for each HDF_TABLE.
    hdf_collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    collect_hspf_tables = defaultdict(list)
    hspf_blocks = set()
    for (hspf_block, hspf_table, table_num), df in ncollected.items():
        hdf_table: str = _tables.loc[
            (_tables["HSPF_BLOCK"] == hspf_block)
            & (
                (_tables["HSPF_TABLE_EXT"] == hspf_table)
                | (_tables["HSPF_TABLE_EXT"].isnull())
            ),
            "HDF_TABLE",
        ].iloc[0]
        # MASS-LINK is different where the table_num is stored in the table
        # as MLNO, so we don't append the table_num to the HDF_TABLE name.
        # So just adjust hspf_table for the FTABLE and MONTH-DATA tables.
        if hspf_table in ["FTABLE", "MONTH-DATA"]:
            hdf_table = f"{hdf_table}{table_num:03d}"
        hdf_collected[hdf_table].append(df)
        collect_hspf_tables[hdf_table].append((hspf_block, hspf_table))
        hspf_blocks.add(hspf_block)

    # Now add any required tables that are missing but can be defaulted.
    for _, row in _defaultable_tables.iterrows():
        hdf_table = row["HDF_TABLE"]
        hspf_block = row["REQ_HSPF_BLOCK"]
        if hspf_block not in hspf_blocks:
            continue
        hspf_table = row["REQ_HSPF_TABLE"]
        key = (hspf_block, hspf_table)
        used_keys = collect_hspf_tables.get(hdf_table, [])
        if key not in used_keys:
            # Get parameter names for this block/table
            param_names = _parameters.loc[
                (_parameters["HSPF_BLOCK"] == hspf_block)
                & (_parameters["HSPF_TABLE"] == hspf_table),
                "PARAMETER_NAME",
            ].tolist()
            param_names = [p for p in param_names if p not in ("OPNID", "OPNIDLAST")]
            index = []
            if len(used_keys) == 0:
                index = ncollected[(hspf_block, "ACTIVITY", None)].index
            # Create empty DataFrame with those columns to be filled with
            # defaults later.
            hdf_collected[hdf_table].append(
                pd.DataFrame(index=index, columns=param_names)
            )

    # Merge DataFrames for (HSPF_block, HSPF_table) into single DataFrame for
    # each HDF_TABLE.
    nhdf_collected: dict[str, pd.DataFrame | pd.Series] = {}
    for hdf_table, list_dfs in hdf_collected.items():
        dfs = pd.DataFrame()
        for df in list_dfs:
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
    units = parameters["/CONTROL/GLOBAL"].loc["Units", "Info"]
    if not units or int(units) == 1:
        units = "ENGL"
    elif int(units) == 2:
        units = "METR"

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

        # The parameters for all the /FTABLES/FTXXX tables are stored in the
        # table_name /FTABLES/FT in _parameters.
        table_name = "/FTABLES/FT" if hdf_table.startswith("/FTABLES/FT") else hdf_table

        parameter_data = _parameters[_parameters["HDF_TABLE"] == table_name]
        parameter_data = parameter_data.set_index("PARAMETER_NAME")

        cols = parms.columns.drop(
            ["OPNIDLAST", "TVOLNO", "TOPFST", "TOPLST"], errors="ignore"
        )
        parameter_data = parameter_data.loc[cols]

        # Set the numerical defaults from DEFAULT_ENGL or DEFAULT_METR from the
        # Parameters.csv file and then use fillna.
        defaults = parameter_data[
            [f"DEFAULT_{units}", f"MIN_{units}", f"MAX_{units}"]
        ].astype(float)
        defaults.columns = ["DEFAULT", "MIN", "MAX"]

        # Set the types.
        nparms = parms.astype(
            dict(zip(parameter_data.index, parameter_data["TYPE"].values)),
            errors="ignore",
        )

        # Fill in the numerical defaults.
        nparms = nparms.infer_objects().fillna(
            dict(zip(defaults.index, defaults["DEFAULT"].values))
        )

        # Fill in the text defaults.
        defaults_text = parameter_data[["DEFAULT_TEXT"]].dropna().astype(str)
        nparms = nparms.fillna(
            dict(zip(defaults_text.index, defaults_text["DEFAULT_TEXT"].values))
        )

        # Only fill string-like columns with empty string.  Avoid filling
        # numeric or other dtypes which could coerce types or mask missing
        # numeric values. Use both 'object' and pandas 'string' dtypes so
        # this works whether convert_dtypes produced Python object columns
        # or pandas StringDtype columns.
        str_cols = nparms.select_dtypes(include=["object", "string"]).columns
        if len(str_cols):
            nparms[str_cols] = nparms[str_cols].fillna("")
            nparms[str_cols] = nparms[str_cols].replace("nan", "")
            nparms[str_cols] = nparms[str_cols].astype(object)

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
                            Values for hdf_table={hdf_table}, and
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
            try:
                lseries = series.str.replace("nan", "").str.strip()
            except AttributeError:
                lseries = series
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

        if "Unnamed: 0" in nparms.columns:
            nparms = nparms.drop(columns=["Unnamed: 0"])

        if "OPNID" in nparms.columns:
            nparms["OPNID"] = nparms["OPNID"].astype(str)

        if not nparms.empty:
            nparameters[hdf_table] = pd.DataFrame(nparms)

    # Create the SAVE tables.  These identify the time-series that are saved.
    savetable = defaultdict(dict)
    for row in _save_tables.itertuples():
        savetable[row.OPERATION, row.ACTIVITY][row.NAME] = row.VALUE
    operations_activities = set()
    for key in nparameters.keys():
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
