"""
Copyright 2020 by RESPEC, INC. - see License.txt with this HSP2 distribution
Author: Robert Heaphy, Ph.D.
"""

# Future imports
from __future__ import annotations

# Standard library imports
import contextlib
import os
import re
import textwrap
import urllib.request
from io import StringIO
from pathlib import Path
from typing import TypedDict

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.common import ops
from hsp2.hsp2io.utilities import (
    _continuation_keys,
    _multiple_table_names,
    _number_prefix_columns,
    _operation_prefix_columns,
    _operation_prefix_map,
    _parameters,
    _tables,
    default_and_validation,
    hspf_tables_to_hdf_tables,
    is_valid_url,
)

with contextlib.suppress(Exception):
    # To use "str" type instead of "object" type in pandas 2.3+
    pd.options.future.infer_string = True

wrapper = textwrap.TextWrapper()


def _reader(filename: str | os.PathLike):
    """
    Simple generator that reads a Users Control Input (UCI) file.

    Yields each line that is not a comment or blank.  Also checks for lines
    longer than 80 characters and issues a warning if any are found, since HSPF
    ignores everything after 80 characters on a line.

    Parameters
    ----------
    filename : str or os.PathLike
        The path to the UCI file to read.

    Yields
    ------
    str
        Each line from the UCI file that is not a comment or blank.
    """
    lopen = urllib.request.urlopen if is_valid_url(filename) else open
    with lopen(filename) as uci_file:
        for line in uci_file:
            nline = line.decode("ascii") if isinstance(line, bytes) else line

            # UCI max line length is 80
            nline = nline.rstrip()
            nline = nline[:80].rstrip()

            # If line is a comment or blank, skip it.
            if "***" not in nline and nline:
                yield nline


def read_uci_parameters(uciname: str | os.PathLike) -> dict:
    """
    Read and parse a Users Control Input (UCI) file.

    Parameters
    ----------
    uciname : str or os.PathLike
        The path to the local UCI file or http:// or https:// URI.

    Returns
    -------
    dict
        Dictionary of pandas DataFrames keyed to the path that HSP2 expects
        within the HDF5 file.  The DataFrames are merged from multiple HSPF
        tables present in the UCI file.
    """
    block_names = _tables["HSPF_BLOCK"].unique()

    # Parse UCI File
    #
    # `collected` is a dictionary that will store the data from the UCI file,
    # key = (HSPF_BLOCK, HSPF_TABLE, table_num)
    # value = list of lines from the UCI file that belong to that table
    #
    # 'table_num' is only used for FTABLE, MONTH-DATA, and MASS-LINK tables.
    block_name = ""
    hspf_table = ""
    table_num = None
    collected: dict[tuple[str, str | None, int | None], list] = {}

    _table_counter: dict[tuple, int] = {}

    for line in _reader(str(uciname)):
        # Identify the start of a block.
        if line in block_names and not block_name:
            block_name = line
            hspf_table = ""
            table_num = None

            # Find all the table names for this block from _tables
            table_names = _tables[(_tables["HSPF_BLOCK"] == block_name)][
                "HSPF_TABLE_EXT"
            ].unique()
            # SPEC-ACTIONS has no explicit tables; lines are classified later.
            if block_name == "SPEC-ACTIONS":
                table_names = []

            continue

        if block_name:
            words = line.split()

            # Identify the end of a block.
            if line == f"END {block_name}":
                block_name = ""
                continue

            # Identify the end of a table.
            if hspf_table and words[0].strip() == "END":
                hspf_table = ""
                continue

            # Rename tables that are dependent on order in the UCI file.  The
            # new table name is in the _tables DataFrame and therefore in the
            # table_names variable.
            baseword = (block_name, words[0])
            if baseword in _multiple_table_names:
                cnt = _table_counter.get(baseword, 0)
                words[0] = f"{words[0]}{_multiple_table_names[baseword][cnt]}"
                _table_counter[baseword] = cnt + 1

            # We are inside a block, now identify the start of a table.
            if words[0] in table_names and not hspf_table:
                hspf_table = words[0].strip()
                table_num = None
                # For FTABLE, MONTH-DATA, and MASS-LINK tables, get the table
                # number.
                if hspf_table in ["FTABLE", "MONTH-DATA", "MASS-LINK"]:
                    table_num = int(words[1])
                continue

            # Keep the key as (HSPF_BLOCK, HSPF_TABLE, table_num) until
            # finished processing the individual tables.
            key = (block_name, hspf_table, table_num)
            collected.setdefault(key, []).append(line)

    # At this point the entire UCI file has been read into the `collected`
    # dictionary, and now the `collected` dictionary will be processed.  The
    # values are just lists of lines from the UCI file keyed to (HSPF_BLOCK,
    # HSPF_TABLE, table_num).  The "table_num" is only used for FTABLE,
    # MONTH-DATE, and MASS-LINK tables.

    # Handle PEST Supplemental File
    #
    # See if there is a Parameter ESTimation (PEST) supplemental file.  The
    # PESTSU line in the FILES block specifies the name of the supplemental
    # file.
    #
    # Creates the `pest_sup` dictionary with the record id as the key and list
    # of floats as the value.
    pestsu = ""
    for line in collected[("FILES", "", None)]:
        words = line.split()
        if words[0] == "PESTSU":
            pestsu = words[2]
    pest_sup = {}
    if pestsu:
        # Read entire PEST supplemental file into 'sfplines' which becomes the
        # 'pest_sup' dictionary with the record id as the key and list of
        # floats as the value.
        pestsu = str(Path(uciname).parent / pestsu)
        with open(pestsu, encoding="ascii") as sfp:
            sfplines = sfp.readlines()

        sfplines = [i.strip() for i in sfplines if "***" not in i]
        sfplines = [i.strip() for i in sfplines if i]
        pest_sup = {
            key.split()[0]: [str(float(i)) for i in value.split()]
            for key, value in zip(sfplines[:-1:2], sfplines[1::2])
        }

    # Handle Tables With Continuation Lines
    #
    # GLOBAL has parameters spread over four continuation lines
    # GQ-PHOTPM has parameters spread over three continuation lines
    # GQ-ALPHA has parameters spread over three continuation lines
    # GQ-GAMMA has parameters spread over three continuation lines
    # GQ-DAUGHTER_* has parameters spread over six continuation lines
    # GQ-DELTA has parameters spread over three continuation lines
    # GQ-CLDFACT has parameters spread over three continuation lines
    # LCONC has parameters spread over two continuation lines
    #
    # I handle these by combining the continuation lines into a single line.
    #
    # For example, the GLOBAL table has 4 lines.  I take collected[("GLOBAL",
    # "", None)] which is a list of those four lines and I combine them into
    # a single line of 320 characters (4 * 80).
    #
    # Later when converting to a DataFrame, the START and STOP columns in the
    # Parameters.csv table reflect the new column positions.
    #
    # NOTE: At the present time, you cannot use a pest supplemental file to
    # modify parameters in tables that use continuation lines.
    for key, n_cont_lines in _continuation_keys:
        if key in collected:
            extended = collected[key]
            newlines = []
            for first_operation in range(0, len(extended), n_cont_lines):
                newstr = "".join(
                    f"{extended[first_operation + i]:<80}" for i in range(n_cont_lines)
                )
                newlines.append(newstr)
            collected[key] = newlines

    # FTABLE has a rows/columns row as the first line that we don't want as the
    # first line of the dataframe.  Append instead to each line in the table
    # and define the new 'rows' and 'columns' parameters in Parameters.csv.
    for ftable_key in [k for k in collected if k[0] == "FTABLES"]:
        rows, cols = collected[ftable_key][0].split()
        nlines = [
            f"{line:<80}{int(rows):<10}{int(cols):<10}"
            for line in collected[ftable_key][1:]
        ]
        collected[ftable_key] = nlines

    # Need to add a MLNO column to the MASS-LINK that is the table_num.
    for mass_link_key in [k for k in collected if k[0] == "MASS-LINK"]:
        nlines = [
            f"{line:<80}{int(mass_link_key[2]):<10}"
            for line in collected[mass_link_key]
        ]
        collected[mass_link_key] = nlines

    # Set the INDELT explicitly for each entry in the OPN SEQUENCE table based
    # on the last INDELT entry read in the OPN SEQUENCE block.  This allows us
    # to ignore the INGRP which just sets INDELT for the group and not have to
    # worry about the INGRP groupings when processing the OPN SEQUENCE table.
    if ("OPN SEQUENCE", "", None) in collected:
        new_opn_sequence_lines = []
        for opn_sequence_line in collected[("OPN SEQUENCE", "", None)]:
            words = opn_sequence_line.split()
            if "INDELT" in words:
                # INDELT can be in minutes or hours:minutes and I convert
                # here to total minutes.
                s = words[-1].split(":")
                indelt = int(s[0]) if len(s) == 1 else 60 * int(s[0]) + int(s[1])
            if "INGRP" in words:
                continue
            # Add INDELT to the end of the line, until changed by another
            # INDELT entry.  The "OPN SEQUENCE" block implicitly does this, but
            # we need to make it explicit here.
            new_opn_sequence_lines.append(f"{opn_sequence_line:<80}{indelt:10d}")
        collected[("OPN SEQUENCE", "", None)] = new_opn_sequence_lines

    # Create a dictionary of operations in the OPN SEQUENCE block to be used
    # later to fill out entries in other tables that are keyed to operations.
    # The key is the operation type (e.g. PERLND, IMPLND, RCHRES, etc.) and the
    # value is a list of the operation numbers for that operation type.
    collect_operations: dict[str, list] = {}
    for opsline in collected[("OPN SEQUENCE", "", None)]:
        words = opsline.split()
        if words[0] in ops:
            collect_operations.setdefault(words[0], []).append(int(words[1]))

    # The SPEC-ACTIONS block doesn't have explicit tables identified, but we
    # can split into tables based on the contents of each line.  These implicit
    # tables are: ACTION, DISTRB, UVNAME, and UVQUAN.
    #
    # Each ACTION has a logical condition that is built up as the lines
    # are read to establish whether an action is to be taken or not.  The
    # default logical condition is True for ACTION lines outside of a "IF/END
    # IF" block.  A "conditions" table, linked by "cond_id", is created to store
    # the logical conditions for each ACTION line.
    default_d_info = TypedDict(
        "default_d_info",
        {"cond_id": int, "parent_id": int, "sibling_id": int, "condition": str},
    )

    sa_actions = []
    sa_uvquan = []
    open_conditions: list[int] = []
    sa_distrb = []
    sa_uvname = []
    sa_conditions: list[default_d_info] = []

    default_d: default_d_info = {
        "cond_id": -1,
        "parent_id": -1,
        "sibling_id": -1,
        "condition": "",
    }
    sa_iter = iter(collected.get(("SPEC-ACTIONS", "", None), []))
    for sa_line in sa_iter:
        words = sa_line.split()
        # Notes:
        # - Only "classic" special actions are currently active.
        # - Other types of special actions are recognized by the parser, but
        #   not stored in hdf5
        # - The condition shows parent IF-THEN-ELSE entries if applicable
        #   - Each action "head_[action type]" should include a "condition"
        #     column to match with condition expression if applicable
        #   - The condition matches an index in /SPEC_ACTIONS/conditions table
        if words[0] == "UVQUAN":
            sa_uvquan.append(sa_line)
        elif words[0] == "DISTRB":
            sa_distrb.append(sa_line)
        elif words[0] == "UVNAME":
            sa_uvname.append(sa_line)
        elif words[0] == "IF" or " ".join(words[:2]) == "ELSE IF":
            # The condition for an IF or ELSE IF can be spread over multiple
            # lines until a THEN is encountered at the end of a line.  So we
            # need to loop through the lines until we find the THEN and combine
            # those lines into a single line to get the full condition.
            continuation_lines = [" ".join(sa_line.strip().split())]
            while continuation_lines[-1].split()[-1] != "THEN":
                continuation_lines.append(" ".join(next(sa_iter).strip().split()))
            continuation_line = " ".join(continuation_lines)

            d = default_d.copy()
            d["cond_id"] = len(sa_conditions)
            d["condition"] = (
                continuation_line.removeprefix("IF ")
                .removeprefix("ELSE IF ")
                .removesuffix(" THEN")
                .strip()
            )
            for i, j in [("[", "("), ("{", "("), ("]", ")"), ("}", ")")]:
                d["condition"] = d["condition"].replace(i, j)

            if words[0] == "IF":
                d["parent_id"] = open_conditions[-1] if open_conditions else -1
            else:
                # ELSE IF replaces the branch that is currently open.
                d["sibling_id"] = open_conditions.pop()
                d["parent_id"] = sa_conditions[d["sibling_id"]]["parent_id"]

            open_conditions.append(d["cond_id"])
            sa_conditions.append(d)
        elif words[0] == "ELSE":
            d = default_d.copy()
            d["cond_id"] = len(sa_conditions)
            d["sibling_id"] = open_conditions.pop()
            d["parent_id"] = sa_conditions[d["sibling_id"]]["parent_id"]
            open_conditions.append(d["cond_id"])
            sa_conditions.append(d)
        elif " ".join(words[:2]) == "END IF":
            open_conditions.pop()
        else:
            cond_id = open_conditions[-1] if open_conditions else -1
            sa_actions.append(f"{sa_line:<80}{cond_id:<10}")

    collected.pop(("SPEC-ACTIONS", "", None), None)
    for list_lines, table_name in [
        (sa_uvquan, "UVQUAN"),
        (sa_distrb, "DISTRB"),
        (sa_uvname, "UVNAME"),
        (sa_actions, "ACTION"),
    ]:
        if list_lines:
            collected[("SPEC-ACTIONS", table_name, None)] = list_lines
    if sa_conditions:
        collected[("SPEC-ACTIONS", "conditions", None)] = pd.DataFrame(sa_conditions)

    # Convert 'collected' dictionary of list of strings to 'ncollected'
    # dictionary of DataFrames
    #
    # `ncollected` is a dictionary that will store the data from the UCI file,
    # ncol_key = (HSPF_BLOCK, HSPF_TABLE, table_number)
    #       where HSPF_BLOCK and HSPF_TABLE are in the UCI file and additional
    #       metadata in the Parameters.csv file, and table_number is only used
    #       to identify FTABLE, MONTH-DATA, and MASS-LINK.
    # value = pandas DataFrame with the data for that table from the UCI file
    #         with the columns named according to the Parameters.csv file and
    #         the missing values filled in with the default values from the
    #         Parameters.csv file.  Uses "explode" to expand the (OPNID,
    #         OPNIDLAST) into unique rows for each range of ids in (OPNID,
    #         OPNIDLAST).  The last row for duplicate OPNIDs is kept.
    ncollected: dict[tuple[str, str | None, str | int | None], pd.DataFrame] = {}
    for (hspf_block, hspf_table, table_num), lines in collected.items():
        if isinstance(lines, pd.DataFrame):
            ncollected[(hspf_block, hspf_table, table_num)] = lines
            continue
        table_metadata = _parameters.loc[
            (_parameters["HSPF_BLOCK"] == hspf_block)
            & (
                (_parameters["HSPF_TABLE_EXT"] == hspf_table)
                | (_parameters["HSPF_TABLE_EXT"].isnull())
            )
        ]

        names = table_metadata["PARAMETER_NAME"].to_list()

        if pest_sup and ("OPNID" in names):
            # Have to read separately for PEST supplemental file and create two
            # DataFrames of the same shape as the table.
            #     * The PEST supplemental DataFrame will have parameter values
            #       filled in from the PEST supplemental file for each row that
            #       has a ~XXX~ in the UCI file (where XXX is the record number
            #       in the PEST supplemental file) and the rest of the values
            #       will be filled in with Nones.
            #     * The standard DataFrame will have all values filled in with
            #       the values read from the UCI file and blank values for the
            #       ~XXX~ rows.  When read with pd.read_fwf, the blanks in the
            #       ~XXX~ rows will be filled with Nones.
            #     * The two DataFrames will be combined with the ~XXX~ rows
            #       filled in with the values from the PEST supplemental file
            #       and the rest of the values filled in with the values from
            #       the UCI file.
            #     * Only HSPF_BLOCK/HSPF_TABLE that have an OPNID parameter can
            #       use entries in the PEST supplemental file.
            nlines = []
            pass_through_lines = []
            for line in lines:
                if tilde := re.match("~([0-9][0-9]*)~", line[10:]):
                    tilde = tilde[0][1:-1]
                    # Have to use read_csv instead of read_fwf since the
                    # supplemental file is space delimited instead of fixed
                    # width format.
                    try:
                        nlines.append(
                            ",".join([line[:5], line[5:10]] + pest_sup[tilde])
                        )
                    except KeyError as exc:
                        raise ValueError(
                            wrapper.wrap(
                                textwrap.dedent(f"""\
                                    The record id ~{tilde}~ in the UCI file
                                    "{uciname}" is not in the Parameter ESTimation
                                    (PEST) supplemental file "{pestsu}".
                                    """)
                            )
                        ) from exc
                    pass_through_lines.append(line[:10])
                else:
                    nlines.append(
                        ",".join([line[:5], line[5:10]] + [""] * (len(names) - 2))
                    )
                    pass_through_lines.append(line)
            lines = pass_through_lines
            pest_sup_df = pd.read_csv(
                StringIO("\n".join(nlines)),
                names=names,
                index_col=False,
                skipinitialspace=True,
            )

        # Read the lines into a DataFrame using the START and STOP columns.
        starts = table_metadata["START"].astype(int).values
        stops = table_metadata["STOP"].astype(int).values
        ndf = pd.read_fwf(
            StringIO("\n".join(lines)), colspecs=list(zip(starts, stops)), names=names
        )

        # Merge in any values from the PEST supplemental file by using
        # combine_first to replace missing values in the ndf DataFrame with
        # values from the pest_sup_df DataFrame.
        if pest_sup and "OPNID" in names:
            ndf = ndf.combine_first(pest_sup_df)

        # Process DataFrames that have an OPNID/OPNIDLAST or TVOLNO/TOPLST
        # columns to use ranges of operations.  If there are ranges specified,
        # explode those ranges into individual rows and then drop duplicates
        # keeping the last row for each operation or target volume number.
        for exploder_col, first_operation_col, last_operation_col in [
            ("OPNID", "OPNID", "OPNIDLAST"),
            ("TVOLNO", "TOPFST", "TOPLST"),
        ]:
            if first_operation_col in ndf.columns and last_operation_col in ndf.columns:
                exploder = pd.DataFrame()
                exploder["FIRSTID"] = pd.to_numeric(ndf[first_operation_col]).astype(
                    int
                )
                exploder["LASTID"] = pd.to_numeric(
                    ndf[last_operation_col].fillna(ndf[first_operation_col])
                ).astype(int)

                exploder[exploder_col] = [
                    list(range(i, j + 1)) for i, j in exploder.values
                ]
                exploder = exploder.drop(columns=["FIRSTID", "LASTID"])

                ndf = ndf.drop(columns=[first_operation_col, last_operation_col])
                ndf = pd.concat([exploder, ndf], axis="columns")
                ndf = ndf.explode(exploder_col)
                if exploder_col == "OPNID":
                    ndf = ndf.drop_duplicates(subset=[exploder_col], keep="last")
                ndf[exploder_col] = ndf[exploder_col].astype(int)

        # Right here need to make sure have a row for each operation in the OPN
        # SEQUENCE table.  If don't have a row for an operation, need to add
        # one so `default_and_validation` function will fill in any missing
        # values with the defaults.
        if "OPNID" in ndf.columns:
            ndf = ndf.set_index("OPNID", drop=True)
            ndf = ndf.reindex(index=sorted(set(collect_operations[hspf_block])))
            ndf = ndf.loc[sorted(set(collect_operations[hspf_block])), :]
            ndf.index = hspf_block[0] + ndf.index.astype(str).str.zfill(3)
            ndf.index.name = "OPERATION"

        ncollected[(hspf_block, hspf_table, table_num)] = ndf

    nhdf_collected = hspf_tables_to_hdf_tables(ncollected)

    # There are tables where parameters have to be copied from one table to
    # another. For example, if the parameters LEN, DELTH, and DB50 exist in
    # /RCHRES/HYDR/PARAMETERS, but not in /RCHRES/SEDTRN/PARAMETERS then they
    # have to be copied.
    for source_hdf_table, pars, target_hdf_table in [
        (
            "/RCHRES/HYDR/PARAMETERS",
            ["LEN", "DELTH", "DB50"],
            "/RCHRES/SEDTRN/PARAMETERS",
        ),
    ]:
        if (source_hdf_table not in nhdf_collected) or (
            target_hdf_table not in nhdf_collected
        ):
            continue
        for par in pars:
            if par not in nhdf_collected[target_hdf_table].columns:
                nhdf_collected[target_hdf_table][par] = nhdf_collected[
                    source_hdf_table
                ][par]

    # Special processing of /CONTROL/GLOBAL tables.
    cgl = nhdf_collected["/CONTROL/GLOBAL"]
    cgl["Comment"] = cgl["COMMENT"]

    # Special processing of /CONTROL/GLOBAL tables: start and end times
    cgl = cgl.fillna(
        {
            "SYR": 1900,
            "SMO": 1,
            "SDA": 1,
            "SHR": 0,
            "SMI": 0,
            "EYR": 1900,
            "EMO": 12,
            "EDA": 31,
            "EHR": 24,
            "EMI": 0,
        }
    )
    cgl["Start"] = str(
        pd.Timestamp(
            int(cgl.loc[0, "SYR"]), int(cgl.loc[0, "SMO"]), int(cgl.loc[0, "SDA"])
        )
        + pd.Timedelta(hours=int(cgl.loc[0, "SHR"]))
        + pd.Timedelta(minutes=int(cgl.loc[0, "SMI"]))
    )[:16]
    cgl["Stop"] = str(
        pd.Timestamp(
            int(cgl.loc[0, "EYR"]), int(cgl.loc[0, "EMO"]), int(cgl.loc[0, "EDA"])
        )
        + pd.Timedelta(hours=int(cgl.loc[0, "EHR"]))
        + pd.Timedelta(minutes=int(cgl.loc[0, "EMI"]))
    )[:16]

    # Special processing of /CONTROL/GLOBAL tables: units, transpose, and
    # replace.
    cgl["Units"] = cgl.loc[0, "UFG"] or 1
    cgl = cgl[
        ["Comment", "Start", "Stop", "Units", "LEV", "SPA", "RES", "RUN"]
    ].transpose()
    cgl.columns = ["Info"]
    cgl["Info"] = cgl["Info"].astype(str)
    nhdf_collected["/CONTROL/GLOBAL"] = cgl

    # Special processing of /CONTROL/EXT_SOURCES table to handle multiple WDM
    # files.
    ext_sources = nhdf_collected["/CONTROL/EXT_SOURCES"]
    for svol, baseno in [
        ("WDM1", 100000),
        ("WDM2", 200000),
        ("WDM3", 300000),
        ("WDM4", 400000),
    ]:
        if svol == "WDM1":
            # WDM and WDM1 are synonymous
            mask = ext_sources["SVOL"].isin(("WDM1", "WDM"))
        else:
            mask = ext_sources["SVOL"] == svol
        if not mask.any():
            continue
        ext_sources["SVOLNO"] = ext_sources["SVOLNO"].astype(str)
        tmpcol = baseno + ext_sources.loc[mask, "SVOLNO"].astype(int)
        tmpcol = tmpcol.apply(lambda x: f"TS{x:03}")
        ext_sources.loc[mask, "SVOLNO"] = tmpcol
    nhdf_collected["/CONTROL/EXT_SOURCES"] = ext_sources

    # HSPsquared in several tables require that operation and id be combined.
    for table, tlink, tlabel in _operation_prefix_columns:
        if table not in nhdf_collected:
            continue
        ndf = nhdf_collected[table]
        ndf[tlink] = ndf[tlabel].replace(_operation_prefix_map) + ndf[tlink].apply(
            lambda x: f"{int(x):03d}"
        )
        nhdf_collected[table] = ndf

    # All MASS-LINK need the "ML" prefix added and all FTBUCI need the "FT"
    # prefix added.
    for table, tlink, iprefix in _number_prefix_columns:
        if table not in nhdf_collected:
            continue
        if tlink not in nhdf_collected[table].columns:
            continue
        ndf = nhdf_collected[table]
        mask = ~pd.isna(ndf[tlink])
        tmp_col = f"{tlink}_TMP"
        ndf[tmp_col] = ""
        ndf.loc[mask, tmp_col] = ndf.loc[mask, tlink].apply(
            lambda x, iprefix=iprefix: f"{iprefix}{int(x):03d}"
        )
        ndf[tlink] = ndf[tmp_col]
        ndf = ndf.drop(columns=[tmp_col])
        nhdf_collected[table] = ndf

    # HSP2 needs some parameters NEXITS and LKFG moved from
    # /RCHRES/GENERAL/INFO to /RCHRES/HYDR/PARAMETERS and IREXIT, IRMINV set to
    # 0 if they don't exist.  Don't know why.
    gen_key = "/RCHRES/GENERAL/INFO"
    if gen_key in nhdf_collected:
        hydr_key = "/RCHRES/HYDR/PARAMETERS"
        if hydr_key in nhdf_collected:
            nhdf_collected[hydr_key] = nhdf_collected[hydr_key].drop(
                columns=["NEXITS", "LKFG"]
            )
            nhdf_collected[hydr_key] = pd.merge(
                nhdf_collected[hydr_key],
                nhdf_collected[gen_key][["NEXITS", "LKFG"]],
                left_index=True,
                right_index=True,
                how="left",
            )
            if (
                "IREXIT" not in nhdf_collected[hydr_key].columns
            ):  # didn't read HYDR-IRRIG table
                nhdf_collected[hydr_key]["IREXIT"] = 0
                nhdf_collected[hydr_key]["IRMINV"] = 0.0

    return default_and_validation(nhdf_collected)
