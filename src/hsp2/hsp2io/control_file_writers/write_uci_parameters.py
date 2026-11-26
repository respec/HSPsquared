"""
Write a Users Control Input (UCI) file from a dictionary of parameter DataFrames.

This is the inverse of ``read_uci_parameters``.  The dictionary is keyed by
HDF_TABLE name (for example "/PERLND/PWATER/PARAMETERS") and the layout of the
UCI file (which HSPF_BLOCK and HSPF_TABLE a column belongs to, and at which
character columns) is taken from "Tables.csv" and "Parameters.csv".

Notes on what is, and is not, recreated:

* Tables that contain only default values (and that HSPF does not require) are
  not written, so they are absent when the new UCI file is read.
* Several UCI tables can be spread over continuation lines, and ``read_uci``
  concatenates them into one long line.  The START/STOP columns in
  "Parameters.csv" beyond 80 characters are split back into lines of 80
  characters here.
* Tables that appear more than once in a block (for example "MON-ACCUM" or
  "GQ-DAUGHTER") get their meaning from their order in the block.  The order
  is the order of ``_multiple_table_names`` in ``utilities``.
* "/CONTROL/GLOBAL" only retains the comment, start, stop, and units.
* "PATHNAMES" and "MONTH-DATA" are not written.
* The SPEC-ACTIONS block is written in the order DISTRB, UVQUAN, UVNAME, and
  ACTIONS without table names.  Each ACTIONS record is written inside the
  IF / ELSE IF / ELSE / END IF structure described by the "conditions" table.
  Within one branch the actions are written before any nested IF, so the
  relative position of an action and a nested IF is not preserved.
"""

# Future imports
from __future__ import annotations

# Standard library imports
import contextlib
import re
import warnings
from pathlib import Path

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.utilities import (
    _continuation_lines,
    _multiple_table_names,
    _number_prefix_columns,
    _operation_prefix_columns,
    _operation_prefix_map,
    _parameters,
    _tables,
)

_LINE = 80

# Columns that hold "<operation prefix><number>" or "<prefix><number>" in HDF,
# but a plain number in the UCI file.
_DIGIT_COLUMNS = {
    column for _, column, *_ in [*_operation_prefix_columns, *_number_prefix_columns]
}

# Link-table columns that are left blank if they equal their default.
_BLANK_IF_DEFAULT = {"MFACTOR", "AFACTR", "QLFG"}

# Duplicates of other columns, not written.
_SKIP_COLUMNS = {"TMEMSB"}

# Order of the operation blocks in a UCI file.
_OPERATION_BLOCKS = [
    "PERLND",
    "IMPLND",
    "RCHRES",
    "COPY",
    "PLTGEN",
    "DISPLY",
    "DURANL",
    "GENER",
    "MUTSIN",
    "BMPRAC",
    "REPORT",
]
assert set(_OPERATION_BLOCKS) == set(_operation_prefix_map)

_UNSUPPORTED_PREFIXES = ("/MONTH-DATA", "/CONTROL/PATHNAMES")


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------
def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", "nan", "<NA>", "None")
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _digits(value) -> int:
    """Number from a number, or from text like "P001", "FT001", "TS100039"."""
    if isinstance(value, str):
        return int(re.search(r"\d+", value).group())
    return round(float(value))


def _fmt_float(value, width: int) -> str:
    value = float(value)
    text = repr(value)
    if len(text) <= width:
        return text
    short = re.sub(r"^(-?)0\.", r"\1.", text)
    if len(short) <= width:
        return short
    for precision in range(16, 0, -1):
        text = f"{value:.{precision}g}"
        if len(text) <= width:
            return text
        # "0.0012" -> ".0012" and "-0.5" -> "-.5"
        short = re.sub(r"^(-?)0\.", r"\1.", text)
        if len(short) <= width:
            return short
    return text[:width]


def _field(value, name: str, ptype: str, width: int) -> str:
    """Text of exactly ``width`` characters for one value."""
    if _is_blank(value):
        return " " * width

    numeric = name in _DIGIT_COLUMNS or ptype == "int"
    if numeric and isinstance(value, str) and not re.search(r"\d", value):
        # Misaligned UCI input can leave text in a numeric column; keep it.
        numeric = False
    if numeric:
        text, right = str(_digits(value)), True
    elif ptype == "float":
        text, right = _fmt_float(value, width), True
    else:
        text = str(value).strip()
        # Subscripts and similar can arrive as "1.0".
        if width <= 3 and re.fullmatch(r"-?\d+\.0+", text):
            text = text.split(".", maxsplit=1)[0]
        right = text.lstrip("-").isdigit()

    text = text[:width]
    return f"{text:>{width}}" if right else f"{text:<{width}}"


def _put(buf: list[str], start: int, text: str) -> None:
    buf[start : start + len(text)] = list(text)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def _is_default(values: pd.Series, par: pd.Series, units: str) -> bool:
    """True if every value is blank or equal to the parameter default."""
    number = par[f"DEFAULT_{units}"]
    text = par["DEFAULT_TEXT"]
    for value in values:
        if _is_blank(value):
            continue
        if not _is_blank(text) and str(value).strip() == str(text).strip():
            continue
        if not _is_blank(number):
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                if float(_digits(value) if isinstance(value, str) else value) == float(
                    number
                ):
                    continue
        return False
    return True


def _units(parameter_dict) -> str:
    try:
        units = parameter_dict["/CONTROL/GLOBAL"].loc["Units", "Info"]
        return "METR" if int(float(units)) == 2 else "ENGL"
    except (KeyError, ValueError, TypeError):
        return "ENGL"


# ---------------------------------------------------------------------------
# Rows to lines
# ---------------------------------------------------------------------------
def _comment_line(text) -> str:
    """Make sure a header line is a comment that survives the 80 char limit."""
    text = str(text).rstrip()
    return text[:_LINE] if "***" in text[:_LINE] else f"***{text[3:]}"[:_LINE]


def _header_lines(block: str, ext: str | None) -> list[str]:
    rows = _tables.loc[_tables["HSPF_BLOCK"] == block]
    if ext is not None:
        rows = rows.loc[rows["HSPF_TABLE_EXT"] == ext]
    if rows.empty:
        return []
    header = rows.iloc[0]["TABLE_HEADER"]
    if _is_blank(header):
        return []
    return [_comment_line(line) for line in str(header).split("\n")]


def _params(block: str, ext: str | None) -> pd.DataFrame:
    pars = _parameters.loc[_parameters["HSPF_BLOCK"] == block]
    if ext is not None:
        pars = pars.loc[pars["HSPF_TABLE_EXT"] == ext]
    pars = pars.drop_duplicates(subset="PARAMETER_NAME")
    return pars.loc[pars["PARAMETER_NAME"].notna()]


def _render(
    df: pd.DataFrame,
    pars: pd.DataFrame,
    n_lines: int = 1,
    blank_default: set[str] | None = None,
    units: str = "ENGL",
) -> list[str]:
    """Convert each row of ``df`` to fixed width lines using START/STOP."""
    blank_default = blank_default or set()
    fields = []
    for par in pars.itertuples():
        name = par.PARAMETER_NAME
        if name in _SKIP_COLUMNS or name not in df.columns:
            continue
        start, stop = int(par.START), int(par.STOP)
        if stop > n_lines * _LINE:
            continue
        default = getattr(par, f"DEFAULT_{units}")
        fields.append((name, par.TYPE, start, stop, default))

    lines = []
    for _, row in df.iterrows():
        buf = [" "] * (_LINE * n_lines)
        for name, ptype, start, stop, default in fields:
            value = row[name]
            if (
                name in blank_default
                and not _is_blank(value)
                and not _is_blank(default)
                and ptype in ("int", "float")
                and float(value) == float(default)
            ):
                continue
            text = _field(value, name, ptype, stop - start)
            _put(buf, start, text)
            if name in ("OPNID", "OPNIDLAST"):
                # Every continuation line needs the operation numbers.
                for i in range(1, n_lines):
                    _put(buf, start + i * _LINE, text)
        for i in range(n_lines):
            chunk = "".join(buf[i * _LINE : (i + 1) * _LINE]).rstrip()
            # A completely blank line would be dropped when read.
            lines.append(chunk or " ")
    return lines


def _operation_ids(index) -> list[int]:
    return [_digits(str(i)) for i in index]


def _group_ranges(
    df: pd.DataFrame,
    ids: list[int],
    first_col: str,
    last_col: str,
    sort: bool,
    allow_gaps: bool = False,
) -> pd.DataFrame:
    """
    Combine consecutive identical rows into (first, last) id ranges.

    ``df`` must not contain ``first_col`` or ``last_col``.  With
    ``allow_gaps`` identical rows are merged even if ids are missing between
    them (the sorted ids must then be increasing).
    """
    data = df.reset_index(drop=True)
    order = sorted(range(len(ids)), key=lambda i: ids[i]) if sort else range(len(ids))

    keep, firsts, lasts = [], [], []
    prev_key = None
    for i in order:
        key = tuple("" if _is_blank(v) else str(v) for v in data.iloc[i])
        if (
            keep
            and key == prev_key
            and (ids[i] > lasts[-1] if allow_gaps else ids[i] == lasts[-1] + 1)
        ):
            lasts[-1] = ids[i]
        else:
            keep.append(i)
            firsts.append(ids[i])
            lasts.append(ids[i])
        prev_key = key

    out = data.iloc[keep].copy().reset_index(drop=True)
    out[first_col] = firsts
    out[last_col] = [
        last if last != first else pd.NA for first, last in zip(firsts, lasts)
    ]
    return out


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _global_lines(df: pd.DataFrame) -> list[str]:
    info = df["Info"]
    comment = str(info.get("Comment", "")).strip() or "HSP2 generated UCI file"
    start = pd.Timestamp(info["Start"])
    stop = pd.Timestamp(info["Stop"])
    # HSPF expresses midnight at the end of a period as 24:00 of the last day.
    if stop.hour == 0 and stop.minute == 0:
        stop -= pd.Timedelta(days=1)
        ehr, emi = 24, 0
    else:
        ehr, emi = stop.hour, stop.minute
    units = 1 if _is_blank(info.get("Units")) else int(float(info.get("Units", 1)))
    lev = 0 if _is_blank(info.get("LEV")) else int(float(info["LEV"]))
    spa = 2 if _is_blank(info.get("SPA")) else int(float(info["SPA"]))
    res = 0 if _is_blank(info.get("RES")) else int(float(info["RES"]))
    run = 0 if _is_blank(info.get("RUN")) else int(float(info["RUN"]))

    buf = [" "] * (4 * _LINE)

    def num(start_col, stop_col, value):
        _put(
            buf,
            start_col,
            f"{value:>{stop_col - start_col}}"[-(stop_col - start_col) :],
        )

    _put(buf, 2, comment[: _LINE - 2])

    line = _LINE
    _put(buf, line + 2, "START")
    num(line + 10, line + 18, start.year)
    _put(buf, line + 18, "/")
    num(line + 19, line + 21, f"{start.month:02d}")
    _put(buf, line + 21, "/")
    num(line + 22, line + 24, f"{start.day:02d}")
    num(line + 25, line + 27, f"{start.hour:02d}")
    _put(buf, line + 27, ":")
    num(line + 28, line + 30, f"{start.minute:02d}")
    _put(buf, line + 32, "END")
    num(line + 35, line + 43, stop.year)
    _put(buf, line + 43, "/")
    num(line + 44, line + 46, f"{stop.month:02d}")
    _put(buf, line + 46, "/")
    num(line + 47, line + 49, f"{stop.day:02d}")
    num(line + 50, line + 52, f"{ehr:02d}")
    _put(buf, line + 52, ":")
    num(line + 53, line + 55, f"{emi:02d}")

    line = 2 * _LINE
    _put(buf, line + 2, "RUN INTERP OUTPUT LEVEL")
    num(line + 26, line + 30, f"{lev:5d}")
    num(line + 31, line + 35, f"{spa:5d}")

    line = 3 * _LINE
    _put(buf, line + 2, "RESUME")
    num(line + 9, line + 14, f"{res:5d}")
    _put(buf, line + 16, "RUN")
    num(line + 19, line + 24, f"{run:5d}")
    _put(buf, line + 42, "UNIT SYSTEM")
    num(line + 55, line + 60, units)

    out = ["GLOBAL"]
    for i in range(4):
        chunk = "".join(buf[i * _LINE : (i + 1) * _LINE]).rstrip()
        out.append(chunk)
    out.append("END GLOBAL")
    return out


def _op_sequence_lines(df: pd.DataFrame) -> list[str]:
    out = ["OPN SEQUENCE"]
    previous = None
    for _, row in df.iterrows():
        indelt = int(row["INDELT_minutes"])
        if indelt != previous:
            if previous is not None:
                out.append("    END INGRP")
            out.append(f"    INGRP{'':14}INDELT {indelt // 60:02d}:{indelt % 60:02d}")
            previous = indelt
        buf = [" "] * _LINE
        _put(buf, 6, f"{row['OPERATION']:<6}"[:6])
        _put(buf, 17, f"{_digits(row['SEGMENT']):>3}")
        out.append("".join(buf).rstrip())
    if previous is not None:
        out.append("    END INGRP")
    out.append("END OPN SEQUENCE")
    return out


def _simple_block(
    block: str,
    df: pd.DataFrame,
    units: str,
    blank_default: set[str] | None = None,
) -> list[str]:
    if lines := _render(df, _params(block, None), 1, blank_default, units):
        return [block, *_header_lines(block, None), *lines, f"END {block}"]
    else:
        return []


def _links(df: pd.DataFrame, units: str) -> tuple[list[str], list[str]]:
    """NETWORK and SCHEMATIC blocks from "/CONTROL/LINKS"."""
    if "MLNO" in df.columns:
        mlno = df["MLNO"]
        is_schematic = ~mlno.map(_is_blank)
    else:
        is_schematic = pd.Series(False, index=df.index)

    out: dict[str, list[str]] = {"NETWORK": [], "SCHEMATIC": []}
    for block, subset in [
        ("SCHEMATIC", df.loc[is_schematic]),
        ("NETWORK", df.loc[~is_schematic]),
    ]:
        if subset.empty:
            continue
        subset = subset.copy()
        if block == "NETWORK":
            ids = _operation_ids(subset["TVOLNO"])
            subset = _group_ranges(
                subset.drop(columns=["TVOLNO"]), ids, "TOPFST", "TOPLST", sort=False
            )
        out[block] = _simple_block(block, subset, units, _BLANK_IF_DEFAULT)
    return out["NETWORK"], out["SCHEMATIC"]


def _ext_sources_lines(df: pd.DataFrame, units: str) -> list[str]:
    df = df.copy()
    wdm = df["SVOL"].astype(str).str.startswith("WDM")
    df["SVOLNO"] = [
        _digits(n) % 100000 if is_wdm else _digits(n)
        for n, is_wdm in zip(df["SVOLNO"], wdm)
    ]
    ids = _operation_ids(df["TVOLNO"])
    df = _group_ranges(df.drop(columns=["TVOLNO"]), ids, "TOPFST", "TOPLST", False)
    return _simple_block("EXT SOURCES", df, units, _BLANK_IF_DEFAULT)


def _mass_link_lines(df: pd.DataFrame, units: str) -> list[str]:
    pars = _params("MASS-LINK", None)
    header = _header_lines("MASS-LINK", "MASS-LINK")
    out = ["MASS-LINK"]
    numbers = df["MLNO"].map(_digits)
    for number in dict.fromkeys(numbers):
        rows = df.loc[numbers == number]
        out.append(f"  MASS-LINK{number:>9}")
        out.extend(header)
        out.extend(_render(rows, pars, 1, _BLANK_IF_DEFAULT, units))
        out.append(f"  END MASS-LINK{number:>5}")
    out.append("END MASS-LINK")
    return out


def _ftables_lines(parameter_dict) -> list[str]:
    keys = sorted(k for k in parameter_dict if k.startswith("/FTABLES/FT"))
    if not keys:
        return []
    pars = _params("FTABLES", None)
    out = ["FTABLES"]
    for key in keys:
        df = parameter_dict[key]
        number = _digits(key.rsplit("/", maxsplit=1)[-1])
        # "rows" and "cols" are what HSPF uses and can be smaller than the
        # table; the whole table is still written.
        rows = int(df["rows"].iloc[0]) if "rows" in df.columns else len(df)
        cols = int(df["cols"].iloc[0]) if "cols" in df.columns else 6

        # FTABLE columns are positional: depth, area, volume, then discharges.
        data_pars = pars.loc[~pars["PARAMETER_NAME"].isin(["rows", "cols"])]
        # The flow-through time can be present beyond the stated columns.
        shown = df
        ncols = cols
        for i in range(len(data_pars), cols, -1):
            name = data_pars["PARAMETER_NAME"].iloc[i - 1]
            if name in shown.columns and (shown[name].fillna(0) != 0).any():
                ncols = i
                break
        data_pars = data_pars.iloc[:ncols]

        out.extend(
            (
                f"  FTABLE{number:>7}",
                " ROWS COLS ***",
                f"{rows:>5}{cols:>5}",
                _comment_line(
                    "".join(f"{n.upper():>10}" for n in data_pars["PARAMETER_NAME"])
                ),
            )
        )
        out.extend(_render(df, data_pars, 1, None, "ENGL"))
        out.append(f"  END FTABLE{number:>7}")
    out.extend(("END FTABLES", ""))
    return out


def _family_position(block: str, base: str, ext: str):
    """Position of a table within the repeated tables with the same name."""
    mapping = _multiple_table_names.get((block, base))
    if mapping is None:
        return None
    return next(
        (position for position, suffix in mapping.items() if ext == f"{base}{suffix}"),
        None,
    )


def _table_lines(
    block: str,
    ext: str,
    base: str,
    hdf_table: str,
    parameter_dict,
    units: str,
) -> list[str] | None:
    """
    Lines for one table (without the table name lines).

    Returns None if the table is not in the dictionary or has only defaults.
    """
    df = parameter_dict.get(hdf_table)
    if df is None or isinstance(df, pd.Series) or df.empty:
        return None

    pars = _params(block, ext)
    names = [n for n in pars["PARAMETER_NAME"] if n not in ("OPNID", "OPNIDLAST")]
    present = [n for n in names if n in df.columns]
    if not present:
        return None

    required = _tables.loc[
        (_tables["HSPF_BLOCK"] == block) & (_tables["HSPF_TABLE_EXT"] == ext),
        "REQ_TABLE",
    ]
    defaultable = not required.empty and not _is_blank(required.iloc[0])
    if defaultable and all(
        _is_default(df[p.PARAMETER_NAME], p, units)
        for _, p in pars.iterrows()
        if p.PARAMETER_NAME in present and int(p.START) < _LINE * 6
    ):
        return None

    n_lines = _continuation_lines.get((block, base), 1)
    data = df[present].copy()
    if "OPNID" in set(pars["PARAMETER_NAME"]):
        ids = _operation_ids(df.index)
        data = _group_ranges(
            data, ids, "OPNID", "OPNIDLAST", sort=True, allow_gaps=True
        )
    return _render(data, pars, n_lines, None, units)


def _operation_block(block: str, parameter_dict, units: str) -> list[str]:
    tables = _tables.loc[_tables["HSPF_BLOCK"] == block].sort_values(
        "HSPF_MANUAL_SORT_ORDER", kind="stable"
    )
    out: list[str] = []
    done_families: set[str] = set()

    for row in tables.itertuples():
        base, ext, hdf_table = row.HSPF_TABLE, row.HSPF_TABLE_EXT, row.HDF_TABLE
        if _is_blank(ext):
            continue

        members = [
            (_family_position(block, base, r.HSPF_TABLE_EXT), r)
            for r in tables.loc[tables["HSPF_TABLE"] == base].itertuples()
        ]
        is_family = any(pos is not None for pos, _ in members)

        if not is_family:
            if lines := _table_lines(
                block, ext, base, hdf_table, parameter_dict, units
            ):
                out += [
                    f"  {base}",
                    *_header_lines(block, ext),
                    *lines,
                    f"  END {base}",
                    "",
                ]
            continue

        # Tables with the same name are identified by their order.
        if base in done_families:
            continue
        done_families.add(base)
        members = sorted(
            (m for m in members if m[0] is not None), key=lambda item: item[0]
        )
        rendered = {
            pos: _table_lines(
                block, r.HSPF_TABLE_EXT, base, r.HDF_TABLE, parameter_dict, units
            )
            for pos, r in members
        }
        used = [pos for pos, lines in rendered.items() if lines]
        if not used:
            continue
        for pos, r in members:
            if pos > max(used):
                break
            out.append(f"  {base}")
            if rendered[pos]:
                out += [*_header_lines(block, r.HSPF_TABLE_EXT), *rendered[pos]]
            out.extend((f"  END {base}", ""))

    return [block, *out, f"END {block}"] if out else []


def _wrap_condition(prefix: str, condition: str, indent: int) -> list[str]:
    """Split "<prefix> <condition> THEN" at spaces into lines of <= 80 characters."""
    words = [*prefix.split(), *condition.split(), "THEN"]
    lines, current = [], " " * indent
    for word in words:
        if current.strip() and len(current) + 1 + len(word) > _LINE:
            lines.append(current)
            current = " " * (indent + 4)
        current += word if current.endswith(" ") else f" {word}"
    lines.append(current)
    return lines


def _spec_actions_lines(parameter_dict, units: str) -> list[str]:
    block = "SPEC-ACTIONS"
    out: list[str] = []
    actions = None
    for ext, hdf_table in [
        ("DISTRB", "/SPEC_ACTIONS/DISTRB"),
        ("UVQUAN", "/SPEC_ACTIONS/UVQUAN"),
        ("UVNAME", "/SPEC_ACTIONS/UVNAME"),
        ("ACTION", "/SPEC_ACTIONS/ACTIONS"),
    ]:
        df = parameter_dict.get(hdf_table)
        if df is None or isinstance(df, pd.Series) or df.empty:
            continue
        pars = _params(block, ext)
        if ext == "ACTION":
            actions = df
            continue
        out += [*_header_lines(block, ext), *_render(df, pars, 1, None, units)]

    if actions is None:
        return [block, *out, f"END {block}"] if out else []

    pars = _params(block, "ACTION")
    rendered = _render(actions, pars, 1, None, units)
    cond_ids = (
        pd.to_numeric(actions["cond_id"], errors="coerce").fillna(-1).astype(int)
        if "cond_id" in actions.columns
        else pd.Series(-1, index=actions.index)
    ).tolist()

    conds = parameter_dict.get("/SPEC_ACTIONS/conditions")
    nodes: dict[int, tuple[int, int, str]] = {}
    if conds is not None and not conds.empty:
        for row in conds.itertuples():
            nodes[int(row.cond_id)] = (
                int(row.parent_id),
                int(row.sibling_id),
                "" if _is_blank(row.condition) else str(row.condition),
            )
    # Actions that point at an unknown condition are written unconditionally.
    by_cond: dict[int, list[str]] = {}
    for cond_id, line in zip(cond_ids, rendered):
        by_cond.setdefault(cond_id if cond_id in nodes else -1, []).append(line)

    next_sibling = {sib: cid for cid, (_, sib, _) in nodes.items() if sib != -1}

    def body(parent: int, depth: int) -> list[str]:
        lines = list(by_cond.get(parent, []))
        starts = [
            c
            for c, (par, sib, _) in sorted(nodes.items())
            if par == parent and sib == -1
        ]
        indent = 2 + 2 * depth
        for node in starts:
            keyword = "IF"
            while node is not None:
                _, _, condition = nodes[node]
                if condition:
                    lines += _wrap_condition(keyword, condition, indent)
                else:
                    lines.append(" " * indent + "ELSE")
                lines += body(node, depth + 1)
                node = next_sibling.get(node)
                keyword = "ELSE IF"
            lines.append(" " * indent + "END IF")
        return lines

    out += [*_header_lines(block, "ACTION"), *body(-1, 0)]
    return [block, *out, f"END {block}"] if out else []


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def write_uci_parameters(uci_name: str | Path, parameter_dict: dict) -> None:
    """
    Write a UCI file from a dictionary of pandas DataFrames.

    Parameters
    ----------
    uci_name : str or Path
        The name of the UCI file to create.
    parameter_dict : dict
        Dictionary of pandas DataFrames keyed by HDF_TABLE, such as returned
        by ``read_uci_parameters``, ``read_hdf_parameters`` or
        ``read_csv_parameters``.
    """
    units = _units(parameter_dict)

    for key in parameter_dict:
        if key.startswith(_UNSUPPORTED_PREFIXES):
            warnings.warn(
                f"The table {key} is not written to a UCI file.", stacklevel=2
            )

    blocks: list[list[str]] = []

    if "/CONTROL/GLOBAL" in parameter_dict:
        blocks.append(_global_lines(parameter_dict["/CONTROL/GLOBAL"]))

    if "/FILES/FILES" in parameter_dict:
        blocks.append(_simple_block("FILES", parameter_dict["/FILES/FILES"], units))

    if "/CONTROL/OP_SEQUENCE" in parameter_dict:
        blocks.append(_op_sequence_lines(parameter_dict["/CONTROL/OP_SEQUENCE"]))

    blocks.append(_spec_actions_lines(parameter_dict, units))

    blocks.extend(
        _operation_block(block, parameter_dict, units) for block in _OPERATION_BLOCKS
    )

    if "/CATEGORY" in parameter_dict:
        blocks.append(_simple_block("CATEGORY", parameter_dict["/CATEGORY"], units))

    blocks.append(_ftables_lines(parameter_dict))

    if "/CONTROL/EXT_SOURCES" in parameter_dict:
        blocks.append(_ext_sources_lines(parameter_dict["/CONTROL/EXT_SOURCES"], units))

    if "/CONTROL/LINKS" in parameter_dict:
        network, schematic = _links(parameter_dict["/CONTROL/LINKS"], units)
        blocks.extend([network, schematic])

    if "/CONTROL/MASS_LINKS" in parameter_dict:
        blocks.append(_mass_link_lines(parameter_dict["/CONTROL/MASS_LINKS"], units))

    if "/CONTROL/EXT_TARGETS" in parameter_dict:
        blocks.append(
            _simple_block(
                "EXT TARGETS",
                parameter_dict["/CONTROL/EXT_TARGETS"],
                units,
                _BLANK_IF_DEFAULT,
            )
        )

    lines = ["RUN", ""]
    for block in blocks:
        if block:
            lines.extend([*block, ""])
    lines.append("END RUN")

    text = "\n".join(lines) + "\n"
    Path(uci_name).write_text(text.encode("ascii", "replace").decode("ascii"))
