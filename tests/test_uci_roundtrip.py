"""
Round trip selected UCI files in tests/ through H5 and CSV and back to UCI.

The original and round-tripped UCI files are both read with
``read_uci_parameters`` and the resulting DataFrames are compared, so comment
lines and number formatting ("32" vs "32.0") do not cause false failures.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd
import pytest

# First party imports
from hsp2.hsp2io.control_file_readers.read_uci_parameters import read_uci_parameters
from hsp2.hsp2io.control_file_writers.write_uci_parameters import _is_default, _units
from hsp2.hsp2io.convert_control_files import convert_control_files
from hsp2.hsp2io.utilities import _parameters

_TESTS_DIR = Path(__file__).parent

# Chosen for the broadest range of tables; the rest of the UCI files in tests/
# are near duplicates of these.
UCI_FILES = [
    # Small and simple.
    "test05/HSPFresults/TEST05.UCI",
    # Operations, PLTGEN (CURV-DATA), and FTABLES.
    "ipwater/data/HSPF_Test240.uci",
    # Most tables of the PERLND, IMPLND and RCHRES operations.
    "test10/HSPFresults/test10.uci",
    # Special actions with conditions, nitrogen and phosphorus tables.
    "land_spec/HSPFresults/hwmA51800.uci",
    # Most FTABLES, and many monthly tables.
    "Calleg/HSPFresults_monthly/calleg.uci",
    # Many IQUAL and PQUAL tables.
    "ZRW_WestIndian/HSPF.uci",
]

# Subset also sent through H5 and CSV in the same round trip.
H5_CSV_UCI_FILES = [
    "test10/HSPFresults/test10.uci",
    "land_spec/HSPFresults/hwmA51800.uci",
]


# Tables whose rows are a set of independent records; their order and index
# have no meaning.
_UNORDERED = (
    "/CONTROL/EXT_SOURCES",
    "/CONTROL/LINKS",
    "/CONTROL/EXT_TARGETS",
    "/SPEC_ACTIONS/ACTIONS",
)


def _clean(value):
    """Strip text and treat numeric-looking text ("32", "32.0") as numbers."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def _normalize(df: pd.DataFrame, unordered: bool = False) -> pd.DataFrame:
    """Make values comparable regardless of how they were read."""
    # Column order is not meaningful.
    df = df[sorted(df.columns)].copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(
            df[col]
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
            df[col] = [_clean(v) for v in df[col].tolist()]
    if unordered:
        df = df.reset_index(drop=True)
        df = df.loc[df.astype(str).sort_values(list(df.columns)).index]
        df = df.reset_index(drop=True)
    return df


def _resolve_actions(parameters: dict) -> pd.DataFrame:
    """
    ACTIONS with ``cond_id`` replaced by the text describing the IF structure.

    The numbering of conditions follows the order in the file, which is not
    meaningful by itself.
    """
    actions = parameters["/SPEC_ACTIONS/ACTIONS"].copy()
    conds = parameters.get("/SPEC_ACTIONS/conditions")
    nodes = (
        {}
        if conds is None
        else {int(r.cond_id): r for r in conds.itertuples(index=False)}
    )

    def describe(cond_id: int) -> str:
        if cond_id not in nodes:
            return ""
        node = nodes[cond_id]
        text = str(node.condition).strip() or "ELSE"
        sibling = int(node.sibling_id)
        return f"[{describe(int(node.parent_id))}]{text}<{describe(sibling)}>"

    actions["cond_id"] = [describe(int(c)) for c in actions["cond_id"]]
    return actions


def _all_default(key: str, df, units: str) -> bool:
    if isinstance(df, pd.Series):
        return False
    pars = _parameters.loc[_parameters["HDF_TABLE"] == key].drop_duplicates(
        "PARAMETER_NAME"
    )
    return not any(
        par.PARAMETER_NAME in df.columns
        and not _is_default(df[par.PARAMETER_NAME], pd.Series(par._asdict()), units)
        for par in pars.itertuples(index=False)
    )


def _assert_same(original: dict, roundtrip: dict) -> None:
    # The writer leaves out tables that only hold default values, and they
    # are not recreated by the reader.  "SAVE" tables are generated.
    units = _units(original)
    for key in set(original) - set(roundtrip):
        if key.endswith("/SAVE"):
            continue
        assert _all_default(key, original[key], units), f"{key} was lost"
    assert set(roundtrip) <= set(original)

    for key in sorted(roundtrip):
        if key == "/SPEC_ACTIONS/conditions":
            continue
        a, b = original[key], roundtrip[key]
        if key == "/SPEC_ACTIONS/ACTIONS":
            a, b = _resolve_actions(original), _resolve_actions(roundtrip)
        if isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, check_dtype=False, obj=key)
            continue
        assert set(a.columns) == set(b.columns), key
        unordered = key in _UNORDERED
        pd.testing.assert_frame_equal(
            _normalize(a, unordered),
            _normalize(b, unordered),
            check_dtype=False,
            obj=key,
        )


def _roundtrip(uci_name: str, steps: list[str], tmp_path: Path) -> None:
    """UCI -> each of ``steps`` ("h5", "csv") -> UCI, then compare."""
    uci = _TESTS_DIR / uci_name
    source = uci
    for step in steps:
        target = tmp_path / step / ("RUN.csv" if step == "csv" else "roundtrip.h5")
        convert_control_files(source, target)
        source = target
    uci2 = tmp_path / "roundtrip.uci"
    convert_control_files(source, uci2)
    _assert_same(read_uci_parameters(uci), read_uci_parameters(uci2))


@pytest.mark.parametrize("uci_name", UCI_FILES)
def test_uci_h5_uci(uci_name: str, tmp_path: Path) -> None:
    (tmp_path / "h5").mkdir()
    _roundtrip(uci_name, ["h5"], tmp_path)


@pytest.mark.parametrize("uci_name", UCI_FILES)
def test_uci_csv_uci(uci_name: str, tmp_path: Path) -> None:
    _roundtrip(uci_name, ["csv"], tmp_path)


@pytest.mark.parametrize("uci_name", H5_CSV_UCI_FILES)
def test_uci_h5_csv_uci(uci_name: str, tmp_path: Path) -> None:
    (tmp_path / "h5").mkdir()
    _roundtrip(uci_name, ["h5", "csv"], tmp_path)
