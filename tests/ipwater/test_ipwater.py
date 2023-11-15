import json
from pathlib import Path

import pandas as pd
import pytest

from numba import types
from numba.typed import Dict

from HSP2.PWATER import pwater
from HSP2.IWATER import iwater

data_path = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def uci_data():
    with open(data_path / "uci_data.json") as f:
        data = json.load(f)
    yield data


@pytest.fixture(scope="module")
def ts_table():
    ts_table_path = data_path / "Flow_Test.plt"
    cols = [
        "REACH01",
        "PERLND1_SURO",
        "PERLND1_IFWO",
        "PERLND1_AGWO",
        "IMPLND1_SURO",
        "PREC",
        "PETINP",
    ]

    table = pd.read_table(
        ts_table_path, sep=r"\s+", skiprows=26, names=range(13)
    ).rename(columns={i + 6: c for i, c in enumerate(cols)})

    yield table


def test_iwater(uci_data, ts_table):
    siminfo = uci_data["siminfo"]
    siminfo["start"], siminfo["stop"] = (
        pd.to_datetime(siminfo["start"]),
        pd.to_datetime(siminfo["stop"]),
    )
    siminfo["delt"] = 60
    siminfo["steps"] = len(ts_table)

    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for param in ["PREC", "PETINP"]:
        ts[param] = ts_table[param].values

    err, errm = iwater(None, siminfo, uci_data["IMPLND"], ts)

    for errorcnt, errormsg in zip(err, errm):
        assert errorcnt == 0, errormsg

    assert abs((ts["SURO"] - ts_table["IMPLND1_SURO"]).sum()) < 1e-3


def test_pwater(uci_data, ts_table):
    siminfo = uci_data["siminfo"]
    siminfo["start"], siminfo["stop"] = (
        pd.to_datetime(siminfo["start"]),
        pd.to_datetime(siminfo["stop"]),
    )
    siminfo["delt"] = 60
    siminfo["steps"] = len(ts_table)

    ts = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    for param in ["PREC", "PETINP"]:
        ts[param] = ts_table[param].values

    err, errm = pwater(None, siminfo, uci_data["PERLND"], ts)

    for errorcnt, errormsg in zip(err, errm):
        assert errorcnt == 0, errormsg

    assert abs((ts["SURO"] - ts_table["PERLND1_SURO"]).sum()) < 1e-3
    assert abs((ts["IFWO"] - ts_table["PERLND1_IFWO"]).sum()) < 1e-3
    assert abs((ts["AGWO"] - ts_table["PERLND1_AGWO"]).sum()) < 1e-3
