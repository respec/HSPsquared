# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.time_series_readers.read_hbn_ts import HBNReader, read_hbn_ts


def test_read_hbn_ts_selects_all_constituents():
    hbn_file = Path(__file__).parent / "test05" / "HSPFresults" / "test05.hbn"

    reader = HBNReader(str(hbn_file))
    reader.read_data()
    expected = reader.get_time_series_data_frame("PERLND", 1, "SNOW", "monthly")
    data_frame = read_hbn_ts(
        str(hbn_file),
        operation="PERLND",
        segment="P001",
        activity="SNOW",
        time_interval="monthly",
    )

    pd.testing.assert_frame_equal(data_frame, expected)


def test_read_hbn_ts_selects_one_constituent_as_dataframe():
    hbn_file = Path(__file__).parent / "test05" / "HSPFresults" / "test05.hbn"

    data_frame = read_hbn_ts(
        str(hbn_file),
        operation="PERLND",
        segment=1,
        activity="SNOW",
        t_cons="PACK",
        time_interval="monthly",
    )

    assert data_frame.shape[1] == 1
    assert data_frame.columns.tolist() == ["PACK"]


def test_read_hbn_ts_without_selectors_returns_summary():
    hbn_file = Path(__file__).parent / "test05" / "HSPFresults" / "test05.hbn"

    summary = read_hbn_ts(str(hbn_file))

    assert summary.shape == (12, 7)
    assert "Frequency" in summary.columns
