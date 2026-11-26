# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2io.io import IOManager
from hsp2.hsp2tools.commands import import_uci, run


def test_io_manager_with_separate_parameter_and_time_series_sources(
    tmp_path: Path,
) -> None:
    uci_file = (
        Path(__file__).parent / "test10" / "HSPFresults" / "test10.uci"
    ).resolve()
    ts_source = tmp_path / "input_timeseries.h5"
    ts_target = tmp_path / "output_timeseries.h5"
    combined_io = tmp_path / "combined_io.h5"

    if not ts_source.exists():
        import_uci(str(uci_file), str(ts_source))
    assert ts_source.is_file()

    import_uci(str(uci_file), str(combined_io))
    ts_target.touch()

    separate_io_manager = IOManager(
        parameter_source=uci_file, ts_source=ts_source, ts_target=ts_target
    )
    model = separate_io_manager.read_parameters()
    assert model.siminfo
    assert model.ddext_sources

    run(
        parameter_source=str(uci_file),
        ts_source=str(ts_source),
        ts_target=str(ts_target),
        saveall=True,
        compress=False,
    )
    run(io_combined=str(combined_io), saveall=True, compress=False)

    assert ts_target.is_file()
    with pd.HDFStore(ts_target, "r") as separate_store, pd.HDFStore(
        combined_io, "r"
    ) as combined_store:
        separate_keys = sorted(
            key for key in separate_store.keys() if key.startswith("/RESULTS/")
        )
        combined_keys = sorted(
            key for key in combined_store.keys() if key.startswith("/RESULTS/")
        )
        assert separate_keys
        assert separate_keys == combined_keys
        for key in separate_keys:
            pd.testing.assert_frame_equal(
                separate_store[key], combined_store[key], obj=key
            )
