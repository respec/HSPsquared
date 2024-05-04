from pathlib import Path

import pytest

from HSP2.main import main
from HSP2tools.readUCI import readUCI
from HSP2tools.readWDM import readWDM
from HSP2IO import hdf
from HSP2tools.HDF5 import HDF5

from HSP2IO.io import IOManager

from .convert.regression_base import RegressTest as RegressTestBase


def import_uci(ucifile, h5file):
    """Import UCI and WDM files into HDF5 file.

    Parameters
    ----------
    ucifile: str
        The UCI file to import into HDF file.
    h5file: str
        The destination HDF5 file.
    """

    readUCI(ucifile, h5file)

    with open(ucifile, "r") as fp:
        uci = []
        for line in fp.readlines():
            if "***" in line[:81]:
                continue
            if not line[:81].strip():
                continue
            uci.append(line[:81].rstrip())

    files_start = uci.index("FILES")
    files_end = uci.index("END FILES")

    uci_dir = Path(ucifile).parent
    for nline in uci[files_start : files_end + 1]:
        if (nline[:10].strip())[:3] == "WDM":
            wdmfile = (uci_dir / nline[16:].strip()).resolve()
            if wdmfile.exists():
                readWDM(wdmfile, h5file)


def run(h5file, saveall=True, compress=True):
    """Run a HSPsquared model.

    Parameters
    ----------
    h5file: str
        HDF5 (path) filename used for both input and output.
    saveall: bool
        [optional] Default is True.
        Saves all calculated data ignoring SAVE tables.
    compression: bool
        [optional] Default is True.
        use compression on the save h5 file.
    """
    hdf5_instance = hdf.HDF5(h5file)
    io_manager = IOManager(hdf5_instance)
    main(io_manager, saveall=saveall, jupyterlab=compress)


class RegressTest(RegressTestBase):
    def _get_hsp2_data(self, test_root) -> None:
        test_root_hspf = Path(test_root) / "HSPFresults"
        hspf_uci = test_root_hspf.resolve() / f"{self.compare_case}.uci"
        assert hspf_uci.exists()

        temp_h5file = test_root_hspf / f"_temp_{self.compare_case}.h5"
        if temp_h5file.exists():
            temp_h5file.unlink()

        self.temp_h5file = temp_h5file

        import_uci(str(hspf_uci), str(self.temp_h5file))
        run(self.temp_h5file, saveall=True, compress=False)
        self.hsp2_data = HDF5(str(self.temp_h5file))

    def _init_files(self):
        test_dir = Path(__file__).resolve().parent
        assert test_dir.name == "tests"

        test_root = test_dir / self.compare_case
        assert test_root.exists()

        self._get_hbn_data(str(test_root))
        self._get_hsp2_data(str(test_root))


@pytest.mark.parametrize(
    "case",
    [
        # "test05",
        # "test09",
        "test10",
        # "test10b",
    ],
)
class TestRegression:
    results: dict[tuple, tuple] = {}

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, case):
        test = RegressTest(case, threads=1)
        self.results = test.run_test()
        yield
        test.temp_h5file.unlink()

    def test_case(self, case):
        found = False
        for key, results in self.results.items():
            no_data_hsp2, no_data_hspf, match, diff = results
            if any([no_data_hsp2, no_data_hspf]):
                continue

            assert match, (case, key, f"{diff:0.00%}")
            found = True
        assert found
