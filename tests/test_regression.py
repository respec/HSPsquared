from pathlib import Path

import pytest

from HSP2tools.commands import import_uci, run
from HSP2tools.HDF5 import HDF5

from .convert.regression_base import RegressTest as RegressTestBase


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
