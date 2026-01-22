from pathlib import Path
import os

import pytest
from hsp2.hsp2tools.commands import import_uci, run
from hsp2.hsp2tools.HDF5 import HDF5

from .convert.regression_base import RegressTest as RegressTestBase


class RegressTest(RegressTestBase):
    def __init__(
        self,
        compare_case: str,
        operations: List[str] = [],
        activities: List[str] = [],
        tcodes: List[str] = ["2"],
        ids: List[str] = [],
        threads: int = os.cpu_count() - 1,
        tests_root_dir = None
    ) -> None:
        if tests_root_dir is None:
            tests_root_dir = Path(__file__).resolve().parent
        super(RegressTest, self).__init__(
            compare_case, operations, activities, 
            tcodes, ids, threads, tests_root_dir
        )

    def _get_hsp2_data(self, test_root) -> None:
        test_root_hspf = Path(test_root) / "HSPFresults"
        hspf_uci = test_root_hspf.resolve() / f"{self.compare_case}.uci"
        assert hspf_uci.exists()

        temp_h5file = test_root_hspf / f"{self.compare_case}.h5"
        if temp_h5file.exists():
            temp_h5file.unlink()

        self.temp_h5file = temp_h5file

        import_uci(str(hspf_uci), str(self.temp_h5file))
        run(self.temp_h5file, saveall=True, compress=False)
        self.hsp2_data = HDF5(str(self.temp_h5file))

    def _init_files(self):
        print("Checking tests_root_dir has tests in the name")
        assert self.tests_root_dir.name == "tests"
        test_root = self.tests_root_dir / self.compare_case
        print("Checking case tests_dir exists:", test_root)
        assert test_root.exists()

        self._get_hbn_data(str(test_root))
        self._get_hsp2_data(str(test_root))


@pytest.mark.parametrize(
    "case",
    [
        # "test05",
        # "test09",
        "test10",
        "test10specl",
        # "test10b",
    ],
)
def test_case(case):
    test = RegressTest(case, threads=1)
    results = test.run_test()
    test.temp_h5file.unlink()

    found = False
    mismatches = []
    for key, results in results.items():
        no_data_hsp2, no_data_hspf, match, diff = results
        if any([no_data_hsp2, no_data_hspf]):
            continue
        if not match:
            mismatches.append((case, key, results))
        found = True
    assert found

    if mismatches:
        for case, key, results in mismatches:
            _, _, _, diff = results
            print(case, key, f"{diff:0.00%}")
        raise ValueError("results don't match hspf output")
