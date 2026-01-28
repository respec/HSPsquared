# Note: This code must be run from the tests dir due to importing
# the `convert` directory where the regression_base file lives
# todo: make this convert path passable by argument 
from pathlib import Path
import os

import pytest
from hsp2.hsp2tools.commands import import_uci, run
from hsp2.hsp2tools.HDF5 import HDF5
from typing import Dict, List, Tuple, Union

from convert.regression_base import RegressTest


case = "test10"

test = RegressTest(case, threads=1)
test.run_hsp2() # todo: this should go away and be called outside the test class
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
