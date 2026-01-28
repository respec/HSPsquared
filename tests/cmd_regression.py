# Note: This code must be run from the tests dir due to importing
# the `convert` directory where the regression_base file lives
# todo: make this convert path passable by argument 
from pathlib import Path
import os
import pandas as pd
import numpy as np
from hsp2.hsp2tools.commands import import_uci, run
from hsp2.hsp2tools.HDF5 import HDF5 # note: has handy function get_time_series() 
from hsp2.hsp2tools.HBNOutput import HBNOutput # Also has function get_time_series()

from tests.convert.regression_base import RegressTest


#case = "test10"
case = "testcbp"
tdir = "/opt/model/HSPsquared/tests"
#import_uci(str(hsp2_specl_uci), str(temp_specl_h5file))
#run(temp_specl_h5file, saveall=True, compress=False)

## Manual Data outside of RegressTest

############# HSPF NO SPECL
hspf_hbn_path = os.path.join(tdir, case, "HSPFresults", case + 'R.hbn')
hsp2_hdf_path = os.path.join(tdir, case, "HSP2results", case + '.h5')
hspf_hbn = HBNOutput(hspf_hbn_path)
hspf_hbn.read_data()
rchres_hydr_hspf_ovol = hspf_hbn.get_time_series('RCHRES', 1, 'OVOL', 'HYDR', 'full')
rchres_hydr_hspf = hspf_hbn._read_table('RCHRES', '001', 'HYDR', 'Hourly')
np.quantile(rchres_hydr_hsp2['ROVOL'], [0,0.25,0.5,0.75,1.0])


test = RegressTest(case, threads=1, tests_root_dir = tdir)
results = test.run_test()
test.temp_h5file.unlink()

# test object hydr
rchres_hydr_test = test.hsp2_data.data[('RCHRES', '005', 'HYDR')]
test_dir = os.path.join(test.tests_root_dir, test.compare_case)
# test_dir=tdir + '/' + case 
dstore_hsp2 = pd.HDFStore(str(tdir) + '/HSP2results/' + case + '.h5', mode='r')
rchres_hydr_hsp2 = pd.read_hdf(dstore_hsp2, '/RESULTS/RCHRES_R001/HYDR')
perlnd_pwater_hsp2 = pd.read_hdf(dstore_hsp2, '/RESULTS/PERLND_P001/PWATER')
dstore_hsp2.close() # tidy up
np.quantile(rchres_hydr_hsp2['ROVOL'], [0,0.25,0.5,0.75,1.0])
np.quantile(perlnd_pwater_hsp2['SURO'], [0,0.25,0.5,0.75,1.0])


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
