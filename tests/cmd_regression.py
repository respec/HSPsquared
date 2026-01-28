# Note: This code must be run from the tests dir due to importing
# the `convert` directory where the regression_base file lives
# todo: make this convert path passable by argument 
import numpy as np
from tests.convert.regression_base import RegressTest


case = "test10specl"
base_case = "test10"
#case = "testcbp"
tdir = "/opt/model/HSPsquared/tests"
#import_uci(str(hsp2_specl_uci), str(temp_specl_h5file))
#run(temp_specl_h5file, saveall=True, compress=False)

############# RegressTest data loader
## TEST CASE
test = RegressTest(case, threads=1, tests_root_dir = tdir)
# test object hydr
rchres_hydr_hsp2_test_table = test.hsp2_data._read_table('RCHRES', '001', 'HYDR')
perlnd_pwat_hsp2_test_table = test.hsp2_data._read_table('PERLND', '001', 'PWATER')
rchres_hydr_hsp2_test = test.hsp2_data.get_time_series('RCHRES', '001', 'RO', 'HYDR')
rchres_hydr_hspf_test = test.get_hspf_time_series( ('RCHRES', 'HYDR', '001', 'RO', 2)) #Note: the order of arguments is wonky in Regressbase
rchres_hydr_hsp2_test_mo = rchres_hydr_hsp2_test.resample('MS').mean()
rchres_hydr_hspf_test_mo = rchres_hydr_hspf_test.resample('MS').mean()
## BASE CASE
base = RegressTest(base_case, threads=1, tests_root_dir = tdir)
# base object hydr
rchres_hydr_hsp2_base_table = base.hsp2_data._read_table('RCHRES', '001', 'HYDR')
perlnd_pwat_hsp2_base_table = base.hsp2_data._read_table('PERLND', '001', 'PWATER')
rchres_hydr_hsp2_base = base.hsp2_data.get_time_series('RCHRES', '001', 'RO', 'HYDR')
rchres_hydr_hspf_base = base.get_hspf_time_series( ('RCHRES', 'HYDR', '001', 'RO', 2))  #Note: the order of arguments is wonky in Regressbase
rchres_hydr_hsp2_base_mo = rchres_hydr_hsp2_base.resample('MS').mean()
rchres_hydr_hspf_base_mo = rchres_hydr_hspf_base.resample('MS').mean()

# Show quantiles
np.quantile(rchres_hydr_hsp2_test, [0,0.25,0.5,0.75,1.0])
np.quantile(rchres_hydr_hspf_test, [0,0.25,0.5,0.75,1.0])
np.quantile(rchres_hydr_hsp2_base, [0,0.25,0.5,0.75,1.0])
np.quantile(rchres_hydr_hspf_base, [0,0.25,0.5,0.75,1.0])
# Monthly mean value comparisons
rchres_hydr_hsp2_test_mo
rchres_hydr_hspf_test_mo
rchres_hydr_hsp2_base_mo
rchres_hydr_hspf_base_mo

# Compare ANY arbitrary timeseries, not just the ones coded into the RegressTest object
# 3rd argument is tolerance to use
test.compare_time_series(rchres_hydr_hsp2_base_table['RO'], rchres_hydr_hsp2_test_table['RO'], 10.0)
# Example:  (True, 8.7855425)

# Compare inflows and outflows
np.mean(perlnd_pwat_hsp2_test_table['PERO']) * 6000 * 0.0833
np.mean(rchres_hydr_hsp2_test_table['IVOL'])
np.mean(rchres_hydr_hsp2_test_table['ROVOL'])
np.mean(perlnd_pwat_hsp2_base_table['PERO']) * 6000 * 0.0833
np.mean(rchres_hydr_hsp2_base_table['IVOL'])
np.mean(rchres_hydr_hsp2_base_table['ROVOL'])

# now do a comparison
# HYDR diff should be almost nonexistent
test.check_con(params = ('RCHRES', 'HYDR', '001', 'ROVOL', 2))
# this is very large for PWTGAS
test.check_con(params = ('PERLND', 'PWTGAS', '001', 'POHT', '2'))

# Now run the full test
results = test.run_test()
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
