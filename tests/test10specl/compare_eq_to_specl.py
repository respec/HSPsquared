# if testing manually you may need to os.chdir('./tests/test10specl/HSP2results')
import os
import pandas as pd
import numpy as np
from pandas import read_hdf
import h5py
#from pandas.io.pytables import read_hdf
#from HSP2IO.hdf import HDF5
from pathlib import Path
from src.hsp2.hsp2tools.commands import import_uci, run
from src.hsp2.hsp2tools.HDF5 import HDF5
from src.hsp2.hsp2tools.HBNOutput import HBNOutput

def test_h5_file_exists():
    assert os.path.exists('test10.h5')


############# HSPF With SPECIAL ACTION 
test_root = Path("tests/test10specl")
test_root.exists()
test_root_hspf = Path(test_root) / "HSPFresults"
hspf_data = HBNOutput(os.path.join(test_root_hspf, 'test10speclR.hbn'))
hspf_data.read_data()
# get RCHRES 5 sedtrn silt
ts_silt_hspf = hspf_data.get_time_series('RCHRES', 5, 'RSEDTOTSILT', 'SEDTRN', 'full')
total_silt_hspf = ts_silt_hspf.mean()
quantile_silt_hspf = np.quantile(ts_silt_hspf,[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])

############# HSPF NO SPECL
hspf_nospecl_root = Path("tests/test10/HSPFresults")
hspf_nospecl_data = HBNOutput(os.path.join(hspf_nospecl_root, 'test10R.hbn'))
hspf_nospecl_data.read_data()
ts_silt_nospecl_hspf = hspf_nospecl_data.get_time_series('RCHRES', 5, 'RSEDTOTSILT', 'SEDTRN', 'full')
total_silt_nospecl_hspf = ts_silt_nospecl_hspf.mean()
quantile_silt_nospecl_hspf = np.quantile(ts_silt_nospecl_hspf,[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])

############# hsp2 SPECL 
# Run and Analyze hsp2 WITH SPECL actions
specl_root = Path("tests/test10specl")
specl_root.exists()
specl_root_hsp2 = Path(specl_root) / "HSP2results"
hsp2_specl_uci = specl_root_hsp2.resolve() / "test10specl.uci"
hsp2_specl_uci.exists()
temp_specl_h5file = specl_root_hsp2 / "test10specl.h5"

# IF we want to run it from python, do this:
    # if temp_specl_h5file.exists():
    #    temp_specl_h5file.unlink()
    # load the UCI into the h5 then run it
    # import_uci(str(hsp2_specl_uci), str(temp_specl_h5file))
    # run(temp_specl_h5file, saveall=True, compress=False)
# Load Data from hdf5 & Analyze
dstore_specl = pd.HDFStore(str(temp_specl_h5file), mode='r')
hsp2_specl_hydr5 = read_hdf(dstore_specl, '/RESULTS/RCHRES_R005/HYDR')
hsp2_specl_sedtrn5 = read_hdf(dstore_specl, '/RESULTS/RCHRES_R005/SEDTRN')
hsp2_specl_rsed5 = hsp2_specl_sedtrn5['RSED5']
quantile_silt_hsp2 = np.quantile(hsp2_specl_rsed5,[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
quantile_ro_hsp2 = np.quantile(hsp2_specl_hydr5['RO'],[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
total_silt_hsp2 = hsp2_specl_rsed5.mean()

############# hsp2 w/out SPECL 
# Run and Analyze hsp2 without SPECL actions
nospecl_root = Path("tests/test10")
nospecl_root.exists()
nospecl_root_hspf = Path(nospecl_root) / "HSPFresults"
hsp2_nospecl_uci = nospecl_root_hspf.resolve() / "test10.uci"
hsp2_nospecl_uci.exists()
temp_nospecl_h5file = nospecl_root_hspf / "nospecl_case.h5"

# IF we want to run it from python, do this:
    #if temp_nospecl_h5file.exists():
    #    temp_nospecl_h5file.unlink()
    # load the UCI into the h5 then run it
    #import_uci(str(hsp2_nospecl_uci), str(temp_nospecl_h5file))
    #run(temp_nospecl_h5file, saveall=True, compress=False)
# Load Data from hdf5 & Analyze
dstore_nospecl = pd.HDFStore(str(temp_nospecl_h5file), mode='r')
hsp2_nospecl_hydr5 = read_hdf(dstore_nospecl, '/RESULTS/RCHRES_R005/HYDR')
hsp2_nospecl_sedtrn5 = read_hdf(dstore_nospecl, '/RESULTS/RCHRES_R005/SEDTRN')
hsp2_nospecl_rsed5 = hsp2_nospecl_sedtrn5['RSED5']
np.quantile(hsp2_nospecl_rsed5,[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
np.quantile(hsp2_nospecl_hydr5['RO'],[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
total_silt_nospecl_hsp2 = hsp2_nospecl_rsed5.mean()


############# hsp2 w/equation replicating SPECL 
# Run and Analyze hsp2 WITH SPECL actions
eq_root = Path("tests/test10eq")
eq_root.exists()
eq_root_hsp2 = Path(eq_root) / "HSP2results"
hsp2_eq_uci = eq_root_hsp2.resolve() / "test10eq.uci"
hsp2_eq_uci.exists()
temp_eq_h5file = eq_root_hsp2 / "test10eq.h5"
# IF we want to run it from python, do this:
    #if temp_eq_h5file.exists():
    #    temp_eq_h5file.unlink()

    # load the UCI into the h5 then run it
    #import_uci(str(hsp2_eq_uci), str(temp_eq_h5file))
    #run(temp_eq_h5file, saveall=True, compress=False)
# Load Data from hdf5 & Analyze
dstore_eq_specl = pd.HDFStore(str(temp_eq_h5file), mode='r')
hsp2_eq_hydr5 = read_hdf(dstore_eq_specl, '/RESULTS/RCHRES_R005/HYDR')
hsp2_eq_sedtrn5 = read_hdf(dstore_eq_specl, '/RESULTS/RCHRES_R005/SEDTRN')
hsp2_eq_rsed5 = hsp2_eq_sedtrn5['RSED5']
quantile_silt_eq_hsp2 = np.quantile(hsp2_eq_rsed5,[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
quantile_ro_eq_hsp2 = np.quantile(hsp2_eq_hydr5['RO'],[0.0,0.10,0.5,0.5,0.75,0.9,0.95,1.0])
total_silt_eq_hsp2 = hsp2_eq_rsed5.mean()
dstore_eq_specl.close()

# Calculate the % difference
print("Total Silt ")
print([
  {'HSPF no specl': total_silt_nospecl_hspf},
  {'HSPF specl': total_silt_hspf},
  {'HSP2 specl': total_silt_hsp2}
])
print(
  {'HSPF no specl': total_silt_nospecl_hspf}
)
print([
  {'HSP2 eq': total_silt_eq_hsp2},
  {'HSP2 no specl': total_silt_nospecl_hsp2},
  {'HSPF no specl': total_silt_nospecl_hspf}
])
pct_dif_specl = round(100.0 * (total_silt_hsp2.mean() - total_silt_hspf.mean()) / total_silt_hsp2.mean(), 3)
pct_dif_nospecl = round(100.0 * (total_silt_nospecl_hsp2.mean() - total_silt_nospecl_hspf.mean()) / total_silt_nospecl_hspf.mean(), 3)
print("Total SiltHSP2 vs. HSPF, % difference = ", pct_dif_specl, "%")
print("No SPECL: Total SiltHSP2 vs. HSPF, % difference = ", pct_dif_nospecl, "%")

