"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
"""

import pandas as pd

from hsp2.hsp2.utilities import deprecated
from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts


@deprecated("""

readWDM is deprecated and will be removed in a future release.
Use hsp2.hsp2io.time_series_readers.read_wdm_ts.read_wdm_ts to read the time
series data from the WDM file, and then write the data to an HDF file using
pandas.

""")
def readWDM(wdmfile, hdffile, compress_output=False):
    time_series = read_wdm_ts(wdmfile)

    with pd.HDFStore(hdffile) as store:
        for path, series in time_series.items():
            if compress_output:
                series.to_hdf(
                    store, key=f"/TIMESERIES/{path}", complib="blosc", complevel=9
                )
            else:
                series.to_hdf(
                    store, key=f"/TIMESERIES/{path}", data_columns=True, format="table"
                )
