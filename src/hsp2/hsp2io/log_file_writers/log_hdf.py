# Third party imports
import pandas as pd


def write_hdf_log(hdffile, hsp2_log: pd.DataFrame) -> None:
    hsp2_log.to_hdf(hdffile, key="RUN_INFO/LOGFILE", data_columns=True, format="table")
