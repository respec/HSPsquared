# Third party imports
import pandas as pd


def write_csv_log(csvfile, hsp2_log: pd.DataFrame) -> None:
    hsp2_log.to_csv(csvfile)
