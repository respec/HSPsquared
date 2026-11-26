"""For reading HSPF mutsin files."""

# Third party imports
import pandas as pd


def read_mutsin_ts(filename):
    """
    Reads HSPF PLTGEN/MUTSIN files and creates a DataFrame.

    Parameters
    ----------
    filename : str
        The path to the PLTGEN/MUTSIN file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the time series data from the PLTGEN/MUTSIN file
        with column names as specified in the file and a DatetimeIndex.
    """
    found_column_names = False
    column_names = []
    with open(filename, encoding="ascii") as fpointer:
        for i, line in enumerate(fpointer):
            if "LINTYP" in line:
                found_column_names = True
                continue
            if line[5:].startswith("Time series"):
                break
            if found_column_names and (column_name := line[4:30].strip()):
                column_names.append(column_name)
        pgdf = pd.read_fwf(
            fpointer,
            colspecs=[(5, 10), (10, 13), (13, 16), (16, 19), (19, 22)]
            + [(22 + i * 14, 36 + i * 14) for i in range(len(column_names))],
            skiprows=3,
            names=["Year", "Month", "Day", "Hour", "Minute"] + column_names,
        )

    # .replace causes RecursionError in 1.5.* versions of pandas and so use
    # .where instead.
    pgdf = pgdf.where(pgdf != -1e30, pd.NA).dropna(how="all", subset=column_names)

    # Can't let read_fwf parse dates because HSPF can use 24:00 for midnight of
    # the following day, where pandas can't work with that. So we create
    # manually here by creating an HH:MM delta and adding to the date.
    pgdf["delta"] = pd.to_timedelta(
        pgdf["Hour"].astype(int), unit="h"
    ) + pd.to_timedelta(pgdf["Minute"].astype(int), unit="m")
    pgdf["Datetime"] = pd.to_datetime(pgdf[["Year", "Month", "Day"]]) + pgdf["delta"]

    pgdf = pgdf.drop(columns=["Year", "Month", "Day", "Hour", "Minute", "delta"])
    pgdf = pgdf.set_index(["Datetime"])

    return pgdf
