# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd

pd.set_option("io.hdf.default_format", "table")

_data_dir = Path(__file__).parent.parent.parent / "data"
_parameters = pd.read_csv(_data_dir / "Parameters.csv")
_tables = pd.read_csv(_data_dir / "Tables.csv")

Lapse = pd.Series(
    [
        0.0035,
        0.0035,
        0.0035,
        0.0035,
        0.0035,
        0.0035,
        0.0037,
        0.0040,
        0.0041,
        0.0043,
        0.0046,
        0.0047,
        0.0048,
        0.0049,
        0.0050,
        0.0050,
        0.0048,
        0.0046,
        0.0044,
        0.0042,
        0.0040,
        0.0038,
        0.0037,
        0.0036,
    ]
)

Seasons = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]).astype(bool)

Svp = pd.Series(
    [
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.005,
        1.01,
        1.01,
        1.015,
        1.02,
        1.03,
        1.04,
        1.06,
        1.08,
        1.1,
        1.29,
        1.66,
        2.13,
        2.74,
        3.49,
        4.40,
        5.55,
        6.87,
        8.36,
        10.1,
        12.2,
        14.6,
        17.5,
        20.9,
        24.8,
        29.3,
        34.6,
        40.7,
        47.7,
        55.7,
        64.9,
    ]
)


def write_hdf_parameters(hdf_fname, parameters, mode="w"):
    """
    Write hsp2 model parameters to an HDF5 file.

    Parameters
    ----------
    hdf_fname : str
        The name of the HDF5 file to write to.
    parameters : dict
        A dictionary of pandas DataFrames containing the model parameters.  The
        keys should be the HDF5 keys used in HSPsquared and the values should
        be the corresponding DataFrames.
    mode : str, optional
        The file mode, either 'a' for append or 'w' for write (default is 'w').
    """
    with pd.HDFStore(hdf_fname, mode=mode) as hdf_store:
        Lapse.to_hdf(hdf_store, key="TIMESERIES/LAPSE_Table")
        Seasons.to_hdf(hdf_store, key="TIMESERIES/SEASONS_Table")
        Svp.to_hdf(hdf_store, key="TIMESERIES/Saturated_Vapor_Pressure_Table")

        for key, table in parameters.items():
            print(f"Writing {key} to HDF5...")
            table.to_hdf(hdf_store, key=key, data_columns=True, format="table")
