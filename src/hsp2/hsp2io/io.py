"""
IOManager classes for HSP2 model parameter and time series management.

This module provides the base IOManager class and validation functionality for
HSP2 models. For specific implementations, see:
- HDF5IOManager in io_classes.py for HDF5 file format
- UCIIOManager in io_classes.py for UCI control files with WDM/HBN time series
- CSVIOManager in io_classes.py for CSV-based data storage
"""

# Future imports
from __future__ import annotations

# Standard library imports
from functools import partial
from pathlib import Path, PurePath
from textwrap import dedent

# Third party imports
import pandas as pd

# First party imports
from hsp2.hsp2.model import Model
from hsp2.hsp2.utilities import pandas_offset_by_version
from hsp2.hsp2io.convert_control_files import get_parameter_reader
from hsp2.hsp2io.convert_time_series import get_ts_reader, get_ts_writer
from hsp2.hsp2io.get_log_writer import get_log_writer
from hsp2.hsp2io.protocols import (
    Category,
    SupportsWriteLogging,
)


def _log_nop(*args, **kwds):
    """
    A no-operation function for logging when no logging is desired.
    """


class IOManager:
    """Management class for IO operations needed to execute the HSP2 model"""

    def __init__(
        self,
        io_combined: bytes | str | Path | None = None,
        parameter_source: bytes | str | Path | None = None,
        ts_source: bytes | str | Path | None = None,
        ts_target: bytes | str | Path | None = None,
        log: SupportsWriteLogging | None = None,
    ) -> None:
        """
        Initialize the IOManager for HSP2 model IO operations.

        R = Read
        W = Write
        L = Local file
        U = URL or remote source
        * = Currently the WDM file is downloaded locally and read from there.

        +------------------+------+-------+
        | Parameter        | Mode | .hdf5 |
        |                  |      | .h5   |
        +==================+======+=======+
        | io_combined      | R W  | L     |
        +------------------+------+-------+

        +------------------+------+-------+------+------+------+------+
        | Parameter        | Mode | .hdf5 | .uci | .csv | .wdm | .plt |
        |                  |      | .h5   |      |      |      |      |
        +==================+======+=======+======+======+======+======+
        | parameter_source | R    | L     | L U  | L U  |      |      |
        +------------------+------+-------+------+------+------+------+
        | ts_source        | R    | L     |      |      | L U* |      |
        +------------------+------+-------+------+------+------+------+
        | ts_target        | R W  | L     |      |      |      |      |
        +------------------+------+-------+------+------+------+------+
        | log              | W    | L     |      | L    |      |      |
        +------------------+------+-------+------+------+------+------+

        `parameter_source`, `ts_source`, `ts_target`, are all required if
        `io_combined` is None.

        Parameters
        ----------
        io_combined
            Object that combines protocols for Parameters, Input, Output, and
            Log. If `read_parameters`, `ts_source`, `ts_target`, or `log` are
            not specified, this argument will be used as the default for those.
        parameter_source
            File path or URL to read model parameters from. Required if
            `io_combined` is not provided.  Allowable source are listed in the
            table above.
        ts_source
            File path or URL to read time series data from. Required if
            `io_combined` is not provided. Allowable sources are listed in the
            table above.
        ts_target
            File path to write time series data to. Required if `io_combined`
            is not provided. Allowable targets are listed in the table above.
        log
            Object or file path to write log data to. Optional. Allowable sources
            are listed in the table above.

        Attributes
        ----------
        self._store : dict
            A dictionary storing parameter data read from the source. Keys are
            HDF styled paths within the data source, and values are pandas
            DataFrames.
        self.model : Model
            An instance of the HSP2 Model class populated with parameters
            read from self._store.
        """
        if io_combined is None and any(
            v is None for v in [parameter_source, ts_source, ts_target]
        ):
            raise ValueError(
                dedent("""\

                Either io_combined, or all of the keyword arguments
                parameter_source, ts_source, and ts_target must be provided.

                """)
            )

        self._log_method = _log_nop
        self._log_file_path = ""

        if io_combined is None:
            self._read_parameter_source = get_parameter_reader(parameter_source)
            self._store = self._read_parameter_source(parameter_source)
            self.parameter_source = parameter_source

            self._input_method = get_ts_reader(ts_source)
            self._input_file_path = ts_source

            self._output_method = get_ts_writer(ts_target)
            self._output_file_path = ts_target

            if log:
                self._log_method = get_log_writer(log)
                self._log_file_path = log

        elif isinstance(io_combined, (str, PurePath)):
            self._read_parameter_source = get_parameter_reader(io_combined)
            self._store = self._read_parameter_source(io_combined)
            self.parameter_source = io_combined

            self._input_method = get_ts_reader(io_combined)
            self._input_file_path = io_combined

            self._output_method = get_ts_writer(io_combined)
            self._output_file_path = io_combined

            self._log_method = get_log_writer(io_combined)
            self._log_file_path = io_combined

        elif (
            io_combined.hasattr("read_parameters")
            and io_combined.hasattr("read_ts")
            and io_combined.hasattr("write_ts")
        ):
            self.read_parameters = io_combined.read_parameters
            self.parameter_source = io_combined.file_path
            self._store = self._read_parameter_source(self.parameter_source)

            self.read_ts = io_combined.read_ts
            self._input_file_path = io_combined.read_ts.file_path

            self.write_ts = io_combined.write_ts
            self._output_file_path = io_combined.write_ts.file_path

            if io_combined.hasattr("write_log"):
                self.write_log = io_combined.write_log
                self._log_file_path = io_combined.write_log.file_path

        self._in_memory = {}

    def _input(
        self,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        return partial(self._input_method, self._input_file_path)(
            category,
            operation,
            segment,
            activity,
            *args,
            **kwargs,
        )

    def _output(
        self,
        data_frame: pd.DataFrame,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
        outstep: int = 2,
        *args,
        **kwargs,
    ) -> None:
        return partial(self._output_method, self._output_file_path)(
            data_frame,
            category=category,
            operation=operation,
            segment=segment,
            activity=activity,
            outstep=outstep,
            *args,
            **kwargs,
        )

    def _log(
        self,
        dataframe,
    ):
        return partial(self._log_method, self._log_file_path)(dataframe)

    def read_parameters(self) -> Model:
        """
        Read parameters from self._store and populate Model instance.
        """
        model = Model()
        for path in self._store:
            op, module, *other = path[1:].split(sep="/", maxsplit=3)
            s = "_".join(other)
            if op == "CONTROL":
                if module == "GLOBAL":
                    temp = self._store[path].to_dict()["Info"]
                    model.siminfo["start"] = pd.Timestamp(temp["Start"])
                    model.siminfo["stop"] = pd.Timestamp(temp["Stop"])
                    model.siminfo["units"] = 1
                    if "Units" in temp and int(temp["Units"]):
                        model.siminfo["units"] = int(temp["Units"])
                elif module == "LINKS":
                    for row in self._store[path].fillna("").itertuples():
                        if row.TVOLNO:
                            model.ddlinks[f"{row.TVOLNO}"].append(row)
                        else:
                            model.ddlinks[f"{row.TOPFST}"].append(row)

                elif module == "MASS_LINKS":
                    for row in self._store[path].replace("na", "").itertuples():
                        model.ddmasslinks[row.MLNO].append(row)
                elif module == "EXT_SOURCES":
                    for row in self._store[path].replace("na", "").itertuples():
                        model.ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
                elif module == "OP_SEQUENCE":
                    model.opseq = self._store[path]
            elif op in {"PERLND", "IMPLND", "RCHRES"}:
                for op_id, vdict in self._store[path].to_dict("index").items():
                    model.model[(op, module, op_id)][s] = vdict
            elif op == "GENER":
                for index, row in self._store[path].iterrows():
                    if module == "COEFFS":
                        for it in range(1, 8):
                            model.ddgener[f"K{it:01d}"][index] = row[f"K{it:01d}"]
                    else:
                        model.ddgener[module][index] = row.iloc[0]
            elif op == "FTABLES":
                model.ftables[module] = self._store[path]
            elif op == "SPEC_ACTIONS":
                model.specactions[module] = self._store[path]
            elif op == "MONTHDATA":
                if not model.monthdata:
                    model.monthdata = {}
                model.monthdata[f"{op}/{module}"] = self._store[path]
        return model

    def write_ts(
        self,
        data_frame: pd.DataFrame,
        save_columns: list[str],
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
        outstep: int = 2,
        *args,
        **kwargs,
    ) -> None:
        key = (category, operation, segment, activity)
        self._in_memory[key] = data_frame.copy(deep=True)

        if drop_columns := [c for c in data_frame.columns if c not in save_columns]:
            data_frame = data_frame.drop(columns=drop_columns)

        if not isinstance(data_frame.index, pd.core.indexes.datetimes.DatetimeIndex):
            data_frame = data_frame.to_timestamp()

        if outstep == 3:
            # change time step of output to daily
            sumdf1 = data_frame.resample("D", origin="start").sum()
            lastdf2 = data_frame.resample("D", origin="start").last()
            meandf3 = data_frame.resample("D", origin="start").mean()
            data_frame = pd.merge(
                lastdf2.add_suffix("_last"),
                sumdf1.add_suffix("_sum"),
                left_index=True,
                right_index=True,
            )
            data_frame = pd.merge(
                data_frame,
                meandf3.add_suffix("_aver"),
                left_index=True,
                right_index=True,
            )
        elif outstep == 4:
            # change to monthly
            sumdf1 = data_frame.resample(
                pandas_offset_by_version("ME"), origin="start"
            ).sum()
            lastdf2 = data_frame.resample(
                pandas_offset_by_version("ME"), origin="start"
            ).last()
            meandf3 = data_frame.resample(
                pandas_offset_by_version("ME"), origin="start"
            ).mean()
            data_frame = pd.merge(
                lastdf2.add_suffix("_last"),
                sumdf1.add_suffix("_sum"),
                left_index=True,
                right_index=True,
            )
            data_frame = pd.merge(
                data_frame,
                meandf3.add_suffix("_aver"),
                left_index=True,
                right_index=True,
            )
        elif outstep == 5:
            # change to annual
            sumdf1 = data_frame.resample(
                pandas_offset_by_version("YE"), origin="start"
            ).sum()
            lastdf2 = data_frame.resample(
                pandas_offset_by_version("YE"), origin="start"
            ).last()
            meandf3 = data_frame.resample(
                pandas_offset_by_version("YE"), origin="start"
            ).mean()
            data_frame = pd.merge(
                lastdf2.add_suffix("_last"),
                sumdf1.add_suffix("_sum"),
                left_index=True,
                right_index=True,
            )
            data_frame = pd.merge(
                data_frame,
                meandf3.add_suffix("_aver"),
                left_index=True,
                right_index=True,
            )
        self._output(
            data_frame,
            category,
            operation,
            segment,
            activity,
            outstep,
            *args,
            **kwargs,
        )

    def read_ts(
        self,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        data_frame = self._get_in_memory(category, operation, segment, activity)
        if data_frame is not None:
            return data_frame
        if category == Category.INPUTS:
            data_frame = self._input(category, operation, segment, activity)
            key = (category, operation, segment, activity)
            self._in_memory[key] = data_frame.copy(deep=True)
            return data_frame
        if category == Category.RESULTS:
            return self._output(
                None,
                category=category,
                operation=operation,
                segment=segment,
                activity=activity,
            )
        return pd.DataFrame

    def write_log(self, data_frame) -> None:
        self._log(data_frame)

    def write_versioning(self, data_frame) -> None:
        self._log(data_frame)

    def _get_in_memory(
        self,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
    ) -> pd.DataFrame | None:
        key = (category, operation, segment, activity)
        try:
            return self._in_memory[key].copy(deep=True)
        except KeyError:
            return None
