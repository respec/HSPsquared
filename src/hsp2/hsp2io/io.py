from typing import List, Union

import pandas as pd

from hsp2.hsp2.model import Model
from hsp2.hsp2.utilities import pandas_offset_by_version
from hsp2.hsp2io.protocols import (
    Category,
    SupportsReadTS,
    SupportsReadParameters,
    SupportsWriteLogging,
    SupportsWriteTS,
)


class IOManager:
    """Management class for IO operations needed to execute the HSP2 model"""

    def __init__(
        self,
        io_combined: Union[
            SupportsReadParameters, SupportsReadTS, SupportsWriteTS, None
        ] = None,
        parameters: Union[SupportsReadParameters, None] = None,
        input: Union[SupportsReadTS, None] = None,
        output: Union[SupportsReadTS, SupportsWriteTS, None] = None,
        log: Union[SupportsWriteLogging, None] = None,
    ) -> None:
        """
        Initialize the IOManager for HSP2 model IO operations.

        Parameters
        ----------
        io_combined : SupportsReadParameters or SupportsReadTS or SupportsWriteTS or None, optional
            Object that combines protocols for Parameters, Input, Output, and
            Log. If `parameters`, `input`, `output`, or `log` are not
            specified, this argument will be used as the default for those.
        parameters : SupportsReadParameters or None, optional
            Instance implementing the SupportReadParameters protocol. Acts as
            the data source for parameter information.  If not specified,
            `io_combined` will be used by default.
        input : SupportsReadTS or None, optional
            Instance implementing the SupportReadTS protocol. Acts as the data
            source for any input timeseries.  If not specified, `io_combined`
            will be used by default.
        output : SupportsWriteTS or SupportsReadTS or None, optional
            Instance implementing the SupportsWriteTS and/or SupportReadTS
            protocol. Acts as the location for outputting result timeseries and
            as the data source for result timeseries needed as inputs to model
            modules.
            If not specified, `io_combined` will be used by default.
        log : SupportsWriteLogging or None, optional
            Instance implementing the SupportWriteLogging protocol. Acts as the
            location to output logging information.  If not specified,
            `io_combined` will be used by default.
        """

        self._input = io_combined if input is None else input
        self._output = io_combined if output is None else output
        self._parameters = io_combined if parameters is None else parameters
        self._log = io_combined if log is None else log

        self._in_memory = {}

    def __del__(self):
        del self._input
        del self._output
        del self._parameters
        del self._log

    def read_parameters(self, *args, **kwargs) -> Model:
        return self._parameters.read_parameters()

    def write_ts(
        self,
        data_frame: pd.DataFrame,
        save_columns: List[str],
        category: Category,
        operation: Union[str, None] = None,
        segment: Union[str, None] = None,
        activity: Union[str, None] = None,
        outstep: int = 2,
        *args,
        **kwargs,
    ) -> None:
        key = (category, operation, segment, activity)
        self._in_memory[key] = data_frame.copy(deep=True)

        drop_columns = [c for c in data_frame.columns if c not in save_columns]
        if drop_columns:
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
        self._output.write_ts(data_frame, category, operation, segment, activity)

    def read_ts(
        self,
        category: Category,
        operation: Union[str, None] = None,
        segment: Union[str, None] = None,
        activity: Union[str, None] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        data_frame = self._get_in_memory(category, operation, segment, activity)
        if data_frame is not None:
            return data_frame
        if category == Category.INPUTS:
            data_frame = self._input.read_ts(category, operation, segment, activity)
            key = (category, operation, segment, activity)
            self._in_memory[key] = data_frame.copy(deep=True)
            return data_frame
        if category == Category.RESULTS:
            return self._output.read_ts(category, operation, segment, activity)
        return pd.DataFrame

    def write_log(self, data_frame) -> None:
        if self._log:
            self._log.write_log(data_frame)

    def write_versioning(self, data_frame) -> None:
        if self._log:
            self._log.write_versioning(data_frame)

    def _get_in_memory(
        self,
        category: Category,
        operation: Union[str, None] = None,
        segment: Union[str, None] = None,
        activity: Union[str, None] = None,
    ) -> Union[pd.DataFrame, None]:
        key = (category, operation, segment, activity)
        try:
            return self._in_memory[key].copy(deep=True)
        except KeyError:
            return None
