# Future imports
from __future__ import annotations

# Standard library imports
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Third party imports
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # First party imports
    from hsp2.hsp2.model import Model

TimeSeriesDict = dict[str, np.float64]


class Category(Enum):
    RESULTS = "RESULT"
    INPUTS = "INPUT"


@runtime_checkable
class SupportsReadParameters(Protocol):
    def read_parameters(self) -> Model: ...


@runtime_checkable
class SupportsWriteParameters(Protocol):
    def write_parameters(self, model: Model) -> None: ...


@runtime_checkable
class SupportsReadTS(Protocol):
    def read_ts(
        self,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
    ) -> pd.DataFrame: ...


@runtime_checkable
class SupportsWriteTS(Protocol):
    def write_ts(
        self,
        data_frame: pd.DataFrame,
        category: Category,
        operation: str | None = None,
        segment: str | None = None,
        activity: str | None = None,
    ) -> None: ...


@runtime_checkable
class SupportsWriteLogging(Protocol):
    def write_log(self, hsp2_log: pd.DataFrame) -> None: ...

    def write_versioning(self, versions: pd.DataFrame) -> None: ...
