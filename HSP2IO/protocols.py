from typing import Protocol, Dict, Any, List, Union, runtime_checkable
from collections import defaultdict
import pandas as pd
import numpy as np
from enum import Enum
from HSP2.uci import UCI

TimeSeriesDict = Dict[str,np.float64]

class Category(Enum):
	RESULTS = 'RESULT'
	INPUTS = 'INPUT'

@runtime_checkable
class SupportsReadUCI(Protocol):
	def read_uci(self) -> UCI:
		...

@runtime_checkable
class SupportsWriteUCI(Protocol):
	def write_uci(self, UCI) -> None:
		...

@runtime_checkable
class SupportsReadTS(Protocol):
	def read_ts(self,
		category:Category,
		operation:Union[str,None]=None, 
		segment:Union[str,None]=None, 
		activity:Union[str,None]=None) -> pd.DataFrame:
		...	

@runtime_checkable
class SupportsWriteTS(Protocol):

	def write_ts(self, 
		data_frame:pd.DataFrame, 
		category:Category,
		operation:Union[str,None]=None, 
		segment:Union[str,None]=None, 
		activity:Union[str,None]=None) -> None:
		...

@runtime_checkable
class SupportsWriteLogging(Protocol):
	
	def write_log(self, hsp2_log:pd.DataFrame) -> None:
		...

	def write_versioning(self, versions:pd.DataFrame) -> None:
		...