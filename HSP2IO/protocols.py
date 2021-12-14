from typing import Protocol, Dict, Any, List, Union, runtime_checkable
from collections import defaultdict
import pandas as pd
import numpy as np
from enum import Enum


UCITuple = [defaultdict(dict), defaultdict(list), defaultdict(list),
	defaultdict(list), defaultdict(dict), dict, int] 
TimeSeriesDict = Dict[str,np.float64]

class Category(Enum):
	RESULTS = 'RESULT'
	INPUTS = 'INPUT'

@runtime_checkable
class SupportsReadUCI(Protocol):
	def read_uci(self) -> UCITuple:
		...

@runtime_checkable
class SupportsWritUCI(Protocol):
	def write_uci(self, UCITuple) -> None:
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

### Potentially need to add get_flows method as well