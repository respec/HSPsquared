from typing import Protocol, Dict, Any, List, Union
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

class ReadableUCI(Protocol):
	def read_uci(self) -> UCITuple:
		...

class WriteableUCI(Protocol):
	def write_uci(self, UCITuple) -> None:
		...

class ReadableTimeseries(Protocol):
	def read_timeseries(self,
		category:Category,
		operation:Union[str,None]=None, 
		segment:Union[str,None]=None, 
		activity:Union[str,None]=None) -> pd.DataFrame:
		...	

class WriteableTimeseries(Protocol):

	def write_timeseries(self, 
		data_frame:pd.DataFrame, 
		category:Category,
		operation:Union[str,None]=None, 
		segment:Union[str,None]=None, 
		activity:Union[str,None]=None) -> None:
		...

### Potentially need to add get_flows method as well