from typing import Protocol, Tuple, Dict, Any, List
from collections import defaultdict
import pandas as pd
import numpy as np
from pandas.core.frame import Pandas 

UCITuple = [defaultdict(dict), defaultdict(list), defaultdict(list),
	defaultdict(list), defaultdict(dict), dict, int] 
TimeSeriesDict = Dict[np.types.unicode_type,np.types.float64]

class ReadableUCI(Protocol):
	def read_uci(self) -> UCITuple:
		...

class WriteableUCI(Protocol):
	def write_uci(self, UCITuple) -> None:
		...

class ReadableTSStorage(Protocol):
	def read_timeseries(self, ext_sourcesdd:List[Pandas], siminfo:Dict[str,Any]) -> TimeSeriesDict:
		...	

class WriteableTimeseries(Protocol):

	def write_timeseries(self, ts:TimeSeriesDict, siminfo:Dict[str,Any], saveall:bool,
		operation:str, segment:str, activity:str) -> None:
		...

### Potentially need to add get_flows method as well