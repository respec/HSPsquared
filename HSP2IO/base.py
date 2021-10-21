from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from collections import defaultdict
import pandas as pd
import numba.typed
import numpy as np

from pandas.core.frame import Pandas 

UCITuple = [defaultdict(dict), defaultdict(list), defaultdict(list),
	defaultdict(list), defaultdict(dict), dict, int] 
TimeSeriesDict = Dict[np.types.unicode_type,np.types.float64]

class IOBase(ABC):

	@abstractmethod
	def read_uci(self) -> UCITuple:
		""""""
		
	@abstractmethod
	def write_uci(self, UCITuple) -> None:
		""""""

	@abstractmethod
	def read_timeseries(self, ext_sourcesdd:List[Pandas], siminfo:Dict[str,Any]) -> TimeSeriesDict:
		""""""

	@abstractmethod
	#ts type hint in incorrect
	def write_timeseries(self, ts:TimeSeriesDict, siminfo:Dict[str,Any], saveall:bool,
		operation:str, segment:str, activity:str) -> None:
		""""""
	
	### Potentially need to add get_flows method as well

class IOHDF(IOBase):

	def __init__(self):
		pass

	def read_uci(self) -> UCITuple:
		pass
	
	def write_uci(self, UCITuple) -> None:
		pass

	def read_timeseries(self, ext_sourcesdd: List[Pandas], siminfo: Dict[str, Any]) -> TimeSeriesDict:
		pass

	def write_timeseries(self, ts:TimeSeriesDict, siminfo:Dict[str,Any], saveall:bool,
		operation:str, segment:str, activity:str) -> None:
		pass