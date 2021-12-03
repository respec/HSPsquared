import pandas as pd
from pandas.core.frame import DataFrame
from protocols import UCITuple, Category, SupportsReadUCI, SupportsReadTS, SupportsWriteTS
from typing import Union

class IOManager:
	"""Management class for IO operations needed to execute the HSP2 model"""

	def __init__(self,
			io_all: Union[SupportsReadUCI, SupportsReadTS, SupportsWriteTS, None] = None,
			io_uci: Union[SupportsReadUCI,None]=None, 
			io_input: Union[SupportsReadTS,None]=None,
			io_output: Union[SupportsReadTS,SupportsWriteTS,None]=None) -> None:
		""" io_all: SupportsReadUCI, SupportsReadTS, SupportsWriteTS/None 
			This parameter is intended to allow users with a single file that 
			combined UCI, Input and Output a short cut to specify a single argument. 
		io_uci: SupportsReadUCI/None (Default None)
			A class implementing SupportReadUCI protocol, io_all used in place of 
			this parameter if not specified.
		io_input: SupportsReadUCI/None (Default None)
			A class implementing SupportReadTS protocol, io_all used in place of 
			this parameter if not specified. This parameter is where the input 
			timeseries will be read from. 
		io_output: SupportsReadUCI/None (Default None)
			A class implementing SupportReadUCI protocol, io_all used in place of 
			this parameter if not specified. This parameter is where the output 
			timeseries will be written to and read from. 
		"""

		self.io_input = io_input if io_uci is None else io_all
		self.io_output = io_output if io_uci is None else io_all
		self.io_uci = io_uci if io_uci is None else io_all

		self._in_memory = {} 

	def read_uci(self) -> UCITuple:
		self.io_uci.read_uci()

	def write_ts(self,
			data_frame:pd.DataFrame, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> None:
		key = (category, operation, segment, activity)
		self.io_output.write_timeseries(data_frame, category, operation, segment, activity)
		self._in_memory[key] = data_frame

	def read_ts(self,
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> pd.DataFrame:
		data_frame = self._get_in_memory(category, operation, segment, activity)
		if data_frame: return data_frame
		if category == Category.INPUTS: 
			data_frame = self.io_input.read_timeseries(category, operation, segment, activity)
			key = (category, operation, segment, activity)
			self._in_memory[key] = data_frame
			return data_frame
		if category == Category.RESULTS:
			return self.io_output.read_timeseries(category, operation, segment, activity)
		return pd.DataFrame

	def _get_in_memory(self, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> Union[pd.DataFrame, None]:
		key = (category, operation, segment, activity)
		try:
			return self._in_memory[key]
		except KeyError:
			return None
