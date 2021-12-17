import pandas as pd
from pandas.core.frame import DataFrame
from HSP2IO.protocols import Category, SupportsReadUCI, SupportsReadTS, SupportsWriteTS, SupportsWriteLog
from typing import Union

from HSP2.uci import UCI

class IOManager:
	"""Management class for IO operations needed to execute the HSP2 model"""

	def __init__(self,
			io_all: Union[SupportsReadUCI, SupportsReadTS, SupportsWriteTS, None] = None,
			io_uci: Union[SupportsReadUCI,None]=None, 
			io_input: Union[SupportsReadTS,None]=None,
			io_output: Union[SupportsReadTS,SupportsWriteTS,None]=None,
			io_log: Union[SupportsWriteLog,None]=None,) -> None:
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

		self._input = io_all if io_input is None else io_input
		self._output = io_all if io_output is None else io_output
		self._uci = io_all if io_uci is None else io_uci

		self._in_memory = {} 

	def __def__(self):
		del(self._input)
		del(self._output)
		del(self._uci)

	def read_uci(self, *args, **kwargs) -> UCI:
		return self._uci.read_uci()

	def write_ts(self,
			data_frame:pd.DataFrame, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None,
			*args, **kwargs) -> None:
		key = (category, operation, segment, activity)
		self._output.write_ts(data_frame, category, operation, segment, activity)
		self._in_memory[key] = data_frame.copy(deep=True)

	def read_ts(self,
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None,
			*args, **kwargs) -> pd.DataFrame:
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

	def _get_in_memory(self, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> Union[pd.DataFrame, None]:
		key = (category, operation, segment, activity)
		try:
			return self._in_memory[key].copy(deep=True)
		except KeyError:
			return None
