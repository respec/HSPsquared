import pandas as pd
from pandas.core.frame import DataFrame
from HSP2IO.protocols import Category, SupportsReadUCI, SupportsReadTS, SupportsWriteTS, SupportsWriteLogging
from typing import Union, List

from HSP2.uci import UCI

class IOManager:
	"""Management class for IO operations needed to execute the HSP2 model"""

	def __init__(self,
			io_combined: Union[SupportsReadUCI, SupportsReadTS, SupportsWriteTS, None] = None,
			uci: Union[SupportsReadUCI,None]=None, 
			input: Union[SupportsReadTS,None]=None,
			output: Union[SupportsReadTS,SupportsWriteTS,None]=None,
			log: Union[SupportsWriteLogging,None]=None,) -> None:
		""" io_combined: SupportsReadUCI & SupportsReadTS & SupportsWriteTS & SupportsWriteLogging / None 
			Intended to allow users with a object that combines protocols for 
			UCI, Input, Output and Log a shortcut where only a 
			single argument needs to be provided. If UCI, Input, Output and/or 
			Log are not specified this argument will be used as the default.   
		uci: SupportsReadUCI/None (Default None)
			A class instance implementing the SupportReadUCI protocol. 
			This class acts as the data source for UCI information. 
			The argument io_combined be used in place by default if this argument is not specified.
		input: SupportsReadUCI/None (Default None)
			A class instance implementing SupportReadTS protocol. 
			This class acts as the data source for any input timeseries.  
			The argument io_combined be used in place by default if this argument is not specified.
		output: SupportsWriteTS & SupportsReadTS / None (Default None)
			A class implementing SupportsWriteTS & SupportReadTS protocol
			This class acts as the location for outputing result timeseries as 
			well as the data source should those result timeseries be needed for 
			inputs into a model modules.  
			The argument io_combined be used in place by default if this argument is not specified.
		log: SupportsWriteLogging/None (Default None)
			A class implementing SupportWriteLogging protocol. This class
			This class acts as the location to output logging information.
			The argument io_combined be used in place by default if this argument is not specified.
		"""

		self._input = io_combined if input is None else input
		self._output = io_combined if output is None else output
		self._uci = io_combined if uci is None else uci
		self._log = io_combined if log is None else log

		self._in_memory = {} 

	def __del__(self):
		del(self._input)
		del(self._output)
		del(self._uci)
		del(self._log)

	def read_uci(self, *args, **kwargs) -> UCI:
		return self._uci.read_uci()

	def write_ts(self,
			data_frame:pd.DataFrame,
			save_columns: List[str], 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None,
			*args, **kwargs) -> None:
		key = (category, operation, segment, activity)
		self._in_memory[key] = data_frame.copy(deep=True)
		
		drop_columns = [c for c in data_frame.columns if c not in save_columns ] 
		if drop_columns:
			data_frame = data_frame.drop(columns=drop_columns)

		self._output.write_ts(data_frame, category, operation, segment, activity)

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

	def write_log(self, data_frame)-> None:
		if self._log: self._log.write_log(data_frame)

	def write_versioning(self, data_frame)-> None:
		if self._log: self._log.write_versioning(data_frame)	

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
