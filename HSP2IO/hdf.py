import pandas as pd
from pandas.io.pytables import read_hdf
from HSP2IO.protocols import Category
from collections import defaultdict
from typing import Union, Any

from HSP2.uci import UCI

class HDF5():

	def __init__(self, file_path:str) -> None:
		self.file_path = file_path
		self._store = pd.HDFStore(file_path)
		None

	def __del__(self):
		self._store.close()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, trace):
		self.__del__()

	def read_uci(self) -> UCI:
		"""Read UCI related tables
		
		Parameters: None
		
		Returns: UCITuple

		"""
		uci = UCI()
		print("Called read_uci()")
		for path in self._store.keys():   # finds ALL data sets into HDF5 file
			op, module, *other = path[1:].split(sep='/', maxsplit=3)
			s = '_'.join(other)
			if op == 'CONTROL':
				if module =='GLOBAL':
					temp = self._store[path].to_dict()['Info']
					uci.siminfo['start'] = pd.Timestamp(temp['Start'])
					uci.siminfo['stop']  = pd.Timestamp(temp['Stop'])
					uci.siminfo['units'] = 1
					if 'Units' in temp:
						if int(temp['Units']):
							uci.siminfo['units'] = int(temp['Units'])
				elif module == 'LINKS':
					for row in self._store[path].fillna('').itertuples():
						if row.TVOLNO != '':
							uci.ddlinks[f'{row.TVOLNO}'].append(row)
						else:
							uci.ddlinks[f'{row.TOPFST}'].append(row)

				elif module == 'MASS_LINKS':
					for row in self._store[path].replace('na','').itertuples():
						uci.ddmasslinks[row.MLNO].append(row)
				elif module == 'EXT_SOURCES':
					for row in self._store[path].replace('na','').itertuples():
						uci.ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
				elif module == 'OP_SEQUENCE':
					uci.opseq = self._store[path]
			elif op in {'PERLND', 'IMPLND', 'RCHRES'}:
				for id, vdict in self._store[path].to_dict('index').items():
					uci.uci[(op, module, id)][s] = vdict
			elif op == 'GENER':
				for row in self._store[path].itertuples():
					if len(row.OPNID.split()) == 1:
						start = int(row.OPNID)
						stop = start
					else:
						start, stop = row.OPNID.split()
					for i in range(int(start), int(stop)+1): uci.ddgener[module][f'G{i:03d}'] = row[2]
			elif op == 'FTABLES':
				uci.ftables[module] = self._store[path]
			elif op == 'SPEC_ACTIONS':
				uci.specactions[module] = self._store[path]
			elif op == 'MONTHDATA':
				if not uci.monthdata: uci.monthdata = {}
				uci.monthdata[f'{op}/{module}'] = self._store[path]
		return uci

	def read_ts(self, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> pd.DataFrame:
		try:
			path = ''
			if category == category.INPUTS:
				path = f'TIMESERIES/{segment}'
			elif category == category.RESULTS:
				path = f'RESULTS/{operation}_{segment}/{activity}'
			return read_hdf(self._store, path)
		except KeyError:
			return pd.DataFrame()

	def write_ts(self, 
			data_frame:pd.DataFrame, 
			category: Category,
			operation:str, 
			segment:str, 
			activity:str, 
			*args:Any, 
			**kwargs:Any) -> None:
		"""Saves timeseries to HDF5"""
		path=f'{operation}_{segment}/{activity}'
		if category:
			path = 'RESULTS/' + path
		complevel = None 
		if 'compress' in kwargs:
			if kwargs['compress']:
				complevel = 9
		data_frame.to_hdf(self._store, path, format='t', data_columns=True, complevel=complevel)
		#data_frame.to_hdf(self._store, path)

	def write_log(self, hsp2_log:pd.DataFrame) -> None:
		hsp2_log.to_hdf(self._store, 'RUN_INFO/LOGFILE', data_columns=True, format='t')

	def write_versioning(self, versioning:pd.DataFrame) -> None:
		versioning.to_hdf(self._store, 'RUN_INFO/VERSIONS', data_columns=True, format='t')


