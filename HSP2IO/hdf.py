import pandas as pd
from pandas.io.pytables import read_hdf
from HSP2IO.protocols import UCITuple, Category
from collections import defaultdict
from typing import Union, Any



class HDF5():

	def __init__(self, file_path:str) -> None:
		self.file_path = file_path
		self._store = pd.HDFStore(file_path)
		None

	def __del__(self):
		self._store.close()

	def read_uci(self) -> UCITuple:
		"""Read UCI related tables
		
		Parameters: None
		
		Returns: UCITuple

		"""
		uci = defaultdict(dict)
		ddlinks = defaultdict(list)
		ddmasslinks = defaultdict(list)
		ddext_sources = defaultdict(list)
		ddgener =defaultdict(dict)
		siminfo = {}
		opseq = 0

		for path in self._store.keys():   # finds ALL data sets into HDF5 file
			op, module, *other = path[1:].split(sep='/', maxsplit=3)
			s = '_'.join(other)
			if op == 'CONTROL':
				if module =='GLOBAL':
					temp = self._store[path].to_dict()['Info']
					siminfo['start'] = pd.Timestamp(temp['Start'])
					siminfo['stop']  = pd.Timestamp(temp['Stop'])
					siminfo['units'] = 1
					if 'Units' in temp:
						if int(temp['Units']):
							siminfo['units'] = int(temp['Units'])
				elif module == 'LINKS':
					for row in self._store[path].fillna('').itertuples():
						if row.TVOLNO != '':
							ddlinks[f'{row.TVOLNO}'].append(row)
						else:
							ddlinks[f'{row.TOPFST}'].append(row)

				elif module == 'MASS_LINKS':
					for row in self._store[path].replace('na','').itertuples():
						ddmasslinks[row.MLNO].append(row)
				elif module == 'EXT_SOURCES':
					for row in self._store[path].replace('na','').itertuples():
						ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
				elif module == 'OP_SEQUENCE':
					opseq = self._store[path]
			elif op in {'PERLND', 'IMPLND', 'RCHRES'}:
				for id, vdict in self._store[path].to_dict('index').items():
					uci[(op, module, id)][s] = vdict
			elif op == 'GENER':
				for row in self._store[path].itertuples():
					if len(row.OPNID.split()) == 1:
						start = int(row.OPNID)
						stop = start
					else:
						start, stop = row.OPNID.split()
					for i in range(int(start), int(stop)+1): ddgener[module][f'G{i:03d}'] = row[2]
		return (opseq, ddlinks, ddmasslinks, ddext_sources, ddgener, uci, siminfo)

	def read_timeseries(self, 
			category:Category,
			operation:Union[str,None]=None, 
			segment:Union[str,None]=None, 
			activity:Union[str,None]=None) -> pd.DataFrame:
		path = ''
		if category == category.INPUTS:
			path = f'TIMESERIES/{segment}'
		elif category == category.RESULTS:
			path = f'RESULTS/{operation}_{segment}/{activity}'
		return read_hdf(self._store, path)

	def write_timeseries(self, 
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

