from weakref import finalize

from pandas.core.frame import DataFrame
from hsp2 import hsp2tools
import pandas as pd
import os
from typing import Union, Dict, Tuple #, final

#turns out that HDF5 is not threading safe, need to implement manual locking
from threading import Lock

class HDF5:

    REQUIRES_MAPPING = ['GQUAL','CONS','IQUAL','PQUAL']

    def __init__(self, file_name:str) -> None:
        self.file_name = file_name
        self.aliases = self._read_aliases_csv()
        self.data = {}
        self.lock = Lock()

        self.gqual_prefixes = self._read_gqual_mapping()
        self.cons_prefixes = self._read_cons_mapping()
        self.iqual_prefixes = self._read_iqual_mapping()
        self.pqual_prefixes = self._read_pqual_mapping()

    def _read_nqual_mapping(self, key:str, target_col:str, nquals:int = 10) -> Dict[str,str]:
        """Some modules, like GQUAL, allow for number which corresponds to the consistent
        being modeled. However which number is assoicated with which parameter changes
        based on the UCI file. Need to read from specification tables
        """
        dict_mappings = {}
        for i in range(1,nquals):
            try:
                with pd.HDFStore(self.file_name,'r') as store:
                    if key.endswith('IQUAL') or key.endswith('PQUAL'):
                        df = pd.read_hdf(store, f'{key}{i}'+'/FLAGS')
                    else:
                        df = pd.read_hdf(store,f'{key}{i}')
                    row = df.iloc[0]
                    gqid = row[target_col]
                    dict_mappings[gqid] = str(i)
            except KeyError:
                #Mean no nqual number (e.g. GQUAL3) for this run
                pass
        return dict_mappings

    def _read_gqual_mapping(self) -> Dict[str,str]:
        return self._read_nqual_mapping(R'RCHRES/GQUAL/GQUAL', 'GQID', 7)

    def _read_cons_mapping(self) -> Dict[str,str]:
        return self._read_nqual_mapping(R'RCHRES/CONS/CONS','CONID', 7)

    def _read_iqual_mapping(self) -> Dict[str,str]:
        return self._read_nqual_mapping(R'IMPLND/IQUAL/IQUAL','QUALID', 10)

    def _read_pqual_mapping(self) -> Dict[str,str]:
        return self._read_nqual_mapping(R'PERLND/PQUAL/PQUAL','QUALID', 10)

    def _read_aliases_csv(self) -> Dict[Tuple[str,str,str],str]:
        datapath = os.path.join(hsp2tools.__path__[0], 'data', 'HBNAliases.csv')
        df = pd.read_csv(datapath)
        df = df.set_index(['operation','activity','hspf_name'])
        df_dict = df['hsp2_name'].to_dict()
        return df_dict

    def get_time_series(self, operation:str, id:str, constituent:str, activity:str) -> Union[pd.Series, None]:
        """Reads timeseries from HDF5 are returns the desired."""
        operation = operation.upper()
        constituent = constituent.upper()
        activity = activity.upper()

        constituent_prefix = ''
        if activity in self.REQUIRES_MAPPING:
            constituent_prefix = ''
            prefix_dict = getattr(self, f'{activity.lower()}_prefixes')
            for key, value in prefix_dict.items():
                if activity == 'PQUAL' or activity == 'IQUAL':
                    if constituent.endswith(key):
                        constituent_prefix = f'{activity}{value}_'
                        constituent = constituent.replace(key,'')
                else:
                    if constituent.startswith(key):
                        constituent_prefix = f'{activity}{value}_'
                        constituent = constituent.replace(key,'')

        key = (operation,id,activity)
        try:
            self.lock.acquire()
            if key not in self.data.keys():
                self.data[key] = self._read_table(operation, id, activity)
            self.lock.release()

            df = self.data[key]
            if constituent_prefix + constituent in df.columns:
                return df[constituent_prefix + constituent]
            else:
                constituent_alias = self.aliases[(operation,activity,constituent)]
                return df[constituent_prefix + constituent_alias]
        except KeyError:
            return None

    def _read_table(self, operation:str, id:str, activity:str) -> pd.DataFrame:
        key = f'RESULTS/{operation}_{operation[0]}{id}/{activity}/table'

        try:
            with pd.HDFStore(self.file_name, 'r') as store:
                df = pd.read_hdf(store, key=key)
                df['index'] = pd.to_datetime(df['index'], unit='ns')
                df = df.set_index('index')
                return df
        except KeyError:
            return pd.DataFrame()
