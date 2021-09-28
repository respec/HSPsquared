from weakref import finalize

from pandas.core.frame import DataFrame
import HSP2tools
import pandas as pd
import os
from typing import Union, Dict, Tuple, final

#turns out that HDF5 is not threading safe, need to implement manual locking
from threading import Lock

class HDF5:
    def __init__(self, file_name:str) -> None:
        self.file_name = file_name
        self.aliases = self._read_aliases_csv()
        self.data = {}
        self.lock = Lock()

    def _read_aliases_csv(self) -> Dict[Tuple[str,str,str],str]:
        datapath = os.path.join(HSP2tools.__path__[0], 'data', 'HBNAliases.csv')
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

        #We still need a special case for RCHES/GQUAL, IMPLAND/IQUAL and PERLAND/PQUAL
        if activity == 'GQUAL': 
            constituent_prefix = 'GQUAL1_'

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