import HSP2tools
import pandas as pd
import os
from typing import Union, Dict, Tuple

class HDF5:
    def __init__(self, file_name:str) -> None:
        self.file_name = file_name
        self.aliases = self._read_aliases_csv()

    def _read_aliases_csv(self) -> Dict[Tuple[str,str,str],str]:
        datapath = os.path.join(HSP2tools.__path__[0], 'data', 'HBNAliases.csv')
        df = pd.read_csv()
        df = df.set_index(['operation','activity','hspf_name'])
        df_dict = df['hsp2'].to_dict()
        return df_dict

    def get_time_series(self, operation:str, id:str, constituent:str, activity:str) -> Union[pd.Series, None]:
        """Reads timeseries from HDF5 are returns the desired."""
        operation = operation.upper()
        constituent = constituent.upper()
        activity = activity.upper()

        #We still need a special case for RCHES/GQUAL, IMPLAND/IQUAL and PERLAND/PQUAL

        key = f'RESULTS/{operation}_{operation[0]}{id}/{activity}/table'
        try:
            with pd.HDFStore(self.file_name, 'r') as store:
                df = pd.read_hdf(store, key=key)
                df['index'] = pd.to_datetime(df['index'], unit='ns')
                df = df.set_index('index')
            
            if constituent in df.columns:
                return df[constituent]
            else:
                constituent_alias = self.aliases[(operation,activity,constituent)]
                return df[constituent_alias]
        except KeyError:
            #Meaning pathway to table or the requested constituent is not in the HDF5 file
            return None
