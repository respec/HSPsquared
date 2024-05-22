from hsp2.hsp2.utilities import get_timeseries
import pandas as pd
from typing import List, Dict

from hsp2.hsp2io.protocols import SupportsReadTS

class Copy():
    """
    Partial implementation of the COPY module. 
    In HSPF, COPY module supports ability able to 'copy' in this case output timeseries to 
    locations specified in NETWORK block or to EXTERNAL SOURCES. 
    This functionality is not currently implemented, presently only loading from EXT SOURCES
    """

    def __init__(self, io:SupportsReadTS, sim_info: Dict, ext_sources: List) -> None:
        self._ts = {}
        self._ts['MEAN'] = {}
        self._ts['POINT'] = {}

        ts = get_timeseries(io, ext_sources, sim_info)
        for source in ext_sources:
            themn = source.TMEMN
            themsb = source.TMEMSB
            self.set_ts(ts[f'{themn}{themsb}'], themn, themsb)

    def set_ts(self, ts: pd.Series, themn: str, themsb: str) -> None:
        """Set the provided timeseries to ts dictionary
        
        ts: pd.Series
            pandas Series class instance corresponding to a timeseries
        tmemn: str, {'MEAN', 'POINT'} 
            Target member name, specifies if target timeseries is in mean-valued 
            or point-valued dictionaries
        tmemsb: str, 
            Target member name subscripts, acts as key for mean-valued and point-valued dictionaries
            Original HSPF restricts this to 0-20 but no limit enforced in HSP2
        """
        self._ts[themn][themsb] = ts

    def get_ts(self, tmemn: str, tmemsb: str) -> pd.Series:
        """Gets the specified timeseries from copy class instance based
        
        tmemn: str, {'MEAN', 'POINT'} 
            Target member name, specifies if target timeseries is in mean-valued 
            or point-valued dictionaries
        tmemsb: str, 
            Target member name subscripts, acts as key for mean-valued and point-valued dictionaries
            Original HSPF restricts this to 0-20 but no limit enforced in HSP2
        """
        return self._ts[tmemn][tmemsb]
