import numpy as np
import pandas as pd
from typing import Dict

class Gener():
    """
    Partial implementation of the GENER module. 
    Currently supports OPCODES 1-7, 9-23, & 25-26
    """

    def __init__(self, segment: str, copies: Dict, geners: Dict, ddlinks: Dict, ddgener: Dict) -> None:
        self.ts_input_1 = pd.Series() # type: pd.Series
        self. ts_input_2 = pd.Series() # type: pd.Series
        self.ts_output = pd.Series()  # type: pd.Series
        self.opcode = 0               # type: int
        self.k = -1.0E30              # type: float 

        self.opcode = ddgener['OPCODE'][segment]
        if self.opcode in [9,10,11,24,25,26]:
            self.k = ddgener['PARM'][segment]

        for link in ddlinks[segment]:
            ts = pd.Series()
            if link.SVOL == 'COPY': 
                copy = copies[link.SVOLNO]
                ts = copy.get_ts(link.SMEMN,link.SMEMSB1)
                if link.MFACTOR != 1: ts *= link.MFACTOR
            elif link.SVOL == 'GENER':
                gener = geners[link.SVOLNO]
                ts = gener.get_ts()
                if link.MFACTOR != 1: ts *= link.MFACTOR
            else:
                raise NotImplementedError(f"Invalid SVOL. GENER module does not currently support reading TimeSeries for '{link.SVOL}'")
                
            if link.TGRPN == 'INPUT' and link.TMEMN == 'ONE':
                self.ts_input_1 = ts            
            elif link.TGRPN == 'INPUT' and link.TMEMN == 'TWO':
                self.ts_input_2 = ts
            else:
                raise AttributeError(f"No attribute {link.TGRPN}{link.THEMN} to assign TimeSeries. Should be either 'INPUTONE' or 'INPUTWO'")
                
        self._execute_gener()

    def get_ts(self) -> pd.Series:
        """
        Returns the result TimeSeries generated from executing the operation specified by the OPCODE.
        """
        return self.ts_output

    def _execute_gener(self) -> None:
        gener_op = getattr(self, f'_opcode{self.opcode}')
        ts_result = gener_op()
        #May need additional logic here to set default of 1.0E30 to be consistent with FORTRAN code
        self.ts_output = ts_result

    def _opcode1(self) -> pd.Series:
        return np.abs(self.ts_input_1)

    def _opcode2(self) -> pd.Series:
        return np.sqrt(self.ts_input_1)

    def _opcode3(self) -> pd.Series: 
        return np.trunc(self.ts_input_1)

    def _opcode4(self) -> pd.Series:
        return np.ceil(self.ts_input_1)

    def _opcode5(self) -> pd.Series:
        return np.floor(self.ts_input_1)

    def _opcode6(self) -> pd.Series:
        return np.log(self.ts_input_1)
    
    def _opcode7(self) -> pd.Series:
        return np.log10(self.ts_input_1)

    def _opcode8(self) -> pd.Series:
		#Not presently implemented, read UCI would need to modify to 
        #process NTERMS and COEFFS sub blocks of GENER block
        raise NotImplementedError("GENER OPCODE 8 is not currently supported")

    def _opcode9(self) -> pd.Series:
        return np.power(self.k , self.ts_input_1)

    def _opcode10(self) -> pd.Series:
        return np.power(self.ts_input_1, self.k)

    def _opcode11(self) -> pd.Series:
        return np.add(self.ts_input_1, self.k)

    def _opcode12(self) -> pd.Series:
        return np.sin(self.ts_input_1)

    def _opcode13(self) -> pd.Series:
        return np.cos(self.ts_input_1)

    def _opcode14(self) -> pd.Series:
        return np.tan(self.ts_input_1)

    def _opcode15(self) -> pd.Series:
        return np.cumsum(self.ts_input_1)

    def _opcode16(self) -> pd.Series:
        return np.add(self.ts_input_1, self.ts_input_2)

    def _opcode17(self) -> pd.Series:
        return np.subtract(self.ts_input_1, self.ts_input_2)

    def _opcode18(self) -> pd.Series:
        return np.multiply(self.ts_input_1, self.ts_input_2)

    def _opcode19(self) -> pd.Series:
        return np.divide(self.ts_input_1, self.ts_input_2)

    def _opcode20(self) -> pd.Series:
        return np.maximum(self.ts_input_1, self.ts_input_2)

    def _opcode21(self) -> pd.Series:
        return np.minimum(self.ts_input_1, self.ts_input_2)

    def _opcode22(self) -> pd.Series:
        return np.power(self.ts_input_1, self.ts_input_2)

    def _opcode23(self) -> pd.Series:
        ts_out = pd.Series(index=self.ts_input_1.index)
        s = 0        
        for idx in self.ts_input_1.index:
            result = self.ts_input_1[idx] - self.ts_input_2[idx] - s
            if result < 0:
                s = s - self.ts_input_1[idx] + self.ts_input_2[idx]
                result = 0
            else:
                s = 0
            ts_out[idx] = result
        return ts_out

    def _opcode24(self) -> pd.Series:
        #skip for now 
        #would need to figure out timeseries length component
        raise NotImplementedError("GENER OPCODE 24 is not currently supported")

    def _opcode25(self) -> pd.Series:
        return np.maximum(self.ts_input_1, self.k)

    def _opcode26(self) -> pd.Series:
        return np.minimum(self.ts_input_1, self.k)
