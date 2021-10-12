from datetime import time
import os
import inspect
import webbrowser
from HSP2tools.HBNOutput import HBNOutput
from HSP2tools.HDF5 import HDF5
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Union

from concurrent.futures import ThreadPoolExecutor, as_completed, thread


OperationsTuple = Tuple[str,str,str,str,str]
ResultsTuple = Tuple[bool,bool,bool,float]

class RegressTest(object):
    def __init__(self, compare_case:str, operations:List[str]=[], activities:List[str]=[], 
            tcodes:List[str] = ['2'], ids:List[str] = [], threads:int=os.cpu_count() - 1) -> None:
        self.compare_case = compare_case
        self.operations = operations
        self.activities = activities
        self.tcodes = tcodes
        self.ids = ids
        self.threads = threads

        self._init_files()

    def _init_files(self):
        current_directory = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
        source_root_path = os.path.split(os.path.split(current_directory)[0])[0]
        tests_root_dir = os.path.join(source_root_path, "tests")
        self.html_file = os.path.join(tests_root_dir, f'HSPF_HSP2_{self.compare_case}.html')

        test_dirs = os.listdir(tests_root_dir)
        for test_dir in test_dirs:
            if test_dir == self.compare_case:
                test_root = os.path.join(tests_root_dir, test_dir)
        
        self._get_hdf5_data(test_root)
        self._get_hbn_data(test_root)

    def _get_hbn_data(self, test_dir: str) -> None:
        sub_dir = os.path.join(test_dir, 'HSPFresults')
        self.hspf_data_collection = {}
        for file in os.listdir(sub_dir):
            if file.lower().endswith('.hbn'):
                hspf_data = HBNOutput(os.path.join(test_dir, sub_dir, file))
                hspf_data.read_data()
                for key in hspf_data.output_dictionary.keys():
                    self.hspf_data_collection[key] = hspf_data

    def get_hspf_time_series(self, ops:OperationsTuple) -> Union[pd.Series,None]:
        operation, activity, id, constituent, tcode = ops
        key = f'{operation}_{activity}_{id}_{tcode}'
        hspf_data = self.hspf_data_collection[key]     
        series = hspf_data.get_time_series(operation, int(id), constituent, activity, 'hourly')
        return series        

    def _get_hdf5_data(self, test_dir: str) -> None:
        sub_dir = os.path.join(test_dir, 'HSP2results')
        for file in os.listdir(sub_dir):
            if file.lower().endswith('.h5') or file.lower().endswith('.hdf'):
                self.hsp2_data = HDF5(os.path.join(sub_dir, file))
                break

    def should_compare(self, operation:str, activity:str, id:str, tcode:str) -> bool:
        if len(self.operations) > 0 and operation not in self.operations:
            return False
        if len(self.activities) > 0 and activity not in self.activities:
            return False
        if len(self.ids) > 0 and id not in self.ids:
            return False
        if len(self.tcodes) > 0 and tcode not in self.tcodes:
            return False
        return True

    def generate_report(self, file:str, results: Dict[OperationsTuple,ResultsTuple]) -> None:
        html = self.make_html_report(results)
        self.write_html(file,html)
        webbrowser.open_new_tab('file://' + file)

    def make_html_report(self, results_dict:Dict[OperationsTuple,ResultsTuple]) -> str:
        """populates html table"""
        style_th = 'style="text-align:left"'
        style_header = 'style="border:1px solid; background-color:#EEEEEE"'
        
        html = f'<html><header><h1>CONVERSION TEST REPORT</h1></header><body>\n'
        html += f'<table style="border:1px solid">\n'

        for key in self.hspf_data_collection.keys():
            operation, activity, opn_id, tcode = key.split('_')
            if not self.should_compare(operation, activity, opn_id, tcode):
                continue 
            html += f'<tr><th colspan=5 {style_header}>{key}</th></tr>\n'
            html += f'<tr><th></th><th {style_th}>Constituent</th><th {style_th}>Max Diff</th><th>Match</th><th>Note</th></tr>\n'
            hspf_data = self.hspf_data_collection[key]
            for cons in hspf_data.output_dictionary[key]:
                result = results_dict[(operation,activity,opn_id, cons, tcode)]
                no_data_hsp2, no_data_hspf, match, diff = result
                html += self.make_html_comp_row(cons, no_data_hsp2, no_data_hspf, match, diff)

        html += f'</table>\n'
        html += f"</body></html>\n"
        return html

    def make_html_comp_row(self, con:str, no_data_hsp2:bool, 
            no_data_hspf:bool, match:bool, diff:float) -> str:
        """populates each constituents rows"""
        if no_data_hsp2 or no_data_hspf:
            html = f'<tr><td>-</td><td>{con}</td><td>NA</td><td>NA</td><td>'
            html += f'{"Not in HSP2" if no_data_hsp2 else ""}<br>'
            html += f'{"Not in HSPF" if no_data_hspf else ""}'
            html += f'</td></tr>\n'
        else:
            if match:
                match_symbol = f'<span style="font-weight:bold;color:green">&#10003;</span>'
            else:
                match_symbol = f'<span style="font-weight:bold;color:red">X</span>'
            html = f'<tr><td>-</td><td>{con}</td><td>{diff}</td><td>{match_symbol}</td><td></td></tr>\n'
        return html 

    def write_html(self, file:str, html:str) -> None:
        with open(file, 'w') as f:
            f.write(html)

    def run_test(self) -> Dict[OperationsTuple,ResultsTuple]:
        futures = {}
        results_dict = {}

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            for key in self.hspf_data_collection.keys():
                (operation, activity, opn_id, tcode) = key.split('_')
                if not self.should_compare(operation, activity, opn_id, tcode):
                    continue 
                hspf_data = self.hspf_data_collection[key]
                for cons in hspf_data.output_dictionary[key]:
                    params = (operation,activity,opn_id,cons,tcode)
                    futures[executor.submit(self.check_con, params)] = params 
            
            for future in as_completed(futures):
                key = futures[future]
                results_dict[key] = future.result()

        return results_dict 

    def check_con(self, params:OperationsTuple) -> ResultsTuple:
        """Performs comparision of single constituent"""
        operation, activity, id, constituent, tcode = params
        print(f'    {operation}_{id}  {activity}  {constituent}\n')
        
        ts_hsp2 = self.hsp2_data.get_time_series(operation, id, constituent, activity)
        ts_hspf = self.get_hspf_time_series(params)

        no_data_hsp2 = ts_hsp2 is None
        no_data_hspf = ts_hspf is None
        
        if no_data_hsp2 or no_data_hspf:
            return (no_data_hsp2, no_data_hspf, False, 0)
        else:
            #Special case, for some parameters (e.g PLANK.BENAL1) HSPF results look to be array.
            #Working assumption is that only the first index of that array are the values of interest.
            if len(ts_hspf.shape) > 1:
                ts_hspf = ts_hspf.iloc[:,0]
           
            tolerance = 1e-2
            # if heat related term, compute special tolerance
            if constituent == 'IHEAT' or constituent == 'ROHEAT' or constituent.startswith('OHEAT') or \
                constituent == 'QSOLAR' or constituent == 'QLONGW' or constituent == 'QEVAP' or \
                constituent == 'QCON' or constituent == 'QPREC' or constituent == 'QBED':
                tolerance = max(abs(ts_hsp2.values.min()), abs(ts_hsp2.values.max())) * 1e-4
            elif constituent == 'QTOTAL' or constituent == 'HTEXCH' :
                tolerance = max(abs(ts_hsp2.values.min()), abs(ts_hsp2.values.max())) * 1e-3    

            ts_hsp2, ts_hspf = self.validate_time_series(ts_hsp2, ts_hspf, operation, activity, id, constituent)
                
            match, diff = self.compare_time_series(ts_hsp2, ts_hspf, tolerance)
        
        return (no_data_hsp2, no_data_hspf, match, diff)

    def fill_nan_and_null(self, timeseries:pd.Series, replacement_value:float = 0.0) -> pd.Series:
        """Replaces any nan or HSPF nulls -1.0e30 with provided replacement_value"""
        timeseries = timeseries.fillna(replacement_value)
        timeseries = timeseries.where(timeseries > -1.0e30, replacement_value)
        return timeseries

    def validate_time_series(self, ts_hsp2:pd.Series, ts_hspf:pd.Series, operation:str, 
            activity:str, id:str, cons:str) -> Tuple[pd.Series, pd.Series]:
        """ validates a corrects time series to avoid false differences """
   
        # In some test cases it looked like HSP2 was executing for a single extra time step 
        # Trim h5 (HSP2) results to be same length as hbn (HSPF)
        # This is a bandaid to get testing working. Long term should identify why HSP2 runs for additional time step.
        if len(ts_hsp2) > len(ts_hspf):
            ts_hsp2 = ts_hsp2[0:len(ts_hspf)]

        ts_hsp2 = self.fill_nan_and_null(ts_hsp2)
        ts_hspf = self.fill_nan_and_null(ts_hspf)
        
        ### special cases
        # if tiny suro in one and no suro in the other, don't trigger on suro-dependent numbers
        if activity == 'PWTGAS' and cons in ['SOTMP', 'SODOX', 'SOCO2']:  
            ts_suro_hsp2 = self.hsp2_data.get_time_series(operation, id, 'SURO', 'PWATER')
            ts_suro_hsp2 = self.fill_nan_and_null(ts_suro_hsp2)            
            ts_suro_hspf = self.get_hspf_time_series((operation, 'PWATER', id, 'SURO', 2))
            ts_suro_hspf = self.fill_nan_and_null(ts_suro_hspf)

        
            idx_zero_suro_hsp2 = ts_suro_hsp2 == 0
            idx_low_suro_hsp2 = ts_suro_hsp2 < 1.0e-8
            idx_zero_suro_hspf = ts_suro_hspf == 0
            idx_low_suro_hspf = ts_suro_hspf < 1.0e-8
            
            ts_hsp2.loc[idx_zero_suro_hsp2 & idx_low_suro_hspf] = ts_hspf.loc[idx_zero_suro_hsp2 & idx_low_suro_hspf] = 0
            ts_hspf.loc[idx_zero_suro_hspf & idx_low_suro_hsp2] = ts_hsp2.loc[idx_zero_suro_hspf & idx_low_suro_hsp2] = 0

        # if volume in reach is going to zero, small concentration differences are not signficant
        if (activity == 'SEDTRN' and cons in ['SSEDCLAY', 'SSEDTOT']) or \
                (activity == 'NUTRX' and cons in ['TAMCONCDIS', 'NH4CONCDIS', 'NH3CONCDIS', 'NO3CONCDIS', 'NO2CONCDIS', 'PO4CONCDIS']):
            ts_vol_hsp2 = self.hsp2_data.get_time_series(operation, id, "VOL", "HYDR")
            ts_vol_hsp2 = self.fill_nan_and_null(ts_vol_hsp2)
            
            idx_low_vol = ts_vol_hsp2 < 1.0e-4
            ts_hsp2.loc[idx_low_vol] = ts_hsp2.loc[idx_low_vol] = 0
            ts_hspf.loc[idx_low_vol] = ts_hspf.loc[idx_low_vol] = 0
        ### end special cases 

        return ts_hsp2, ts_hspf
       
    def compare_time_series(self, ts_hsp2:pd.Series, ts_hspf:pd.Series, tol:float) -> Tuple[bool, float]:
        
        max_diff1 = (ts_hspf.values - ts_hsp2.values).max()
        max_diff2 = (ts_hsp2.values - ts_hspf.values).max()
        max_diff = max(max_diff1, max_diff2)
        
        match = np.allclose(ts_hspf, ts_hsp2, rtol=1e-2, atol=tol, equal_nan=False)
        return (match, max_diff)