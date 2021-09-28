from concurrent import futures
import concurrent
import os
import inspect
import webbrowser
from HSP2tools.readWDM import get_wdm_data_set
from HSP2tools.HBNOutput import HBNOutput
from HSP2tools.HDF5 import HDF5
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy


class RegressTest(object):
    def __init__(self, compare_case:str) -> None:
        self.compare_case = compare_case
        self.test_meta = {}
        self.compare_tcodes = [2]  # control time unit for comparison
        self.compare_ids = [434]

    def _get_hbn_data(self, test_dir: str) -> None:
        sub_dirs = os.listdir(test_dir)
        hbn_files = []
        for sub_dir in sub_dirs:
            if sub_dir.__contains__('HSPFresults'):
                files = os.listdir(os.path.join(test_dir, sub_dir))
                for file in files:
                    if file.lower().endswith('.hbn'):
                        self.hspf_data = HBNOutput(os.path.join(test_dir, sub_dir, file))
                        break 

        self.hspf_data.read_data()

    def _get_hdf5_data(self, test_dir: str) -> List[HDF5]:
        sub_dirs = os.listdir(test_dir)
        h5_files = []
        for sub_dir in sub_dirs:
            if sub_dir.__contains__('HSP2results'):
                files = os.listdir(os.path.join(test_dir, sub_dir))
                for file in files:
                    if file.lower().endswith('.h5') or file.lower().endswith('.hdf'):
                        self.hsp2_data = HDF5(os.path.join(test_dir, sub_dir, file))
                        break

    def make_html_report(self, results_dict):
        """populates html table"""
        style_th = 'style="text-align:left"'
        style_header = 'style="border:1px solid; background-color:#EEEEEE"'
        
        html = f'<table style="border:1px solid">\n'

        for key in self.hspf_data.output_dictionary.keys():
            operation, activity, opn_id, tcode = key.split('_')
            if int(tcode) not in self.compare_tcodes or int(opn_id) not in self.compare_ids:
                continue 
            html += f'<tr><th colspan=5 {style_header}>{key}</th></tr>\n'
            html += f'<tr><th></th><th {style_th}>Constituent</th><th {style_th}>Max Diff</th><th>Match</th><th>Note</th></tr>\n'
            for cons in self.hspf_data.output_dictionary[key]:
                result = results_dict[(operation,activity,opn_id, cons, tcode)]
                no_data_hsp2, no_data_hspf, match, diff = result
                html += self.make_html_comp_row(cons, no_data_hsp2, no_data_hspf, match, diff)

        html += f'</table>\n'
        return html

    def make_html_comp_row(self, con:str, no_data_hsp2:bool, 
            no_data_hspf:bool, match:bool, diff:float) -> str:
        """populates each constituents rows"""
        if no_data_hsp2 or no_data_hspf:
            html = f'<tr><td>-</td><td>{con}</td><td>NA</td><td>NA</td><td>'
            html += f'{no_data_hsp2}<br>'
            html += f'{no_data_hspf}'
            html += f'</td></tr>\n'
        else:
            if match:
                match_symbol = f'<span style="font-weight:bold;color:red">X</span>'
            else:
                match_symbol = f'<span style="font-weight:bold;color:green">&#10003;</span>'
            html = f'<tr><td>-</td><td>{con}</td><td>{diff}</td><td>{match_symbol}</td><td></td></tr>\n'
        return html 

    def _run_test(self) -> str:
        futures = {}
        results_dict = {}

        #with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for key in self.hspf_data.output_dictionary.keys():
                (operation, activity, opn_id, tcode) = key.split('_')
                if int(tcode) not in self.compare_tcodes or int(opn_id) not in self.compare_ids:
                    continue 
                for cons in self.hspf_data.output_dictionary[key]:
                    params = (operation,activity,opn_id,cons,tcode)
                    futures[executor.submit(self.check_con, params)] = params 
            
            for future in as_completed(futures):
                key = futures[future]
                results_dict[key] = future.result()

        html = self.make_html_report(results_dict) 
        return html

    def check_con(self, params:Tuple[str,str,str,str,str]) -> Tuple[bool,bool,bool,float]:
        """Performs comparision of single constituent"""
        operation, activity, id, constituent, tcode = params
        print(f'    {operation}_{id}  {activity}  {constituent}\n')
        
        ts_hsp2 = self.hsp2_data.get_time_series(operation, id, constituent, activity)
        ts_hspf = self.hspf_data.get_time_series(operation, int(id), constituent, activity, 'hourly')

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
                abstol = max(abs(ts_hsp2.values.min()), abs(ts_hsp2.values.max())) * 1e-4
            elif constituent == 'QTOTAL' or constituent == 'HTEXCH' :
                abstol = max(abs(ts_hsp2.values.min()), abs(ts_hsp2.values.max())) * 1e-3    

            ts_hsp2, ts_hspf = self.validate_time_series(ts_hsp2, ts_hspf,
                self.hsp2_data, self.hspf_data, operation, activity, id, constituent)
                
            match, diff = self.compare_time_series(ts_hsp2, ts_hspf, tolerance)
        
        return (no_data_hsp2, no_data_hspf, match, diff)

    def fill_nan_and_null(self, timeseries:pd.Series, replacement_value:float = 0.0) -> pd.Series:
        """Replaces any nan or HSPF nulls -1.0e26 with provided replacement_value"""
        timeseries = timeseries.fillna(replacement_value)
        timeseries = timeseries.replace(-1.0e26, replacement_value)
        return timeseries


    def validate_time_series(self, ts_hsp2:pd.Series, ts_hspf:pd.Series, 
            hsp2_data:HDF5, hspf_data:HBNOutput, operation:str, activity:str, 
            id:str, cons:str) -> Tuple[pd.Series, pd.Series]:
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
            ts_suro_hsp2 = hsp2_data.get_time_series(operation, id, "SURO", "PWATER")
            ts_suro_hspf = hspf_data.get_time_series(operation, int(id), "SURO", "PWATER", 'hourly')
        
            idx_zero_suro_hsp2 = ts_suro_hsp2 == 0
            idx_low_suro_hsp2 = ts_suro_hsp2 < 1.0e-8
            idx_zero_suro_hspf = ts_suro_hspf == 0
            idx_low_suro_hspf = ts_suro_hspf < 1.0e-8
            
            ts_hsp2.loc[idx_zero_suro_hsp2 & idx_low_suro_hspf] = ts_hsp2.loc[idx_zero_suro_hsp2 & idx_low_suro_hspf] = 0
            ts_hsp2.loc[idx_zero_suro_hspf & idx_low_suro_hsp2] = ts_hsp2.loc[idx_zero_suro_hspf & idx_low_suro_hsp2] = 0
       
        # if volume in reach is going to zero, small concentration differences are not signficant
        if activity == 'SEDTRN' and cons in ['SSEDCLAY', 'SSEDTOT']: 
            ts_vol_hsp2 = hsp2_data.get_time_series(operation, id, "VOL", "HYDR")
            
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


    def run_test(self) -> None:
        # find all tests
        current_directory = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
        source_root_path = os.path.split(os.path.split(current_directory)[0])[0]
        test_path = os.path.join(source_root_path, "tests")

        test_dirs = os.listdir(test_path)
        for test_dir in test_dirs:
            if test_dir == self.compare_case:
                test_path = os.path.join(test_path, test_dir)
        
        self._get_hdf5_data(test_path)
        self._get_hbn_data(test_path)
        #PRT - everything above this should to initializer 

        html_file = os.path.join(test_path, 'test_report_conversion.html')
        text_file = open(html_file, "w")
        text_file.write('<html><header><h1>CONVERSION TEST REPORT</h1></header><body>\n')

        print(f'conversion test case: {self.compare_case}')
        text_file.write(f'<h3>{os.path.join(test_path, test_dir)}</h3>\n')
        one_test_report = self._run_test()
        text_file.write(one_test_report)

        text_file.write("</body></html>\n")
        text_file.close()
        print('conversion tests are done.')

        try:
            webbrowser.open_new_tab('file://' + html_file)
        except:
            print("Error writing test results to " + html_file)
