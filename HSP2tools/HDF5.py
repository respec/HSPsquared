from struct import unpack

import h5py
from numpy import fromfile
from pandas import DataFrame
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

class HDF5:
    def __init__(self, file_name):
        self.data_frames = []
        self.file_name = file_name
        self.simulation_duration_count = 0
        self.summary = []
        self.summarycols = []
        self.summaryindx = []

        self.time_index = [] # this will be shared with all time series
        self.data_dictionary = {}
        # self.dd_implnd = {}
        # self.dd_perlnd = {}
        # self.dd_rchres = {}
        self.dd_key_separator = ':'

        self.tcodes = {1: 'Minutely', 2: 'Hourly', 3: 'Daily', 4: 'Monthly', 5: 'Yearly'}

    def open_output(self):
        """
        Reads ALL data dictionary from hdf5_file's /RESULTS group

        Parameters
        ----------
        hdf5_file : str
            Name/path of HBN created by HSPF.

        Populate
        -------
        data_dictionary : {}
            Summary information of data found in HDF5 file HSP2 outputs
        """
        with h5py.File(self.file_name, "r") as f:
            str_starttime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[1].astype('datetime64[D]')
            str_endtime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[2].astype('datetime64[D]')
            start_time = pd.to_datetime(str_starttime)
            end_time = pd.to_datetime(str_endtime)
            section = f.get('/RESULTS')
            opn_keys = list(section.keys())
            for opn_key in opn_keys:
                opn_output_grp = section[opn_key]   # e.g. opn_key = IMPLND_I001
                opn_output_keys = list(opn_output_grp.keys())
                for opn_output_key in opn_output_keys:
                    dd_key = opn_key + self.dd_key_separator + opn_output_key
                    data_table = section[opn_key][opn_output_key]['table']  # e.g. opn_output_key = IQUAL
                    all_table_attrs = list(data_table.attrs)
                    field_indices = {}
                    for table_attr in all_table_attrs:
                        str_attr_value = ''
                        try:
                            str_attr_value = data_table.attrs[table_attr].astype('unicode') # e.g. table_attr = FIELD_2_NAME
                        except:
                            str_attr_value = ''
                        if (not str_attr_value == '') and table_attr.startswith('FIELD') and table_attr.endswith('NAME'):
                            # convert FIELD_n_NAME to lookup of field index <-> field name'
                            name_parts = table_attr.split('_')
                            field_indices[int(name_parts[1])] = str_attr_value
                    self.data_dictionary[dd_key] = field_indices
                    self.data_dictionary[dd_key + f'{self.dd_key_separator}values'] = None
                    if len(self.time_index) == 0:
                        # alternatively, could construct the time index from the start and end times above
                        self.time_index = list(pd.date_range(start_time, end_time, freq='H')[:-1])  # issue in HDF5 table!
                        '''
                        for row in range(data_table.attrs['NROWS']):
                            dt = pd.to_datetime(data_table.fields('index')[row].astype('datetime64[D]'))
                            self.time_index.append(dt)
                        '''
                    pass
                pass
            pass
        pass

    def screen_dd_key(self, opn_type, opn_ids):
        dd_keys_to_read = []
        key_prefix = opn_type
        if opn_type == 'IMPLND':
            key_prefix += '_I'
        elif opn_type == 'PERLND':
            key_prefix += '_P'
        elif opn_type == 'RCHRES':
            key_prefix += '_R'

        for key in self.data_dictionary.keys():
            if not key.startswith(opn_type):
                continue
            if key.endswith('values'):
                continue
            parts = key.split(self.dd_key_separator)
            try:
                opn_id = int(parts[0][len(key_prefix):])
                if opn_ids is None or len(opn_ids) == 0:
                    dd_keys_to_read.append(key)
                elif opn_id in opn_ids:
                    dd_keys_to_read.append(key)
            except:
                pass

        return dd_keys_to_read

    def read_output_from_table(self, table_key):
        (opn_key, activity_key) = table_key.split(self.dd_key_separator)
        mapn = []
        mapn_keys = list(self.data_dictionary[table_key].keys())
        mapn_keys.sort()
        for mapn_key in mapn_keys:
            mapn.append(self.data_dictionary[table_key][mapn_key])
        with h5py.File(self.file_name, "r") as f:
            str_starttime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[1].astype('datetime64[D]')
            str_endtime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[2].astype('datetime64[D]')
            start_time = pd.to_datetime(str_starttime)
            end_time = pd.to_datetime(str_endtime)
            section = f.get('/RESULTS')
            data_table = section[opn_key][activity_key]['table']  # e.g. activity_key = IQUAL
            data_table_rows = list(data_table)
            rows = []
            for row in data_table_rows:
                rows.append(list(row)[1:])
            self.data_dictionary[table_key + f'{self.dd_key_separator}values'] = \
                DataFrame(rows, index=self.time_index, columns=mapn[1:])

    def read_output(self, opn_type, opn_ids=None):
        if len(self.data_dictionary) == 0:
            return
        dd_keys_to_read = self.screen_dd_key(opn_type, opn_ids)
        for dd_key_to_read in dd_keys_to_read:
            self.read_output_from_table(dd_key_to_read)

    def get_time_series(self, name, duration):
        """
        get a single time series based on:
        1. constituent name
        2. duration: yearly, monthly, full (default is 'full' simulation duration)
        """
        search_shape = self.simulation_duration_count
        if duration == 'yearly':
            search_shape = 366
        elif duration == 'monthly':
            search_shape = 12

        for data_frame in self.data_frames:
            for key in data_frame.keys():
                if key == name and data_frame[key].shape[0] == search_shape:
                    return data_frame[key]

        return None

    @staticmethod
    def save_time_series_to_file(file_name, time_series):
        with open(file_name, 'w+') as f:
            for row in range(len(time_series.index)):
                dt = time_series.index[row]
                dv = time_series.values[row]
                # f.write(f'{dt},{"{:.2f}".format(dv)}\n')
                f.write(f'{dt},{dv}\n')