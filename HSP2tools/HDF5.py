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
        self.start_time = None
        self.end_time = None

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
            if self.start_time is None or self.end_time is None:
                str_starttime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[1].astype('datetime64[D]')
                str_endtime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[2].astype('datetime64[D]')
                self.start_time = pd.to_datetime(str_starttime)
                self.end_time = pd.to_datetime(str_endtime)
                if len(self.time_index) == 0:
                    self.time_index = list(
                        pd.date_range(self.start_time, self.end_time, freq='H')[:-1])  # issue in HDF5 table!

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
                            str_attr_value = data_table.attrs[table_attr].astype(
                                'unicode')  # e.g. table_attr = FIELD_2_NAME
                        except:
                            str_attr_value = ''
                        if (not str_attr_value == '') and table_attr.startswith('FIELD') and table_attr.endswith('NAME'):
                            # convert FIELD_n_NAME to lookup of field index <-> field name'
                            name_parts = table_attr.split('_')
                            field_indices[int(name_parts[1])] = str_attr_value
                    self.data_dictionary[dd_key] = field_indices
                    self.data_dictionary[dd_key + f'{self.dd_key_separator}values'] = None
                    '''
                    if len(self.time_index) == 0:
                        # alternatively, could construct the time index from the start and end times above
                        self.time_index = list(
                            pd.date_range(self.start_time, self.end_time, freq='H')[:-1])  # issue in HDF5 table!
                        # reading row by row is VERY VERY SLOW! so this is left here to warn you.
                        # for row in range(data_table.attrs['NROWS']):
                        #     dt = pd.to_datetime(data_table.fields('index')[row].astype('datetime64[D]'))
                        #     self.time_index.append(dt)
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

    def find_output_activity(self, constituent):
        # has to search for it based on constituent name
        # assuming all of the output items are unique in any given operation
        if self.data_dictionary is None or len(self.data_dictionary) == 0:
            return None
        for key in self.data_dictionary.keys():
            if key.endswith('values'):
                continue
            name_indices = self.data_dictionary[key].keys()
            for name_index in name_indices:
                if self.data_dictionary[key][name_index] == constituent.upper():
                    (opn_key, activity) = key.split(self.dd_key_separator)
                    return activity
        return None

    def get_time_series(self, operation, id, constituent, activity=None):
        """
        get a single time series based on:
        1.   operation: e.g. IMPLND, PERLND, RCHRES
        2.          id: e.g. 1, 2, 3, ...
        3. constituent: e.g. SUPY, PERO, SOQUAL etc
        4.    activity: e.g. IQUAL, IWATER, PWATER, PWTGAS, SNOW etc, could leave blank, the program will look for it

        the above inputs will be used to build data_table_key: e.g. IMPLND_I001:IQUAL
        5. duration: yearly, monthly, full (default is 'full' simulation duration) <-- hdf5 output currently only has 2
        """
        data_table_key = operation.upper()
        key_opn = operation.upper()
        key_id = f'{id:03}'

        if activity is None:
            activity = self.find_output_activity(constituent)

        key_act = activity.upper()
        if key_opn == 'IMPLND':
            data_table_key += f'_I{key_id}' + self.dd_key_separator + key_act
        elif key_opn == 'PERLND':
            data_table_key += f'_P{key_id}' + self.dd_key_separator + key_act
        elif key_opn == 'RCHRES':
            data_table_key += f'_R{key_id}' + self.dd_key_separator + key_act

        data_value_key = f'{data_table_key}{self.dd_key_separator}values'
        if data_value_key in self.data_dictionary:
            if self.data_dictionary[data_value_key] is None:
                self.read_output_from_table(data_table_key)
        else:
            return None

        # the data frames in the in-memory collection for a table
        # don't have the first 'index' column from the h5 files
        # so the actual index in the in-memory collection will be (df_index - 1)
        # regardless, the collection is keyed on constituent name, so it's easy to find.
        # this search is just to be double sure that the constituent is legit
        df_index = -1
        for key in self.data_dictionary[data_table_key].keys():
            if operation.upper() == 'IMPLND' and key_act == 'IQUAL':
                if self.data_dictionary[data_table_key][key].endswith(constituent.upper()):
                    df_index = key
            elif self.data_dictionary[data_table_key][key] == constituent.upper():
                df_index = key

        if df_index >= 0:
            return self.data_dictionary[data_value_key][constituent.upper()]
        else:
            return None

    @staticmethod
    def save_time_series_to_file(file_name, time_series):
        with open(file_name, 'w+') as f:
            for row in range(len(time_series.index)):
                dt = time_series.index[row]
                dv = time_series.values[row]
                # f.write(f'{dt},{"{:.2f}".format(dv)}\n')
                f.write(f'{dt},{dv}\n')