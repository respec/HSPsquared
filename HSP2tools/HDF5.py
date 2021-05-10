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
        self.dd_key_implnd_iqual_ids = 'IMPLND_IQUAL_ID'
        self.dd_key_rchres_cons_ids = 'RCHRES_CONS_ID'
        self.dd_key_rchres_gqual_ids = 'RCHRES_GQUAL_ID'
        self.dd_key_rchres_gqual_alias = {}
        self.dd_key_rchres_sedtrn_alias = {}
        self.start_time = None
        self.end_time = None

        self.tcodes = {1: 'Minutely', 2: 'Hourly', 3: 'Daily', 4: 'Monthly', 5: 'Yearly'}

    def set_qual_alias_to_hspf(self):
        self.dd_key_rchres_gqual_alias['ADQAL1'] = 'ADQALSUSPSAND'
        self.dd_key_rchres_gqual_alias['ADQAL2'] = 'ADQALSUSPSILT'
        self.dd_key_rchres_gqual_alias['ADQAL3'] = 'ADQALSUSPCLAY'
        self.dd_key_rchres_gqual_alias['ADQAL4'] = 'ADQALBEDSAND'
        self.dd_key_rchres_gqual_alias['ADQAL5'] = 'ADQALBEDSILT'
        self.dd_key_rchres_gqual_alias['ADQAL6'] = 'ADQALBEDCLAY'
        self.dd_key_rchres_gqual_alias['ADQAL7'] = 'ADQALTOT'

        self.dd_key_rchres_gqual_alias['DDQAL1'] = 'DDQALHYDROL'
        self.dd_key_rchres_gqual_alias['DDQAL2'] = 'DDQALOXID'
        self.dd_key_rchres_gqual_alias['DDQAL3'] = 'DDQALPHOTOL'
        self.dd_key_rchres_gqual_alias['DDQAL4'] = 'DDQALVOLAT'
        self.dd_key_rchres_gqual_alias['DDQAL5'] = 'DDQALBIODEG'
        self.dd_key_rchres_gqual_alias['DDQAL6'] = 'DDQALGEN'
        self.dd_key_rchres_gqual_alias['DDQAL7'] = 'DDQALTOT'

        self.dd_key_rchres_gqual_alias['DSQAL1'] = 'DSQALSAND'
        self.dd_key_rchres_gqual_alias['DSQAL2'] = 'DSQALSILT'
        self.dd_key_rchres_gqual_alias['DSQAL3'] = 'DSQALCLAY'
        self.dd_key_rchres_gqual_alias['DSQAL4'] = 'DSQALTOT'

        self.dd_key_rchres_gqual_alias['ISQAL1'] = 'ISQALSAND'
        self.dd_key_rchres_gqual_alias['ISQAL2'] = 'ISQALSILT'
        self.dd_key_rchres_gqual_alias['ISQAL3'] = 'ISQALCLAY'
        self.dd_key_rchres_gqual_alias['ISQAL4'] = 'ISQALTOT'

        self.dd_key_rchres_gqual_alias['ROSQAL1'] = 'ROSQALSAND'
        self.dd_key_rchres_gqual_alias['ROSQAL2'] = 'ROSQALSILT'
        self.dd_key_rchres_gqual_alias['ROSQAL3'] = 'ROSQALCLAY'
        self.dd_key_rchres_gqual_alias['ROSQAL4'] = 'ROSQALTOT'

        self.dd_key_rchres_gqual_alias['RSQAL1'] = 'RSQALSUSPSAND'
        self.dd_key_rchres_gqual_alias['RSQAL2'] = 'RSQALSUSPSILT'
        self.dd_key_rchres_gqual_alias['RSQAL3'] = 'RSQALSUSPCLAY'
        self.dd_key_rchres_gqual_alias['RSQAL4'] = 'RSQALSUSPTOT'
        self.dd_key_rchres_gqual_alias['RSQAL5'] = 'RSQALBEDSAND'
        self.dd_key_rchres_gqual_alias['RSQAL6'] = 'RSQALBEDSILT'
        self.dd_key_rchres_gqual_alias['RSQAL7'] = 'RSQALBEDCLAY'
        self.dd_key_rchres_gqual_alias['RSQAL8'] = 'RSQALBEDTOT'
        self.dd_key_rchres_gqual_alias['RSQAL9'] = 'RSQALTOTSAND'
        self.dd_key_rchres_gqual_alias['RSQAL10'] = 'RSQALTOTSILT'
        self.dd_key_rchres_gqual_alias['RSQAL11'] = 'RSQALTOTCLAY'
        self.dd_key_rchres_gqual_alias['RSQAL12'] = 'RSQALTOTTOT'

        self.dd_key_rchres_gqual_alias['SQAL1'] = 'SQALSUSPSAND'
        self.dd_key_rchres_gqual_alias['SQAL2'] = 'SQALSUSPSILT'
        self.dd_key_rchres_gqual_alias['SQAL3'] = 'SQALSUSPCLAY'
        self.dd_key_rchres_gqual_alias['SQAL4'] = 'SQALBEDSAND'
        self.dd_key_rchres_gqual_alias['SQAL5'] = 'SQALBEDSILT'
        self.dd_key_rchres_gqual_alias['SQAL6'] = 'SQALBEDCLAY'

        self.dd_key_rchres_gqual_alias['SQDEC1'] = 'SQDECSUSPSAND'
        self.dd_key_rchres_gqual_alias['SQDEC2'] = 'SQDECSUSPSILT'
        self.dd_key_rchres_gqual_alias['SQDEC3'] = 'SQDECSUSPCLAY'
        self.dd_key_rchres_gqual_alias['SQDEC4'] = 'SQDECBEDSAND'
        self.dd_key_rchres_gqual_alias['SQDEC5'] = 'SQDECBEDSILT'
        self.dd_key_rchres_gqual_alias['SQDEC6'] = 'SQDECBEDCLAY'
        self.dd_key_rchres_gqual_alias['SQDEC7'] = 'SQDECBEDTOT'

        self.dd_key_rchres_sedtrn_alias['ISED1'] = 'ISEDSAND'
        self.dd_key_rchres_sedtrn_alias['ISED2'] = 'ISEDSILT'
        self.dd_key_rchres_sedtrn_alias['ISED3'] = 'ISEDCLAY'
        self.dd_key_rchres_sedtrn_alias['DEPSCR1'] = 'DEPSCOURSAND'
        self.dd_key_rchres_sedtrn_alias['DEPSCR2'] = 'DEPSCOURSILT'
        self.dd_key_rchres_sedtrn_alias['DEPSCR3'] = 'DEPSCOURCLAY'
        self.dd_key_rchres_sedtrn_alias['DEPSCR4'] = 'DEPSCOURTOT'
        self.dd_key_rchres_sedtrn_alias['ROSED1'] = 'ROSEDSAND'
        self.dd_key_rchres_sedtrn_alias['ROSED2'] = 'ROSEDSILT'
        self.dd_key_rchres_sedtrn_alias['ROSED3'] = 'ROSEDCLAY'
        self.dd_key_rchres_sedtrn_alias['ROSED4'] = 'ROSEDTOT'
        self.dd_key_rchres_sedtrn_alias['SSED1'] = 'SSEDSAND'
        self.dd_key_rchres_sedtrn_alias['SSED2'] = 'SSEDSILT'
        self.dd_key_rchres_sedtrn_alias['SSED3'] = 'SSEDCLAY'
        self.dd_key_rchres_sedtrn_alias['SSED4'] = 'SSEDTOT'
        self.dd_key_rchres_sedtrn_alias['RSED1'] = 'RSEDSUSPSAND'
        self.dd_key_rchres_sedtrn_alias['RSED2'] = 'RSEDSUSPSILT'
        self.dd_key_rchres_sedtrn_alias['RSED3'] = 'RSEDSUSPCLAY'
        self.dd_key_rchres_sedtrn_alias['RSED4'] = 'RSEDBEDSAND'
        self.dd_key_rchres_sedtrn_alias['RSED5'] = 'RSEDBEDSILT'
        self.dd_key_rchres_sedtrn_alias['RSED6'] = 'RSEDBEDCLAY'
        self.dd_key_rchres_sedtrn_alias['RSED7'] = 'RSEDTOTSAND'
        self.dd_key_rchres_sedtrn_alias['RSED8'] = 'RSEDTOTSILT'
        self.dd_key_rchres_sedtrn_alias['RSED9'] = 'RSEDTOTCLAY'
        self.dd_key_rchres_sedtrn_alias['RSED10'] = 'RSEDTOTTOT'

        self.dd_key_rchres_sedtrn_alias['OSED11'] = 'OSEDSANDEXIT1'
        self.dd_key_rchres_sedtrn_alias['OSED21'] = 'OSEDSILTEXIT1'
        self.dd_key_rchres_sedtrn_alias['OSED31'] = 'OSEDCLAYEXIT1'
        self.dd_key_rchres_sedtrn_alias['OSED41'] = 'OSEDTOTEXIT1'
        self.dd_key_rchres_sedtrn_alias['OSED12'] = 'OSEDSANDEXIT2'
        self.dd_key_rchres_sedtrn_alias['OSED22'] = 'OSEDSILTEXIT2'
        self.dd_key_rchres_sedtrn_alias['OSED32'] = 'OSEDCLAYEXIT2'
        self.dd_key_rchres_sedtrn_alias['OSED42'] = 'OSEDTOTEXIT2'
        self.dd_key_rchres_sedtrn_alias['OSED13'] = 'OSEDSANDEXIT3'
        self.dd_key_rchres_sedtrn_alias['OSED23'] = 'OSEDSILTEXIT3'
        self.dd_key_rchres_sedtrn_alias['OSED33'] = 'OSEDCLAYEXIT3'
        self.dd_key_rchres_sedtrn_alias['OSED43'] = 'OSEDTOTEXIT3'
        self.dd_key_rchres_sedtrn_alias['OSED14'] = 'OSEDSANDEXIT4'
        self.dd_key_rchres_sedtrn_alias['OSED24'] = 'OSEDSILTEXIT4'
        self.dd_key_rchres_sedtrn_alias['OSED34'] = 'OSEDCLAYEXIT4'
        self.dd_key_rchres_sedtrn_alias['OSED44'] = 'OSEDTOTEXIT4'
        self.dd_key_rchres_sedtrn_alias['OSED15'] = 'OSEDSANDEXIT5'
        self.dd_key_rchres_sedtrn_alias['OSED25'] = 'OSEDSILTEXIT5'
        self.dd_key_rchres_sedtrn_alias['OSED35'] = 'OSEDCLAYEXIT5'
        self.dd_key_rchres_sedtrn_alias['OSED45'] = 'OSEDTOTEXIT5'

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
        self.set_qual_alias_to_hspf()
        with h5py.File(self.file_name, "r") as f:
            if self.start_time is None or self.end_time is None:
                str_starttime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[1].astype('datetime64[D]')
                str_endtime = f.get('/CONTROL/GLOBAL')['table'].fields('Info')[2].astype('datetime64[D]')
                # start at end of 1st simulation period
                self.start_time = pd.to_datetime(str_starttime) + pd.to_timedelta(1, unit='h')
                self.end_time = pd.to_datetime(str_endtime)
                if len(self.time_index) == 0:
                    self.time_index = list(
                        pd.date_range(self.start_time, self.end_time, freq='H'))

            # look at IQUAL's QUALID for naming of h5 IQUAL constituents in the RESULT group
            iqual_flag_grp_rows = f.get('/IMPLND/IQUAL/IQUAL1/FLAGS')['table']['index','QUALID']
            self.data_dictionary[self.dd_key_implnd_iqual_ids] = {}
            for (bindex, bqualid) in iqual_flag_grp_rows:
                self.data_dictionary[self.dd_key_implnd_iqual_ids][bindex.astype('unicode')] = bqualid.astype('unicode')

            # get rchres conservative cons names, not sure why it would be per rchres as diff cons in diff table!
            rcons_groups = f.get('/RCHRES/CONS')
            self.data_dictionary[self.dd_key_rchres_cons_ids] = {}
            for rcon_group_key in rcons_groups.keys():
                if rcon_group_key.startswith('CONS'):
                    self.data_dictionary[self.dd_key_rchres_cons_ids][rcon_group_key] = {}
                    rcons_grp_rows = f.get('/RCHRES/CONS/' + rcon_group_key)['table']['index','CONID']
                    for (bindex, bqualid) in rcons_grp_rows:
                        self.data_dictionary[self.dd_key_rchres_cons_ids][rcon_group_key][bindex.astype('unicode')] = \
                            bqualid.astype('unicode')

            # get rchres gqual names, not sure why it would be per rchres as diff gqual cons in diff table!
            rgqual_groups = f.get('/RCHRES/GQUAL')
            self.data_dictionary[self.dd_key_rchres_gqual_ids] = {}
            for rgqual_group_key in rgqual_groups.keys():
                if rgqual_group_key.startswith('GQUAL'):
                    self.data_dictionary[self.dd_key_rchres_gqual_ids][rgqual_group_key] = {}
                    rgqual_grp_rows = f.get('/RCHRES/GQUAL/' + rgqual_group_key)['table']['index', 'GQID']
                    for (bindex, bqualid) in rgqual_grp_rows:
                        self.data_dictionary[self.dd_key_rchres_gqual_ids][rgqual_group_key][
                            bindex.astype('unicode')] = \
                            bqualid.astype('unicode')

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
        col_key = ''
        for key in self.data_dictionary[data_table_key].keys():
            if operation.upper() == 'IMPLND' and key_act == 'IQUAL': # special matching
                iqual_id = self.data_dictionary[self.dd_key_implnd_iqual_ids][key_opn[0:1] + key_id]
                if self.data_dictionary[data_table_key][key].endswith('_' + constituent.upper().replace(iqual_id, '')):
                    df_index = key
                    col_key = self.data_dictionary[data_table_key][key]
                    break
            elif operation.upper() == 'RCHRES' and key_act == 'CONS': # special matching
                if '_' not in self.data_dictionary[data_table_key][key]:
                    continue
                (o_cons_id, o_cons_name) = self.data_dictionary[data_table_key][key].split('_') # e.g. CONS1_ROCON
                p_cons_id = self.data_dictionary[self.dd_key_rchres_cons_ids][o_cons_id][f'R{key_id}']
                if o_cons_name == 'CON':
                    o_cons_name = 'CONC'
                if o_cons_name == constituent.upper().replace(p_cons_id, ''):
                    df_index = key
                    col_key = self.data_dictionary[data_table_key][key]
                    break
            elif operation.upper() == 'RCHRES' and key_act == 'GQUAL': # special matching
                if '_' not in self.data_dictionary[data_table_key][key]:
                    continue
                (o_cons_id, o_cons_name) = self.data_dictionary[data_table_key][key].split('_') # e.g. GQUAL1_SQAL6
                p_cons_id = self.data_dictionary[self.dd_key_rchres_gqual_ids][o_cons_id][f'R{key_id}']
                if o_cons_name in self.dd_key_rchres_gqual_alias:
                    # switch to HSPF output name
                    o_cons_name = self.dd_key_rchres_gqual_alias[o_cons_name]
                if o_cons_name == constituent.upper().replace(p_cons_id, ''):
                    df_index = key
                    col_key = self.data_dictionary[data_table_key][key]
                    break
            elif operation.upper() == 'RCHRES' and key_act == 'SEDTRN': # special matching
                o_cons_name = self.data_dictionary[data_table_key][key]
                if o_cons_name in self.dd_key_rchres_sedtrn_alias:
                    # switch to HSPF output name
                    o_cons_name = self.dd_key_rchres_sedtrn_alias[o_cons_name]
                if o_cons_name == constituent.upper():
                    df_index = key
                    col_key = self.data_dictionary[data_table_key][key]
                    break
            elif self.data_dictionary[data_table_key][key] == constituent.upper():
                df_index = key
                col_key = constituent.upper()
                break

        if df_index >= 0:
            return self.data_dictionary[data_value_key][col_key]
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