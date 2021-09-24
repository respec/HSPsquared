from struct import unpack
from numpy import fromfile
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from collections import defaultdict

from typing import Union

class HBNOutput:
    def __init__(self, file_name:str) -> None:
        self.data_frames = []
        self.file_name = file_name
        self.simulation_duration_count = 0
        self.summary = []
        self.summarycols = []
        self.summaryindx = []

        self.output_dictionary = {}

        self.tcodes = {1: 'Minutely', 2: 'Hourly', 3: 'Daily', 4: 'Monthly', 5: 'Yearly'}

    def read_data(self) -> Noner:
        """
        Reads ALL data from hbn_file and return them in DataFrame

        Parameters
        ----------
        hbn_file : str
            Name/path of HBN created by HSPF.

        Returns
        -------
        df_summary : DataFrame
            Summary information of data found in HBN file (also saved to HDF5 file.)
        """

        data = fromfile(self.file_name, 'B')
        if data[0] != 0xFD:
            print('BAD HBN FILE - must start with magic number 0xFD')
            return

        # Build layout maps of the file's contents
        mapn = defaultdict(list)
        mapd = defaultdict(list)
        index = 1  # already used first byte (magic number)
        while index < len(data):
            rc1, rc2, rc3, rc, rectype, operation, id, activity = unpack('4BI8sI8s', data[index:index + 28])
            rc1 = int(rc1 >> 2)
            rc2 = int(rc2) * 64 + rc1  # 2**6
            rc3 = int(rc3) * 16384 + rc2  # 2**14
            reclen = int(rc) * 4194304 + rc3 - 24  # 2**22

            operation = operation.decode('ascii').strip()  # Python3 converts to bytearray not string
            activity = activity.decode('ascii').strip()

            if operation not in {'PERLND', 'IMPLND', 'RCHRES'}:
                print('ALIGNMENT ERROR', operation)

            if rectype == 1:  # data record
                tcode = unpack('I', data[index + 32: index + 36])[0]
                mapd[operation, id, activity, tcode].append((index, reclen))
            elif rectype == 0:  # data names record
                i = index + 28
                slen = 0
                while slen < reclen:
                    ln = unpack('I', data[i + slen: i + slen + 4])[0]
                    n = unpack(f'{ln}s', data[i + slen + 4: i + slen + 4 + ln])[0].decode('ascii').strip()
                    mapn[operation, id, activity].append(n.replace('-', ''))
                    slen += 4 + ln
            else:
                print('UNKNOW RECTYPE', rectype)
            if reclen < 36:
                index += reclen + 29  # found by trial and error
            else:
                index += reclen + 30

        self.data_frames = []
        self.summary = []
        self.summarycols = ['Operation', 'Activity', 'segment', 'Frequency', 'Shape', 'Start', 'Stop']
        self.summaryindx = []
        for (operation, id, activity, tcode) in mapd:
            rows = []
            times = []
            nvals = len(mapn[operation, id, activity])
            for (index, reclen) in mapd[operation, id, activity, tcode]:
                yr, mo, dy, hr, mn = unpack('5I', data[index + 36: index + 56])
                dt = datetime(yr, mo, dy, 0, mn) + timedelta(hours=hr)
                times.append(dt)

                index += 56
                row = unpack(f'{nvals}f', data[index:index + (4 * nvals)])
                rows.append(row)
            dfname = f'{operation}_{activity}_{id:03d}_{tcode}'
            if self.simulation_duration_count == 0:
                self.simulation_duration_count = len(times)
            df = DataFrame(rows, index=times, columns=mapn[operation, id, activity]).sort_index('index')
            self.data_frames.append(df)

            self.summaryindx.append(dfname)
            self.summary.append((operation, activity, str(id), self.tcodes[tcode], str(df.shape), df.index[0], df.index[-1]))
            self.output_dictionary[dfname] = mapn[operation, id, activity]

    def get_time_series(self, t_opn:str, t_opn_id:int, t_cons:str, t_activity:str, time_unit:str) -> Union[pd.Series, None]:
        """
        get a single time series based on:
        1.      t_opn: RCHRES, IMPLND, PERLND
        2.   t_opn_id: 1, 2, 3, etc
        3.     t_cons: target constituent name
        4. t_activity: HYDR, IQUAL, etc
        5.  time_unit: yearly, monthly, full (default is 'full' simulation duration)
        """
        target_tcode = 2
        for tcode_key in self.tcodes.keys():
            if self.tcodes[tcode_key].lower() == time_unit:
                target_tcode = tcode_key
                break

        target_data_frames = {}
        for index_group_key in self.summaryindx:
            if index_group_key.endswith(str(target_tcode)):
                group_index = self.summaryindx.index(index_group_key)
                target_data_frames[index_group_key] = self.data_frames[group_index]

        for key in target_data_frames.keys():
            (operation, activity, oid, tu) = key.split('_')
            if operation == t_opn and int(oid) == t_opn_id and activity == t_activity:
                data_frame = target_data_frames[key]
                for cons_key in data_frame.keys():
                    if cons_key == t_cons:
                        return data_frame[cons_key]

        return None

    #PRT - I think we'll deprecate this method
    @staticmethod
    def save_time_series_to_file(file_name:str, time_series:pd.Series) -> None:
        with open(file_name, 'w+') as f:
            for row in range(len(time_series.index)):
                dt = time_series.index[row]
                dv = time_series.values[row]
                # f.write(f'{dt},{"{:.2f}".format(dv)}\n')
                f.write(f'{dt},{dv}\n')