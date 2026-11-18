from collections import defaultdict
from datetime import datetime, timedelta
from struct import unpack
from typing import Union

import pandas as pd
from numpy import fromfile
from pandas import DataFrame


def _unformatted_sequential_positions(data) -> Union[list, None]:
    """Records framed by 4-byte length markers, or None if the data are not."""
    positions = []
    index = 0
    while index < len(data):
        if index + 8 > len(data):
            return None
        (length,) = unpack("<i", data[index : index + 4])
        end = index + 4 + length
        if length < 24 or end + 4 > len(data):
            return None
        if unpack("<i", data[end : end + 4])[0] != length:
            return None
        positions.append((index, length - 24))
        index = end + 4
    return positions if positions else None


def _lahey_positions(data) -> list:
    """Records of the Lahey layout, after the first byte 0xFD."""
    positions = []
    index = 1  # already used first byte (magic number)
    while index < len(data):
        rc1, rc2, rc3, rc = unpack("4B", data[index : index + 4])
        rc1 = int(rc1 >> 2)
        rc2 = int(rc2) * 64 + rc1  # 2**6
        rc3 = int(rc3) * 16384 + rc2  # 2**14
        reclen = int(rc) * 4194304 + rc3 - 24  # 2**22
        positions.append((index, reclen))
        if reclen < 36:
            index += reclen + 29  # found by trial and error
        else:
            index += reclen + 30
    return positions


def record_positions(data) -> Union[list, None]:
    """Positions of the records of an HSPF binary output (HBN) file.

    Returns a list of (index, reclen) pairs, one per record: the record body
    (record type, operation, id, activity, ...) starts at index + 4, and reclen is
    the length of the body less its first 24 bytes. Two layouts are read:

    - the unformatted sequential layout of compilers such as gfortran, in which
      4-byte length markers frame each record; HSPF builds that write the binary
      output with plain unformatted WRITE statements give it;
    - the layout of the Lahey Fortran runtime, which HSPF also writes with other
      compilers through its compiler independent binary output routines: a first
      byte 0xFD, then for each record a 4-byte packed length, the body and a
      1- or 2-byte back pointer.

    The markers are checked first, because the first marker of an unformatted
    sequential file can also begin with the byte 0xFD.
    Returns None when the data are in neither layout.
    """
    positions = _unformatted_sequential_positions(data)
    if positions is None and len(data) > 0 and data[0] == 0xFD:
        positions = _lahey_positions(data)
    return positions


class HBNOutput:
    def __init__(self, file_name: str) -> None:
        self.data_frames = []
        self.file_name = file_name
        self.simulation_duration_count = 0
        self.summary = []
        self.summarycols = []
        self.summaryindx = []

        self.output_dictionary = {}

        self.tcodes = {
            1: "Minutely",
            2: "Hourly",
            3: "Daily",
            4: "Monthly",
            5: "Yearly",
        }

    def read_data(self) -> None:
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

        data = fromfile(self.file_name, "B")
        positions = record_positions(data)
        if positions is None:
            print(
                "BAD HBN FILE - must start with magic number 0xFD, or be an "
                "unformatted sequential file with 4-byte record markers"
            )
            return

        # Build layout maps of the file's contents
        mapn = defaultdict(list)
        mapd = defaultdict(list)
        for index, reclen in positions:
            rectype, operation, id, activity = unpack(
                "I8sI8s", data[index + 4 : index + 28]
            )

            operation = operation.decode(
                "ascii"
            ).strip()  # Python3 converts to bytearray not string
            activity = activity.decode("ascii").strip()

            if operation not in {"PERLND", "IMPLND", "RCHRES"}:
                print("ALIGNMENT ERROR", operation)

            if rectype == 1:  # data record
                tcode = unpack("I", data[index + 32 : index + 36])[0]
                mapd[operation, id, activity, tcode].append((index, reclen))
            elif rectype == 0:  # data names record
                i = index + 28
                slen = 0
                while slen < reclen:
                    ln = unpack("I", data[i + slen : i + slen + 4])[0]
                    n = (
                        unpack(f"{ln}s", data[i + slen + 4 : i + slen + 4 + ln])[0]
                        .decode("ascii")
                        .strip()
                    )
                    mapn[operation, id, activity].append(n.replace("-", ""))
                    slen += 4 + ln
            else:
                print("UNKNOW RECTYPE", rectype)

        self.data_frames = []
        self.summary = []
        self.summarycols = [
            "Operation",
            "Activity",
            "segment",
            "Frequency",
            "Shape",
            "Start",
            "Stop",
        ]
        self.summaryindx = []
        for operation, id, activity, tcode in mapd:
            rows = []
            times = []
            nvals = len(mapn[operation, id, activity])
            for index, reclen in mapd[operation, id, activity, tcode]:
                yr, mo, dy, hr, mn = unpack("5I", data[index + 36 : index + 56])
                dt = datetime(yr, mo, dy, 0, mn) + timedelta(hours=hr)
                times.append(dt)

                index += 56
                row = unpack(f"{nvals}f", data[index : index + (4 * nvals)])
                rows.append(row)
            dfname = f"{operation}_{activity}_{id:03d}_{tcode}"
            if self.simulation_duration_count == 0:
                self.simulation_duration_count = len(times)
            # df = DataFrame(rows, index=times, columns=mapn[operation, id, activity]).sort_index('index')
            df = DataFrame(
                rows, index=times, columns=mapn[operation, id, activity]
            ).sort_index()
            self.data_frames.append(df)

            self.summaryindx.append(dfname)
            self.summary.append(
                (
                    operation,
                    activity,
                    str(id),
                    self.tcodes[tcode],
                    str(df.shape),
                    df.index[0],
                    df.index[-1],
                )
            )
            self.output_dictionary[dfname] = mapn[operation, id, activity]

    def get_time_series(
        self, t_opn: str, t_opn_id: int, t_cons: str, t_activity: str, time_unit: str
    ) -> Union[pd.Series, None]:
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
            (operation, activity, oid, tu) = key.split("_")
            if operation == t_opn and int(oid) == t_opn_id and activity == t_activity:
                data_frame = target_data_frames[key]
                for cons_key in data_frame.keys():
                    if cons_key == t_cons:
                        return data_frame[cons_key]

        return None

    # PRT - I think we'll deprecate this method
    @staticmethod
    def save_time_series_to_file(file_name: str, time_series: pd.Series) -> None:
        with open(file_name, "w+") as f:
            for row in range(len(time_series.index)):
                dt = time_series.index[row]
                dv = time_series.values[row]
                # f.write(f'{dt},{"{:.2f}".format(dv)}\n')
                f.write(f"{dt},{dv}\n")
