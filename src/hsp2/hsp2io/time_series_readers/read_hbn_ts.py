# Future imports
from __future__ import annotations

# Standard library imports
from collections import defaultdict
from datetime import datetime, timedelta
from struct import unpack

# Third party imports
import pandas as pd
from numpy import fromfile
from pandas import DataFrame


class HBNReader:
    def __init__(self, file_name: str) -> None:
        self.data_frames: list[pd.DataFrame] = []
        self.file_name = file_name
        self.simulation_duration_count = 0
        self.summary: list[tuple] = []
        self.summarycols: list[str] = []
        self.summaryindx: list[str] = []

        self.output_dictionary: dict[str, list[str]] = {}

        self.tcodes = {
            1: "minutely",
            2: "hourly",
            3: "daily",
            4: "monthly",
            5: "yearly",
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
            Summary information of data found in HBN file (also saved to HDF5
            file.)
        """

        data = fromfile(self.file_name, "B")
        if data[0] != 0xFD:
            print("BAD HBN FILE - must start with magic number 0xFD")
            return

        # Build layout maps of the file's contents
        mapn = defaultdict(list)
        mapd = defaultdict(list)
        index = 1  # already used first byte (magic number)
        while index < len(data):
            rc1, rc2, rc3, rc, rectype, operation, id, activity = unpack(
                "4BI8sI8s", data[index : index + 28]
            )
            rc1 = int(rc1 >> 2)
            rc2 = int(rc2) * 64 + rc1  # 2**6
            rc3 = int(rc3) * 16384 + rc2  # 2**14
            reclen = int(rc) * 4194304 + rc3 - 24  # 2**22

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
            index += reclen + 29 if reclen < 36 else reclen + 30
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
        self.output_dictionary = {}
        self.simulation_duration_count = 0
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
        self,
        t_opn: str,
        t_opn_id: int,
        t_cons: str,
        t_activity: str,
        time_interval: str,
    ) -> pd.Series | None:
        """
        get a single time series based on:
        1.      t_opn: RCHRES, IMPLND, PERLND
        2.   t_opn_id: 1, 2, 3, etc
        3.     t_cons: target constituent name
        4. t_activity: HYDR, IQUAL, etc
        5.  time_interval: hourly, daily, monthly, yearly
        """
        data_frame = self.get_time_series_data_frame(
            t_opn, t_opn_id, t_activity, time_interval
        )
        if data_frame is not None and t_cons in data_frame.columns:
            return data_frame[t_cons]
        return None

    def get_time_series_data_frame(
        self,
        t_opn: str,
        t_opn_id: int,
        t_activity: str,
        time_interval: str,
    ) -> pd.DataFrame | None:
        """Return the requested operation, activity, and frequency data frame."""
        # This next bit just does an inverse lookup to find the tcode corresponding to the requested timer
        # interval.
        target_tcode = next(
            (
                tcode_key
                for tcode_key in self.tcodes
                if self.tcodes[tcode_key].lower() == time_interval
            ),
            2,
        )
        for key, data_frame in zip(self.summaryindx, self.data_frames):
            operation, activity, oid, tcode = key.split("_")
            if (
                operation == t_opn
                and int(oid) == t_opn_id
                and activity == t_activity
                and int(tcode) == target_tcode
            ):
                return data_frame
        return None


def read_hbn_ts(
    file_name: str,
    operation: str | None = None,
    segment: str | int | None = None,
    activity: str | None = None,
    t_cons: str | None = None,
    time_interval: str = "hourly",
) -> DataFrame:
    """Read an HBN time series data frame.

    When no operation, segment, or activity is supplied, return the HBN file
    summary for compatibility with existing callers. Otherwise, all three
    selectors are required. Segment may be the HSP2 name (for example, ``P001``)
    or the numeric HBN identifier (for example, ``1``).

    If ``t_cons`` is supplied, return a one-column data frame for that
    constituent; otherwise return every constituent for the selected group.
    """
    reader = HBNReader(file_name)
    reader.read_data()
    if operation is None and segment is None and activity is None:
        if t_cons is not None:
            raise ValueError(
                "operation, segment, and activity are required with t_cons"
            )
        return pd.DataFrame(
            reader.summary, index=reader.summaryindx, columns=reader.summarycols
        )
    if operation is None or segment is None or activity is None:
        raise ValueError("operation, segment, and activity must all be provided")

    normalized_operation = operation.upper()
    normalized_activity = activity.upper()
    segment_text = str(segment)
    if segment_text.upper().startswith(normalized_operation[0]):
        segment_text = segment_text[1:]
    try:
        segment_id = int(segment_text)
    except ValueError as exc:
        raise ValueError(
            f"segment must be a numeric HBN identifier or start with "
            f"{normalized_operation[0]!r}: {segment!r}"
        ) from exc

    data_frame = reader.get_time_series_data_frame(
        normalized_operation, segment_id, normalized_activity, time_interval.lower()
    )
    if data_frame is None:
        raise KeyError(
            f"No {time_interval} HBN data for {normalized_operation} "
            f"segment {segment} activity {normalized_activity}"
        )
    if t_cons is not None:
        if t_cons not in data_frame.columns:
            raise KeyError(f"Constituent {t_cons!r} is not present in the HBN data")
        return data_frame[[t_cons]].copy()
    return data_frame.copy()
