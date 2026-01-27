from typing import Any, Union
import warnings

import pandas as pd

from hsp2.hsp2.model import Model
from hsp2.hsp2io.protocols import Category


class HDF5:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._store = pd.HDFStore(file_path)
        None

    def close(self):
        self._store.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.__del__()

    def read_uci(self) -> Model:
        """
        DEPRECATED: use read_parameters instead
        """
        warnings.warn(
            """
* 
* DEPRECATED: use read_parameters instead
* read_uci will be removed in future releases
*
""",
            DeprecationWarning,
        )
        return self.read_parameters()

    def read_parameters(self) -> Model:
        """
        Read parameter tables from HDF5 and populate Model instance.
        """
        model = Model()
        for path in self._store.keys():  # finds ALL data sets into HDF5 file
            op, module, *other = path[1:].split(sep="/", maxsplit=3)
            s = "_".join(other)
            if op == "CONTROL":
                if module == "GLOBAL":
                    temp = self._store[path].to_dict()["Info"]
                    model.siminfo["start"] = pd.Timestamp(temp["Start"])
                    model.siminfo["stop"] = pd.Timestamp(temp["Stop"])
                    model.siminfo["units"] = 1
                    if "Units" in temp:
                        if int(temp["Units"]):
                            model.siminfo["units"] = int(temp["Units"])
                elif module == "LINKS":
                    for row in self._store[path].fillna("").itertuples():
                        if row.TVOLNO != "":
                            model.ddlinks[f"{row.TVOLNO}"].append(row)
                        else:
                            model.ddlinks[f"{row.TOPFST}"].append(row)

                elif module == "MASS_LINKS":
                    for row in self._store[path].replace("na", "").itertuples():
                        model.ddmasslinks[row.MLNO].append(row)
                elif module == "EXT_SOURCES":
                    for row in self._store[path].replace("na", "").itertuples():
                        model.ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
                elif module == "OP_SEQUENCE":
                    model.opseq = self._store[path]
            elif op in {"PERLND", "IMPLND", "RCHRES"}:
                for id, vdict in self._store[path].to_dict("index").items():
                    model.model[(op, module, id)][s] = vdict
            elif op == "GENER":
                for row in self._store[path].itertuples():
                    if len(row.OPNID.split()) == 1:
                        start = int(row.OPNID)
                        stop = start
                    else:
                        start, stop = row.OPNID.split()
                    for i in range(int(start), int(stop) + 1):
                        if module != "COEFFS":
                            model.ddgener[module][f"G{i:03d}"] = row[2]
                        else:
                            for it in range(1, 8):
                                model.ddgener[f"K{it:01d}"][f"G{i:03d}"] = row[it + 1]
            elif op == "FTABLES":
                model.ftables[module] = self._store[path]
            elif op == "SPEC_ACTIONS":
                model.specactions[module] = self._store[path]
            elif op == "MONTHDATA":
                if not model.monthdata:
                    model.monthdata = {}
                model.monthdata[f"{op}/{module}"] = self._store[path]
        return model

    def read_ts(
        self,
        category: Category,
        operation: Union[str, None] = None,
        segment: Union[str, None] = None,
        activity: Union[str, None] = None,
    ) -> pd.DataFrame:
        try:
            path = ""
            if category == category.INPUTS:
                path = f"TIMESERIES/{segment}"
            elif category == category.RESULTS:
                path = f"RESULTS/{operation}_{segment}/{activity}"
            return pd.read_hdf(self._store, path)
        except KeyError:
            return pd.DataFrame()

    def write_ts(
        self,
        data_frame: pd.DataFrame,
        category: Category,
        operation: str,
        segment: str,
        activity: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Saves timeseries to HDF5"""
        path = f"{operation}_{segment}/{activity}"
        if category:
            path = "RESULTS/" + path
        complevel = None
        if "compress" in kwargs:
            if kwargs["compress"]:
                complevel = 9
        data_frame.to_hdf(
            self._store, key=path, format="t", data_columns=True, complevel=complevel
        )
        # data_frame.to_hdf(self._store, key=path)

    def write_log(self, hsp2_log: pd.DataFrame) -> None:
        hsp2_log.to_hdf(
            self._store, key="RUN_INFO/LOGFILE", data_columns=True, format="t"
        )

    def write_versioning(self, versioning: pd.DataFrame) -> None:
        versioning.to_hdf(
            self._store, key="RUN_INFO/VERSIONS", data_columns=True, format="t"
        )
