'''
Convert a dict of UCI data frames into the uci object as used in HSP2
'''

from hsp2.hsp2.uci import UCI
import pandas as pd

def convert_uci_InMem(uciDict):
    """
    Convert a dict of UCI data frames into the uci object as used in HSP2
    Cousin of read_uci in HSP2 hdf.py, which converts HDF5 format UCI into uci object

    Parameters
    ----------
    uciDict : dict of pandas dataframes as returned by readUci

    Returns
    -------
    uci object as used by HSP2
    """

    uci = UCI()
    for path in uciDict.keys():
        op, module, *other = path[1:].split(sep="/", maxsplit=3)
        s = "_".join(other)
        if op == "CONTROL":
            if module == "GLOBAL":
                temp = uciDict[path].to_dict()["Info"]
                uci.siminfo["start"] = pd.Timestamp(temp["Start"])
                uci.siminfo["stop"] = pd.Timestamp(temp["Stop"])
                uci.siminfo["units"] = 1
                if "Units" in temp:
                    if int(temp["Units"]):
                        uci.siminfo["units"] = int(temp["Units"])
            elif module == "LINKS":
                for row in uciDict[path].fillna("").itertuples():
                    if row.TVOLNO != "":
                        uci.ddlinks[f"{row.TVOLNO}"].append(row)
                    else:
                        uci.ddlinks[f"{row.TOPFST}"].append(row)

            elif module == "MASS_LINKS":
                for row in uciDict[path].replace("na", "").itertuples():
                    uci.ddmasslinks[row.MLNO].append(row)
            elif module == "EXT_SOURCES":
                for row in uciDict[path].replace("na", "").itertuples():
                    uci.ddext_sources[(row.TVOL, row.TVOLNO)].append(row)
            elif module == "OP_SEQUENCE":
                uci.opseq = uciDict[path]
        elif op in {"PERLND", "IMPLND", "RCHRES"}:
            for id, vdict in uciDict[path].to_dict("index").items():
                uci.uci[(op, module, id)][s] = vdict
        elif op == "GENER":
            for row in uciDict[path].itertuples():
                if len(row) > 1:
                    if len(row.OPNID.split()) == 1:
                        start = int(row.OPNID)
                        stop = start
                    else:
                        start, stop = row.OPNID.split()
                    for i in range(int(start), int(stop) + 1):
                        if module != "COEFFS":
                            uci.ddgener[module][f"G{i:03d}"] = row[2]
                        else:
                            for it in range(1, 8):
                                uci.ddgener[f"K{it:01d}"][f"G{i:03d}"] = row[it + 1]
        elif op == "FTABLES":
            uci.ftables[module] = uciDict[path]
        elif op == "SPEC_ACTIONS":
            uci.specactions[module] = uciDict[path]
        elif op == "MONTHDATA":
            if not uci.monthdata:
                uci.monthdata = {}
            uci.monthdata[f"{op}/{module}"] = uciDict[path]

    return uci
