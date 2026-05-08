"""Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
"""

import os
from datetime import datetime as dt
from typing import Union

from numpy import float64
from pandas import DataFrame, date_range
from pandas.tseries.offsets import Minute

from hsp2.hsp2.configuration import activities, expand_masslinks, noop
from hsp2.hsp2.om import (
    om_init_state,
    state_load_dynamics_om,
    state_om_model_run_finish,
    state_om_model_run_prep,
)
from hsp2.hsp2.SPECL import specl_load_state
from hsp2.hsp2.utilities import (
    expand_timeseries_names,
    get_gener_timeseries,
    get_timeseries,
    save_timeseries,
    versions,
)
from hsp2.hsp2io.hdf import HDF5
from hsp2.hsp2io.io import Category, IOManager, SupportsReadTS
from hsp2.state.state import (
    init_state_dicts,
    state_context_hsp2,
    state_init_hsp2,
    state_load_dynamics_hsp2,
    state_siminfo_hsp2,
)


def main(
    io_manager: Union[str, IOManager], saveall: bool = False, jupyterlab: bool = True
) -> None:
    """
    Run main HSP2 program.
    Parameters
    ----------
    io_manager
        An instance of IOManager class.
    saveall: bool, default=False
        Saves all calculated data ignoring SAVE tables.
    jupyterlab: bool, default=True
        Flag for specific output behavior for  jupyter lab.

    Return
    ------------
    None

    """
    if isinstance(io_manager, str):
        hdf5_instance = HDF5(io_manager)
        io_manager = IOManager(hdf5_instance)
    hdfname = io_manager._input.file_path
    if not os.path.exists(hdfname):
        raise FileNotFoundError(f"{hdfname} HDF5 File Not Found")

    msg = messages()
    msg(1, f"Processing started for file {hdfname}; saveall={saveall}")

    # read user control, parameters, states, and flags parameters and map to local variables
    parameter_obj = io_manager.read_parameters()
    opseq = parameter_obj.opseq
    ddlinks = parameter_obj.ddlinks
    ddmasslinks = parameter_obj.ddmasslinks
    ddext_sources = parameter_obj.ddext_sources
    ddgener = parameter_obj.ddgener
    model = parameter_obj.model
    siminfo = parameter_obj.siminfo
    ftables = parameter_obj.ftables
    specactions = parameter_obj.specactions
    monthdata = parameter_obj.monthdata

    start, stop = siminfo["start"], siminfo["stop"]

    copy_instances = {}
    gener_instances = {}

    #######################################################################################
    # initialize STATE dicts
    #######################################################################################
    # Set up Things in state that will be used in all modular activities like SPECL
    state = init_state_dicts()
    state_siminfo_hsp2(parameter_obj, siminfo, io_manager, state)
    # Add support for dynamic functions to operate on STATE
    # - Load any dynamic components if present, and store variables on objects
    state_load_dynamics_hsp2(state, io_manager, siminfo)
    # Iterate through all segments and add crucial paths to state
    # before loading dynamic components that may reference them
    state_init_hsp2(state, opseq, activities)
    # - finally stash specactions in state, not domain (segment) dependent so do it once
    state["specactions"] = specactions  # stash the specaction dict in state
    om_init_state(state)  # set up operational model specific state entries
    specl_load_state(state, io_manager, siminfo)  # traditional special actions
    state_load_dynamics_om(
        state, io_manager, siminfo
    )  # operational model for custom python
    # finalize all dynamically loaded components and prepare to run the model
    state_om_model_run_prep(state, io_manager, siminfo)
    #######################################################################################

    # main processing loop
    msg(1, f"Simulation Start: {start}, Stop: {stop}")
    for _, operation, segment, delt in opseq.itertuples():
        msg(2, f"{operation} {segment} DELT(minutes): {delt}")
        siminfo["delt"] = delt
        siminfo["tindex"] = date_range(start, stop, freq=Minute(delt))[1:]
        siminfo["steps"] = len(siminfo["tindex"])

        if operation in ["DISPLY", "PLTGEN"]:
            # HSPF only operations that are kept in HSP2 only in order to be
            # able to move parameters from UCI -> HDF5 -> UCI.
            continue

        if operation == "COPY":
            copy_instances[segment] = activities[operation](
                io_manager, siminfo, ddext_sources[(operation, segment)]
            )
        elif operation == "GENER":
            try:
                ts = get_timeseries(
                    io_manager, ddext_sources[(operation, segment)], siminfo
                )
                ts = get_gener_timeseries(
                    ts, gener_instances, ddlinks[segment], ddmasslinks
                )
                get_flows(
                    io_manager,
                    ts,
                    {},
                    model,
                    segment,
                    ddlinks,
                    ddmasslinks,
                    siminfo["steps"],
                    msg,
                )
                gener_instances[segment] = activities[operation](
                    segment,
                    siminfo,
                    copy_instances,
                    gener_instances,
                    ddlinks,
                    ddmasslinks,
                    ts,
                    ddgener,
                )
            except NotImplementedError as e:
                print(f"GENER '{segment}' may not function correctly. '{e}'")
        else:
            # now conditionally execute all activity modules for the op, segment
            ts = get_timeseries(
                io_manager, ddext_sources[(operation, segment)], siminfo
            )
            ts = get_gener_timeseries(
                ts, gener_instances, ddlinks[segment], ddmasslinks
            )
            flags = model[(operation, "GENERAL", segment)]["ACTIVITY"]
            if operation == "RCHRES":
                # Add nutrient adsorption flags:
                if flags["NUTRX"] == 1:
                    flags["TAMFG"] = model[(operation, "NUTRX", segment)]["FLAGS"][
                        "NH3FG"
                    ]
                    flags["ADNHFG"] = model[(operation, "NUTRX", segment)]["FLAGS"][
                        "ADNHFG"
                    ]
                    flags["PO4FG"] = model[(operation, "NUTRX", segment)]["FLAGS"][
                        "PO4FG"
                    ]
                    flags["ADPOFG"] = model[(operation, "NUTRX", segment)]["FLAGS"][
                        "ADPOFG"
                    ]

                get_flows(
                    io_manager,
                    ts,
                    flags,
                    model,
                    segment,
                    ddlinks,
                    ddmasslinks,
                    siminfo["steps"],
                    msg,
                )

            for activity, function in activities[operation].items():
                if function == noop:  # or not flags[activity]:
                    continue

                if (activity in flags) and (not flags[activity]):
                    continue

                if (
                    (activity == "RQUAL")
                    and (not flags["OXRX"])
                    and (not flags["NUTRX"])
                    and (not flags["PLANK"])
                    and (not flags["PHCARB"])
                ):
                    continue

                msg(3, f"{activity}")
                # Set context for dynamic executables and special actions
                state_context_hsp2(state, operation, segment, activity)

                ui = model[(operation, activity, segment)]  # ui is a dictionary
                if operation == "PERLND":
                    if activity == "PSTEMP":
                        # special exception here to make AIRTFG available
                        ui["PARAMETERS"]["AIRTFG"] = flags["ATEMP"]
                    elif activity in ["SEDMNT", "PWTGAS"]:
                        # special exception here to make CSNOFG available
                        ui["PARAMETERS"]["CSNOFG"] = model[
                            (operation, "PWATER", segment)
                        ]["PARAMETERS"]["CSNOFG"]
                if operation == "RCHRES":
                    if "PARAMETERS" not in ui:
                        ui["PARAMETERS"] = {}
                    ui["PARAMETERS"]["NEXITS"] = model[(operation, "HYDR", segment)][
                        "PARAMETERS"
                    ]["NEXITS"]
                    if activity == "ADCALC":
                        ui["PARAMETERS"]["ADFG"] = flags["ADCALC"]
                        ui["PARAMETERS"]["KS"] = model[(operation, "HYDR", segment)][
                            "PARAMETERS"
                        ]["KS"]
                        ui["PARAMETERS"]["VOL"] = model[(operation, "HYDR", segment)][
                            "STATES"
                        ]["VOL"]
                        ui["PARAMETERS"]["ROS"] = model[(operation, "HYDR", segment)][
                            "PARAMETERS"
                        ]["ROS"]
                        nexits = model[(operation, "HYDR", segment)]["PARAMETERS"][
                            "NEXITS"
                        ]
                        for index in range(nexits):
                            ui["PARAMETERS"][f"OS{str(index + 1)}"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"][f"OS{str(index + 1)}"]
                    elif activity == "CONS":
                        ui["advectData"] = model[(operation, "ADCALC", segment)][
                            "adcalcData"
                        ]
                    elif activity == "GQUAL":
                        ui["advectData"] = model[(operation, "ADCALC", segment)][
                            "adcalcData"
                        ]
                        ui["PARAMETERS"]["HTFG"] = flags["HTRCH"]
                        ui["PARAMETERS"]["SEDFG"] = flags["SEDTRN"]
                        # ui['PARAMETERS']['REAMFG'] = parameters[(operation, 'OXRX', segment)]['PARAMETERS']['REAMFG']
                        ui["PARAMETERS"]["HYDRFG"] = flags["HYDR"]
                        if flags["HYDR"]:
                            ui["PARAMETERS"]["LKFG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LKFG"]
                            ui["PARAMETERS"]["AUX1FG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["AUX1FG"]
                            ui["PARAMETERS"]["AUX2FG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["AUX2FG"]
                            ui["PARAMETERS"]["LEN"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LEN"]
                            ui["PARAMETERS"]["DELTH"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["DELTH"]
                        if flags["OXRX"]:
                            ui["PARAMETERS"]["LKFG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LKFG"]
                            ui["PARAMETERS"]["CFOREA"] = model[
                                (operation, "OXRX", segment)
                            ]["PARAMETERS"]["CFOREA"]
                        if flags["SEDTRN"]:
                            ui["PARAMETERS"]["SSED1"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED1"]
                            ui["PARAMETERS"]["SSED2"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED2"]
                            ui["PARAMETERS"]["SSED3"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED3"]
                        if flags["HTRCH"]:
                            ui["PARAMETERS"]["CFSAEX"] = model[
                                (operation, "HTRCH", segment)
                            ]["PARAMETERS"]["CFSAEX"]
                        elif flags["PLANK"]:
                            if (
                                "CFSAEX"
                                in model[(operation, "PLANK", segment)]["PARAMETERS"]
                            ):
                                ui["PARAMETERS"]["CFSAEX"] = model[
                                    (operation, "PLANK", segment)
                                ]["PARAMETERS"]["CFSAEX"]

                    elif activity == "HTRCH":
                        ui["PARAMETERS"]["ADFG"] = flags["ADCALC"]
                        ui["advectData"] = model[(operation, "ADCALC", segment)][
                            "adcalcData"
                        ]
                    elif activity == "RQUAL":
                        # RQUAL inputs:
                        ui["advectData"] = model[(operation, "ADCALC", segment)][
                            "adcalcData"
                        ]
                        if flags["HYDR"]:
                            ui["PARAMETERS"]["LKFG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LKFG"]

                        ui["FLAGS"]["HTFG"] = flags["HTRCH"]
                        ui["FLAGS"]["SEDFG"] = flags["SEDTRN"]
                        ui["FLAGS"]["GQFG"] = flags["GQUAL"]
                        ui["FLAGS"]["OXFG"] = flags["OXRX"]
                        ui["FLAGS"]["NUTFG"] = flags["NUTRX"]
                        ui["FLAGS"]["PLKFG"] = flags["PLANK"]
                        ui["FLAGS"]["PHFG"] = flags["PHCARB"]
                        if flags["CONS"] and (
                            "PARAMETERS" in model[(operation, "CONS", segment)]
                            and (
                                "NCONS"
                                in model[(operation, "CONS", segment)]["PARAMETERS"]
                            )
                        ):
                            ui["PARAMETERS"]["NCONS"] = model[
                                (operation, "CONS", segment)
                            ]["PARAMETERS"]["NCONS"]

                        # OXRX module inputs:
                        ui_oxrx = model[(operation, "OXRX", segment)]

                        if flags["HYDR"]:
                            ui_oxrx["PARAMETERS"]["LEN"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LEN"]
                            ui_oxrx["PARAMETERS"]["DELTH"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["DELTH"]

                        if flags["HTRCH"]:
                            ui_oxrx["PARAMETERS"]["ELEV"] = model[
                                (operation, "HTRCH", segment)
                            ]["PARAMETERS"]["ELEV"]

                        if flags["SEDTRN"]:
                            ui["PARAMETERS"]["SSED1"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED1"]
                            ui["PARAMETERS"]["SSED2"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED2"]
                            ui["PARAMETERS"]["SSED3"] = model[
                                (operation, "SEDTRN", segment)
                            ]["STATES"]["SSED3"]

                        # PLANK module inputs:
                        if flags["HTRCH"]:
                            ui["PARAMETERS"]["CFSAEX"] = model[
                                (operation, "HTRCH", segment)
                            ]["PARAMETERS"]["CFSAEX"]

                        # NUTRX, PLANK, PHCARB module inputs:
                        ui_nutrx = model[(operation, "NUTRX", segment)]
                        ui_plank = model[(operation, "PLANK", segment)]
                        ui_phcarb = model[(operation, "PHCARB", segment)]

                    elif activity == "SEDTRN":
                        ui["PARAMETERS"]["ADFG"] = flags["ADCALC"]
                        ui["advectData"] = model[(operation, "ADCALC", segment)][
                            "adcalcData"
                        ]
                        # ui['STATES']['VOL'] = parameters[(operation, 'HYDR', segment)]['STATES']['VOL']
                        ui["PARAMETERS"]["HTFG"] = flags["HTRCH"]
                        ui["PARAMETERS"]["AUX3FG"] = 0
                        if flags["HYDR"]:
                            ui["PARAMETERS"]["LEN"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["LEN"]
                            ui["PARAMETERS"]["DELTH"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["DELTH"]
                            ui["PARAMETERS"]["DB50"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["DB50"]
                            ui["PARAMETERS"]["AUX3FG"] = model[
                                (operation, "HYDR", segment)
                            ]["PARAMETERS"]["AUX3FG"]
                ############ calls activity function like snow() ##############
                if operation not in ["COPY", "GENER"]:
                    if activity == "HYDR":
                        errors, errmessages = function(
                            io_manager, siminfo, ui, ts, ftables, state
                        )
                    elif activity in ["SEDTRN", "SEDMNT"]:
                        errors, errmessages = function(
                            io_manager, siminfo, ui, ts, state
                        )
                    elif activity != "RQUAL":
                        errors, errmessages = function(io_manager, siminfo, ui, ts)
                    else:
                        errors, errmessages = function(
                            io_manager,
                            siminfo,
                            ui,
                            ui_oxrx,
                            ui_nutrx,
                            ui_plank,
                            ui_phcarb,
                            ts,
                            monthdata,
                            state,
                        )
                ###############################################################

                for errorcnt, errormsg in zip(errors, errmessages):
                    if errorcnt > 0:
                        msg(4, f"Error count {errorcnt}: {errormsg}")

                # default to hourly output
                outstep = 2
                outstep_oxrx = 2
                outstep_nutrx = 2
                outstep_plank = 2
                outstep_phcarb = 2
                if "BINOUT" in model[(operation, "GENERAL", segment)]:
                    if activity in model[(operation, "GENERAL", segment)]["BINOUT"]:
                        outstep = model[(operation, "GENERAL", segment)]["BINOUT"][
                            activity
                        ]
                    elif activity == "RQUAL":
                        outstep_oxrx = model[(operation, "GENERAL", segment)]["BINOUT"][
                            "OXRX"
                        ]
                        outstep_nutrx = model[(operation, "GENERAL", segment)][
                            "BINOUT"
                        ]["NUTRX"]
                        outstep_plank = model[(operation, "GENERAL", segment)][
                            "BINOUT"
                        ]["PLANK"]
                        outstep_phcarb = model[(operation, "GENERAL", segment)][
                            "BINOUT"
                        ]["PHCARB"]

                if "SAVE" in ui:
                    save_timeseries(
                        io_manager,
                        ts,
                        ui["SAVE"],
                        siminfo,
                        saveall,
                        operation,
                        segment,
                        activity,
                        jupyterlab,
                        outstep,
                    )

                if activity == "RQUAL":
                    if "SAVE" in ui_oxrx:
                        save_timeseries(
                            io_manager,
                            ts,
                            ui_oxrx["SAVE"],
                            siminfo,
                            saveall,
                            operation,
                            segment,
                            "OXRX",
                            jupyterlab,
                            outstep_oxrx,
                        )
                    if "SAVE" in ui_nutrx and flags["NUTRX"] == 1:
                        save_timeseries(
                            io_manager,
                            ts,
                            ui_nutrx["SAVE"],
                            siminfo,
                            saveall,
                            operation,
                            segment,
                            "NUTRX",
                            jupyterlab,
                            outstep_nutrx,
                        )
                    if "SAVE" in ui_plank and flags["PLANK"] == 1:
                        save_timeseries(
                            io_manager,
                            ts,
                            ui_plank["SAVE"],
                            siminfo,
                            saveall,
                            operation,
                            segment,
                            "PLANK",
                            jupyterlab,
                            outstep_plank,
                        )
                    if "SAVE" in ui_phcarb and flags["PHCARB"] == 1:
                        save_timeseries(
                            io_manager,
                            ts,
                            ui_phcarb["SAVE"],
                            siminfo,
                            saveall,
                            operation,
                            segment,
                            "PHCARB",
                            jupyterlab,
                            outstep_phcarb,
                        )

    msglist = msg(1, "Done", final=True)

    # Finish operational models
    state_om_model_run_finish(state, io_manager, siminfo)

    df = DataFrame(msglist, columns=["logfile"])
    io_manager.write_log(df)

    if jupyterlab:
        df = versions(["jupyterlab", "notebook"])
        io_manager.write_versioning(df)
        print("\n\n", df)
    return


def messages():
    """Closure routine; msg() prints messages to screen and run log"""
    start = dt.now()
    mlist = []

    def msg(indent, message, final=False):
        now = dt.now()
        m = str(now)[:22] + "   " * indent + message
        if final:
            mn, sc = divmod((now - start).seconds, 60)
            ms = (now - start).microseconds // 100_000
            m = "; ".join((m, f"Run time is about {mn:02}:{sc:02}.{ms} (mm:ss)"))
        print(m)
        mlist.append(m)
        return mlist

    return msg


def get_flows(
    io_manager: SupportsReadTS,
    ts,
    flags,
    parameters,
    segment,
    ddlinks,
    ddmasslinks,
    steps,
    msg,
):
    # get inflows to this operation
    for x in ddlinks[segment]:
        if x.SVOL != "GENER":  # gener already handled in get_gener_timeseries
            recs = []
            if x.MLNO == "":  # Data from NETWORK part of Links table
                rec = {
                    "MFACTOR": x.MFACTOR,
                    "SGRPN": x.SGRPN,
                    "SMEMN": x.SMEMN,
                    "SMEMSB1": x.SMEMSB1,
                    "SMEMSB2": x.SMEMSB2,
                    "TMEMN": x.TMEMN,
                    "TMEMSB1": x.TMEMSB1,
                    "TMEMSB2": x.TMEMSB2,
                    "SVOL": x.SVOL,
                }
                recs.append(rec)
            else:  # Data from SCHEMATIC part of Links table
                mldata = ddmasslinks[x.MLNO]
                for dat in mldata:
                    if dat.SMEMN != "":
                        rec = {
                            "MFACTOR": dat.MFACTOR,
                            "SGRPN": dat.SGRPN,
                            "SMEMN": dat.SMEMN,
                            "SMEMSB1": dat.SMEMSB1,
                            "SMEMSB2": dat.SMEMSB2,
                            "TMEMN": dat.TMEMN,
                            "TMEMSB1": dat.TMEMSB1,
                            "TMEMSB2": dat.TMEMSB2,
                            "SVOL": dat.SVOL,
                        }
                        recs.append(rec)
                    elif dat.SGRPN in ["ROFLOW", "OFLOW"]:
                        recs = expand_masslinks(flags, parameters, dat, recs)

            for rec in recs:
                mfactor = rec["MFACTOR"]
                sgrpn = rec["SGRPN"]
                smemn = rec["SMEMN"]
                smemsb1 = rec["SMEMSB1"]
                smemsb2 = rec["SMEMSB2"]
                tmemn = rec["TMEMN"]
                tmemsb1 = rec["TMEMSB1"]
                tmemsb2 = rec["TMEMSB2"]

                if x.AFACTR != "":
                    afactr = x.AFACTR
                    factor = afactr * mfactor
                else:
                    factor = mfactor

                # KLUDGE until remaining HSP2 modules are available.
                if tmemn not in {
                    "IVOL",
                    "ICON",
                    "IHEAT",
                    "ISED",
                    "ISED1",
                    "ISED2",
                    "ISED3",
                    "IDQAL",
                    "ISQAL1",
                    "ISQAL2",
                    "ISQAL3",
                    "OXIF",
                    "NUIF1",
                    "NUIF2",
                    "PKIF",
                    "PHIF",
                    "ONE",
                    "TWO",
                }:
                    continue
                if (sgrpn == "OFLOW" and smemn == "OVOL") or (
                    sgrpn == "ROFLOW" and smemn == "ROVOL"
                ):
                    sgrpn = "HYDR"
                if (sgrpn == "OFLOW" and smemn == "OHEAT") or (
                    sgrpn == "ROFLOW" and smemn == "ROHEAT"
                ):
                    sgrpn = "HTRCH"
                if (sgrpn == "OFLOW" and smemn == "OSED") or (
                    sgrpn == "ROFLOW" and smemn == "ROSED"
                ):
                    sgrpn = "SEDTRN"
                if (sgrpn == "OFLOW" and smemn == "ODQAL") or (
                    sgrpn == "ROFLOW" and smemn == "RODQAL"
                ):
                    sgrpn = "GQUAL"
                if (sgrpn == "OFLOW" and smemn == "OSQAL") or (
                    sgrpn == "ROFLOW" and smemn == "ROSQAL"
                ):
                    sgrpn = "GQUAL"
                if (sgrpn == "OFLOW" and smemn == "OXCF2") or (
                    sgrpn == "ROFLOW" and smemn == "OXCF1"
                ):
                    sgrpn = "OXRX"
                if (
                    sgrpn == "OFLOW"
                    and smemn in ["NUCF9", "OSNH4", "OSPO4"]
                    or (sgrpn == "ROFLOW" and smemn in ["NUCF1", "NUFCF2"])
                ):
                    sgrpn = "NUTRX"
                if (sgrpn == "OFLOW" and smemn == "PKCF2") or (
                    sgrpn == "ROFLOW" and smemn == "PKCF1"
                ):
                    sgrpn = "PLANK"
                if (sgrpn == "OFLOW" and smemn == "PHCF2") or (
                    sgrpn == "ROFLOW" and smemn == "PHCF1"
                ):
                    sgrpn = "PHCARB"

                if tmemn in ["ISED", "ISQAL"]:
                    tmemn = tmemn + str(
                        int(float(tmemsb1))
                    )  # need to add sand, silt, clay subscript
                if (sgrpn == "HYDR" and smemn == "OVOL") or (
                    sgrpn == "HTRCH" and smemn == "OHEAT"
                ):
                    smemsb2 = ""
                if sgrpn == "GQUAL" and smemsb2 == "":
                    smemsb2 = "1"

                smemn, tmemn = expand_timeseries_names(
                    sgrpn, smemn, smemsb1, smemsb2, tmemn, tmemsb1, tmemsb2
                )

                path = f"RESULTS/{x.SVOL}_{x.SVOLNO}/{sgrpn}"
                MFname = f"{x.SVOL}{x.SVOLNO}_MFACTOR"
                AFname = f"{x.SVOL}{x.SVOLNO}_AFACTR"
                data = f"{smemn}{smemsb1}{smemsb2}"

                data_frame = io_manager.read_ts(
                    Category.RESULTS, x.SVOL, x.SVOLNO, sgrpn
                )
                try:
                    if data in data_frame.columns:
                        t = data_frame[data].astype(float64).to_numpy()[:steps]
                    else:
                        t = data_frame[smemn].astype(float64).to_numpy()[:steps]

                    if MFname in ts and AFname in ts:
                        t *= ts[MFname][:steps] * ts[AFname][:steps]
                        msg(4, f"MFACTOR modified by timeseries {MFname}")
                        msg(4, f"AFACTR modified by timeseries {AFname}")
                    elif MFname in ts:
                        t *= afactr * ts[MFname][:steps]
                        msg(4, f"MFACTOR modified by timeseries {MFname}")
                    elif AFname in ts:
                        t *= mfactor * ts[AFname][:steps]
                        msg(4, f"AFACTR modified by timeseries {AFname}")
                    else:
                        t *= factor

                    # if poht to iheat, imprecision in hspf conversion factor requires a slight adjustment
                    if smemn in ["POHT", "SOHT"] and tmemn == "IHEAT":
                        t *= 0.998553
                    if smemn in ["PODOXM", "SODOXM"] and tmemn == "OXIF1":
                        t *= 1.000565

                    # ??? ISSUE: can fetched data be at different frequency - don't know how to transform.
                    if tmemn in ts:
                        ts[tmemn] += t
                    else:
                        ts[tmemn] = t

                except KeyError:
                    print("ERROR in FLOWS, cant resolve ", path + " " + smemn)

    return
