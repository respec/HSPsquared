"""Write the input of the test05atemp and test05atemp2h cases from test05.

usage (from the root of the repository):
    python tests/test05atemp/make_cases.py tests/test05/HSPFresults <work folder>

writes into <work folder>/test05atemp and <work folder>/test05atemp2h:
    <case>.uci           the input file of HSPF and HSP2
    list.txt, dsn*.txt   the input data sets at the run step, in the form read by
                         wdm_write of the HSPF build named in the README files,
                         which writes them to <case>.wdm

The changes to test05 are listed in tests/test05atemp/README.md and
tests/test05atemp2h/README.md.
"""

import pathlib
import sys

import numpy as np

from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts

END = "1976/04/30 24:00"

# data sets of test05 used by the cases: TSTYPE, TSFORM and kind of value
DATA_SETS = {
    39: ("PREC", 2, "total"),  # hourly precipitation, PERLND 1 and 2
    131: ("PREC", 2, "total"),  # hourly precipitation, PERLND 3
    123: ("ATMP", 1, "temperature"),  # 2-hour air temperature, PERLND 1 and 2
    122: ("ATMP", 1, "temperature"),  # 2-hour air temperature, PERLND 3
    46: ("SOLR", 2, "total"),  # 2-hour solar radiation
    41: ("EVAP", 2, "total"),  # daily potential evaporation
    42: ("WIND", 2, "total"),  # daily wind movement
    126: ("DEWP", 1, "mean"),  # daily dew point, PERLND 1 and 2
    125: ("DEWP", 1, "mean"),  # daily dew point, PERLND 3
    136: ("FLOW", 1, "mean"),  # daily flow, PLTGEN only
}

CASES = {
    # A1 and E1: 30-minute step, air temperature changing within the hour; the
    # first snowfall of the run begins within an hour, after rain
    "test05atemp": dict(
        step=30,
        start="1976/02/20 00:00",
        title="  test05, ATEMP at a 30-minute step, ELDAT 0, 1111 and -1111 ft, no ice",
        binary=f"{2:5d}{2:5d}" + " " * 50,  # ATEMP and SNOW
    ),
    # A2: 2-hour step
    "test05atemp2h": dict(
        step=120,
        start="1976/01/01 00:00",
        title="  test05, ATEMP at a 2-hour step, ELDAT 0, 1111 and -1111 ft, no ice",
        binary=f"{2:5d}{2:5d}{2:5d}" + " " * 45,  # ATEMP, SNOW and PWATER
    ),
}


def make_uci(text, case, step, start, title, binary):
    text = text.replace("\r\n", "\n")

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    rep(
        "  Version 11 test run:  PERLND w/ SNOW, PWATER, COPY, DISPLY, PLTGEN\n",
        title + "\n",
    )
    rep(
        "  START       1976/01/01 00:00  END    1976/12/31 24:00\n",
        f"  START       {start}  END    {END}\n",
    )
    rep(
        "WDM        21   test.wdm\n"
        "MESSU      22   test05.ech\n"
        "            1   test05.out\n"
        "           62   test05.d62\n"
        "           92   test05.p92\n"
        "BINO       93   test05.hbn\n",
        f"WDM        21   {case}.wdm\n"
        f"MESSU      22   {case}.ech\n"
        f"            1   {case}.out\n"
        f"           92   {case}.p92\n"
        f"BINO       93   {case}.hbn\n",
    )
    rep(
        "    INGRP              INDELT 00:30\n",
        f"    INGRP              INDELT {step // 60:02d}:{step % 60:02d}\n",
    )
    # ATEMP active (test05: inactive, air temperature read as AIRTMP)
    rep(
        "    # -  # ATMP SNOW PWAT  SED  PST  PWG PQAL MSTL PEST NITR PHOS TRAC ***\n"
        "    1    3         1    1\n",
        "    # -  # ATMP SNOW PWAT  SED  PST  PWG PQAL MSTL PEST NITR PHOS TRAC ***\n"
        "    1    3    1    1    1\n",
    )
    # binary output at every interval (flag 2, PIVL 1)
    rep(
        "    1    3         4    4" + " " * 53 + "12\n  END BINARY-INFO",
        "    1    3" + binary + f"{1:5d}{12:5d}\n  END BINARY-INFO",
    )
    # ICEFG 0: the ground-ice items of issue 232 (F2, P1) do not enter these cases;
    # ATEMP-DAT: the land segment 1111 ft above (PERLND 2) and below (PERLND 3) the
    # temperature gage, PERLND 1 at its height. With a round height difference the
    # air temperature falls exactly on PETMIN in some intervals, where the single
    # precision of HSPF and the double precision of HSP2 can decide differently.
    rep(
        "  ICE-FLAG\n",
        "  ATEMP-DAT\n"
        "    <PLS >     ELDAT    AIRTMP ***\n"
        "    # -  #      (ft)   (deg F) ***\n"
        "    1             0.       20.\n"
        "    2          1111.       20.\n"
        "    3         -1111.       20.\n"
        "  END ATEMP-DAT\n"
        "\n"
        "  ICE-FLAG\n",
    )
    rep(
        "    # -  #ICEFG" + " " * 42 + "***\n    1    3    1\n  END ICE-FLAG",
        "    # -  #ICEFG" + " " * 42 + "***\n    1    3    0\n  END ICE-FLAG",
    )
    # air temperature of the gage goes to ATEMP (EXTNL GATMP), which computes AIRTMP
    for dsn, pls in ((123, "1   2"), (122, "3    ")):
        head = f"WDM    {dsn} ATMP     ENGL              SAME PERLND   {pls} "
        rep(head + "ATEMP  AIRTMP\n", head + "EXTNL  GATMP\n")
    # the data sets are written at the run step: no transformation (HSPF does not
    # accept DIV between equal time steps)
    assert text.count("DIV  PERLND") == 6
    text = text.replace("DIV  PERLND", "     PERLND")
    # DISPLY dropped: HSP2 does not run it, and the gfortran build of HSPF used for
    # the reference output stops in its annual summary
    a = text.index("\nDISPLY\n")
    b = text.index("END DISPLY\n") + len("END DISPLY\n")
    text = text[: a + 1] + text[b:]
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if not (line.strip() == "DISPLY       1" or "DISPLY   1     INPUT" in line)
    ]
    return "\n".join(lines) + "\n"


def at_step(series, kind, step):
    """Values of 1976 of a test05 data set at a step of `step` minutes.

    Totals are divided evenly when the data step is longer than `step` and summed
    when it is shorter; means are repeated or averaged. Air temperature at a step
    shorter than its 2 hours is interpolated linearly in time, each 2-hour mean
    placed at the middle of its interval, so that it changes within the hour.
    """
    s = series["1976-01-01":"1976-12-31 23:59"].astype(float)
    data_step = int((s.index[1] - s.index[0]).total_seconds() // 60)
    n = 366 * 1440 // step
    if data_step == step:
        v = s.to_numpy()
    elif data_step > step:
        k = data_step // step
        if kind == "temperature":
            t_data = np.arange(len(s)) * data_step + data_step / 2.0
            t_run = np.arange(n) * step + step / 2.0
            v = np.interp(t_run, t_data, s.to_numpy())
        elif kind == "total":
            v = np.repeat(s.to_numpy() / k, k)
        else:
            v = np.repeat(s.to_numpy(), k)
    else:
        k = step // data_step
        a = s.to_numpy().reshape(-1, k)
        v = a.sum(axis=1) if kind == "total" else a.mean(axis=1)
    assert len(v) == n and np.isfinite(v).all()
    return v


if __name__ == "__main__":
    src, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    with open(src / "TEST05bin.UCI") as f:
        test05 = f.read()
    data = read_wdm_ts(src / "TEST.WDM", dsns=None)
    for case, c in CASES.items():
        out = work / case
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{case}.uci", "w", newline="\n") as f:
            f.write(
                make_uci(test05, case, c["step"], c["start"], c["title"], c["binary"])
            )
        with open(out / "list.txt", "w", newline="\n") as lst:
            for dsn, (tstype, tsform, kind) in DATA_SETS.items():
                v = at_step(data[f"TS{dsn:03d}"], kind, c["step"])
                # time step and unit of the data set: minutes (2) below an hour,
                # hours (3) otherwise
                step, unit = (c["step"], 2) if c["step"] < 60 else (c["step"] // 60, 3)
                with open(out / f"dsn{dsn}.txt", "w", newline="\n") as f:
                    # first line: start of the first interval, number of values,
                    # time step and time unit
                    f.write(f"1976 1 1 0 0 {len(v)} {step} {unit}\n")
                    f.write("\n".join(f"{x:.9g}" for x in v) + "\n")
                lst.write(f"{dsn} '{tstype}' {tsform} 'dsn{dsn}.txt'\n")
