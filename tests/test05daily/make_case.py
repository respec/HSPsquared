"""Write the input of the test05daily case from the input of test05.

usage (from the root of the repository):
    python tests/test05daily/make_case.py tests/test05/HSPFresults <work folder>

writes into the work folder:
    test05daily.uci      the input file of HSPF and HSP2
    list.txt, dsn*.txt   the daily data sets, in the form read by wdm_write of the
                         HSPF build named in tests/test05daily/README.md, which
                         writes them to test05daily.wdm

The changes to test05 are listed in tests/test05daily/README.md.
"""

import pathlib
import sys

from hsp2.hsp2io.time_series_readers.read_wdm_ts import read_wdm_ts

START, END = "1976/01/01 00:00", "1976/12/31 24:00"

# data sets of test05 used by the case: (TSTYPE, TSFORM, rule to a daily value)
DATA_SETS = {
    39: ("PREC", 2, "sum"),  # hourly precipitation, PERLND 1 and 2
    131: ("PREC", 2, "sum"),  # hourly precipitation, PERLND 3
    123: ("ATMP", 1, "mean"),  # 2-hour air temperature, PERLND 1 and 2
    122: ("ATMP", 1, "mean"),  # 2-hour air temperature, PERLND 3
    46: ("SOLR", 2, "sum"),  # 2-hour solar radiation
    41: ("EVAP", 2, "day"),  # daily potential evaporation
    42: ("WIND", 2, "day"),  # daily wind movement
    126: ("DEWP", 1, "day"),  # daily dew point, PERLND 1 and 2
    125: ("DEWP", 1, "day"),  # daily dew point, PERLND 3
    136: ("FLOW", 1, "day"),  # daily flow, PLTGEN only
}


def make_uci(text):
    text = text.replace("\r\n", "\n")

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    rep(
        "  Version 11 test run:  PERLND w/ SNOW, PWATER, COPY, DISPLY, PLTGEN\n",
        "  test05 at a daily step, LAT -43 on PERLND 2, high SNOEVP on PERLND 3,"
        " no ice\n",
    )
    rep(
        "  START       1976/01/01 00:00  END    1976/12/31 24:00\n",
        f"  START       {START}  END    {END}\n",
    )
    rep(
        "WDM        21   test.wdm\n"
        "MESSU      22   test05.ech\n"
        "            1   test05.out\n"
        "           62   test05.d62\n"
        "           92   test05.p92\n"
        "BINO       93   test05.hbn\n",
        "WDM        21   test05daily.wdm\n"
        "MESSU      22   test05daily.ech\n"
        "            1   test05daily.out\n"
        "           92   test05daily.p92\n"
        "BINO       93   test05daily.hbn\n",
    )
    rep(
        "    INGRP              INDELT 00:30\n", "    INGRP              INDELT 24:00\n"
    )
    # binary output of SNOW and PWATER at every interval (flag 2, PIVL 1)
    rep(
        "    1    3         4    4" + " " * 53 + "12\n  END BINARY-INFO",
        "    1    3" + " " * 5 + f"{2:5d}{2:5d}" + " " * 45 + f"{1:5d}{12:5d}"
        "\n  END BINARY-INFO",
    )
    # ICEFG 0: ground ice is not simulated at steps longer than 6 hours
    rep(
        "    # -  #ICEFG" + " " * 42 + "***\n    1    3    1\n  END ICE-FLAG",
        "    # -  #ICEFG" + " " * 42 + "***\n    1    3    0\n  END ICE-FLAG",
    )
    # PERLND 1 as in test05; PERLND 2 south of the equator (LAT -43); PERLND 3 with
    # COVIND 0.1 in and SNOEVP 1.0 (the upper limit), so that a thin pack covers
    # the whole segment and a day of evaporation can remove it
    rep(
        "    1    3       43.     1225.       0.0      1.45       0.5\n"
        "  END SNOW-PARM1",
        "    1            43.     1225.       0.0      1.45       0.5\n"
        "    2           -43.     1225.       0.0      1.45       0.5\n"
        "    3            43.     1225.       0.0      1.45       0.1\n"
        "  END SNOW-PARM1",
    )
    rep(
        "    1    3      0.12       32.      0.05       0.5      0.08    0.0001\n"
        "  END SNOW-PARM2",
        "    1    2      0.12       32.      0.05       0.5      0.08    0.0001\n"
        "    3           0.12       32.       1.0       0.5      0.08    0.0001\n"
        "  END SNOW-PARM2",
    )
    # the data sets are daily, as the run: no transformation (HSPF does not accept
    # DIV between equal time steps)
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


def daily_values(series, rule):
    """Daily values of 1976 from an hourly, 2-hour or daily series of test05."""
    s = series["1976-01-01":"1976-12-31 23:59"].astype(float)
    if rule == "day":
        d = s
    else:
        g = s.groupby(s.index.normalize())
        d = g.sum() if rule == "sum" else g.mean()
    assert len(d) == 366 and not d.isna().any()
    return d.to_numpy()


if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with open(src / "TEST05bin.UCI") as f:
        uci = make_uci(f.read())
    with open(out / "test05daily.uci", "w", newline="\n") as f:
        f.write(uci)
    data = read_wdm_ts(src / "TEST.WDM", dsns=None)
    with open(out / "list.txt", "w", newline="\n") as lst:
        for dsn, (tstype, tsform, rule) in DATA_SETS.items():
            v = daily_values(data[f"TS{dsn:03d}"], rule)
            with open(out / f"dsn{dsn}.txt", "w", newline="\n") as f:
                # first line: start of the first interval, number of values,
                # time step and time unit (4 = day)
                f.write(f"1976 1 1 0 0 {len(v)} 1 4\n")
                f.write("\n".join(f"{x:.7g}" for x in v) + "\n")
            lst.write(f"{dsn} '{tstype}' {tsform} 'dsn{dsn}.txt'\n")
