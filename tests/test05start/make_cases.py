"""Write the input files of the test05start and test05start30 cases from the input
file of test05.

usage (from the root of the repository):
    python tests/test05start/make_cases.py tests/test05/HSPFresults/TEST05bin.UCI \
        tests

writes tests/test05start/HSPFresults/test05start.uci and
tests/test05start30/HSPFresults/test05start30.uci. The changes to test05 are listed
in the README files of the two cases.
"""

import pathlib
import sys


def make_case(text, case):
    text = text.replace("\r\n", "\n")

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    if case == "test05start":
        title = "  test05 at a 1-hour step, from 06:00 to a December end, initial packs"
        start, end = "1976/11/15 06:00", "1976/12/15 24:00"
        binary = f"{2:5d}{2:5d}" + " " * 45  # SNOW and PWATER
    else:
        title = (
            "  test05, 30-minute step, start at 00:30, no ice: PERLND w/ SNOW, PWATER"
        )
        start, end = "1976/01/02 00:30", "1976/01/31 24:00"
        binary = f"{2:5d}" + " " * 50  # SNOW
    rep(
        "  Version 11 test run:  PERLND w/ SNOW, PWATER, COPY, DISPLY, PLTGEN\n",
        title + "\n",
    )
    rep(
        "  START       1976/01/01 00:00  END    1976/12/31 24:00\n",
        f"  START       {start}  END    {end}\n",
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
    # binary output at every interval (flag 2, PIVL 1)
    rep(
        "    1    3         4    4" + " " * 53 + "12\n  END BINARY-INFO",
        "    1    3" + " " * 5 + binary + f"{1:5d}{12:5d}\n  END BINARY-INFO",
    )
    if case == "test05start":
        rep(
            "    INGRP              INDELT 00:30\n",
            "    INGRP              INDELT 01:00\n",
        )
        # initial pack: thin, with partial cover, on PERLND 1; none, with an
        # initial freezing capacity XLNMLT of 0.5 in, on PERLND 2; as in test05 on
        # PERLND 3
        rep(
            "    1    3       1.4       0.2       0.1       0.2      375.      27.5\n"
            "  END SNOW-INIT1",
            "    1            0.1       0.0       0.0       0.2      375.      27.5\n"
            "    2            0.0       0.0       0.0       0.2      375.      27.5\n"
            "    3            1.4       0.2       0.1       0.2      375.      27.5\n"
            "  END SNOW-INIT1",
        )
        rep(
            "    1    3      0.50       0.0       1.0\n  END SNOW-INIT2",
            "    1           0.50       0.0       1.0\n"
            "    2           0.50       0.5       1.0\n"
            "    3           0.50       0.0       1.0\n"
            "  END SNOW-INIT2",
        )
        # interflow storage at the start on PERLND 3 (test05: 0.0), so that the
        # interflow recession, computed when the day flag is on, acts in the first
        # interval
        rep(
            "    3           0.05       0.0      0.30       0.0       6.5      0.15"
            "      0.10\n",
            "    3           0.05       0.0      0.30       0.1       6.5      0.15"
            "      0.10\n",
        )
    else:
        # ICEFG 0: the ground-ice items of a sub-hourly step (F2 and P1 of issue
        # 232) do not enter this case
        rep(
            "    # -  #ICEFG" + " " * 42 + "***\n    1    3    1\n  END ICE-FLAG",
            "    # -  #ICEFG" + " " * 42 + "***\n    1    3    0\n  END ICE-FLAG",
        )
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


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        test05 = f.read()
    tests = pathlib.Path(sys.argv[2])
    for case in ("test05start", "test05start30"):
        with open(tests / case / "HSPFresults" / f"{case}.uci", "w", newline="\n") as f:
            f.write(make_case(test05, case))
