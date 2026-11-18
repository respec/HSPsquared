"""Write the input file of the test05monthly case from the input file of test05.

usage (from the root of the repository):
    python tests/test05monthly/make_case.py tests/test05/HSPFresults/TEST05bin.UCI \
        tests/test05monthly/HSPFresults/test05monthly.uci

The changes to test05 are listed in tests/test05monthly/README.md.
"""

import sys

START, END = "1976/04/01 00:00", "1976/07/31 24:00"


def make_case(text):
    text = text.replace("\r\n", "\n")

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    rep(
        "  Version 11 test run:  PERLND w/ SNOW, PWATER, COPY, DISPLY, PLTGEN\n",
        "  test05 at a 1-hour step, monthly tables left out while their flags are on\n",
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
        "WDM        21   test05monthly.wdm\n"
        "MESSU      22   test05monthly.ech\n"
        "            1   test05monthly.out\n"
        "           92   test05monthly.p92\n"
        "BINO       93   test05monthly.hbn\n",
    )
    rep(
        "    INGRP              INDELT 00:30\n", "    INGRP              INDELT 01:00\n"
    )
    # binary output of PWATER at every interval (flag 2, PIVL 1)
    rep(
        "    1    3         4    4" + " " * 53 + "12\n  END BINARY-INFO",
        "    1    3" + " " * 10 + f"{2:5d}" + " " * 45 + f"{1:5d}{12:5d}"
        "\n  END BINARY-INFO",
    )
    # constant values that differ from the defaults of the monthly tables: NSUR 0.3
    # on PERLND 1, CEPSC 0.1 on PERLND 2, LZETP 0.5 on PERLND 3
    rep(
        "    1    2                0.01       0.1       1.0      0.60\n"
        "    3                     0.01       0.1       1.2      0.80\n"
        "  END PWAT-PARM4",
        "    1                     0.01       0.3       1.0      0.60\n"
        "    2            0.1      0.01       0.1       1.0      0.60\n"
        "    3                     0.01       0.1       1.2      0.80       0.5\n"
        "  END PWAT-PARM4",
    )
    # monthly tables whose flag is on, left out: MON-INTERCEP (VCSFG), MON-MANNING
    # (VNNFG) and MON-LZETPARM (VLEFG); HSPF then uses the defaults of the tables
    for table in ("MON-INTERCEP", "MON-MANNING", "MON-LZETPARM"):
        a = text.index(f"  {table}\n")
        b = text.index(f"  END {table}\n") + len(f"  END {table}\n") + 1
        text = text[:a] + text[b:]
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
        uci = make_case(f.read())
    with open(sys.argv[2], "w", newline="\n") as f:
        f.write(uci)
