"""Write the input file of the test05ice case from the input file of test05.

usage (from the root of the repository):
    python tests/test05ice/make_case.py tests/test05/HSPFresults/TEST05bin.UCI \
        tests/test05ice/HSPFresults/test05ice.uci

The changes to test05 are listed in tests/test05ice/README.md.
"""

import sys

START, END = "1976/01/02 00:00", "1976/04/30 24:00"


def make_case(text):
    text = text.replace("\r\n", "\n")

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)

    rep(
        "  Version 11 test run:  PERLND w/ SNOW, PWATER, COPY, DISPLY, PLTGEN\n",
        "  test05, ground ice at a 30-minute step: PERLND w/ SNOW, PWATER\n",
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
        "WDM        21   test05ice.wdm\n"
        "MESSU      22   test05ice.ech\n"
        "            1   test05ice.out\n"
        "           92   test05ice.p92\n"
        "BINO       93   test05ice.hbn\n",
    )
    # binary output of SNOW and PWATER at every interval (flag 2, PIVL 1)
    rep(
        "    1    3         4    4" + " " * 53 + "12\n  END BINARY-INFO",
        "    1    3" + " " * 5 + f"{2:5d}{2:5d}" + " " * 45 + f"{1:5d}{12:5d}"
        "\n  END BINARY-INFO",
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
        uci = make_case(f.read())
    with open(sys.argv[2], "w", newline="\n") as f:
        f.write(uci)
