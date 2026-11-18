"""Rewrite HSPF binary output written by a gfortran build in the layout of the Windows build.

A gfortran build of HSPF writes the binary output file (.hbn) as unformatted sequential
records, each framed by 4-byte length markers. hsp2.hsp2tools.HBNOutput reads the layout
of the Windows build: a first byte 0xFD, then for every record a 4-byte little-endian lead
word (length << 2 | 3), the record body, and a trailing word ((length + 4) << 2 | (n - 1))
of n bytes, big-endian, n being the smallest number of bytes (1 to 4) that holds it. The
record bodies are copied unchanged.

usage: python tests/convert/hbn_to_windows.py <gfortran.hbn> <windows.hbn>
"""

import sys
from struct import pack, unpack


def gfortran_records(buf):
    i = 0
    while i < len(buf):
        n = unpack("<i", buf[i : i + 4])[0]
        body = buf[i + 4 : i + 4 + n]
        assert unpack("<i", buf[i + 4 + n : i + 8 + n])[0] == n, f"bad marker at {i}"
        yield body
        i += 8 + n


def windows_record(body):
    lead = pack("<I", (len(body) << 2) | 3)
    total = len(body) + 4
    nbytes = next(nb for nb in (1, 2, 3, 4) if total < 1 << (8 * nb - 2))
    trail = ((total << 2) | (nbytes - 1)).to_bytes(nbytes, "big")
    return lead + body + trail


def convert(src, dst):
    with open(src, "rb") as f:
        buf = f.read()
    out = bytearray(b"\xfd")
    for body in gfortran_records(buf):
        out += windows_record(body)
    with open(dst, "wb") as f:
        f.write(bytes(out))


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
