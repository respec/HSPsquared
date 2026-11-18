"""HBNOutput reads both layouts of the HSPF binary output (HBN) file.

HSPF writes the layout of the Lahey Fortran runtime (first byte 0xFD) through its
compiler independent binary output routines. Builds of HSPF sources that write the
binary output with plain unformatted WRITE statements give, with compilers such as
gfortran, an unformatted sequential file in which each record is framed by 4-byte
length markers. The test rewrites the records of test05.hbn in that layout and
checks that HBNOutput reads the same data from both files.
"""

import os
from struct import pack

import numpy as np
import pandas as pd
from hsp2.hsp2tools.HBNOutput import HBNOutput, record_positions

TEST05_HBN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test05", "HSPFresults", "test05.hbn"
)


def test_unformatted_sequential_layout(tmp_path):
    lahey = np.fromfile(TEST05_HBN, "B")
    positions = record_positions(lahey)
    assert positions

    sequential = bytearray()
    for index, reclen in positions:
        body = lahey[index + 4 : index + 28 + reclen].tobytes()
        sequential += pack("<i", len(body)) + body + pack("<i", len(body))
    path = tmp_path / "test05_sequential.hbn"
    path.write_bytes(bytes(sequential))

    first = HBNOutput(TEST05_HBN)
    first.read_data()
    second = HBNOutput(str(path))
    second.read_data()

    assert len(first.data_frames) > 0
    assert first.summaryindx == second.summaryindx
    assert first.output_dictionary == second.output_dictionary
    for a, b in zip(first.data_frames, second.data_frames):
        pd.testing.assert_frame_equal(a, b)


def test_first_marker_beginning_with_0xfd():
    # one record of 253 bytes: its first length marker begins with the byte 0xFD
    raw = pack("<i", 253) + bytes(253) + pack("<i", 253)
    data = np.frombuffer(raw, dtype=np.uint8)
    assert data[0] == 0xFD
    assert record_positions(data) == [(0, 229)]


def test_unknown_layout(tmp_path):
    path = tmp_path / "not_hbn.hbn"
    path.write_bytes(b"\x01\x02\x03\x04\x05\x06\x07\x08")
    hbn = HBNOutput(str(path))
    hbn.read_data()
    assert hbn.data_frames == []
