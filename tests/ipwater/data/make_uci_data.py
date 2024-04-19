import json
import os

from pathlib import Path

from HSP2tools import readUCI
from HSP2IO.hdf import HDF5

this_dir = Path(__file__).parent


def dump_uci_data(uci_file, hdf_file):
    readUCI(uci_file, hdf_file)

    UCI = HDF5(hdf_file).read_uci()
    uci = {}
    uci["IMPLND"] = UCI.uci[("IMPLND", "IWATER", "I001")]
    uci["PERLND"] = UCI.uci[("PERLND", "PWATER", "P001")]
    uci["siminfo"] = UCI.siminfo

    with open("./uci_data.json", "w") as f:
        json.dump(uci, f, indent=2, default=str)

    os.remove(hdf_file)


if __name__ == "__main__":
    uci_file = this_dir / "HSPF_Test240.uci"
    hdf_file = this_dir / "hdf_test.h5"

    dump_uci_data(uci_file, hdf_file)
