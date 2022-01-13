import mando

from HSP2.main import main as hsp2main
from HSP2tools.readUCI import readUCI
from HSP2tools.readWDM import readWDM
from HSP2IO.hdf import HDF5
from HSP2IO.io import IOManager


@mando.command(doctype="numpy")
def run(hdfname, saveall=True, jupyterlab=False):
    """Run a HSPsquared model.

    Parameters
    ----------
    hdfname: str
        HDF5 (path) filename used for both input and output.
    saveall: bool
        [optional] Default is False.
        Saves all calculated data ignoring SAVE tables.
    jupyterlab: bool
        Jupyterlab
    """
    hdf5_instance = HDF5(hdfname)
    io_manager = IOManager(hdf5_instance)
    hsp2main(io_manager, saveall=saveall, jupyterlab=jupyterlab)


@mando.command(doctype="numpy")
def import_uci(ucifile, h5file):
    """Import UCI and WDM files into HDF5 file.

    Parameters
    ----------
    ucifile: str
        The UCI file to import into HDF file.
    h5file: str
        The destination HDF5 file.
    """
    readUCI(ucifile, h5file)

    with open(ucifile, "r") as fp:
        uci = []
        for line in fp.readlines():
            if '***' in line[:81]:
                continue
            if not line[:81].strip():
                continue
            uci.append(line[:81].rstrip())

    files_start = uci.index("FILES")
    files_end = uci.index("END FILES")

    for nline in uci[files_start: files_end+1]:
        if nline[:10].strip() == "WDM":
            readWDM(nline[16:].strip(), h5file)


def main():
    mando.main()


if __name__ == "__main__":
    main()
