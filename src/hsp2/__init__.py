"""The Hydrologic Simulation Program - Python (HSP2) watershed model is is a
port of the well-established Hydrological Simulation Program - FORTRAN (HSPF),
re-coded with modern scientific Python and data formats.

Modules:
   - HSP2 contains the hydrology and water quality code modules converted from
   HSPF, along with the main programs to run HSP2.
   - HSP2tools contains supporting software modules such as the code to convert
   legacy WDM and UCI files to HDF5 files for HSP2, and to provide additional
   new and legacy capabilities.
   - HSP2IO is new in v0.10 and contains an abstracted approach to getting data
   in and out of HSP2 for flexibility and performance and also to support future
   automation and model coupling.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hsp2")
except PackageNotFoundError:
    import os

    with open(
        os.path.join(os.path.dirname(__file__), "../..", "VERSION"), encoding="ascii"
    ) as version_file:
        __version__ = version_file.read().strip()
