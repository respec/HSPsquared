"""The `hsp2` module contains the hydrology and water quality code modules
converted from HSPF, along with the main programs to run HSP2
"""

from hsp2.hsp2.main import main
from hsp2.hsp2.mainDoE import main as mainDoE
from hsp2.hsp2.utilities import flowtype, versions

from hsp2 import __version__
